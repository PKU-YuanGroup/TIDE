"""
Timestep-Aware Interpolation Distillation (TAID) loss for diffusion LMs.

Creates an interpolated soft target by mixing student and teacher logits
with a time-adaptive lambda schedule, then distills the student toward
that target via KL divergence.  The lambda grows over training (shifting
from student-reliant to teacher-reliant) and is further modulated by the
diffusion timestep so that nearly-clean positions lean more on the teacher.

References:
- TAID: https://arxiv.org/abs/2501.16937
"""

import math

import torch
import torch.nn.functional as F


def compute_taid_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    masked_mask: torch.BoolTensor,
    t: torch.Tensor,
    config: dict,
    global_step: int,
    max_steps: int,
    entropy_weights: torch.Tensor = None,
) -> torch.Tensor:
    """
    Compute TAID distillation loss on masked positions.

    The loss creates a soft interpolated target from teacher and student
    logits, then minimises KL(interpolated_target || student).  A growing
    lambda controls how much the target leans on the teacher vs the student
    over the course of training, and is further scaled per-sample by the
    diffusion timestep.

    Args:
        student_logits: Student model logits, shape ``[b, l, V]``.
        teacher_logits: Teacher model logits, shape ``[b, l, V]``.
        masked_mask: Boolean mask of noised (masked) positions, ``[b, l]``.
        t: Per-sample diffusion timestep (masking ratio), ``[b]``.
        config: Dict with keys:
            - ``lambda_init``  (float): initial lambda (default 0.1).
            - ``lambda_max``   (float): final lambda (default 0.9).
            - ``lambda_decay`` (str): ``"cosine"`` or ``"linear"``.
            - ``timestep_weight`` (str): ``"uniform"`` or ``"midrange"``.
            - ``shared_vocab_size`` (int | None): slice both logits to this
              vocab size when teacher/student vocabs differ.
            - ``temperature`` (float, optional): softmax temperature
              (default 2.0).
        global_step: Current optimiser step.
        max_steps: Total number of training steps.
        entropy_weights: Optional ``[b, l]`` per-token weights derived from
            teacher entropy (higher for confident teacher predictions).
            When provided, each masked position's KL loss is scaled by
            the corresponding weight.  Weights should be pre-normalised
            to mean 1.0 over masked positions.

    Returns:
        Scalar TAID loss (already multiplied by T**2).
    """
    # --- unpack config ---
    lambda_init = config.get("lambda_init", 0.1)
    lambda_max = config.get("lambda_max", 0.9)
    lambda_decay = config.get("lambda_decay", "cosine")
    timestep_weight = config.get("timestep_weight", "uniform")
    shared_vocab_size = config.get("shared_vocab_size", None)
    T = config.get("temperature", 2.0)

    device = student_logits.device

    # --- edge case: no masked positions ---
    if not masked_mask.any():
        return torch.tensor(0.0, device=device)

    # --- optional vocab slicing ---
    if shared_vocab_size is not None:
        student_logits = student_logits[:, :, :shared_vocab_size]
        teacher_logits = teacher_logits[:, :, :shared_vocab_size]

    # --- 1. training-level lambda schedule ---
    # growth_val goes from 0 → 1 over training, so lambda goes from lambda_init → lambda_max
    training_progress = global_step / max(max_steps, 1)
    if lambda_decay == "cosine":
        growth_val = 0.5 * (1.0 - math.cos(math.pi * training_progress))
    elif lambda_decay == "linear":
        growth_val = training_progress
    else:
        raise ValueError(f"Unknown lambda_decay: {lambda_decay}")
    lambda_train = lambda_init + (lambda_max - lambda_init) * growth_val

    # --- 2. modulate by diffusion timestep ---
    # t ≈ 0 → mostly unmasked → use more teacher (higher λ)
    # t ≈ 1 → mostly masked   → rely more on student (lower λ)
    axis_mode = config.get("axis_mode", "both")
    if axis_mode == "both":
        lambda_t = lambda_train * (1.0 - t)  # [b] — full dual-axis
    elif axis_mode == "training_only":
        lambda_t = lambda_train * torch.ones_like(t)  # [b] — no timestep modulation
    elif axis_mode == "timestep_only":
        lambda_t = lambda_max * (1.0 - t)  # [b] — no training progression
    else:
        raise ValueError(f"Unknown taid_axis_mode: {axis_mode}")

    # --- 3. Gather only masked positions to save memory ---
    # Instead of computing on full [b, l, V] and masking at the end,
    # select only masked positions first to avoid OOM on large vocab.
    b, l, V = student_logits.shape
    mask_flat = masked_mask.reshape(-1)  # [b*l]
    n_masked = mask_flat.sum()

    # Per-position batch index to look up lambda_t and t per masked position
    batch_idx = (
        torch.arange(b, device=device).unsqueeze(1).expand(b, l).reshape(-1)
    )  # [b*l]
    batch_idx_masked = batch_idx[mask_flat]  # [n_masked]

    # Select only masked positions — [n_masked, V]
    s_masked = student_logits.reshape(b * l, V)[mask_flat]
    t_masked = teacher_logits.reshape(b * l, V)[mask_flat]

    # Per-position lambda — [n_masked, 1]
    lam = lambda_t[batch_idx_masked].float().unsqueeze(-1)

    # --- 4. interpolated target (logit mixture) ---
    # Cast to float32 for numerical stability in softmax/KL
    s_scaled = (s_masked / T).float()  # [n_masked, V]
    t_scaled = (t_masked / T).float()  # [n_masked, V]
    del s_masked, t_masked

    mixed_logits = (1.0 - lam) * s_scaled + lam * t_scaled  # [n_masked, V]
    del t_scaled
    r_t = F.softmax(mixed_logits, dim=-1).detach()  # [n_masked, V]
    del mixed_logits

    # --- 5. per-position KL(r_t || student) ---
    log_p_student = F.log_softmax(s_scaled, dim=-1)  # [n_masked, V]
    del s_scaled
    # KL(p || q) = sum p * (log p - log q)
    kl_per_pos = (r_t * (r_t.clamp_min(1e-8).log() - log_p_student)).sum(
        dim=-1
    )  # [n_masked]
    del r_t, log_p_student

    # --- 6. timestep-aware weighting ---
    t_per_pos = t[batch_idx_masked]  # [n_masked]
    if timestep_weight == "uniform":
        w = torch.ones_like(t_per_pos)
    elif timestep_weight == "midrange":
        # Gaussian bell curve centred at t=0.5, σ=0.15
        w = torch.exp(-((t_per_pos - 0.5) ** 2) / (2 * 0.15**2))
    else:
        raise ValueError(f"Unknown timestep_weight: {timestep_weight}")

    # --- 6b. entropy-based per-token weighting ---
    if entropy_weights is not None:
        ew_flat = entropy_weights.reshape(-1)[mask_flat]  # [n_masked]
        w = w * ew_flat

    # --- 7. average over masked positions ---
    taid_loss = (kl_per_pos * w).sum() / n_masked.clamp_min(1)

    # --- 8. temperature scaling ---
    taid_loss = taid_loss * (T * T)

    return taid_loss
