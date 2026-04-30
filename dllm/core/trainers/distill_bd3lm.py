"""
Distillation trainer for A2D BD3LM supporting multiple modes:
- ALM (cross-tokenizer): LLaDA2.0 teacher with chunk-level likelihood matching via tokenkit
- reverse_ALM (cross-tokenizer): ALM with reversed BCE direction -(s*log(t) + (1-s)*log(1-t))
- ALM_TAID (cross-tokenizer): ALM with TAID-style lambda decay and timestep-aware weighting
- KL, reverse_KL, TAID (same-tokenizer): direct logit distillation
- *_aligned variants (e.g., kl_aligned, taid_aligned): different chat templates with
  position-level alignment — any mode except ALM can be used with alignment data
- Composite modes via '+' (e.g., "taid+kl")

References:
- Block Diffusion: https://arxiv.org/abs/2503.09573
- ALM / tokenkit: cross-tokenizer knowledge distillation via chunk-level likelihood matching
- WeDLM: https://arxiv.org/abs/2502.11475
"""

import math
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from .bd3lm import BD3LMConfig, BD3LMTrainer, _create_bd3lm_attention_mask


class DistillBD3LMTrainer(BD3LMTrainer):

    @dataclass
    class DistillBD3LMConfig(BD3LMConfig):
        distill_mode: str = (
            "alm"  # "alm", "reverse_alm", "alm_taid", "reverse_alm_taid", "kl", "reverse_kl", "taid", or *_aligned variants
        )
        concat_order: str = (
            "xt_x0"  # teacher concat order: "xt_x0" or "x0_xt" (student always [xt, x0])
        )
        # ALM-specific
        alm_weight: float = 1.0
        alm_temperature: float = 2.0
        # KL-specific
        kl_weight: float = 1.0
        kl_temperature: float = 2.0
        shared_vocab_size: int = None
        # Shared
        teacher_mask_token_id: int = None
        # --- TAID (Timestep-Aware Interpolation Distillation) ---
        taid_lambda_init: float = 0.1  # initial lambda (near student)
        taid_lambda_max: float = 0.9  # final lambda (near teacher)
        taid_lambda_decay: str = "cosine"  # "cosine" or "linear"
        taid_timestep_weight: str = (
            "uniform"  # "uniform" or "midrange" (higher weight for t in [0.3, 0.7])
        )
        taid_weight: float = 1.0  # weight for TAID loss term
        taid_axis_mode: str = (
            "both"  # "both", "training_only", or "timestep_only" (for axis ablation)
        )
        # --- Complementary Demonstration-Conditioned Denoising ---
        use_complementary_demo: bool = (
            False  # enable 2-pass teacher with complementary demos
        )
        demo_ratio: float = (
            0.5  # fraction of M assigned to M_A (0=standard, 0.5=symmetric)
        )

    def __init__(
        self,
        args: "DistillBD3LMTrainer.DistillBD3LMConfig",
        teacher_model: nn.Module,
        teacher_tokenizer: Any = None,
        *pargs,
        **kwargs,
    ):
        super().__init__(args=args, *pargs, **kwargs)
        self.teacher_model = teacher_model
        self.teacher_tokenizer = teacher_tokenizer
        self.distill_mode = args.distill_mode
        self.concat_order = args.concat_order
        # ALM
        self.alm_weight = args.alm_weight
        self.alm_temperature = args.alm_temperature
        # KL
        self.kl_weight = args.kl_weight
        self.kl_temperature = args.kl_temperature
        self.teacher_mask_token_id = args.teacher_mask_token_id
        self.shared_vocab_size = args.shared_vocab_size
        # TAID
        self.taid_lambda_init = args.taid_lambda_init
        self.taid_lambda_max = args.taid_lambda_max
        self.taid_lambda_decay = args.taid_lambda_decay
        self.taid_timestep_weight = args.taid_timestep_weight
        self.taid_weight = args.taid_weight
        self.taid_axis_mode = args.taid_axis_mode
        # Complementary demo
        self.use_complementary_demo = args.use_complementary_demo
        self.demo_ratio = args.demo_ratio

    def _split_complementary_masks(self, masked_mask):
        """Split masked positions into two complementary subsets M_A, M_B.

        Each sample in the batch gets an independent random split.
        M_A | M_B = M, M_A & M_B = empty.
        demo_ratio controls |M_A| / |M|.
        """
        random_assign = torch.rand_like(masked_mask.float()) < self.demo_ratio
        M_A = masked_mask & random_assign
        M_B = masked_mask & ~random_assign
        return M_A, M_B

    @staticmethod
    def log1mexp(x: torch.Tensor) -> torch.Tensor:
        """Numerically stable computation of log(1 - exp(x)) for x <= 0."""
        # For x close to 0, use log1p(-exp(x)); for x << 0, use log(-expm1(x))
        mask = x > -0.6931  # -ln(2)
        result = torch.empty_like(x)
        result[mask] = torch.log1p(-torch.exp(x[mask]))
        result[~mask] = torch.log(-torch.expm1(x[~mask]))
        return result

    @staticmethod
    def compute_alm_loss(
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_input_ids: torch.Tensor,
        teacher_input_ids: torch.Tensor,
        alignment_matrix_student: torch.Tensor,
        alignment_matrix_teacher: torch.Tensor,
        chunk_mask: torch.Tensor,
        temperature: float = 2.0,
        chunk_weights: torch.Tensor = None,
        reverse: bool = False,
    ) -> torch.Tensor:
        """
        Compute ALM (Approximate Likelihood Matching) chunk-level distillation loss.

        Aggregates per-token log-probs of correct tokens into chunk-level log-probs
        via alignment matrices, then applies a BCE-like distance with temperature scaling.

        Args:
            student_logits: [b, l_s, V_s] student model logits
            teacher_logits: [b, l_t, V_t] teacher model logits
            student_input_ids: [b, l_s] student clean token ids
            teacher_input_ids: [b, l_t] teacher clean token ids
            alignment_matrix_student: [b, l_s, n_chunks] bool
            alignment_matrix_teacher: [b, l_t, n_chunks] bool
            chunk_mask: [b, n_chunks] bool — which chunks are masked (active for loss)
            temperature: temperature scaling for chunk probabilities
            reverse: if True, use reverse BCE -(s*log(t) + (1-s)*log(1-t))
                     instead of forward BCE -(t*log(s) + (1-t)*log(1-s))

        Returns:
            Scalar ALM loss.
        """
        # Per-token log-probs of correct tokens
        # Memory-efficient: gather + logsumexp instead of full log_softmax + gather
        # log_softmax(x)[i] = x[i] - logsumexp(x), avoids allocating full [b, l, V] output
        student_token_logits = student_logits.gather(
            2, student_input_ids.unsqueeze(-1)
        ).squeeze(
            -1
        )  # [b, l_s]
        student_token_lp = student_token_logits - torch.logsumexp(
            student_logits, dim=-1
        )

        with torch.no_grad():
            teacher_token_logits = teacher_logits.gather(
                2, teacher_input_ids.unsqueeze(-1)
            ).squeeze(
                -1
            )  # [b, l_t]
            teacher_token_lp = teacher_token_logits - torch.logsumexp(
                teacher_logits, dim=-1
            )

        # Aggregate to chunk-level via alignment matrices (sum of log-probs within each chunk)
        # student_token_lp: [b, l_s], alignment_matrix_student: [b, l_s, n_chunks]
        align_s = alignment_matrix_student.to(
            dtype=student_token_lp.dtype
        )  # [b, l_s, n_chunks]
        align_t = alignment_matrix_teacher.to(
            dtype=teacher_token_lp.dtype
        )  # [b, l_t, n_chunks]

        student_chunk_lp = torch.bmm(student_token_lp.unsqueeze(1), align_s).squeeze(
            1
        )  # [b, n_chunks]
        with torch.no_grad():
            teacher_chunk_lp = torch.bmm(
                teacher_token_lp.unsqueeze(1), align_t
            ).squeeze(
                1
            )  # [b, n_chunks]

        # BCE distance with temperature on chunk probabilities
        T = temperature
        # Cast to float32 for numerical stability (bfloat16 cannot represent 1 - 1e-7 != 1)
        s_chunk_lp_scaled = (student_chunk_lp / T).float()
        t_chunk_lp_scaled = (teacher_chunk_lp / T).float()

        # Convert log-probs to probs, clamped for numerical stability
        s_p = torch.exp(s_chunk_lp_scaled).clamp(1e-7, 1 - 1e-7)
        t_p = torch.exp(t_chunk_lp_scaled).clamp(1e-7, 1 - 1e-7)

        if reverse:
            # Reverse BCE: -(s * log(t) + (1-s) * log(1-t))
            alm_loss_per_chunk = -(
                s_p * torch.log(t_p) + (1 - s_p) * torch.log(1 - t_p)
            )
        else:
            # Forward BCE: -(t * log(s) + (1-t) * log(1-s))
            alm_loss_per_chunk = -(
                t_p * torch.log(s_p) + (1 - t_p) * torch.log(1 - s_p)
            )

        # Apply optional chunk-level weights (e.g., entropy-based)
        if chunk_weights is not None:
            alm_loss_per_chunk = alm_loss_per_chunk * chunk_weights.to(
                alm_loss_per_chunk.dtype
            )

        # Mask to only masked chunks and average
        chunk_mask_f = chunk_mask.float()  # also in float32 for consistency
        alm_loss = (
            alm_loss_per_chunk * chunk_mask_f
        ).sum() / chunk_mask_f.sum().clamp_min(1)

        return alm_loss

    def compute_loss(
        self,
        model: transformers.PreTrainedModel | nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        **kwargs,
    ):
        """
        Route to appropriate loss computation method based on distill_mode.

        Supports composition via '+' syntax (e.g., "taid+kl" computes both
        losses and sums them).
        """
        # Route all aligned modes (except alm which has its own mechanism)
        if "_aligned" in self.distill_mode and "alm" not in self.distill_mode:
            return self._compute_loss_aligned(model, inputs, return_outputs, **kwargs)

        # Handle composite modes (e.g., "taid+kl")
        # Compute shared forward pass and CE loss once, then accumulate
        # distillation-specific losses to avoid double-counting CE.
        if "+" in self.distill_mode:
            modes = [m.strip() for m in self.distill_mode.split("+")]

            if "alm" in modes:
                raise ValueError(
                    "ALM modes use separate forward passes "
                    "and cannot be composed with other modes via '+'. Use them standalone."
                )

            assert self.processing_class.padding_side == "right"
            inputs = self._preprocess_inputs(inputs)

            forward_result = self._forward_student_teacher(model, inputs)
            ce_loss = self._compute_ce_loss(model, forward_result, inputs)

            total_loss = ce_loss
            combined_log = {"ce_loss": ce_loss.detach().item()}

            for mode in modes:
                distill_loss = self._compute_distill_component(
                    mode, model, inputs, forward_result
                )
                total_loss = total_loss + distill_loss
                combined_log[f"{mode}_loss"] = distill_loss.detach().item()

            combined_log["loss"] = total_loss.detach().item()
            self._distill_log = combined_log

            return (
                (total_loss, forward_result["outputs"])
                if return_outputs
                else total_loss
            )

        # Single mode
        if self.distill_mode == "kl":
            return self._compute_loss_kl(model, inputs, return_outputs, **kwargs)
        elif self.distill_mode == "reverse_kl":
            return self._compute_loss_reverse_kl(model, inputs, return_outputs, **kwargs)
        elif self.distill_mode in ("alm", "reverse_alm"):
            return self._compute_loss_alm(model, inputs, return_outputs, **kwargs)
        elif self.distill_mode in ("alm_taid", "reverse_alm_taid"):
            return self._compute_loss_alm_taid(model, inputs, return_outputs, **kwargs)
        elif self.distill_mode == "taid":
            return self._compute_loss_taid(model, inputs, return_outputs, **kwargs)
        else:
            raise ValueError(f"Unknown distill_mode: {self.distill_mode}")

    def _compute_ce_loss(self, model, forward_result, inputs):
        """
        Compute CE loss from forward pass result. Shared by all distillation wrappers.

        Args:
            model: Student model (used to determine train/eval split for metrics).
            forward_result: Dict returned by _forward_student_teacher().
            inputs: Preprocessed input dict (needs 'labels' for assertion).

        Returns:
            Scalar CE loss tensor.
        """
        student_logits = forward_result["student_logits"]
        input_ids = forward_result["input_ids"]
        maskable_mask = forward_result["maskable_mask"]
        masked_mask = forward_result["masked_mask"]
        loss_weights = forward_result["loss_weights"]
        b = forward_result["b"]

        assert (
            input_ids[maskable_mask] == inputs["labels"][maskable_mask]
        ).all(), "Mismatch between input_ids and labels at valid positions"

        token_nll = F.cross_entropy(
            student_logits.transpose(1, 2),  # [b, V, l]
            input_ids,  # [b, l]
            reduction="none",  # [b, l]
        )
        token_nll = token_nll * loss_weights * masked_mask.to(token_nll.dtype)

        self.meter.update(
            split="train" if model.training else "eval",
            value=token_nll.detach(),
            weight=maskable_mask.to(dtype=student_logits.dtype).detach(),
        )

        if self.loss_norm_type == "token":
            token_nll /= maskable_mask.sum().clamp_min(1)
        elif self.loss_norm_type == "sequence":
            token_nll /= maskable_mask.sum(-1, keepdim=True).clamp_min(1) * b
        elif self.loss_norm_type == "batch":
            token_nll /= b
        else:
            raise ValueError("Invalid loss_norm_type.")

        return token_nll.sum()

    def _compute_distill_component(self, mode, model, inputs, forward_result):
        """Compute only the distillation-specific loss for a given mode.

        Used by composite mode to avoid double-counting CE loss.
        Returns the weighted distillation loss (without CE).
        """
        if mode == "kl" or mode == "reverse_kl":
            T = self.kl_temperature
            student_logits = forward_result["student_logits"]
            teacher_logits = forward_result["teacher_logits"]
            masked_mask = forward_result["masked_mask"]
            device = forward_result["device"]

            student_logits_shared = student_logits[:, :, : self.shared_vocab_size]
            teacher_logits_shared = teacher_logits[:, :, : self.shared_vocab_size]
            mask_flat = masked_mask.view(-1)
            student_masked = student_logits_shared.reshape(-1, self.shared_vocab_size)[
                mask_flat
            ]
            teacher_masked = teacher_logits_shared.reshape(-1, self.shared_vocab_size)[
                mask_flat
            ]

            if student_masked.numel() > 0:
                log_p_teacher = F.log_softmax(teacher_masked / T, dim=-1)
                log_p_student = F.log_softmax(student_masked / T, dim=-1)
                if mode == "reverse_kl":
                    # Reverse KL: KL(student || teacher)
                    kl_per_pos = F.kl_div(
                        log_p_teacher,
                        log_p_student,
                        log_target=True,
                        reduction="none",
                    ).sum(dim=-1)
                else:
                    # Forward KL: KL(teacher || student)
                    kl_per_pos = F.kl_div(
                        log_p_student,
                        log_p_teacher,
                        log_target=True,
                        reduction="none",
                    ).sum(dim=-1)
                kl_loss = kl_per_pos.mean() * (T * T)
            else:
                kl_loss = torch.tensor(0.0, device=device)
            return self.kl_weight * kl_loss

        elif mode == "taid":
            from dllm.core.trainers.losses.taid import compute_taid_loss

            taid_config = {
                "lambda_init": self.taid_lambda_init,
                "lambda_max": self.taid_lambda_max,
                "lambda_decay": self.taid_lambda_decay,
                "timestep_weight": self.taid_timestep_weight,
                "shared_vocab_size": self.shared_vocab_size,
                "temperature": self.kl_temperature,
                "axis_mode": self.taid_axis_mode,
            }
            taid_loss = compute_taid_loss(
                student_logits=forward_result["student_logits"],
                teacher_logits=forward_result["teacher_logits"],
                masked_mask=forward_result["masked_mask"],
                t=forward_result["t"],
                config=taid_config,
                global_step=self.state.global_step,
                max_steps=self.state.max_steps,
            )
            return self.taid_weight * taid_loss

        else:
            raise ValueError(f"Unknown distill_mode component: {mode}")

    def _compute_loss_taid(
        self,
        model: transformers.PreTrainedModel | nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        **kwargs,
    ):
        """
        Compute TAID (Timestep-Aware Interpolation Distillation) loss.

        Combines CE loss with TAID distillation loss that interpolates between
        student and teacher logits with a decaying lambda schedule.
        """
        assert self.processing_class.padding_side == "right"
        inputs = self._preprocess_inputs(inputs)

        forward_result = self._forward_student_teacher(model, inputs)
        ce_loss = self._compute_ce_loss(model, forward_result, inputs)

        # === TAID loss ===
        from dllm.core.trainers.losses.taid import compute_taid_loss

        taid_config = {
            "lambda_init": self.taid_lambda_init,
            "lambda_max": self.taid_lambda_max,
            "lambda_decay": self.taid_lambda_decay,
            "timestep_weight": self.taid_timestep_weight,
            "shared_vocab_size": self.shared_vocab_size,
            "temperature": self.kl_temperature,
            "axis_mode": self.taid_axis_mode,
        }

        taid_loss = compute_taid_loss(
            student_logits=forward_result["student_logits"],
            teacher_logits=forward_result["teacher_logits"],
            masked_mask=forward_result["masked_mask"],
            t=forward_result["t"],
            config=taid_config,
            global_step=self.state.global_step,
            max_steps=self.state.max_steps,
        )

        loss = ce_loss + self.taid_weight * taid_loss

        distill_log = {
            "ce_loss": ce_loss.detach().item(),
            "taid_loss": taid_loss.detach().item(),
            "loss": loss.detach().item(),
        }
        self._distill_log = distill_log

        outputs = forward_result["outputs"]
        return (loss, outputs) if return_outputs else loss

    def _compute_loss_alm(
        self,
        model: transformers.PreTrainedModel | nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        **kwargs,
    ):
        """
        Compute combined BD3LM CE loss + ALM chunk-level distillation loss.
        Used for cross-tokenizer distillation (e.g. LLaDA2.0 -> Qwen3 BD3LM).
        """
        assert self.processing_class.padding_side == "right"
        inputs = self._preprocess_inputs(inputs)

        input_ids = inputs["input_ids"]  # [b, l_s]
        labels = inputs["labels"]  # [b, l_s]
        teacher_input_ids = inputs["teacher_input_ids"]  # [b, l_t]
        teacher_attention_mask = inputs["teacher_attention_mask"]  # [b, l_t]
        alignment_matrix_student = inputs["alignment_matrix_student"]  # [b, l_s, n_c]
        alignment_matrix_teacher = inputs["alignment_matrix_teacher"]  # [b, l_t, n_c]

        b, l_s = input_ids.shape
        _, l_t = teacher_input_ids.shape
        maskable_mask = labels != -100  # [b, l_s]

        # === 1. Sample diffusion timesteps ===
        t = self.time_epsilon + (1.0 - self.time_epsilon) * torch.rand(
            b, device=input_ids.device
        )  # [b]
        p_mask = 1.0 - self.scheduler(t).unsqueeze(1).expand(b, l_s)  # [b, l_s]

        # === 2. Apply stochastic masking to student ===
        masked_mask = (
            torch.rand((b, l_s), device=input_ids.device) < p_mask
        ) & maskable_mask  # [b, l_s]

        # === 3. Chunk-consistent masking ===
        # If any token in a chunk is masked, mask ALL tokens in that chunk
        # (for both student and teacher) so ALM sees consistent state.
        chunk_has_masked = (masked_mask.unsqueeze(-1) & alignment_matrix_student).any(
            dim=1
        )  # [b, n_c]

        # Expand back to student token positions: mask all student tokens in
        # chunks that have at least one masked token
        student_chunk_masked = (
            chunk_has_masked.unsqueeze(1) & alignment_matrix_student
        ).any(
            dim=-1
        )  # [b, l_s]
        masked_mask = masked_mask | (student_chunk_masked & maskable_mask)  # [b, l_s]

        noised_input_ids = torch.where(
            masked_mask, self.processing_class.mask_token_id, input_ids
        )  # [b, l_s]

        # Safety guard: exclude chunks where any student token has labels=-100
        # (prompt chunks). With two-segment alignment, prompt chunks are entirely
        # masked and response chunks are entirely active.
        chunk_has_prompt = (
            (labels == -100).unsqueeze(-1) & alignment_matrix_student
        ).any(
            dim=1
        )  # [b, n_c]

        # Map to teacher token positions
        teacher_masked_mask = (
            chunk_has_masked.unsqueeze(1) & alignment_matrix_teacher
        ).any(
            dim=-1
        )  # [b, l_t]

        noised_teacher_ids = torch.where(
            teacher_masked_mask, self.teacher_mask_token_id, teacher_input_ids
        )  # [b, l_t]

        # === 4. Student BD3LM forward pass ===
        concat_input_ids = torch.cat([noised_input_ids, input_ids], dim=1)  # [b, 2*l_s]

        if self.accelerator.unwrap_model(model).config._attn_implementation == "sdpa":
            attention_mask = _create_bd3lm_attention_mask(
                b=None,
                h=None,
                q_idx=torch.arange(l_s * 2)[:, None],
                kv_idx=torch.arange(l_s * 2)[None, :],
                block_size=self.block_size,
                n=l_s,
            )
            attention_mask = (
                attention_mask.unsqueeze(0).unsqueeze(0).expand(1, 1, 2 * l_s, 2 * l_s)
            )
            attention_mask = attention_mask.to(input_ids.device)
        elif (
            self.accelerator.unwrap_model(model).config._attn_implementation
            == "flex_attention"
        ):
            from torch.nn.attention.flex_attention import create_block_mask

            attention_mask = create_block_mask(
                partial(
                    _create_bd3lm_attention_mask, block_size=self.block_size, n=l_s
                ),
                B=None,
                H=None,
                Q_LEN=l_s * 2,
                KV_LEN=l_s * 2,
            )
        else:
            raise NotImplementedError(
                f"Unsupported attention implementation: "
                f"{self.accelerator.unwrap_model(model).config._attn_implementation}"
            )

        base_pos = (
            torch.arange(l_s, device=input_ids.device).unsqueeze(0).expand(b, l_s)
        )
        concat_position_ids = torch.cat([base_pos, base_pos], dim=1)  # [b, 2*l_s]

        outputs = model(
            input_ids=concat_input_ids,
            attention_mask=attention_mask,
            position_ids=concat_position_ids,
        )
        outputs = self._postprocess_outputs(outputs)
        student_logits = outputs.logits[:, :l_s]  # [b, l_s, V_s]

        # === 5. Teacher forward pass (no grad, bidirectional attention) ===
        with torch.no_grad():
            teacher_pad_mask = ~teacher_attention_mask.bool()  # [b, l_t]
            model_dtype = next(self.teacher_model.parameters()).dtype
            teacher_4d_mask = torch.zeros(
                b, 1, l_t, l_t, dtype=model_dtype, device=input_ids.device
            )
            teacher_4d_mask.masked_fill_(
                teacher_pad_mask[:, None, None, :], float("-inf")
            )
            teacher_4d_mask.masked_fill_(
                teacher_pad_mask[:, None, :, None], float("-inf")
            )

            if self.use_complementary_demo and chunk_has_masked.any():
                # --- Complementary demonstration at chunk level ---
                chunk_random = (
                    torch.rand_like(chunk_has_masked.float()) < self.demo_ratio
                )
                chunk_A = chunk_has_masked & chunk_random  # demo in pass 1
                chunk_B = chunk_has_masked & ~chunk_random  # demo in pass 2

                # Map chunks to teacher token positions
                teacher_M_B = (chunk_B.unsqueeze(1) & alignment_matrix_teacher).any(
                    dim=-1
                )  # [b, l_t]
                teacher_M_A = (chunk_A.unsqueeze(1) & alignment_matrix_teacher).any(
                    dim=-1
                )  # [b, l_t]

                # Pass 1: chunk_A clean (demo), chunk_B masked
                noised_teacher_pass1 = torch.where(
                    teacher_M_B, self.teacher_mask_token_id, teacher_input_ids
                )
                # Pass 2: chunk_B clean (demo), chunk_A masked
                noised_teacher_pass2 = torch.where(
                    teacher_M_A, self.teacher_mask_token_id, teacher_input_ids
                )

                teacher_out_1 = self.teacher_model(
                    input_ids=noised_teacher_pass1,
                    attention_mask=teacher_4d_mask,
                )
                teacher_out_2 = self.teacher_model(
                    input_ids=noised_teacher_pass2,
                    attention_mask=teacher_4d_mask,
                )

                # Combine: chunk_A positions from pass 2, chunk_B from pass 1
                teacher_logits = teacher_out_1.logits
                teacher_logits[teacher_M_A] = teacher_out_2.logits[teacher_M_A]
                del teacher_out_1, teacher_out_2
            else:
                teacher_outputs = self.teacher_model(
                    input_ids=noised_teacher_ids,
                    attention_mask=teacher_4d_mask,
                )
                teacher_logits = teacher_outputs.logits  # [b, l_t, V_t]

        # === 6. CE loss (standard BD3LM) ===
        loss_weights = self._compute_loss_weights(
            t=t, inputs=inputs, masked_mask=masked_mask
        )

        assert (
            input_ids[maskable_mask] == labels[maskable_mask]
        ).all(), "Mismatch between input_ids and labels at valid positions"

        token_nll = F.cross_entropy(
            student_logits.transpose(1, 2),  # [b, V_s, l_s]
            input_ids,  # [b, l_s]
            reduction="none",  # [b, l_s]
        )
        token_nll = (
            token_nll * loss_weights * masked_mask.to(token_nll.dtype)
        )  # [b, l_s]

        self.meter.update(
            split="train" if model.training else "eval",
            value=token_nll.detach(),
            weight=maskable_mask.to(dtype=student_logits.dtype).detach(),
        )

        if self.loss_norm_type == "token":
            token_nll /= maskable_mask.sum().clamp_min(1)
        elif self.loss_norm_type == "sequence":
            token_nll /= maskable_mask.sum(-1, keepdim=True).clamp_min(1) * b
        elif self.loss_norm_type == "batch":
            token_nll /= b
        else:
            raise ValueError("Invalid loss_norm_type.")
        ce_loss = token_nll.sum()

        # === 7. ALM loss ===
        active_chunk_mask = chunk_has_masked & ~chunk_has_prompt
        use_reverse = self.distill_mode == "reverse_alm"
        alm_loss = self.compute_alm_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            student_input_ids=input_ids,
            teacher_input_ids=teacher_input_ids,
            alignment_matrix_student=alignment_matrix_student,
            alignment_matrix_teacher=alignment_matrix_teacher,
            chunk_mask=active_chunk_mask,
            temperature=self.alm_temperature,
            reverse=use_reverse,
        )

        # === 8. Combine losses ===
        loss = ce_loss + self.alm_weight * alm_loss

        # Log individual losses
        loss_key = "reverse_alm_loss" if use_reverse else "alm_loss"
        self._distill_log = {
            "ce_loss": ce_loss.detach().item(),
            loss_key: alm_loss.detach().item(),
            "loss": loss.detach().item(),
        }

        return (loss, outputs) if return_outputs else loss

    def _compute_loss_alm_taid(
        self,
        model: transformers.PreTrainedModel | nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        **kwargs,
    ):
        """
        Compute BD3LM CE loss + TAID-enhanced ALM chunk-level distillation loss.

        Like _compute_loss_alm but replaces the fixed BCE(teacher, student) with
        BCE(mixed_target, student) where the mix shifts from teacher-reliant to
        student-reliant over training via TAID's lambda decay schedule and
        timestep-aware weighting.
        """
        assert self.processing_class.padding_side == "right"
        inputs = self._preprocess_inputs(inputs)

        input_ids = inputs["input_ids"]  # [b, l_s]
        labels = inputs["labels"]  # [b, l_s]
        teacher_input_ids = inputs["teacher_input_ids"]  # [b, l_t]
        teacher_attention_mask = inputs["teacher_attention_mask"]  # [b, l_t]
        alignment_matrix_student = inputs["alignment_matrix_student"]  # [b, l_s, n_c]
        alignment_matrix_teacher = inputs["alignment_matrix_teacher"]  # [b, l_t, n_c]

        b, l_s = input_ids.shape
        _, l_t = teacher_input_ids.shape
        maskable_mask = labels != -100  # [b, l_s]

        # === 1. Sample diffusion timesteps ===
        t = self.time_epsilon + (1.0 - self.time_epsilon) * torch.rand(
            b, device=input_ids.device
        )  # [b]
        p_mask = 1.0 - self.scheduler(t).unsqueeze(1).expand(b, l_s)  # [b, l_s]

        # === 2. Apply stochastic masking to student ===
        masked_mask = (
            torch.rand((b, l_s), device=input_ids.device) < p_mask
        ) & maskable_mask  # [b, l_s]

        # === 3. Chunk-consistent masking ===
        chunk_has_masked = (masked_mask.unsqueeze(-1) & alignment_matrix_student).any(
            dim=1
        )  # [b, n_c]

        student_chunk_masked = (
            chunk_has_masked.unsqueeze(1) & alignment_matrix_student
        ).any(
            dim=-1
        )  # [b, l_s]
        masked_mask = masked_mask | (student_chunk_masked & maskable_mask)  # [b, l_s]

        noised_input_ids = torch.where(
            masked_mask, self.processing_class.mask_token_id, input_ids
        )  # [b, l_s]

        chunk_has_prompt = (
            (labels == -100).unsqueeze(-1) & alignment_matrix_student
        ).any(
            dim=1
        )  # [b, n_c]

        teacher_masked_mask = (
            chunk_has_masked.unsqueeze(1) & alignment_matrix_teacher
        ).any(
            dim=-1
        )  # [b, l_t]

        noised_teacher_ids = torch.where(
            teacher_masked_mask, self.teacher_mask_token_id, teacher_input_ids
        )  # [b, l_t]

        # === 4. Student BD3LM forward pass ===
        concat_input_ids = torch.cat([noised_input_ids, input_ids], dim=1)  # [b, 2*l_s]

        if self.accelerator.unwrap_model(model).config._attn_implementation == "sdpa":
            attention_mask = _create_bd3lm_attention_mask(
                b=None,
                h=None,
                q_idx=torch.arange(l_s * 2)[:, None],
                kv_idx=torch.arange(l_s * 2)[None, :],
                block_size=self.block_size,
                n=l_s,
            )
            attention_mask = (
                attention_mask.unsqueeze(0).unsqueeze(0).expand(1, 1, 2 * l_s, 2 * l_s)
            )
            attention_mask = attention_mask.to(input_ids.device)
        elif (
            self.accelerator.unwrap_model(model).config._attn_implementation
            == "flex_attention"
        ):
            from torch.nn.attention.flex_attention import create_block_mask

            attention_mask = create_block_mask(
                partial(
                    _create_bd3lm_attention_mask, block_size=self.block_size, n=l_s
                ),
                B=None,
                H=None,
                Q_LEN=l_s * 2,
                KV_LEN=l_s * 2,
            )
        else:
            raise NotImplementedError(
                f"Unsupported attention implementation: "
                f"{self.accelerator.unwrap_model(model).config._attn_implementation}"
            )

        base_pos = (
            torch.arange(l_s, device=input_ids.device).unsqueeze(0).expand(b, l_s)
        )
        concat_position_ids = torch.cat([base_pos, base_pos], dim=1)  # [b, 2*l_s]

        outputs = model(
            input_ids=concat_input_ids,
            attention_mask=attention_mask,
            position_ids=concat_position_ids,
        )
        outputs = self._postprocess_outputs(outputs)
        student_logits = outputs.logits[:, :l_s]  # [b, l_s, V_s]

        # === 5. Teacher forward pass (no grad, bidirectional attention) ===
        with torch.no_grad():
            teacher_pad_mask = ~teacher_attention_mask.bool()  # [b, l_t]
            model_dtype = next(self.teacher_model.parameters()).dtype
            teacher_4d_mask = torch.zeros(
                b, 1, l_t, l_t, dtype=model_dtype, device=input_ids.device
            )
            teacher_4d_mask.masked_fill_(
                teacher_pad_mask[:, None, None, :], float("-inf")
            )
            teacher_4d_mask.masked_fill_(
                teacher_pad_mask[:, None, :, None], float("-inf")
            )

            if self.use_complementary_demo and chunk_has_masked.any():
                # --- Complementary demonstration at chunk level ---
                chunk_random = (
                    torch.rand_like(chunk_has_masked.float()) < self.demo_ratio
                )
                chunk_A = chunk_has_masked & chunk_random  # demo in pass 1
                chunk_B = chunk_has_masked & ~chunk_random  # demo in pass 2

                # Map chunks to teacher token positions
                teacher_M_B = (chunk_B.unsqueeze(1) & alignment_matrix_teacher).any(
                    dim=-1
                )  # [b, l_t]
                teacher_M_A = (chunk_A.unsqueeze(1) & alignment_matrix_teacher).any(
                    dim=-1
                )  # [b, l_t]

                # Pass 1: chunk_A clean (demo), chunk_B masked
                noised_teacher_pass1 = torch.where(
                    teacher_M_B, self.teacher_mask_token_id, teacher_input_ids
                )
                # Pass 2: chunk_B clean (demo), chunk_A masked
                noised_teacher_pass2 = torch.where(
                    teacher_M_A, self.teacher_mask_token_id, teacher_input_ids
                )

                teacher_out_1 = self.teacher_model(
                    input_ids=noised_teacher_pass1,
                    attention_mask=teacher_4d_mask,
                )
                teacher_out_2 = self.teacher_model(
                    input_ids=noised_teacher_pass2,
                    attention_mask=teacher_4d_mask,
                )

                # Combine: chunk_A positions from pass 2, chunk_B from pass 1
                teacher_logits = teacher_out_1.logits
                teacher_logits[teacher_M_A] = teacher_out_2.logits[teacher_M_A]
                del teacher_out_1, teacher_out_2
            else:
                teacher_outputs = self.teacher_model(
                    input_ids=noised_teacher_ids,
                    attention_mask=teacher_4d_mask,
                )
                teacher_logits = teacher_outputs.logits  # [b, l_t, V_t]

        # === 6. CE loss (standard BD3LM) ===
        loss_weights = self._compute_loss_weights(
            t=t, inputs=inputs, masked_mask=masked_mask
        )

        assert (
            input_ids[maskable_mask] == labels[maskable_mask]
        ).all(), "Mismatch between input_ids and labels at valid positions"

        token_nll = F.cross_entropy(
            student_logits.transpose(1, 2),  # [b, V_s, l_s]
            input_ids,  # [b, l_s]
            reduction="none",  # [b, l_s]
        )
        token_nll = (
            token_nll * loss_weights * masked_mask.to(token_nll.dtype)
        )  # [b, l_s]

        self.meter.update(
            split="train" if model.training else "eval",
            value=token_nll.detach(),
            weight=maskable_mask.to(dtype=student_logits.dtype).detach(),
        )

        if self.loss_norm_type == "token":
            token_nll /= maskable_mask.sum().clamp_min(1)
        elif self.loss_norm_type == "sequence":
            token_nll /= maskable_mask.sum(-1, keepdim=True).clamp_min(1) * b
        elif self.loss_norm_type == "batch":
            token_nll /= b
        else:
            raise ValueError("Invalid loss_norm_type.")
        ce_loss = token_nll.sum()

        # === 7. ALM-TAID loss ===
        # 7a. Per-token log-probs of correct tokens (same as compute_alm_loss)
        student_token_logits = student_logits.gather(
            2, input_ids.unsqueeze(-1)
        ).squeeze(
            -1
        )  # [b, l_s]
        student_token_lp = student_token_logits - torch.logsumexp(
            student_logits, dim=-1
        )

        with torch.no_grad():
            teacher_token_logits = teacher_logits.gather(
                2, teacher_input_ids.unsqueeze(-1)
            ).squeeze(
                -1
            )  # [b, l_t]
            teacher_token_lp = teacher_token_logits - torch.logsumexp(
                teacher_logits, dim=-1
            )

        # 7b. Aggregate to chunk-level via alignment matrices
        align_s = alignment_matrix_student.to(dtype=student_token_lp.dtype)
        align_t = alignment_matrix_teacher.to(dtype=teacher_token_lp.dtype)

        student_chunk_lp = torch.bmm(student_token_lp.unsqueeze(1), align_s).squeeze(
            1
        )  # [b, n_chunks]
        with torch.no_grad():
            teacher_chunk_lp = torch.bmm(
                teacher_token_lp.unsqueeze(1), align_t
            ).squeeze(
                1
            )  # [b, n_chunks]

        # 7c. Temperature scale and convert to probs
        T = self.alm_temperature
        s_chunk_lp_scaled = (student_chunk_lp / T).float()
        t_chunk_lp_scaled = (teacher_chunk_lp / T).float()

        s_p = torch.exp(s_chunk_lp_scaled).clamp(1e-7, 1 - 1e-7)
        t_p = torch.exp(t_chunk_lp_scaled).clamp(1e-7, 1 - 1e-7)

        # 7d. Lambda schedule (training-level growth: student -> teacher)
        training_progress = self.state.global_step / max(self.state.max_steps, 1)
        if self.taid_lambda_decay == "cosine":
            growth_val = 0.5 * (1.0 - math.cos(math.pi * training_progress))
        elif self.taid_lambda_decay == "linear":
            growth_val = training_progress
        else:
            raise ValueError(f"Unknown taid_lambda_decay: {self.taid_lambda_decay}")
        lambda_train = (
            self.taid_lambda_init
            + (self.taid_lambda_max - self.taid_lambda_init) * growth_val
        )

        # 7e. Timestep modulation: nearly-clean positions lean more on teacher
        if self.taid_axis_mode == "both":
            lambda_t = lambda_train * (1.0 - t)  # [b]
        elif self.taid_axis_mode == "training_only":
            lambda_t = lambda_train * torch.ones_like(t)  # [b]
        elif self.taid_axis_mode == "timestep_only":
            lambda_t = self.taid_lambda_max * (1.0 - t)  # [b]
        else:
            raise ValueError(f"Unknown taid_axis_mode: {self.taid_axis_mode}")
        lambda_t = lambda_t.float().unsqueeze(1)  # [b, 1]

        # 7f. Interpolated chunk target (detached -- gradients only through s_p)
        mixed_p = ((1 - lambda_t) * s_p + lambda_t * t_p).detach().clamp(1e-7, 1 - 1e-7)

        # 7g. BCE loss per chunk
        use_reverse = self.distill_mode == "reverse_alm_taid"
        if use_reverse:
            # Reverse BCE: student as label, mixed as prediction
            alm_taid_loss_per_chunk = -(
                s_p * torch.log(mixed_p) + (1 - s_p) * torch.log(1 - mixed_p)
            )
        else:
            # Forward BCE: mixed as label, student as prediction
            alm_taid_loss_per_chunk = -(
                mixed_p * torch.log(s_p) + (1 - mixed_p) * torch.log(1 - s_p)
            )

        # 7h. Timestep weighting
        chunk_mask = chunk_has_masked & ~chunk_has_prompt  # [b, n_chunks]
        chunk_mask_f = chunk_mask.float()
        if self.taid_timestep_weight == "midrange":
            w = (
                torch.exp(-((t - 0.5) ** 2) / (2 * 0.15**2)).float().unsqueeze(1)
            )  # [b, 1]
        elif self.taid_timestep_weight == "uniform":
            w = torch.ones(b, 1, device=input_ids.device)
        else:
            raise ValueError(
                f"Unknown taid_timestep_weight: {self.taid_timestep_weight}"
            )

        alm_taid_loss = (alm_taid_loss_per_chunk * chunk_mask_f * w).sum() / (
            chunk_mask_f.sum().clamp_min(1)
        )

        # === 8. Combine losses ===
        loss = ce_loss + self.alm_weight * alm_taid_loss

        loss_key = "reverse_alm_taid_loss" if use_reverse else "alm_taid_loss"
        distill_log = {
            "ce_loss": ce_loss.detach().item(),
            loss_key: alm_taid_loss.detach().item(),
            "lambda_train": lambda_train,
            "loss": loss.detach().item(),
        }
        self._distill_log = distill_log

        return (loss, outputs) if return_outputs else loss

    def _forward_student_teacher(self, model, inputs):
        """
        Shared forward pass for same-tokenizer distillation modes.

        Performs student and teacher forward passes with BD3LM attention patterns,
        handling the staircase block attention and proper masking.

        Args:
            model: Student model (BD3LM architecture)
            inputs: Preprocessed input dict with input_ids, labels

        Returns:
            dict with keys:
                - student_logits: [b, l, V] student model logits
                - teacher_logits: [b, l, V] teacher model logits
                - masked_mask: [b, l] bool mask of which tokens were masked
                - maskable_mask: [b, l] bool mask of which tokens can be masked
                - t: [b] sampled diffusion timesteps
                - input_ids: [b, l] clean input token ids
                - loss_weights: [b, l] per-token loss weights
                - b: batch size
                - l: sequence length
                - device: torch device
                - outputs: student model outputs object
        """
        input_ids = inputs["input_ids"]  # [b, l]
        labels = inputs["labels"]  # [b, l]
        b, l = input_ids.shape
        device = input_ids.device
        maskable_mask = labels != -100  # [b, l]
        # Student always uses [xt, x0]; teacher concat order is configurable
        teacher_x0_first = self.concat_order == "x0_xt"

        # Sample diffusion timesteps
        t = self.time_epsilon + (1.0 - self.time_epsilon) * torch.rand(
            b, device=device
        )  # [b]
        p_mask = 1.0 - self.scheduler(t).unsqueeze(1).expand(b, l)  # [b, l]

        # Apply stochastic masking
        masked_mask = (
            torch.rand((b, l), device=device) < p_mask
        ) & maskable_mask  # [b, l]

        # Create noised input ids for student and teacher
        student_noised_ids = torch.where(
            masked_mask, self.processing_class.mask_token_id, input_ids
        )  # [b, l]
        teacher_noised_ids = torch.where(
            masked_mask, self.teacher_mask_token_id, input_ids
        )  # [b, l]

        # === Student BD3LM forward pass (always [xt, x0]) ===
        concat_input_ids = torch.cat([student_noised_ids, input_ids], dim=1)  # [b, 2l]

        if self.accelerator.unwrap_model(model).config._attn_implementation == "sdpa":
            attention_mask = _create_bd3lm_attention_mask(
                b=None,
                h=None,
                q_idx=torch.arange(l * 2)[:, None],
                kv_idx=torch.arange(l * 2)[None, :],
                block_size=self.block_size,
                n=l,
                x0_first=False,  # student always [xt, x0]
            )
            attention_mask = (
                attention_mask.unsqueeze(0).unsqueeze(0).expand(1, 1, 2 * l, 2 * l)
            )
            attention_mask = attention_mask.to(device)
        elif (
            self.accelerator.unwrap_model(model).config._attn_implementation
            == "flex_attention"
        ):
            from torch.nn.attention.flex_attention import create_block_mask

            attention_mask = create_block_mask(
                partial(
                    _create_bd3lm_attention_mask,
                    block_size=self.block_size,
                    n=l,
                    x0_first=False,  # student always [xt, x0]
                ),
                B=None,
                H=None,
                Q_LEN=l * 2,
                KV_LEN=l * 2,
            )
        else:
            raise NotImplementedError(
                f"Unsupported attention implementation: "
                f"{self.accelerator.unwrap_model(model).config._attn_implementation}"
            )

        base_pos = torch.arange(l, device=device).unsqueeze(0).expand(b, l)  # [b, l]
        concat_position_ids = torch.cat([base_pos, base_pos], dim=1)  # [b, 2l]

        outputs = model(
            input_ids=concat_input_ids,
            attention_mask=attention_mask,
            position_ids=concat_position_ids,
        )
        outputs = self._postprocess_outputs(outputs)
        student_logits = outputs.logits[:, :l]  # xt is always first half for student

        # === Teacher BD3LM forward pass (no grad, concat order per config) ===
        with torch.no_grad():
            # Build BD3LM 4D float mask for WeDLM teacher (content-independent, reusable)
            # WeDLM checks isinstance(attention_mask, dict) to skip causal mask
            bd3lm_bool = _create_bd3lm_attention_mask(
                b=None,
                h=None,
                q_idx=torch.arange(l * 2, device=device)[:, None],
                kv_idx=torch.arange(l * 2, device=device)[None, :],
                block_size=self.block_size,
                n=l,
                x0_first=teacher_x0_first,
            )
            model_dtype = next(self.teacher_model.parameters()).dtype
            bd3lm_float = torch.zeros(
                1, 1, 2 * l, 2 * l, dtype=model_dtype, device=device
            )
            bd3lm_float.masked_fill_(
                ~bd3lm_bool.unsqueeze(0).unsqueeze(0),
                torch.finfo(model_dtype).min,
            )
            teacher_attn = {
                "full_attention": bd3lm_float,
                "sliding_attention": bd3lm_float,
            }

            if self.use_complementary_demo and masked_mask.any():
                # --- Complementary demonstration: two teacher passes ---
                M_A, M_B = self._split_complementary_masks(masked_mask)

                # Pass 1: M_A is clean (demo), M_B is masked
                teacher_noised_pass1 = torch.where(
                    M_B, self.teacher_mask_token_id, input_ids
                )
                # Pass 2: M_B is clean (demo), M_A is masked
                teacher_noised_pass2 = torch.where(
                    M_A, self.teacher_mask_token_id, input_ids
                )

                # Build concat inputs preserving [xt, x0] or [x0, xt] structure
                if teacher_x0_first:
                    teacher_concat_1 = torch.cat(
                        [input_ids, teacher_noised_pass1], dim=1
                    )
                    teacher_concat_2 = torch.cat(
                        [input_ids, teacher_noised_pass2], dim=1
                    )
                else:
                    teacher_concat_1 = torch.cat(
                        [teacher_noised_pass1, input_ids], dim=1
                    )
                    teacher_concat_2 = torch.cat(
                        [teacher_noised_pass2, input_ids], dim=1
                    )

                # Two sequential teacher forward calls
                teacher_out_1 = self.teacher_model(
                    input_ids=teacher_concat_1,
                    attention_mask=teacher_attn,
                    position_ids=concat_position_ids,
                )
                teacher_out_2 = self.teacher_model(
                    input_ids=teacher_concat_2,
                    attention_mask=teacher_attn,
                    position_ids=concat_position_ids,
                )

                # Extract logits from xt half
                if teacher_x0_first:
                    logits_1 = teacher_out_1.logits[:, l:]
                    logits_2 = teacher_out_2.logits[:, l:]
                else:
                    logits_1 = teacher_out_1.logits[:, :l]
                    logits_2 = teacher_out_2.logits[:, :l]

                # Combine: M_B positions from pass 1 (M_A was demo),
                #          M_A positions from pass 2 (M_B was demo)
                teacher_logits = logits_1
                teacher_logits[M_A] = logits_2[M_A]
                del logits_2  # free early
                del teacher_out_1, teacher_out_2

            else:
                # --- Standard single-pass teacher forward ---
                if teacher_x0_first:
                    teacher_concat_ids = torch.cat(
                        [input_ids, teacher_noised_ids], dim=1
                    )  # [b, 2l]: [x0, xt]
                else:
                    teacher_concat_ids = torch.cat(
                        [teacher_noised_ids, input_ids], dim=1
                    )  # [b, 2l]: [xt, x0]

                teacher_outputs = self.teacher_model(
                    input_ids=teacher_concat_ids,
                    attention_mask=teacher_attn,
                    position_ids=concat_position_ids,
                )

                # Extract teacher logits from the xt half
                if teacher_x0_first:
                    teacher_logits = teacher_outputs.logits[:, l:]  # xt is second half
                else:
                    teacher_logits = teacher_outputs.logits[:, :l]  # xt is first half

        # === Compute loss weights ===
        loss_weights = self._compute_loss_weights(
            t=t, inputs=inputs, masked_mask=masked_mask
        )

        result = {
            "student_logits": student_logits,
            "teacher_logits": teacher_logits,
            "masked_mask": masked_mask,
            "maskable_mask": maskable_mask,
            "t": t,
            "input_ids": input_ids,
            "loss_weights": loss_weights,
            "b": b,
            "l": l,
            "device": device,
            "outputs": outputs,
        }

        return result

    def _forward_student_teacher_aligned(self, model, inputs):
        """
        Shared forward pass for aligned distillation modes (different chat templates).

        Similar to _forward_student_teacher() but handles different sequence lengths
        between student and teacher, using position-level alignment indices to
        coordinate masking and gather logits at aligned positions.

        Args:
            model: Student model (BD3LM architecture)
            inputs: Preprocessed input dict with student and teacher fields

        Returns:
            dict with:
                - Full student data (for CE loss): student_logits, masked_mask, etc.
                - Aligned data (for distill loss): student_logits_aligned,
                  teacher_logits_aligned, masked_mask_aligned
                - Alignment stats: n_aligned_masked, n_aligned_total
        """
        input_ids = inputs["input_ids"]  # [b, l_s]
        labels = inputs["labels"]  # [b, l_s]
        teacher_input_ids = inputs["teacher_input_ids"]  # [b, l_t]
        teacher_attention_mask = inputs["teacher_attention_mask"]  # [b, l_t]
        align_student = inputs["align_student"]  # [b, max_n]
        align_teacher = inputs["align_teacher"]  # [b, max_n]
        n_aligned = inputs["n_aligned"]  # [b]

        b, l_s = input_ids.shape
        _, l_t = teacher_input_ids.shape
        max_n = align_student.shape[1]
        device = input_ids.device
        maskable_mask = labels != -100  # [b, l_s]

        teacher_x0_first = self.concat_order == "x0_xt"

        # === 1. Sample diffusion timesteps & mask student ===
        t = self.time_epsilon + (1.0 - self.time_epsilon) * torch.rand(
            b, device=device
        )  # [b]
        p_mask = 1.0 - self.scheduler(t).unsqueeze(1).expand(b, l_s)  # [b, l_s]
        masked_mask = (
            torch.rand((b, l_s), device=device) < p_mask
        ) & maskable_mask  # [b, l_s]
        student_noised_ids = torch.where(
            masked_mask, self.processing_class.mask_token_id, input_ids
        )  # [b, l_s]

        # === 2. Propagate masking to teacher at aligned positions ===
        align_range = torch.arange(max_n, device=device).unsqueeze(0)  # [1, max_n]
        valid_align = align_range < n_aligned.unsqueeze(1)  # [b, max_n]

        align_s_clamped = align_student.clamp(0, l_s - 1)  # [b, max_n]
        student_masked_at_aligned = masked_mask.gather(1, align_s_clamped)  # [b, max_n]
        student_masked_at_aligned = (
            student_masked_at_aligned & valid_align
        )  # [b, max_n]

        teacher_masked_mask = torch.zeros(b, l_t, dtype=torch.bool, device=device)
        align_t_clamped = align_teacher.clamp(0, l_t - 1)  # [b, max_n]
        teacher_masked_mask.scatter_(1, align_t_clamped, student_masked_at_aligned)

        teacher_noised_ids = torch.where(
            teacher_masked_mask, self.teacher_mask_token_id, teacher_input_ids
        )  # [b, l_t]

        # === 3. Student BD3LM forward pass (always [xt, x0]) ===
        concat_input_ids = torch.cat(
            [student_noised_ids, input_ids], dim=1
        )  # [b, 2*l_s]

        if self.accelerator.unwrap_model(model).config._attn_implementation == "sdpa":
            attention_mask = _create_bd3lm_attention_mask(
                b=None,
                h=None,
                q_idx=torch.arange(l_s * 2)[:, None],
                kv_idx=torch.arange(l_s * 2)[None, :],
                block_size=self.block_size,
                n=l_s,
                x0_first=False,
            )
            attention_mask = (
                attention_mask.unsqueeze(0).unsqueeze(0).expand(1, 1, 2 * l_s, 2 * l_s)
            )
            attention_mask = attention_mask.to(device)
        elif (
            self.accelerator.unwrap_model(model).config._attn_implementation
            == "flex_attention"
        ):
            from torch.nn.attention.flex_attention import create_block_mask

            attention_mask = create_block_mask(
                partial(
                    _create_bd3lm_attention_mask,
                    block_size=self.block_size,
                    n=l_s,
                    x0_first=False,
                ),
                B=None,
                H=None,
                Q_LEN=l_s * 2,
                KV_LEN=l_s * 2,
            )
        else:
            raise NotImplementedError(
                f"Unsupported attention implementation: "
                f"{self.accelerator.unwrap_model(model).config._attn_implementation}"
            )

        base_pos_s = (
            torch.arange(l_s, device=device).unsqueeze(0).expand(b, l_s)
        )  # [b, l_s]
        concat_position_ids_s = torch.cat([base_pos_s, base_pos_s], dim=1)  # [b, 2*l_s]

        outputs = model(
            input_ids=concat_input_ids,
            attention_mask=attention_mask,
            position_ids=concat_position_ids_s,
        )
        outputs = self._postprocess_outputs(outputs)
        student_logits = outputs.logits[:, :l_s]  # [b, l_s, V_s]

        # === 4. Teacher BD3LM forward pass (no grad, independent length) ===
        with torch.no_grad():
            bd3lm_bool = _create_bd3lm_attention_mask(
                b=None,
                h=None,
                q_idx=torch.arange(l_t * 2, device=device)[:, None],
                kv_idx=torch.arange(l_t * 2, device=device)[None, :],
                block_size=self.block_size,
                n=l_t,
                x0_first=teacher_x0_first,
            )
            model_dtype = next(self.teacher_model.parameters()).dtype
            bd3lm_float = torch.zeros(
                1, 1, 2 * l_t, 2 * l_t, dtype=model_dtype, device=device
            )
            bd3lm_float.masked_fill_(
                ~bd3lm_bool.unsqueeze(0).unsqueeze(0),
                torch.finfo(model_dtype).min,
            )

            # Apply teacher padding mask
            teacher_pad_mask = ~teacher_attention_mask.bool()  # [b, l_t]
            teacher_pad_2l = torch.cat(
                [teacher_pad_mask, teacher_pad_mask], dim=1
            )  # [b, 2*l_t]
            bd3lm_float = bd3lm_float.expand(b, 1, 2 * l_t, 2 * l_t).clone()
            bd3lm_float.masked_fill_(
                teacher_pad_2l[:, None, None, :], torch.finfo(model_dtype).min
            )
            bd3lm_float.masked_fill_(
                teacher_pad_2l[:, None, :, None], torch.finfo(model_dtype).min
            )

            teacher_attn = {
                "full_attention": bd3lm_float,
                "sliding_attention": bd3lm_float,
            }

            base_pos_t = (
                torch.arange(l_t, device=device).unsqueeze(0).expand(b, l_t)
            )  # [b, l_t]
            concat_position_ids_t = torch.cat(
                [base_pos_t, base_pos_t], dim=1
            )  # [b, 2*l_t]

            if self.use_complementary_demo and masked_mask.any():
                # --- Complementary demonstration: two teacher passes ---
                M_A_student, M_B_student = self._split_complementary_masks(masked_mask)

                # Propagate M_B to teacher via alignment indices
                student_M_B_at_aligned = (
                    M_B_student.gather(1, align_s_clamped) & valid_align
                )
                teacher_M_B = torch.zeros(b, l_t, dtype=torch.bool, device=device)
                teacher_M_B.scatter_(1, align_t_clamped, student_M_B_at_aligned)

                # Propagate M_A to teacher via alignment indices
                student_M_A_at_aligned = (
                    M_A_student.gather(1, align_s_clamped) & valid_align
                )
                teacher_M_A = torch.zeros(b, l_t, dtype=torch.bool, device=device)
                teacher_M_A.scatter_(1, align_t_clamped, student_M_A_at_aligned)

                # Pass 1: M_A clean (demo), M_B masked
                teacher_noised_pass1 = torch.where(
                    teacher_M_B, self.teacher_mask_token_id, teacher_input_ids
                )
                # Pass 2: M_B clean (demo), M_A masked
                teacher_noised_pass2 = torch.where(
                    teacher_M_A, self.teacher_mask_token_id, teacher_input_ids
                )

                if teacher_x0_first:
                    teacher_concat_1 = torch.cat(
                        [teacher_input_ids, teacher_noised_pass1], dim=1
                    )
                    teacher_concat_2 = torch.cat(
                        [teacher_input_ids, teacher_noised_pass2], dim=1
                    )
                else:
                    teacher_concat_1 = torch.cat(
                        [teacher_noised_pass1, teacher_input_ids], dim=1
                    )
                    teacher_concat_2 = torch.cat(
                        [teacher_noised_pass2, teacher_input_ids], dim=1
                    )

                teacher_out_1 = self.teacher_model(
                    input_ids=teacher_concat_1,
                    attention_mask=teacher_attn,
                    position_ids=concat_position_ids_t,
                )
                teacher_out_2 = self.teacher_model(
                    input_ids=teacher_concat_2,
                    attention_mask=teacher_attn,
                    position_ids=concat_position_ids_t,
                )

                if teacher_x0_first:
                    logits_1 = teacher_out_1.logits[:, l_t:]
                    logits_2 = teacher_out_2.logits[:, l_t:]
                else:
                    logits_1 = teacher_out_1.logits[:, :l_t]
                    logits_2 = teacher_out_2.logits[:, :l_t]

                # Combine at full [b, l_t, V] level
                teacher_logits = logits_1
                teacher_logits[teacher_M_A] = logits_2[teacher_M_A]
                del logits_2  # free early
                del teacher_out_1, teacher_out_2

            else:
                # --- Standard single-pass teacher forward ---
                if teacher_x0_first:
                    teacher_concat_ids = torch.cat(
                        [teacher_input_ids, teacher_noised_ids], dim=1
                    )  # [b, 2*l_t]: [x0, xt]
                else:
                    teacher_concat_ids = torch.cat(
                        [teacher_noised_ids, teacher_input_ids], dim=1
                    )  # [b, 2*l_t]: [xt, x0]

                teacher_outputs = self.teacher_model(
                    input_ids=teacher_concat_ids,
                    attention_mask=teacher_attn,
                    position_ids=concat_position_ids_t,
                )

                if teacher_x0_first:
                    teacher_logits = teacher_outputs.logits[:, l_t:]  # [b, l_t, V_t]
                else:
                    teacher_logits = teacher_outputs.logits[:, :l_t]  # [b, l_t, V_t]

        # === 5. Compute loss weights ===
        loss_weights = self._compute_loss_weights(
            t=t, inputs=inputs, masked_mask=masked_mask
        )

        # === 6. Gather aligned logits ===
        align_s_exp = align_s_clamped.unsqueeze(-1).expand(
            b, max_n, student_logits.shape[-1]
        )
        student_aligned = student_logits.gather(1, align_s_exp)  # [b, max_n, V_s]

        align_t_exp = align_t_clamped.unsqueeze(-1).expand(
            b, max_n, teacher_logits.shape[-1]
        )
        teacher_aligned = teacher_logits.gather(1, align_t_exp)  # [b, max_n, V_t]

        # Slice to shared vocab
        student_aligned = student_aligned[:, :, : self.shared_vocab_size]
        teacher_aligned = teacher_aligned[:, :, : self.shared_vocab_size]

        # Mask: valid alignment AND student was masked at that aligned position
        kl_mask = valid_align & student_masked_at_aligned  # [b, max_n]

        n_aligned_masked = kl_mask.sum().item()
        n_aligned_total = valid_align.sum().item()

        result = {
            # Full student data (for CE loss)
            "student_logits": student_logits,
            "masked_mask": masked_mask,
            "maskable_mask": maskable_mask,
            "input_ids": input_ids,
            "loss_weights": loss_weights,
            "outputs": outputs,
            "b": b,
            "l": l_s,
            "t": t,
            "device": device,
            # Aligned data (for distill loss) -- shape [b, max_n, ...]
            "student_logits_aligned": student_aligned,
            "teacher_logits_aligned": teacher_aligned,
            "masked_mask_aligned": kl_mask,
            # Alignment stats
            "n_aligned_masked": n_aligned_masked,
            "n_aligned_total": n_aligned_total,
        }

        return result

    def _compute_loss_aligned(
        self,
        model: transformers.PreTrainedModel | nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        **kwargs,
    ):
        """
        Generic handler for all *_aligned distillation modes.

        Runs the aligned forward pass once, computes CE loss on full student logits,
        then dispatches distillation loss computation on aligned logits.
        Supports single modes (e.g., "kl_aligned") and composite modes
        (e.g., "taid_aligned+kl_aligned").
        """
        assert self.processing_class.padding_side == "right"
        inputs = self._preprocess_inputs(inputs)

        # Parse base mode(s) -- strip "_aligned" suffix from each component
        if "+" in self.distill_mode:
            modes = [
                m.strip().replace("_aligned", "") for m in self.distill_mode.split("+")
            ]
            if "alm" in modes:
                raise ValueError(
                    "ALM has its own cross-tokenizer mechanism and cannot use "
                    "aligned mode."
                )
        else:
            modes = [self.distill_mode.replace("_aligned", "")]

        forward_result = self._forward_student_teacher_aligned(model, inputs)
        ce_loss = self._compute_ce_loss(model, forward_result, inputs)

        # Repackage aligned data for _compute_distill_component
        aligned_result = {
            "student_logits": forward_result["student_logits_aligned"],
            "teacher_logits": forward_result["teacher_logits_aligned"],
            "masked_mask": forward_result["masked_mask_aligned"],
            "t": forward_result["t"],
            "device": forward_result["device"],
        }

        total_loss = ce_loss
        combined_log = {
            "ce_loss": ce_loss.detach().item(),
            "n_aligned_masked": forward_result["n_aligned_masked"],
            "n_aligned_total": forward_result["n_aligned_total"],
        }

        for mode in modes:
            distill_loss = self._compute_distill_component(
                mode, model, inputs, aligned_result
            )
            total_loss = total_loss + distill_loss
            combined_log[f"{mode}_loss"] = distill_loss.detach().item()

        combined_log["loss"] = total_loss.detach().item()
        self._distill_log = combined_log

        outputs = forward_result["outputs"]
        return (total_loss, outputs) if return_outputs else total_loss

    def _compute_loss_kl(
        self,
        model: transformers.PreTrainedModel | nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        **kwargs,
    ):
        """
        Compute combined BD3LM CE loss + KL divergence distillation loss.
        Used for same-tokenizer distillation (e.g. WeDLM -> Qwen3 BD3LM).
        """
        assert self.processing_class.padding_side == "right"
        inputs = self._preprocess_inputs(inputs)

        forward_result = self._forward_student_teacher(model, inputs)
        ce_loss = self._compute_ce_loss(model, forward_result, inputs)

        student_logits = forward_result["student_logits"]
        teacher_logits = forward_result["teacher_logits"]
        masked_mask = forward_result["masked_mask"]
        device = forward_result["device"]

        # === KL divergence loss on shared vocab ===
        T = self.kl_temperature
        # Slice both logits to shared vocab (first shared_vocab_size tokens are identical)
        student_logits_shared = student_logits[:, :, : self.shared_vocab_size]
        teacher_logits_shared = teacher_logits[:, :, : self.shared_vocab_size]

        # Compute KL only on masked positions
        mask_flat = masked_mask.view(-1)  # [b*l]
        student_flat = student_logits_shared.reshape(-1, self.shared_vocab_size)
        teacher_flat = teacher_logits_shared.reshape(-1, self.shared_vocab_size)

        # Select masked positions only
        student_masked = student_flat[mask_flat]  # [n_masked, V]
        teacher_masked = teacher_flat[mask_flat]  # [n_masked, V]

        if student_masked.numel() > 0:
            # KL(teacher || student) = sum(p_teacher * (log p_teacher - log p_student))
            log_p_teacher = F.log_softmax(teacher_masked / T, dim=-1)
            log_p_student = F.log_softmax(student_masked / T, dim=-1)
            kl_per_pos = F.kl_div(
                log_p_student,
                log_p_teacher,
                log_target=True,
                reduction="none",
            ).sum(
                dim=-1
            )  # [n_masked]
            kl_loss = kl_per_pos.mean() * (T * T)
        else:
            kl_loss = torch.tensor(0.0, device=device)

        # === Combine losses ===
        loss = ce_loss + self.kl_weight * kl_loss

        self._distill_log = {
            "ce_loss": ce_loss.detach().item(),
            "kl_loss": kl_loss.detach().item(),
            "loss": loss.detach().item(),
        }

        outputs = forward_result["outputs"]
        return (loss, outputs) if return_outputs else loss

    def _compute_loss_reverse_kl(
        self,
        model: transformers.PreTrainedModel | nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        **kwargs,
    ):
        """
        Compute combined BD3LM CE loss + reverse KL divergence distillation loss.
        Reverse KL = KL(student || teacher), mode-seeking behavior.
        Used for same-tokenizer distillation (e.g. WeDLM -> Qwen3 BD3LM).
        """
        assert self.processing_class.padding_side == "right"
        inputs = self._preprocess_inputs(inputs)

        forward_result = self._forward_student_teacher(model, inputs)
        ce_loss = self._compute_ce_loss(model, forward_result, inputs)

        student_logits = forward_result["student_logits"]
        teacher_logits = forward_result["teacher_logits"]
        masked_mask = forward_result["masked_mask"]
        device = forward_result["device"]

        # === Reverse KL divergence loss on shared vocab ===
        T = self.kl_temperature
        student_logits_shared = student_logits[:, :, : self.shared_vocab_size]
        teacher_logits_shared = teacher_logits[:, :, : self.shared_vocab_size]

        mask_flat = masked_mask.view(-1)
        student_flat = student_logits_shared.reshape(-1, self.shared_vocab_size)
        teacher_flat = teacher_logits_shared.reshape(-1, self.shared_vocab_size)

        student_masked = student_flat[mask_flat]
        teacher_masked = teacher_flat[mask_flat]

        if student_masked.numel() > 0:
            log_p_teacher = F.log_softmax(teacher_masked / T, dim=-1)
            log_p_student = F.log_softmax(student_masked / T, dim=-1)
            # Reverse KL: KL(student || teacher) = sum(p_student * (log p_student - log p_teacher))
            kl_per_pos = F.kl_div(
                log_p_teacher,
                log_p_student,
                log_target=True,
                reduction="none",
            ).sum(dim=-1)
            reverse_kl_loss = kl_per_pos.mean() * (T * T)
        else:
            reverse_kl_loss = torch.tensor(0.0, device=device)

        loss = ce_loss + self.kl_weight * reverse_kl_loss

        self._distill_log = {
            "ce_loss": ce_loss.detach().item(),
            "reverse_kl_loss": reverse_kl_loss.detach().item(),
            "loss": loss.detach().item(),
        }

        outputs = forward_result["outputs"]
        return (loss, outputs) if return_outputs else loss

    def training_step(self, *args, **kwargs):
        loss = super().training_step(*args, **kwargs)
        if hasattr(self, "_distill_log"):
            self.log(self._distill_log)
        return loss
