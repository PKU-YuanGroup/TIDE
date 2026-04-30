"""
reference: https://github.com/NVlabs/Fast-dLLM/blob/main/llada/generate.py
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union

import torch
import torch.nn.functional as F

from dllm.core.samplers.base import BaseSampler, BaseSamplerConfig, BaseSamplerOutput
from dllm.core.samplers.utils import add_gumbel_noise, get_num_transfer_tokens


def _trim_past_key_values(
    past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
    upto: int,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Keep only KV up to sequence index `upto` (exclusive) along seq_len dim (-2).
    Assumes each K/V is shaped like [B, H, S, D].
    """
    new_pkv = []
    for layer_kv in past_key_values:
        # layer_kv is usually (k, v)
        new_layer = tuple(t[:, :, :upto] for t in layer_kv)
        new_pkv.append(new_layer)  # type: ignore[arg-type]
    return new_pkv


def _get_transfer_index(
    logits: torch.Tensor,
    temperature: float,
    remasking: str,
    mask_index: torch.Tensor,  # (B, L) bool
    x: torch.Tensor,  # (B, L) long
    num_transfer_tokens: Optional[torch.Tensor] = None,  # (B,) long (top-k mode)
    threshold: Optional[float] = None,  # threshold mode
    factor: Optional[float] = None,  # dynamic mode (highest priority)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        x0:            (B, L) long — proposed tokens
        transfer_index:(B, L) bool — which positions to update

    Priority:
      if factor is not None: dynamic schedule
      elif threshold is not None: threshold mode
      else: top-k mode (num_transfer_tokens required)
    """
    # 1) Propose tokens (greedy / gumbel-max)
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)  # (B, L)

    # 2) Confidence (or random)
    if remasking == "low_confidence":
        p = F.softmax(logits.to(torch.float32), dim=-1)
        conf = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(
            -1
        )  # (B, L) float32
    elif remasking == "random":
        conf = torch.rand(x0.shape, device=x0.device, dtype=torch.float32)
    else:
        raise NotImplementedError(remasking)

    # Only propose changes on masked positions
    x0 = torch.where(mask_index, x0, x)

    # Use a very negative value for non-mask positions so they never get selected
    neg = torch.finfo(conf.dtype).min
    confidence = torch.where(
        mask_index, conf, torch.tensor(neg, device=conf.device, dtype=conf.dtype)
    )  # (B, L)

    # --------------------------
    # A) Dynamic factor schedule
    # --------------------------
    if factor is not None:
        B, L = confidence.shape
        values, idx = torch.sort(confidence, dim=1, descending=True)  # (B, L)

        # rank r = 1..L : thr[r] = 1 - factor/(r+1), but force rank-1 always selectable with -1
        ranks = torch.arange(
            1, L + 1, device=confidence.device, dtype=values.dtype
        )  # (L,)
        factor_t = torch.tensor(
            float(factor), device=confidence.device, dtype=values.dtype
        )
        thr = 1.0 - (factor_t / (ranks + 1.0))  # (L,)
        thr[0] = -1.0

        accept = values >= thr.unsqueeze(0)  # (B, L) bool
        k = accept.sum(dim=1).to(torch.long)  # (B,)

        # never select more than masked count
        n_masked = mask_index.sum(dim=1).to(torch.long)
        k = torch.minimum(k, n_masked)

        cols = torch.arange(L, device=confidence.device).unsqueeze(0).expand(B, L)
        select_sorted = cols < k.unsqueeze(1)  # (B, L)

        transfer_int = torch.zeros(B, L, device=confidence.device, dtype=torch.int8)
        transfer_int = transfer_int.scatter(1, idx, select_sorted.to(torch.int8))
        transfer_index = transfer_int.bool() & mask_index
        return x0, transfer_index

    # --------------------------
    # B) Threshold mode
    # --------------------------
    if threshold is not None:
        transfer_index = mask_index & (confidence >= threshold)

        # force at least one transfer per row if there is any mask and none selected
        has_mask = mask_index.any(dim=1)  # (B,)
        selected = transfer_index.any(dim=1)  # (B,)
        need_force = has_mask & (~selected)
        if need_force.any():
            max_idx = torch.argmax(
                confidence, dim=1, keepdim=True
            )  # (B,1) — safe because non-masks are neg
            force = torch.zeros_like(transfer_index).scatter_(1, max_idx, True)
            transfer_index = (transfer_index | force) & mask_index

        return x0, transfer_index

    # --------------------------
    # C) Top-k (quota) mode
    # --------------------------
    if num_transfer_tokens is None:
        raise ValueError(
            "num_transfer_tokens must be provided when threshold is None and factor is None"
        )

    if num_transfer_tokens.dim() == 2 and num_transfer_tokens.size(1) == 1:
        num_transfer_tokens = num_transfer_tokens.squeeze(1)

    num_transfer_tokens = num_transfer_tokens.to(
        dtype=torch.long, device=confidence.device
    )
    num_transfer_tokens = torch.clamp(num_transfer_tokens, min=0)

    values, idx = torch.sort(confidence, dim=1, descending=True)  # (B, L)
    B, L = confidence.shape

    cols = torch.arange(L, device=confidence.device).unsqueeze(0).expand(B, L)
    select_sorted = cols < num_transfer_tokens.unsqueeze(1)  # (B, L)

    transfer_int = torch.zeros(B, L, device=confidence.device, dtype=torch.int8)
    transfer_int = transfer_int.scatter(1, idx, select_sorted.to(torch.int8))
    transfer_index = transfer_int.bool() & mask_index

    return x0, transfer_index


@dataclass
class FastdLLMLLaDASamplerConfig(BaseSamplerConfig):
    max_new_tokens: int = 128
    max_length: int = None
    block_size: int = 128
    steps: int = 128
    temperature: float = 0.0
    remasking: str = "low_confidence"
    stochastic_transfer: bool = False
    cfg_scale: float = 0.0  # Unused within Fast-dLLM
    cfg_keep_tokens: list[int] | None = None
    suppress_tokens: list[int] | None = None
    begin_suppress_tokens: list[int] | None = None
    right_shift_logits: bool = False

    # Can be "prefix" or "dual" or None
    use_cache: str | None = None

    # Remasking knobs (match NVLabs generate.py behavior)
    threshold: float | None = None
    factor: float | None = None


@dataclass
class FastdLLMLLaDASampler(BaseSampler):
    @torch.no_grad()
    def sample(
        self,
        inputs: Union[List[torch.Tensor], List[List[int]], torch.Tensor],
        config: Optional[FastdLLMLLaDASamplerConfig] = None,
        **kwargs,
    ) -> BaseSamplerOutput | torch.Tensor:
        """
        Fast-dLLM v1 sampler.
        Supports:
          - use_cache=None: baseline (no cache)
          - use_cache="prefix": prefix cache
          - use_cache="dual": dual cache (requires model forward supports replace_position)
        """
        if config is None:
            config = FastdLLMLLaDASamplerConfig()

        # ----- pull args from config, allow kwargs to override -----
        steps = kwargs.get("steps", config.steps)
        max_new_tokens = kwargs.get("max_new_tokens", config.max_new_tokens)
        max_length = kwargs.get("max_length", config.max_length)
        block_size = kwargs.get("block_size", config.block_size)
        temperature = kwargs.get("temperature", config.temperature)
        remasking = kwargs.get("remasking", config.remasking)
        stochastic_transfer = kwargs.get(
            "stochastic_transfer", config.stochastic_transfer
        )
        return_dict = kwargs.get("return_dict", config.return_dict)
        right_shift_logits = kwargs.get("right_shift_logits", config.right_shift_logits)
        suppress_tokens = kwargs.get("suppress_tokens", config.suppress_tokens)
        begin_suppress_tokens = kwargs.get(
            "begin_suppress_tokens", config.begin_suppress_tokens
        )

        use_cache = kwargs.get("use_cache", config.use_cache)
        threshold = kwargs.get("threshold", config.threshold)
        factor = kwargs.get("factor", config.factor)

        assert block_size >= 1
        assert steps >= 1
        mask_id = self.tokenizer.mask_token_id
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id

        # ----- Normalize inputs -> list[1D LongTensor] -----
        if isinstance(inputs, torch.Tensor):
            if inputs.dim() == 1:
                inputs = inputs.unsqueeze(0)
            inputs_list = [
                row.to(device=self.model.device, dtype=torch.long) for row in inputs
            ]
        else:
            # list of lists or list of tensors
            if len(inputs) == 0:
                raise ValueError("inputs is empty")
            if isinstance(inputs[0], list):
                inputs_list = [
                    torch.as_tensor(p, dtype=torch.long, device=self.model.device) for p in inputs  # type: ignore[arg-type]
                ]
            else:
                inputs_list = [p.to(device=self.model.device, dtype=torch.long) for p in inputs]  # type: ignore[arg-type]

        prompt_lens = [p.shape[0] for p in inputs_list]
        B = len(inputs_list)
        max_prompt_len = max(prompt_lens)

        # If right_shift_logits and a sequence has length 0, replace that sequence with [bos] (match your MDLM style)
        if right_shift_logits:
            fixed = []
            for p in inputs_list:
                if p.numel() == 0:
                    fixed.append(
                        torch.tensor(
                            [bos_id], device=self.model.device, dtype=torch.long
                        )
                    )
                else:
                    fixed.append(p)
            inputs_list = fixed
            prompt_lens = [p.shape[0] for p in inputs_list]
            max_prompt_len = max(prompt_lens)

        # determine final T
        if max_new_tokens is not None:
            if max_length is None:
                max_length = max_prompt_len + max_new_tokens
            else:
                # respect explicit max_length
                max_new_tokens = max_length - max_prompt_len
        else:
            if max_length is None:
                raise ValueError("Either max_new_tokens or max_length must be set.")
            max_new_tokens = max_length - max_prompt_len

        T = int(max_length)

        # ----- Build canvas x and attention_mask -----
        # x is right-padded with EOS; prompt left-aligned; generation tail initialized as [MASK]
        x = torch.full((B, T), eos_id, dtype=torch.long, device=self.model.device)
        attention_mask = torch.zeros((B, T), dtype=torch.long, device=self.model.device)

        for i, p in enumerate(inputs_list):
            pl = p.shape[0]
            x[i, :pl] = p
            gen_end = min(pl + max_new_tokens, T)
            x[i, pl:gen_end] = mask_id
            attention_mask[i, :gen_end] = 1

        histories = [x.clone()] if return_dict else None

        # ----- Block scheduling -----
        num_blocks = math.ceil(max_new_tokens / block_size)
        steps_per_block = math.ceil(steps / num_blocks)

        # Cache modes assume a single shared prompt length (like NVLabs reference code)
        if use_cache == "none":
            use_cache = None
        if use_cache not in (None, "prefix", "dual"):
            raise RuntimeError(
                f"Unknown use_cache mode: {use_cache}. Expected None, 'prefix', or 'dual'."
            )
        # Fast-dLLM cache modes require batchsize = 1 or equal prompt lengths
        if use_cache is None:
            prompt_len = None
        else:
            if len(set(prompt_lens)) != 1:
                raise ValueError(
                    f"use_cache={use_cache!r} requires equal prompt lengths in batch. "
                    f"Got prompt_lens={prompt_lens}. "
                    f"Either batch by prompt length or set use_cache=None."
                )
            else:
                prompt_len = prompt_lens[0]

        # Helper: apply token suppressions to logits (in-place)
        def _apply_suppressions(logits_: torch.Tensor):
            if suppress_tokens:
                for tid in suppress_tokens:
                    logits_[:, :, tid] = -torch.inf
            if begin_suppress_tokens:
                # Simple interpretation: always suppress these tokens (you can specialize if needed)
                for tid in begin_suppress_tokens:
                    logits_[:, :, tid] = -torch.inf

        # =============================
        # Main block loop
        # =============================
        for b in range(num_blocks):
            # Compute block boundaries
            if prompt_len is not None:
                # cache modes: shared boundaries
                s = prompt_len + b * block_size
                e = min(s + block_size, prompt_len + max_new_tokens, T)
                if s >= e:
                    continue
                block_len = e - s

                # Build block_mask_index for scheduling (B, block_size), padded with False
                block_mask_index = torch.zeros(
                    (B, block_size), dtype=torch.bool, device=x.device
                )
                block_mask_index[:, :block_len] = x[:, s:e] == mask_id

            else:
                # no-cache mode: per-sample boundaries
                # Build a block_mask_index (B, block_size) with per-sample widths
                block_mask_index = torch.zeros(
                    (B, block_size), dtype=torch.bool, device=x.device
                )
                widths = []
                for j in range(B):
                    start_j = prompt_lens[j] + b * block_size
                    end_j = min(
                        start_j + block_size, prompt_lens[j] + max_new_tokens, T
                    )
                    width_j = max(0, end_j - start_j)
                    widths.append((start_j, end_j, width_j))
                    if width_j > 0:
                        block_mask_index[j, :width_j] = x[j, start_j:end_j] == mask_id

            # quotas for this block
            num_transfer_tokens = get_num_transfer_tokens(
                mask_index=block_mask_index,
                steps=steps_per_block,
                scheduler=self.scheduler,
                stochastic=stochastic_transfer,
            )
            effective_steps = num_transfer_tokens.size(1)

            # -------------------------
            # Mode 1: No cache
            # -------------------------
            if use_cache is None:
                i = 0
                while True:
                    # mask only within current block (per-sample)
                    mask_allowed = torch.zeros_like(x, dtype=torch.bool)

                    for j in range(B):
                        start_j, end_j, width_j = widths[j]
                        if width_j > 0:
                            # only masked positions in current block
                            mask_allowed[j, start_j:end_j] = (
                                x[j, start_j:end_j] == mask_id
                            )

                    if mask_allowed.sum() == 0:
                        break

                    out = self.model(x, attention_mask=attention_mask)
                    logits = out.logits
                    _apply_suppressions(logits)

                    if right_shift_logits:
                        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

                    quota = None if threshold is not None else num_transfer_tokens[:, i]
                    x0, transfer_idx = _get_transfer_index(
                        logits=logits,
                        temperature=temperature,
                        remasking=remasking,
                        mask_index=mask_allowed,
                        x=x,
                        num_transfer_tokens=quota,
                        threshold=threshold,
                        factor=factor,
                    )

                    x = torch.where(transfer_idx, x0, x)
                    i += 1

                    if histories is not None:
                        histories.append(x.clone())

                continue  # next block

            # -------------------------
            # Mode 2: Prefix cache
            # -------------------------
            if use_cache == "prefix":
                # Warm cache on full x once per block
                out_full = self.model(x, attention_mask=attention_mask, use_cache=True)
                logits_full = out_full.logits
                past_key_values = out_full.past_key_values

                _apply_suppressions(logits_full)
                if right_shift_logits:
                    logits_full = torch.cat(
                        [logits_full[:, :1], logits_full[:, :-1]], dim=1
                    )

                # Step 0 update on full logits, restricted to [s:e]
                mask_allowed = torch.zeros_like(x, dtype=torch.bool)
                mask_allowed[:, s:e] = x[:, s:e] == mask_id

                if mask_allowed.sum() > 0:
                    quota = None if threshold is not None else num_transfer_tokens[:, 0]
                    x0, transfer_idx = _get_transfer_index(
                        logits=logits_full,
                        temperature=temperature,
                        remasking=remasking,
                        mask_index=mask_allowed,
                        x=x,
                        num_transfer_tokens=quota,
                        threshold=threshold,
                        factor=factor,
                    )

                    x = torch.where(transfer_idx, x0, x)
                    if histories is not None:
                        histories.append(x.clone())

                # Trim cache to prefix only (up to s)
                if past_key_values is None:
                    raise RuntimeError(
                        "Model did not return past_key_values with use_cache=True"
                    )
                past_key_values = _trim_past_key_values(past_key_values, s)

                # Refinement steps on suffix with prefix cache
                i = 1
                while True:
                    if (x[:, s:e] == mask_id).sum() == 0:
                        break

                    x_suffix = x[:, s:]  # (B, T-s)
                    mask_suffix = x_suffix == mask_id
                    # restrict to current block only
                    if x_suffix.size(1) > block_len:
                        mask_suffix[:, block_len:] = False

                    if mask_suffix.sum() == 0:
                        break

                    out_suf = self.model(
                        x_suffix,
                        attention_mask=attention_mask,  # full-length mask is OK for this model
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    logits_suf = out_suf.logits
                    _apply_suppressions(logits_suf)

                    if right_shift_logits:
                        logits_suf = torch.cat(
                            [logits_suf[:, :1], logits_suf[:, :-1]], dim=1
                        )

                    quota = (
                        None
                        if (threshold is not None or factor is not None)
                        else num_transfer_tokens[:, i]
                    )
                    x0_suf, transfer_suf = _get_transfer_index(
                        logits=logits_suf,
                        temperature=temperature,
                        remasking=remasking,
                        mask_index=mask_suffix,
                        x=x_suffix,
                        num_transfer_tokens=quota,
                        threshold=threshold,
                        factor=factor,
                    )

                    x_suffix_new = torch.where(transfer_suf, x0_suf, x_suffix)
                    x = torch.cat([x[:, :s], x_suffix_new], dim=1)

                    i += 1
                    if histories is not None:
                        histories.append(x.clone())

                continue  # next block

            # -------------------------
            # Mode 3: Dual cache
            # -------------------------
            if use_cache == "dual":
                # Warm cache on full x once per block
                out_full = self.model(x, attention_mask=attention_mask, use_cache=True)
                logits_full = out_full.logits
                past_key_values = out_full.past_key_values
                if past_key_values is None:
                    raise RuntimeError(
                        "Model did not return past_key_values with use_cache=True"
                    )

                _apply_suppressions(logits_full)
                if right_shift_logits:
                    logits_full = torch.cat(
                        [logits_full[:, :1], logits_full[:, :-1]], dim=1
                    )

                # replace_position mask for this block (B, T)
                replace_position = torch.zeros_like(x, dtype=torch.bool)
                replace_position[:, s:e] = True

                # Step 0 update on full logits, restricted to [s:e]
                mask_allowed = torch.zeros_like(x, dtype=torch.bool)
                mask_allowed[:, s:e] = x[:, s:e] == mask_id

                if mask_allowed.sum() > 0:
                    quota = None if threshold is not None else num_transfer_tokens[:, 0]
                    x0, transfer_idx = _get_transfer_index(
                        logits=logits_full,
                        temperature=temperature,
                        remasking=remasking,
                        mask_index=mask_allowed,
                        x=x,
                        num_transfer_tokens=quota,
                        threshold=threshold,
                        factor=factor,
                    )

                    x = torch.where(transfer_idx, x0, x)
                    if histories is not None:
                        histories.append(x.clone())

                # Use for loop here for better compilation performance according to original implementation
                for i_step in range(1, effective_steps):
                    blk = x[:, s:e]
                    mask_blk = blk == mask_id
                    if mask_blk.sum() == 0:
                        break

                    # This requires model forward supports replace_position (as in your first modeling_llada.py)
                    out_blk = self.model(
                        blk,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                        replace_position=replace_position,
                    )
                    logits_blk = out_blk.logits
                    _apply_suppressions(logits_blk)

                    if right_shift_logits:
                        logits_blk = torch.cat(
                            [logits_blk[:, :1], logits_blk[:, :-1]], dim=1
                        )

                    quota = (
                        None
                        if threshold is not None
                        else num_transfer_tokens[:, i_step]
                    )
                    x0_blk, transfer_blk = _get_transfer_index(
                        logits=logits_blk,
                        temperature=temperature,
                        remasking=remasking,
                        mask_index=mask_blk,
                        x=blk,
                        num_transfer_tokens=quota,
                        threshold=threshold,
                        factor=factor,
                    )

                    blk_new = torch.where(transfer_blk, x0_blk, blk)
                    x = torch.cat([x[:, :s], blk_new, x[:, e:]], dim=1)

                    if histories is not None:
                        histories.append(x.clone())

                continue  # next block

            raise ValueError(f"Unknown use_cache mode: {use_cache!r}")

        # ----- Output format -----
        if not return_dict:
            return x
        return BaseSamplerOutput(sequences=x, histories=histories)

    @torch.no_grad()
    def infill(
        self,
        inputs: Union[List[torch.Tensor], List[List[int]]],
        config: FastdLLMLLaDASamplerConfig | None = None,
        **kwargs,
    ) -> BaseSamplerOutput:
        raise NotImplementedError
