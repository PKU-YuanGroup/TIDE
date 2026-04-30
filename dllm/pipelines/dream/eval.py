"""
accelerate launch \
    --num_processes 4 \
    dllm/pipelines/dream/eval.py \
    --tasks gsm8k_cot \
    --model dream \
    --apply_chat_template \
    --num_fewshot 0 \
    --model_args "pretrained=Dream-org/Dream-v0-Instruct-7B,max_new_tokens=256,steps=256,temperature=0.1,top_p=0.9,alg=entropy,dtype=bfloat16"
"""

import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.registry import register_model
from tqdm import tqdm

from dllm.core.eval import BaseEvalConfig, BaseEvalHarness
from dllm.pipelines.dream import DreamSampler, DreamSamplerConfig

eval_logger = logging.getLogger(__name__)


@dataclass
class DreamEvalSamplerConfig(DreamSamplerConfig):
    """Default sampler config for Dream eval."""

    max_new_tokens: int = 128
    steps: int = 128
    temperature: float = 0.0
    top_p: float | None = None
    top_k: float | None = None
    alg: str = "entropy"
    cfg_scale: float = 0.0


@dataclass
class DreamEvalConfig(BaseEvalConfig):
    """Eval-only config for Dream models."""

    max_length: int = 4096
    add_bos_token: bool = False
    nll_type: str = "mc"
    log_type: str = "ftb"
    mc_num: int = 128
    sampling_eps: float = 1e-3


@register_model("dream")
class DreamEvalHarness(BaseEvalHarness):
    def __init__(
        self,
        eval_config: DreamEvalConfig | None = None,
        sampler_config: DreamSamplerConfig | None = None,
        sampler_cls: type[DreamSampler] = DreamSampler,
        **kwargs,
    ) -> None:
        eval_config = eval_config or DreamEvalConfig()
        sampler_config = sampler_config or DreamEvalSamplerConfig()

        super().__init__(
            eval_config=eval_config,
            sampler_config=sampler_config,
            sampler_cls=sampler_cls,
            **kwargs,
        )

        # Dream-specific eval params (used by loglikelihood methods)
        self.mask_id = self.tokenizer.mask_token_id
        self.max_length = int(kwargs.get("max_length", eval_config.max_length))
        self.add_bos_token = kwargs.get("add_bos_token", eval_config.add_bos_token)
        self.nll_type = kwargs.get("nll_type", eval_config.nll_type)
        self.log_type = kwargs.get("log_type", eval_config.log_type)
        self.mc_num = int(kwargs.get("mc_num", eval_config.mc_num))
        self.sampling_eps = float(kwargs.get("sampling_eps", eval_config.sampling_eps))

    # ── Public API (lm-eval) ──────────────────────────────────────────

    @torch.no_grad()
    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        out = []
        for instance in tqdm(requests, desc="Computing likelihood..."):
            prefix_ids, target_ids = self._encode_pair(*instance.args)
            assert len(prefix_ids) + len(target_ids) <= self.max_length, (
                f"Context + continuation length exceeds {self.max_length} tokens: "
                f"{len(prefix_ids)} + {len(target_ids)}"
            )
            prefix = torch.tensor(prefix_ids, device=self.device, dtype=torch.long)
            target = torch.tensor(target_ids, device=self.device, dtype=torch.long)

            if self.nll_type == "mc":
                ll = -self._eval_target_nll_mc(prefix, target)
                if self.log_type == "union":
                    ll = ll / (len(target) + len(prefix))
            elif self.nll_type == "ar_ftb" or self.nll_type == "ar_btf":
                ll = -self._eval_target_nll_ar(prefix, target)
            else:
                raise NotImplementedError(self.nll_type)

            out.append((ll, False))
        return out

    # ── Private helpers (loglikelihood) ──────────────────────────────

    @torch.no_grad()
    def _forward_process(
        self, batch: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, l = batch.shape
        u0 = torch.rand(1, device=batch.device, dtype=torch.float32)
        indices = torch.arange(b, device=batch.device).float()
        t = (u0 + indices / b) % 1

        p_mask = (1 - self.sampling_eps) * t + self.sampling_eps
        p_mask = p_mask[:, None].repeat(1, l)

        mask_indices = torch.rand((b, l), device=batch.device) < p_mask
        mask_indices[:, 0] = False
        mask_indices[:, -1] = False

        noisy_batch = torch.where(mask_indices, self.mask_id, batch)
        return noisy_batch, p_mask

    @torch.no_grad()
    def _get_logits(
        self, batch: torch.Tensor, prompt_index: torch.Tensor
    ) -> torch.Tensor:
        """Single conditional forward for loglikelihood; CFG is only used in the sampler (generate_until)."""
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = self.model(batch).logits
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        return logits[:, : batch.shape[1]]

    @torch.no_grad()
    def _eval_target_nll_mc(
        self, prefix: torch.Tensor | None, target: torch.Tensor
    ) -> float:
        if prefix is None:
            seq = target[None, :]
        else:
            seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)

        if self.log_type == "ftb":
            prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        else:
            prompt_index = torch.arange(seq.shape[1], device=self.device) >= len(prefix)

        loss_acc = []
        for _ in range(max(self.mc_num // self.batch_size, 1)):
            perturbed_seq = seq.clone()
            perturbed_seq_, p_mask = self._forward_process(seq)
            if self.log_type == "ftb":
                perturbed_seq[:, -len(target) :] = perturbed_seq_[:, -len(target) :]
            elif self.log_type == "btf":
                perturbed_seq[:, : len(prefix)] = perturbed_seq_[:, : len(prefix)]
            elif self.log_type == "union":
                perturbed_seq = perturbed_seq_
            else:
                raise NotImplementedError(self.log_type)

            mask_indices = perturbed_seq == self.mask_id
            logits = self._get_logits(perturbed_seq, prompt_index)
            loss = (
                F.cross_entropy(
                    logits[mask_indices], seq[mask_indices], reduction="none"
                )
                / p_mask[mask_indices]
            )
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())

        return sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def _eval_target_nll_ar(self, prefix: torch.Tensor, target: torch.Tensor) -> float:
        prefix, target = prefix.unsqueeze(0), target.unsqueeze(0)
        assert self.log_type in ["ftb", "btf"]
        assert self.nll_type in ["ar_ftb", "ar_btf"]

        if self.log_type == "ftb":
            prompt_index = (
                torch.arange(prefix.shape[1] + target.shape[1], device=self.device)
                < prefix.shape[1]
            )
        else:
            prompt_index = (
                torch.arange(prefix.shape[1] + target.shape[1], device=self.device)
                >= prefix.shape[1]
            )

        if self.log_type == "ftb":
            perturbed_ = target.repeat(target.shape[1], 1).clone().contiguous()
        else:
            perturbed_ = prefix.repeat(prefix.shape[1], 1).clone().contiguous()

        mask_index = torch.ones(
            (perturbed_.shape[1], perturbed_.shape[1]), dtype=torch.bool
        )
        if self.nll_type == "ar_ftb":
            mask_index = torch.triu(mask_index)
        else:
            mask_index = torch.tril(mask_index)
        perturbed_[mask_index] = self.mask_id
        if self.log_type == "ftb":
            perturbed_seq = torch.cat(
                [prefix.repeat(perturbed_.shape[0], 1), perturbed_], dim=-1
            )
        else:
            perturbed_seq = torch.cat(
                [perturbed_, target.repeat(perturbed_.shape[0], 1)], dim=-1
            )

        logits_ = []
        num = (
            len(perturbed_seq) // self.batch_size
            if len(perturbed_seq) % self.batch_size == 0
            else len(perturbed_seq) // self.batch_size + 1
        )
        for i in range(num):
            end = (
                (i + 1) * self.batch_size
                if (i + 1) * self.batch_size < len(perturbed_seq)
                else len(perturbed_seq)
            )
            perturbed_seq_ = perturbed_seq[i * self.batch_size : end]
            perturbed_seq_ = perturbed_seq_.to(self.device)
            if len(perturbed_seq_.shape) == 1:
                perturbed_seq_ = perturbed_seq_.unsqueeze(0)
            logits = self._get_logits(perturbed_seq_, prompt_index)
            logits_.append(logits.cpu())
        logits = torch.cat(logits_, dim=0)

        temp_index = torch.ones(
            (perturbed_.shape[1], perturbed_.shape[1]), dtype=torch.bool
        )
        if self.nll_type == "ar_ftb":
            temp_index = torch.triu(temp_index, diagonal=1)
        else:
            temp_index = torch.tril(temp_index, diagonal=-1)
        mask_index[temp_index] = False
        if self.log_type == "ftb":
            logits_index = torch.cat(
                [
                    torch.zeros(
                        (perturbed_.shape[1], prefix.shape[1]), dtype=torch.bool
                    ),
                    mask_index,
                ],
                dim=-1,
            )
        else:
            logits_index = torch.cat(
                [
                    mask_index,
                    torch.zeros(
                        (perturbed_.shape[1], target.shape[1]), dtype=torch.bool
                    ),
                ],
                dim=-1,
            )

        if self.log_type == "ftb":
            loss = (
                F.cross_entropy(logits[logits_index], target[0], reduction="sum")
                .cpu()
                .item()
            )
        else:
            loss = (
                F.cross_entropy(logits[logits_index], prefix[0], reduction="sum")
                .cpu()
                .item()
            )
        return loss

    def _encode_pair(
        self, context: str, continuation: str
    ) -> tuple[list[int], list[int]]:
        if self.add_bos_token:
            context = self.tokenizer.bos_token + context

        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer.encode(context + continuation) + [
            self.tokenizer.eos_token_id
        ]
        context_enc = self.tokenizer.encode(context)
        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        cutoff_length = max(len(whole_enc) - self.max_length, 0)
        if cutoff_length > 0:
            eval_logger.warning(
                f"Text length {len(whole_enc)} is larger than {self.max_length}, cutoff on the left side"
            )
            context_remain = context_enc_len - cutoff_length
            if context_remain > 0:
                context_enc = context_enc[-context_remain:]
            else:
                eval_logger.warning("All context (prompt) is truncated.")
                context_enc = []
                continuation_enc = whole_enc[-self.max_length :]
        return context_enc, continuation_enc


if __name__ == "__main__":
    cli_evaluate()
