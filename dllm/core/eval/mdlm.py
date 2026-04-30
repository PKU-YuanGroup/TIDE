"""
Generic MDLM eval harness with loglikelihood (Monte Carlo) and generate_until.
Pipelines inherit and provide EvalConfig + @register_model.

Run: Not runnable directly; use pipeline eval entrypoints (e.g. dllm.pipelines.llada.eval).
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from lm_eval.api.instance import Instance
from tqdm import tqdm

from dllm.core.eval.base import BaseEvalConfig, BaseEvalHarness
from dllm.core.samplers import MDLMSampler, MDLMSamplerConfig


def _parse_token_list(value):
    """Parse token list from string format like '[126081;126348]' or list."""
    if isinstance(value, str):
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            value = value[1:-1]
        if not value:
            return []
        return [int(x.strip()) for x in value.split(";") if x.strip()]
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return []


@dataclass
class MDLMEvalSamplerConfig(MDLMSamplerConfig):
    """Default sampler config for MDLM eval."""

    max_new_tokens: int = 128
    steps: int = 128
    block_size: int = 128


@dataclass
class MDLMEvalConfig(BaseEvalConfig):
    """Eval-only config for MDLM-style models (batch_size, mc_num, etc.)."""

    max_length: int = 2048
    batch_size: int = 32
    mc_num: int = 128
    is_check_greedy: bool = False


class MDLMEvalHarness(BaseEvalHarness):
    """MDLM eval harness: loglikelihood + generate_until (inherited from BaseEvalHarness)."""

    def __init__(
        self,
        eval_config: MDLMEvalConfig | None = None,
        sampler_config: MDLMSamplerConfig | None = None,
        sampler_cls: type[MDLMSampler] = MDLMSampler,
        **kwargs,
    ):
        eval_config = eval_config or MDLMEvalConfig()
        sampler_config = sampler_config or MDLMEvalSamplerConfig()

        # Parse token list strings so _build_config puts them into sampler_config
        for key in ("suppress_tokens", "begin_suppress_tokens"):
            kwargs[key] = _parse_token_list(
                kwargs.get(key, getattr(sampler_config, key, None))
            )

        super().__init__(
            eval_config=eval_config,
            sampler_config=sampler_config,
            sampler_cls=sampler_cls,
            **kwargs,
        )

        self.mask_id = self.tokenizer.mask_token_id
        self.max_length = int(kwargs.get("max_length", eval_config.max_length))
        self.mc_num = int(kwargs.get("mc_num", eval_config.mc_num))
        self.is_check_greedy = kwargs.get(
            "is_check_greedy", eval_config.is_check_greedy
        )

        assert self.mc_num % self.batch_size == 0

    # ── Private helpers (low-level → high-level) ───────────────────────

    def _encode_pair(
        self, context: str, continuation: str
    ) -> tuple[list[int], list[int]]:
        """Encode context and continuation; move trailing spaces from context to continuation."""
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]
        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]
        return context_enc, continuation_enc

    @torch.no_grad()
    def _get_logits(
        self, batch: torch.Tensor, prompt_index: torch.Tensor
    ) -> torch.Tensor:
        """Plain forward; CFG is handled in the sampler (generate_until)."""
        logits = self.model(batch).logits
        return logits[:, : batch.shape[1]]

    def _forward_process(
        self, batch: torch.Tensor, prompt_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply forward diffusion process by masking a random subset of target tokens."""
        b, l = batch.shape
        target_len = (l - prompt_index.sum()).item()
        k = torch.randint(1, target_len + 1, (), device=batch.device)

        x = torch.round(
            torch.linspace(
                float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device
            )
        ).long()
        x = ((x - 1) % target_len) + 1
        assert x.min() >= 1 and x.max() <= target_len

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)

        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]

        is_mask = torch.cat(
            (
                torch.zeros(
                    b, int(prompt_index.sum()), dtype=torch.bool, device=batch.device
                ),
                is_mask,
            ),
            dim=1,
        )

        noisy_batch = torch.where(is_mask, self.mask_id, batch)
        p_mask = (x / target_len).unsqueeze(1).repeat(1, l)
        return noisy_batch, p_mask

    @torch.no_grad()
    def _get_loglikelihood(self, prefix: torch.Tensor, target: torch.Tensor) -> float:
        """Monte Carlo estimate of log-likelihood via _forward_process + _get_logits."""
        seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            perturbed_seq, p_mask = self._forward_process(seq, prompt_index)
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

        return -sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def _suffix_greedy_prediction(
        self, prefix: torch.Tensor, target: torch.Tensor
    ) -> bool:
        """Greedy unmasking check via _get_logits."""
        if not self.is_check_greedy:
            return False

        seq = torch.full(
            (1, len(prefix) + len(target)), self.mask_id, device=self.device
        )
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        prefix, target = prefix.to(self.device), target.to(self.device)
        seq[0, : len(prefix)] = prefix

        for i in range(len(target)):
            mask_index = seq == self.mask_id
            logits = self._get_logits(seq, prompt_index)[mask_index]
            x0 = torch.argmax(logits, dim=-1)
            p = torch.softmax(logits.to(torch.float32), dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(
                dim=-1
            )
            _, index = torch.sort(confidence, descending=True)
            x0[index[1:]] = self.mask_id
            seq[mask_index] = x0.clone()
        correct = target == seq[0, len(prefix) :]
        return torch.all(correct).item()

    # ── Public API (lm-eval interface) ────────────────────────────────

    @torch.no_grad()
    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        out = []
        for instance in tqdm(requests, desc="Computing likelihood..."):
            context_enc, continuation_enc = self._encode_pair(*instance.args)
            assert len(context_enc) + len(continuation_enc) <= self.max_length, (
                f"Context + continuation length exceeds {self.max_length} tokens: "
                f"{len(context_enc)} + {len(continuation_enc)}"
            )

            context = torch.tensor(context_enc, device=self.device, dtype=torch.long)
            continuation = torch.tensor(
                continuation_enc, device=self.device, dtype=torch.long
            )

            logprob = self._get_loglikelihood(context, continuation)
            isgreedy = self._suffix_greedy_prediction(context, continuation)
            out.append((logprob, isgreedy))
        torch.cuda.empty_cache()
        return out
