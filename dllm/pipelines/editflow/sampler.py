import math
from dataclasses import dataclass

import torch

from dllm.core.samplers.base import BaseSampler, BaseSamplerConfig, BaseSamplerOutput
from dllm.core.schedulers import BaseKappaScheduler, LinearKappaScheduler


def bernoulli_from_rate(rate: torch.Tensor, tau: float) -> torch.Tensor:
    p = (rate.float() * float(tau)).clamp_(0.0, 1.0 - 1e-6)
    return torch.bernoulli(p)


def sample_from_logits(logits_row: torch.Tensor, temperature: float) -> int:
    if temperature <= 0.0:
        return int(torch.argmax(logits_row).item())
    return int(
        torch.distributions.Categorical(logits=(logits_row / temperature))
        .sample()
        .item()
    )


@torch.no_grad()
def tau_leap_step(
    x: torch.Tensor,  # [T]
    model,
    prompt_len: int,
    t: float,
    sched: BaseKappaScheduler,
    tau: float,
    temperature: float,
    edit_prompt: bool,
    prev_out: dict | None = None,
    reuse_prev: bool = False,
) -> tuple[torch.Tensor, bool, dict]:
    """One τ-leap step for bs=1."""
    T = x.numel()
    use_reuse = bool(reuse_prev and (prev_out is not None))
    if use_reuse:
        out = prev_out
    else:
        attn = torch.ones(1, T, dtype=torch.long, device=x.device)
        t_tensor = torch.full((1, 1), float(t), device=x.device)
        out = model(input_ids=x.unsqueeze(0), attention_mask=attn, t=t_tensor)

    del_rate = out["del_rate_hat"]
    sub_rate = out["sub_rate_hat"]
    ins_rate = out["ins_rate_hat"]
    sub_logits = out["sub_logits"]
    ins_logits = out["ins_logits"]

    w = sched.weight(torch.tensor([[t]], device=x.device))
    del_rate = del_rate * w
    sub_rate = sub_rate * w
    ins_rate = ins_rate * w

    prompt_len = int(max(1, min(prompt_len, T)))
    if not edit_prompt:
        del_rate[:, :prompt_len] = 0.0
        sub_rate[:, :prompt_len] = 0.0
        if prompt_len >= 2:
            ins_rate[:, : prompt_len - 1] = 0.0

    comb_rate = (del_rate + sub_rate).squeeze(0)  # [T]
    comb_fire = bernoulli_from_rate(comb_rate, tau).bool()
    p_del = (del_rate.squeeze(0) / (comb_rate + 1e-8)).clamp(0, 1)
    choose_del = (torch.rand_like(p_del) < p_del) & comb_fire
    choose_sub = comb_fire & (~choose_del)
    ins_fire = bernoulli_from_rate(ins_rate.squeeze(0), tau).bool()

    sub_samples: list[int | None] = [
        (sample_from_logits(sub_logits[0, i], temperature) if choose_sub[i] else None)
        for i in range(T)
    ]
    ins_samples: list[int | None] = [
        sample_from_logits(ins_logits[0, i], temperature) if ins_fire[i] else None
        for i in range(T)
    ]

    new_ids: list[int] = []
    for i in range(T):
        if choose_del[i]:
            pass
        elif choose_sub[i]:
            new_ids.append(int(sub_samples[i]))
        else:
            new_ids.append(int(x[i].item()))
        if ins_samples[i] is not None:
            new_ids.append(int(ins_samples[i]))

    x_next = torch.tensor(new_ids, dtype=torch.long, device=x.device)
    any_edit = bool(comb_fire.any().item() or ins_fire.any().item())
    return x_next, any_edit, out


@dataclass
class EditFlowSamplerConfig(BaseSamplerConfig):
    tau: float = 0.01
    time_epsilon: float = 1e-3
    mask_length: int = 128
    temperature: float = 0.0
    edit_prompt: bool = False
    time_independent: bool = True


@dataclass
class EditFlowSampler(BaseSampler):

    @torch.no_grad()
    def sample(
        self,
        inputs: list[torch.Tensor | list],
        config: EditFlowSamplerConfig | None = None,
        **kwargs,
    ) -> BaseSamplerOutput | torch.Tensor:

        if config is None:
            config = EditFlowSamplerConfig()

        tau = kwargs.get("tau", config.tau)
        time_epsilon = kwargs.get("time_epsilon", config.time_epsilon)
        mask_length = kwargs.get("mask_length", config.mask_length)
        temperature = kwargs.get("temperature", config.temperature)
        edit_prompt = kwargs.get("edit_prompt", config.edit_prompt)
        time_independent = kwargs.get("time_independent", config.time_independent)
        return_dict = kwargs.get("return_dict", config.return_dict)

        if len(inputs) != 1:
            raise NotImplementedError("EditFlowSampler only supports bs=1")

        x = inputs[0]
        if isinstance(x, list):
            x = torch.as_tensor(x, dtype=torch.long, device=self.model.device)
        if x.dim() == 2:
            if x.size(0) != 1:
                raise NotImplementedError("EditFlowSampler only supports bs=1")
            x = x.squeeze(0)

        bos = self.tokenizer.bos_token_id
        if bos is None:
            raise ValueError("tokenizer.bos_token_id must be set")
        if x.numel() == 0:
            x = torch.tensor([bos], dtype=torch.long, device=self.model.device)
        elif int(x[0].item()) != int(bos):
            x = torch.cat(
                [torch.tensor([bos], dtype=torch.long, device=x.device), x], dim=0
            )

        prompt_len = int(x.numel())

        if mask_length:
            mask_id = self.tokenizer.mask_token_id
            if mask_id is None:
                raise ValueError(
                    "tokenizer.mask_token_id must be set when mask_length>0"
                )
            x = torch.cat(
                [
                    x,
                    torch.full(
                        (mask_length,), mask_id, dtype=torch.long, device=x.device
                    ),
                ],
                dim=0,
            )

        sched = LinearKappaScheduler()
        steps = math.ceil(1.0 / max(float(tau), 1e-9))
        histories = [x.unsqueeze(0).clone()] if return_dict else None

        prev_out: dict | None = None
        prev_had_edits = True
        t = 0.0
        for _ in range(steps):
            reuse_prev = bool(
                time_independent and (not prev_had_edits) and (prev_out is not None)
            )
            x, prev_had_edits, prev_out = tau_leap_step(
                x=x,
                model=self.model,
                prompt_len=prompt_len,
                t=t,
                sched=sched,
                tau=float(tau),
                temperature=float(temperature),
                edit_prompt=bool(edit_prompt),
                prev_out=prev_out,
                reuse_prev=reuse_prev,
            )
            if histories is not None:
                histories.append(x.unsqueeze(0).clone())

            t = min(1.0, t + float(tau))
            if t >= 1.0 - float(time_epsilon):
                break

        x = x.unsqueeze(0)  # [1, T]
        if not return_dict:
            return x
        return BaseSamplerOutput(sequences=x, histories=histories)

    @torch.no_grad()
    def infill(
        self,
        inputs: list[torch.Tensor | list],
        config: BaseSamplerConfig | None = None,
        **kwargs,
    ) -> BaseSamplerOutput:
        raise NotImplementedError
