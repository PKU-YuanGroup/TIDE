"""
Generic eval harness base: accelerator, rank/world_size, model/tokenizer loading,
device, apply_chat_template, tokenizer_name, unified generate_until scaffolding.
Pipeline-agnostic; no MDLM/Dream specifics.

Run: Not runnable directly; use pipeline eval entrypoints (e.g. dllm.pipelines.llada.eval).
"""

import dataclasses
from dataclasses import dataclass
from typing import Optional, Type, TypeVar

import accelerate
import torch
from lm_eval import utils as lm_eval_utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from tqdm import tqdm

import dllm
from dllm.core.samplers import BaseSampler, BaseSamplerConfig
from dllm.utils.configs import ModelArguments

_T = TypeVar("_T")


@dataclass
class BaseEvalConfig:
    """Minimal config for base eval: device and batch_size."""

    pretrained: str = ""
    device: str = "cuda"
    batch_size: int = 1

    def get_model_config(self, pretrained: str):
        """Optional: return custom model config for loading. Default None (use checkpoint config)."""
        return None


class BaseEvalHarness(LM):
    """
    Pipeline-agnostic eval base: accelerator, rank/world_size, model and tokenizer
    loading, device placement, apply_chat_template, tokenizer_name.
    Subclasses implement loglikelihood (and optionally loglikelihood_rolling);
    generate_until is implemented here and uses sampler + sampler_config.
    """

    @classmethod
    def create_from_arg_string(
        cls: Type[_T],
        arg_string: str,
        additional_config: Optional[dict] = None,
    ) -> _T:
        """Override lm-eval's create_from_arg_string to merge model_args and
        additional_config (CLI defaults like batch_size) without conflict.

        Keys in model_args take precedence over CLI defaults so that
        ``--model_args "...,batch_size=4,..."`` works as expected.
        """
        additional_config = {} if additional_config is None else additional_config
        args = lm_eval_utils.simple_parse_args_string(arg_string)
        args2 = {k: v for k, v in additional_config.items() if v is not None}
        merged = {**args2, **args}  # model_args override CLI defaults
        return cls(**merged)

    @staticmethod
    def _build_config(config_cls, source, kwargs):
        """Build a dataclass *config_cls* by copying fields from *source*, with *kwargs* overrides."""
        init = {}
        for f in dataclasses.fields(config_cls):
            if f.name in kwargs:
                init[f.name] = kwargs[f.name]
            elif hasattr(source, f.name):
                init[f.name] = getattr(source, f.name)
        return config_cls(**init)

    def __init__(
        self,
        eval_config: BaseEvalConfig | None = None,
        model_args: ModelArguments | None = None,
        sampler_config: BaseSamplerConfig | None = None,
        sampler_cls: type[BaseSampler] | None = None,
        **kwargs,
    ):
        super().__init__()
        eval_config = eval_config or BaseEvalConfig()
        # Ensure model path is in kwargs and we have a safe default for ModelArguments(__post_init__).
        model_args = model_args or ModelArguments(
            model_name_or_path=kwargs.get("pretrained")
        )
        device = kwargs.get("device", eval_config.device)

        # ── Distributed ──────────────────────────────────────────
        accelerator = accelerate.Accelerator()
        if torch.distributed.is_initialized():
            self._rank = torch.distributed.get_rank()
            self._world_size = torch.distributed.get_world_size()
        else:
            self._rank = 0
            self._world_size = 1

        # ── Model + tokenizer + sampler ──────────────────────────
        if "pretrained" in kwargs:
            kwargs.setdefault("model_name_or_path", kwargs["pretrained"])
        self.model_args = self._build_config(ModelArguments, model_args, kwargs)
        self.model = dllm.utils.get_model(
            self.model_args,
            config=eval_config.get_model_config(self.model_args.model_name_or_path),
        )
        self.model.eval()
        self.tokenizer = dllm.utils.get_tokenizer(self.model_args)
        if sampler_config is not None:
            self.sampler_config = self._build_config(
                type(sampler_config), sampler_config, kwargs
            )
        if sampler_cls is not None:
            self.sampler = sampler_cls(model=self.model, tokenizer=self.tokenizer)

        # ── Device placement ─────────────────────────────────────
        if accelerator.num_processes > 1:
            self.model = accelerator.prepare(self.model)
            self.device = accelerator.device
            self.accelerator = accelerator
        else:
            self.model = self.model.to(device)
            self.device = torch.device(device)
            self.accelerator = None

        self.batch_size = int(kwargs.get("batch_size", eval_config.batch_size))

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    def apply_chat_template(
        self,
        chat_history: list[dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """Format chat history for input to the LM."""
        return self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )

    # ── Unified generate_until scaffolding ────────────────────────────

    # Qwen3 models: inject empty thinking block (enable_thinking=False
    # equivalent) into the prompt so the model skips thinking and directly
    # generates the answer.  This is done in code so the checkpoint's
    # chat_template.jinja does not need to support enable_thinking.
    # Set INJECT_THINK=0 to disable injection (for ablation).
    _THINK_PREFIX = "<think>\n\n</think>\n\n"

    @torch.no_grad()
    def generate_until(self, requests: list[Instance]) -> list[str]:
        import os
        import re

        out: list[str] = []
        inject_think = os.environ.get("INJECT_THINK", "1") != "0"

        for batch_start in tqdm(
            range(0, len(requests), self.batch_size), desc="Generating..."
        ):
            batch = requests[batch_start : batch_start + self.batch_size]
            contexts, gen_kwargs_list = zip(*[inst.args for inst in batch])

            # Inject empty thinking block if not already present.
            # Use `in` instead of `endswith` because tasks with gen_prefix
            # (e.g. HumanEval, MBPP) append code after the think block,
            # so the prompt contains but does not end with the prefix.
            if inject_think:
                contexts = [
                    ctx if self._THINK_PREFIX in ctx else ctx + self._THINK_PREFIX
                    for ctx in contexts
                ]

            prompts = [
                torch.tensor(
                    self.tokenizer(ctx)["input_ids"],
                    device=self.device,
                    dtype=torch.long,
                )
                for ctx in contexts
            ]

            generated_ids = self.sampler.sample(
                inputs=prompts,
                config=self.sampler_config,
                return_dict=False,
            )
            generated_answers = dllm.utils.sample_trim(
                self.tokenizer,
                generated_ids.tolist(),
                [p.tolist() for p in prompts],
            )

            for answer, gen_kwargs in zip(generated_answers, gen_kwargs_list):
                # Strip <think>...</think> blocks (for long-gen tasks where
                # thinking is auto-generated rather than injected into prompt)
                answer = re.sub(
                    r"<think>.*?</think>\s*", "", answer, flags=re.DOTALL
                )
                for stop_seq in gen_kwargs["until"]:
                    if stop_seq in answer:
                        answer = answer.split(stop_seq)[0]
                out.append(answer)

            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()

        return out

    def loglikelihood(self, requests):
        raise NotImplementedError

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError
