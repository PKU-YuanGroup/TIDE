"""
accelerate launch \
    --num_processes 4 \
    dllm/pipelines/bert/eval.py \
    --tasks gsm8k_bert \
    --model bert \
    --apply_chat_template \
    --num_fewshot 0 \
    --model_args "pretrained=dllm-hub/ModernBERT-base-chat-v0.1,max_new_tokens=256,steps=256,block_size=32"
"""

from dataclasses import dataclass

from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model

from dllm.core.eval import MDLMEvalConfig, MDLMEvalHarness
from dllm.core.samplers import MDLMSampler, MDLMSamplerConfig


@dataclass
class BERTEvalSamplerConfig(MDLMSamplerConfig):
    """Default sampler config for BERT eval (shorter context)."""

    max_new_tokens: int = 128
    steps: int = 128
    block_size: int = 128


@dataclass
class BERTEvalConfig(MDLMEvalConfig):
    """BERT eval config."""

    max_length: int = 4096


@register_model("bert")
class BERTEvalHarness(MDLMEvalHarness):
    def __init__(
        self,
        eval_config: BERTEvalConfig | None = None,
        sampler_config: MDLMSamplerConfig | None = None,
        sampler_cls: type[MDLMSampler] = MDLMSampler,
        **kwargs,
    ):
        eval_config = eval_config or BERTEvalConfig()
        sampler_config = sampler_config or BERTEvalSamplerConfig()
        super().__init__(
            eval_config=eval_config,
            sampler_config=sampler_config,
            sampler_cls=sampler_cls,
            **kwargs,
        )


if __name__ == "__main__":
    cli_evaluate()
