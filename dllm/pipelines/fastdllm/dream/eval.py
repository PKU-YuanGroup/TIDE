"""
accelerate launch \
    --num_processes 4 \
    dllm/pipelines/fastdllm/dream/eval.py \
    --tasks gsm8k_cot \
    --model fastdllm_dream \
    --apply_chat_template \
    --num_fewshot 0 \
    --model_args "pretrained=Dream-org/Dream-v0-Instruct-7B,max_new_tokens=256,steps=256,temperature=0.1,top_p=0.9,alg=entropy,dtype=bfloat16"
"""

from dataclasses import dataclass

from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model

from dllm.pipelines.dream.eval import DreamEvalConfig, DreamEvalHarness
from dllm.pipelines.fastdllm.dream import (
    FastdLLMDreamConfig,
    FastdLLMDreamSampler,
    FastdLLMDreamSamplerConfig,
)


@dataclass
class FastdLLMDreamEvalSamplerConfig(FastdLLMDreamSamplerConfig):
    """Default sampler config for FastdLLM Dream eval."""

    max_new_tokens: int = 128
    steps: int = 128
    temperature: float = 0.0
    top_p: float | None = None
    top_k: float | None = None
    alg: str = "entropy"


@dataclass
class FastdLLMDreamEvalConfig(DreamEvalConfig):
    """FastdLLM Dream eval config."""

    def get_model_config(self, pretrained: str):
        """Return FastdLLM model config so BaseEvalHarness loads the correct model."""
        return FastdLLMDreamConfig.from_pretrained(pretrained)


@register_model("fastdllm_dream")
class FastdLLMDreamEvalHarness(DreamEvalHarness):
    """Dream eval harness for FastdLLM Dream model; inherits from DreamEvalHarness."""

    def __init__(
        self,
        eval_config: FastdLLMDreamEvalConfig | None = None,
        sampler_config: FastdLLMDreamSamplerConfig | None = None,
        sampler_cls: type[FastdLLMDreamSampler] = FastdLLMDreamSampler,
        **kwargs,
    ) -> None:
        eval_config = eval_config or FastdLLMDreamEvalConfig()
        sampler_config = sampler_config or FastdLLMDreamEvalSamplerConfig()

        super().__init__(
            eval_config=eval_config,
            sampler_config=sampler_config,
            sampler_cls=sampler_cls,
            **kwargs,
        )


if __name__ == "__main__":
    cli_evaluate()
