"""
accelerate launch \
    --num_processes 4 \
    dllm/pipelines/fastdllm/llada/eval.py \
    --tasks gsm8k_cot \
    --model fastdllm_llada \
    --apply_chat_template \
    --num_fewshot 5 \
    --model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,max_new_tokens=512,steps=512,block_size=512"
"""

from dataclasses import dataclass

from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model

from dllm.core.eval import MDLMEvalConfig, MDLMEvalHarness
from dllm.pipelines.fastdllm.llada import (
    FastdLLMLLaDAConfig,
    FastdLLMLLaDASampler,
    FastdLLMLLaDASamplerConfig,
)


@dataclass
class FastdLLMLLaDAEvalSamplerConfig(FastdLLMLLaDASamplerConfig):
    """Default sampler config for FastdLLM LLaDA eval."""

    max_new_tokens: int = 1024
    steps: int = 1024
    block_size: int = 1024


@dataclass
class FastdLLMLLaDAEvalConfig(MDLMEvalConfig):
    """FastdLLM LLaDA eval config (eval-only fields)."""

    max_length: int = 4096

    def get_model_config(self, pretrained: str):
        """Return FastdLLM model config so BaseEvalHarness loads the correct model."""
        return FastdLLMLLaDAConfig.from_pretrained(pretrained)


@register_model("fastdllm_llada")
class FastdLLMLLaDAEvalHarness(MDLMEvalHarness):
    """LLaDA eval harness for FastdLLM LLaDA model; inherits from MDLMEvalHarness."""

    def __init__(
        self,
        eval_config: FastdLLMLLaDAEvalConfig | None = None,
        sampler_config: FastdLLMLLaDASamplerConfig | None = None,
        sampler_cls: type[FastdLLMLLaDASampler] = FastdLLMLLaDASampler,
        **kwargs,
    ):
        eval_config = eval_config or FastdLLMLLaDAEvalConfig()
        sampler_config = sampler_config or FastdLLMLLaDAEvalSamplerConfig()

        super().__init__(
            eval_config=eval_config,
            sampler_config=sampler_config,
            sampler_cls=sampler_cls,
            **kwargs,
        )


if __name__ == "__main__":
    cli_evaluate()
