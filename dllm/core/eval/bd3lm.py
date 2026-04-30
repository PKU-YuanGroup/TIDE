"""
Generic BD3LM eval base: inherit BaseEvalHarness, override sampler hooks for BD3LMSampler.
generate_until scaffolding is inherited from BaseEvalHarness.
Loglikelihood is not supported. Pipelines (e.g. a2d) import and register with @register_model.

Run: Not runnable directly; use pipeline eval entrypoints (e.g. dllm.pipelines.a2d.eval).
"""

from dataclasses import dataclass

from dllm.core.eval.base import BaseEvalConfig, BaseEvalHarness
from dllm.core.samplers import BD3LMSampler, BD3LMSamplerConfig


@dataclass
class BD3LMEvalSamplerConfig(BD3LMSamplerConfig):
    """Default sampler config for BD3LM eval."""

    max_new_tokens: int = 128
    steps: int = 128
    block_size: int = 32


@dataclass
class BD3LMEvalConfig(BaseEvalConfig):
    """Eval-only config for BD3LM."""

    max_length: int = 2048


class BD3LMEvalHarness(BaseEvalHarness):
    """
    BD3LM eval: BaseEvalHarness + generate_until via BD3LMSampler.
    loglikelihood / loglikelihood_rolling not supported.
    """

    def __init__(
        self,
        eval_config: BD3LMEvalConfig | None = None,
        sampler_config: BD3LMSamplerConfig | None = None,
        sampler_cls: type[BD3LMSampler] = BD3LMSampler,
        **kwargs,
    ):
        eval_config = eval_config or BD3LMEvalConfig()
        sampler_config = sampler_config or BD3LMEvalSamplerConfig()

        super().__init__(
            eval_config=eval_config,
            sampler_config=sampler_config,
            sampler_cls=sampler_cls,
            **kwargs,
        )
