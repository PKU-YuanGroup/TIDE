"""
LLaDA Fast-dLLM configuration wrapper.
Reuses LLaDAConfig but registers under a different model_type.
"""

from dllm.pipelines.llada.models.configuration_llada import LLaDAConfig


class FastdLLMLLaDAConfig(LLaDAConfig):
    model_type = "fastdllm_llada"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
