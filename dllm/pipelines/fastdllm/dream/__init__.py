from .models import FastdLLMDreamConfig, FastdLLMDreamModel
from .sampler import FastdLLMDreamSampler, FastdLLMDreamSamplerConfig

__all__ = [
    "FastdLLMDreamConfig",
    "FastdLLMDreamModel",
    "FastdLLMDreamSampler",
    "FastdLLMDreamSamplerConfig",
]

# Register with HuggingFace Auto classes for local usage
try:
    from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM

    AutoConfig.register("fastdllm_dream", FastdLLMDreamConfig)
    AutoModel.register(FastdLLMDreamConfig, FastdLLMDreamModel)
    AutoModelForMaskedLM.register(FastdLLMDreamConfig, FastdLLMDreamModel)
except ImportError:
    # transformers not available or Auto classes not imported
    pass
