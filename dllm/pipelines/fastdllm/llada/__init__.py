from .models import FastdLLMLLaDAConfig, FastdLLMLLaDAModelLM
from .sampler import FastdLLMLLaDASampler, FastdLLMLLaDASamplerConfig

# Optional: register with transformers Auto classes when available
try:
    from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM

    AutoConfig.register("fastdllm_llada", FastdLLMLLaDAConfig)
    AutoModel.register(FastdLLMLLaDAConfig, FastdLLMLLaDAModelLM)
    AutoModelForMaskedLM.register(FastdLLMLLaDAConfig, FastdLLMLLaDAModelLM)
except ImportError:
    pass

__all__ = [
    "FastdLLMLLaDAConfig",
    "FastdLLMLLaDAModelLM",
    "FastdLLMLLaDASampler",
    "FastdLLMLLaDASamplerConfig",
]
