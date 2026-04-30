"""
python -u examples/fastdllm/dream/sample.py --model_name_or_path "YOUR_MODEL_PATH"
"""

import time
from dataclasses import dataclass

import transformers

import dllm


@dataclass
class ScriptArguments:
    model_name_or_path: str = "Dream-org/Dream-v0-Instruct-7B"
    seed: int = 42
    visualize: bool = True

    def __post_init__(self):
        self.model_name_or_path = dllm.utils.resolve_with_base_env(
            self.model_name_or_path, "BASE_MODELS_DIR"
        )


@dataclass
class SamplerConfig(dllm.pipelines.fastdllm.dream.FastdLLMDreamSamplerConfig):
    steps: int = 512
    max_new_tokens: int = 512
    temperature: float = 0.0  # Recommended to be 0.0 for alg=="confidence_threshold"
    top_p: float = None
    top_k: int = None
    alg: str = "confidence_threshold"  # "entropy", "confidence_threshold"
    alg_temp: float = 0.0
    threshold: float = 0.9
    use_cache: str = "prefix"  # "none", "prefix", "dual"
    # block_size: int = 32


parser = transformers.HfArgumentParser((ScriptArguments, SamplerConfig))
script_args, sampler_config = parser.parse_args_into_dataclasses()
transformers.set_seed(script_args.seed)
dreamfastdllm_config = (
    dllm.pipelines.fastdllm.dream.FastdLLMDreamConfig.from_pretrained(
        script_args.model_name_or_path
    )
)

# Load model & tokenizer
model = dllm.utils.get_model(model_args=script_args, config=dreamfastdllm_config).eval()
tokenizer = dllm.utils.get_tokenizer(model_args=script_args)
sampler = dllm.pipelines.fastdllm.dream.FastdLLMDreamSampler(
    model=model, tokenizer=tokenizer
)
terminal_visualizer = dllm.utils.TerminalVisualizer(tokenizer=tokenizer)

# --- Example 1: Batch sampling ---
print("\n" + "=" * 80)
print("TEST: dream.sample()".center(80))
print("=" * 80)

messages = [
    [{"role": "user", "content": "Lily runs 12 km/h for 4 hours. How far in 8 hours?"}],
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
)

start = time.time()
outputs = sampler.sample(inputs, sampler_config, return_dict=True)
end = time.time()
sequences = dllm.utils.sample_trim(tokenizer, outputs.sequences.tolist(), inputs)

for iter, s in enumerate(sequences):
    print("\n" + "-" * 80)
    print(f"[Case {iter}]")
    print("-" * 80)
    print(s.strip() if s.strip() else "<empty>")
print("\n" + "=" * 80 + "\n")

if script_args.visualize:
    terminal_visualizer.visualize(outputs.histories, rich=True)

print(
    f"Config: use_cache={sampler_config.use_cache}, threshold={sampler_config.threshold}, steps={sampler_config.steps}"
)
print(
    f"Total NFE:{len(outputs.histories) - 1}. Time taken for sampling: {end - start:.2f} seconds"
)
print(
    f"Token speed: {(len(outputs.sequences[0])-len(inputs[0]))*1.0/(end - start):.2f} tokens/s"
)
