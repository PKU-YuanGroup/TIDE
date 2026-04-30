"""
python -u examples/editflow/sample.py --model_name_or_path "YOUR_MODEL_PATH"
"""

from dataclasses import dataclass

import transformers

import dllm


@dataclass
class ScriptArguments:
    model_name_or_path: str = (
        ".models/editflow/ModernBERT-large/alpaca/checkpoint-final"
    )
    seed: int = 42
    visualize: bool = True

    def __post_init__(self):
        self.model_name_or_path = dllm.utils.resolve_with_base_env(
            self.model_name_or_path, "BASE_MODELS_DIR"
        )


@dataclass
class SamplerConfig(dllm.pipelines.editflow.EditFlowSamplerConfig):
    tau: float = 0.01
    time_epsilon: float = 1e-3
    mask_length: int = 64
    temperature: float = 0.0


parser = transformers.HfArgumentParser((ScriptArguments, SamplerConfig))
script_args, sampler_config = parser.parse_args_into_dataclasses()
transformers.set_seed(script_args.seed)

# Load model & tokenizer
model = dllm.utils.get_model(model_args=script_args).eval()
tokenizer = dllm.utils.get_tokenizer(model_args=script_args)
sampler = dllm.pipelines.editflow.EditFlowSampler(model=model, tokenizer=tokenizer)
terminal_visualizer = dllm.utils.TerminalVisualizer(tokenizer=tokenizer)

# --- Example 1: Batch sampling ---
print("\n" + "=" * 80)
print("TEST: editflow.sample()".center(80))
print("=" * 80)

messages = [
    [{"role": "user", "content": "Lily runs 12 km/h for 4 hours. How far in 8 hours?"}],
    # [{"role": "user", "content": "Please write an educational python function."}],
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
)
outputs = sampler.sample(inputs, sampler_config, return_dict=True)
sequences = dllm.utils.sample_trim(tokenizer, outputs.sequences.tolist(), inputs)

for iter, s in enumerate(sequences):
    print("\n" + "-" * 80)
    print(f"[Case {iter}]")
    print("-" * 80)
    print(s.strip() if s.strip() else "<empty>")
print("\n" + "=" * 80 + "\n")

if script_args.visualize:
    terminal_visualizer.visualize(outputs.histories, rich=True)
