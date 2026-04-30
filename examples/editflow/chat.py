"""
Interactive chat / sampling script for Bert models.

Examples
--------
# Raw multi-turn sampling (default)
python -u examples/editflow/chat.py --model_name_or_path "YOUR_MODEL_PATH"
"""

import sys
from dataclasses import dataclass

import transformers

import dllm


@dataclass
class ScriptArguments:
    model_name_or_path: str = (
        ".models/editflow/ModernBERT-large/alpaca/checkpoint-final"
    )
    seed: int = 42
    chat_template: bool = True
    visualize: bool = True

    def __post_init__(self):
        # same base-path resolution logic as in sample.py
        self.model_name_or_path = dllm.utils.resolve_with_base_env(
            self.model_name_or_path, "BASE_MODELS_DIR"
        )


@dataclass
class SamplerConfig(dllm.pipelines.editflow.EditFlowSamplerConfig):
    tau: float = 0.01
    time_epsilon: float = 1e-3
    mask_length: int = 64
    temperature: float = 0.0


def main():
    parser = transformers.HfArgumentParser((ScriptArguments, SamplerConfig))
    script_args, sampler_config = parser.parse_args_into_dataclasses()
    transformers.set_seed(script_args.seed)

    model = dllm.utils.get_model(model_args=script_args).eval()
    tokenizer = dllm.utils.get_tokenizer(model_args=script_args)
    sampler = dllm.pipelines.editflow.EditFlowSampler(model=model, tokenizer=tokenizer)

    if script_args.chat_template:
        dllm.utils.multi_turn_chat(
            sampler=sampler,
            sampler_config=sampler_config,
            visualize=script_args.visualize,
        )
    else:
        print("\nSingle-turn sampling (no chat template).")
        dllm.utils.single_turn_sampling(
            sampler=sampler,
            sampler_config=sampler_config,
            visualize=script_args.visualize,
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Bye!")
        sys.exit(0)
