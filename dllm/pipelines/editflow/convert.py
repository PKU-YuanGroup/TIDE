"""
python dllm/pipelines/editflow/convert.py --model_name_or_path "answerdotai/ModernBERT-large" --output_dir ".models/editflow/ModernBERT-large"
python dllm/pipelines/editflow/convert.py --model_name_or_path "GSAI-ML/LLaDA-8B-Base" --output_dir ".models/editflow/LLaDA-8B-Base"
python dllm/pipelines/editflow/convert.py --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct" --output_dir ".models/editflow/LLaDA-8B-Instruct"
python dllm/pipelines/editflow/convert.py --model_name_or_path "Dream-org/Dream-v0-Base-7B" --output_dir ".models/editflow/Dream-v0-Base-7B"
python dllm/pipelines/editflow/convert.py --model_name_or_path "Dream-org/Dream-v0-Instruct-7B" --output_dir ".models/editflow/Dream-v0-Instruct-7B"

python dllm/pipelines/a2d/convert.py --model_name_or_path "Qwen/Qwen2.5-0.5B" --output_dir ".models/a2d/Qwen2.5-0.5B"
python dllm/pipelines/editflow/convert.py --model_name_or_path ".models/a2d/Qwen2.5-0.5B" --output_dir ".models/editflow/Qwen2.5-0.5B"

python dllm/pipelines/a2d/convert.py --model_name_or_path "Qwen/Qwen3-0.6B" --output_dir ".models/a2d/Qwen3-0.6B"
python dllm/pipelines/editflow/convert.py --model_name_or_path ".models/a2d/Qwen3-0.6B" --output_dir ".models/editflow/Qwen3-0.6B"
"""

from dataclasses import dataclass

import transformers
import tyro

import dllm

EDITFLOW_MAP = {
    "modernbert": {
        "config": dllm.pipelines.editflow.EditFlowModernBertConfig,
        "lm_head_key": "decoder",
    },
    "llada": {
        "config": dllm.pipelines.editflow.EditFlowLLaDAConfig,
        "lm_head_key": "model.transformer.ff_out",
    },
    "Dream": {
        "config": dllm.pipelines.editflow.EditFlowDreamConfig,
        "lm_head_key": "lm_head",
    },
    "a2d-qwen2": {
        "config": dllm.pipelines.editflow.EditFlowQwen2Config,
        "lm_head_key": "lm_head",
    },
    "a2d-qwen3": {
        "config": dllm.pipelines.editflow.EditFlowQwen3Config,
        "lm_head_key": "lm_head",
    },
}


@dataclass
class ScriptArguments:

    model_name_or_path: str = "answerdotai/ModernBERT-large"
    output_dir: str = ".models/editflow/ModernBERT-large"

    def __post_init__(self):
        self.model_name_or_path = dllm.utils.resolve_with_base_env(
            self.model_name_or_path, "BASE_MODELS_DIR"
        )


def main():

    args = tyro.cli(ScriptArguments)
    dllm.utils.print_args(args)

    # Load source model
    src_model = transformers.AutoModelForMaskedLM.from_pretrained(
        args.model_name_or_path,
        dtype="bfloat16",
    )
    src_tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
    )
    src_config = src_model.config

    # Remove unused HF fields
    for k in ["auto_map", "architectures"]:
        if hasattr(src_config, k):
            delattr(src_config, k)

    # Select corresponding A2D config class
    base_type = src_config.model_type
    tgt_config_cls = EDITFLOW_MAP[base_type]["config"]

    # Build A2D config from source config dict
    cfg_dict = src_config.to_dict()
    cfg_dict.pop("model_type", None)
    tgt_config = tgt_config_cls(**cfg_dict)

    with dllm.utils.init_device_context_manager():
        # Build A2D model
        tgt_model = transformers.AutoModel.from_config(tgt_config)

        src_model = transformers.AutoModelForMaskedLM.from_pretrained(
            args.model_name_or_path,
            dtype="bfloat16",
        )
        # Initialize EditFlow model from the src model: copies backbone & clones lm_head
        dllm.pipelines.editflow.utils.init_editflow_from_src(
            tgt_model,
            src_model,
            lm_head_key=EDITFLOW_MAP[base_type]["lm_head_key"],
        )
        del src_model

        # Save model and config
        tgt_model.save_pretrained(args.output_dir)
        tgt_config.save_pretrained(args.output_dir)
        src_tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
