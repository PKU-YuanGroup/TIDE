"""
Preprocess distillation data to avoid NCCL timeout during training.

This script preprocesses the dataset offline so that training can start
immediately without waiting for data processing.

Example usage:
    PYTHONPATH=$(pwd)/tokenkit:$(pwd) python examples/a2d/bd3lm/preprocess_distill_data.py \
        --model_name_or_path "dllm-collection/Qwen3-0.6B-diffusion-bd3lm-v0.1" \
        --teacher_model_name_or_path "inclusionAI/LLaDA2.0-mini" \
        --dataset_args "allenai/tulu-3-sft-mixture" \
        --max_length 512 \
        --output_dir "data/preprocessed/distill_bd3lm_512"
"""

import os
from dataclasses import dataclass, field
from functools import partial

import transformers

import dllm
from distill_utils import distill_sft_map_fn

logger = dllm.utils.get_default_logger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="dllm-collection/Qwen3-0.6B-diffusion-bd3lm-v0.1",
        metadata={"help": "Student model path (for tokenizer)"},
    )
    teacher_model_name_or_path: str = field(
        default="inclusionAI/LLaDA2.0-mini",
        metadata={"help": "Teacher model path (for tokenizer)"},
    )

    def __post_init__(self):
        from dllm.utils.utils import resolve_with_base_env

        self.model_name_or_path = resolve_with_base_env(
            self.model_name_or_path, "BASE_MODELS_DIR"
        )
        self.teacher_model_name_or_path = resolve_with_base_env(
            self.teacher_model_name_or_path, "BASE_MODELS_DIR"
        )


@dataclass
class DataArguments:
    dataset_args: str = field(
        default="tatsu-lab/alpaca",
        metadata={"help": "Dataset specification (same format as distill.py)"},
    )
    output_dir: str = field(
        default="data/preprocessed/distill_bd3lm",
        metadata={"help": "Output directory for preprocessed data"},
    )
    max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length"},
    )
    mask_prompt_loss: bool = field(
        default=True,
        metadata={"help": "Whether to mask the loss on the prompt tokens"},
    )
    truncation: str = field(
        default="right",
        metadata={"help": "Truncation strategy: 'right' or 'filter'"},
    )
    num_proc: int = field(
        default=None,
        metadata={"help": "Number of processes for dataset mapping (default: all CPUs)"},
    )


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    # Use all CPUs if num_proc not specified
    if data_args.num_proc is None:
        data_args.num_proc = os.cpu_count()

    logger.info(f"Model arguments: {model_args}")
    logger.info(f"Data arguments: {data_args}")

    # ----- Load tokenizers --------------------------------------------------------
    logger.info(f"Loading student tokenizer from {model_args.model_name_or_path}")
    student_tokenizer = dllm.utils.get_tokenizer(
        model_name_or_path=model_args.model_name_or_path,
    )

    logger.info(
        f"Loading teacher tokenizer from {model_args.teacher_model_name_or_path}"
    )
    teacher_tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.teacher_model_name_or_path,
        padding_side="right",
    )
    if not teacher_tokenizer.pad_token:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

    # ----- Load raw dataset -------------------------------------------------------
    logger.info(f"Loading dataset: {data_args.dataset_args}")
    dataset = dllm.data.load_sft_dataset(data_args.dataset_args)
    logger.info(f"Loaded dataset with {len(dataset['train'])} training samples")

    # ----- Apply mapping function -------------------------------------------------
    map_fn = partial(
        distill_sft_map_fn,
        student_tokenizer=student_tokenizer,
        teacher_tokenizer=teacher_tokenizer,
        max_length=data_args.max_length,
        mask_prompt_loss=data_args.mask_prompt_loss,
    )

    keep_cols = {
        "input_ids",
        "labels",
        "attention_mask",
        "teacher_input_ids",
        "teacher_labels",
        "prompt_len",
        "ranges",
    }
    remove_cols = [c for c in dataset["train"].column_names if c not in keep_cols]

    logger.info(f"Mapping dataset with {data_args.num_proc} processes...")
    dataset = dataset.map(
        map_fn,
        num_proc=data_args.num_proc if data_args.num_proc > 1 else None,
        remove_columns=remove_cols,
        load_from_cache_file=False,
        desc="Mapping dataset to distillation SFT format",
    )

    # ----- Filter out discarded samples (prompt >= max_length) --------------------
    if "_discard" in dataset["train"].column_names:
        before = len(dataset["train"])
        dataset = dataset.filter(
            lambda x: not x.get("_discard", False),
            num_proc=data_args.num_proc if data_args.num_proc > 1 else None,
        )
        dataset = dataset.remove_columns(["_discard"])
        after = len(dataset["train"])
        logger.info(
            f"Filtered {before - after} samples where prompt >= max_length "
            f"({before} -> {after})"
        )

    # ----- Post-process (truncate/filter) -----------------------------------------
    logger.info(f"Post-processing with truncation='{data_args.truncation}'...")
    dataset = dllm.utils.post_process_dataset(dataset, data_args)

    if "train" in dataset:
        logger.info(
            f"After post-processing: {len(dataset['train'])} training samples"
        )

    # ----- Save to disk -----------------------------------------------------------
    os.makedirs(data_args.output_dir, exist_ok=True)
    logger.info(f"Saving preprocessed data to {data_args.output_dir}")
    dataset.save_to_disk(data_args.output_dir)
    logger.info("Done!")


if __name__ == "__main__":
    main()
