"""
Cross-tokenizer distillation: LLaDA2.0-mini (teacher) -> Qwen3-0.6B BD3LM (student).

Local users
------------
- 8 GPUs (ZeRO-2):
    accelerate launch \
        --config_file scripts/accelerate_configs/zero2.yaml --num_processes 8 \
        examples/a2d/bd3lm/distill.py \
        --teacher_model_name_or_path "inclusionAI/LLaDA2.0-mini" \
        --dataset_args "tatsu-lab/alpaca" \
        --max_length 512

Slurm users
------------
- 1 Node, 8 GPUs (ZeRO-2):
    sbatch --gres=gpu:8 scripts/train.slurm.sh \
        --accelerate_config "zero2" \
        --script_path "examples/a2d/bd3lm/distill.py" \
        -- --teacher_model_name_or_path "inclusionAI/LLaDA2.0-mini"
"""

import os
from dataclasses import dataclass, field
from functools import partial

import accelerate
import torch
import transformers

import dllm
from dllm.core.trainers.distill_bd3lm import DistillBD3LMTrainer
from dllm.core.trainers.distill_collator import DistillCollator
from dllm.core.trainers.bd3lm import AppendEOSBlockWrapper
from dllm.utils.collators import FixedLengthPadWrapper
from distill_utils import distill_sft_map_fn

logger = dllm.utils.get_default_logger(__name__)


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = "dllm-collection/Qwen3-0.6B-diffusion-bd3lm-v0.1"


@dataclass
class TeacherModelArguments:
    teacher_model_name_or_path: str = "inclusionAI/LLaDA2.0-mini"
    teacher_dtype: str = "bfloat16"
    teacher_load_in_4bit: bool = True
    teacher_byteify_spec: str = "inclusionAI/LLaDA2.0-mini:source=LLaDA2"
    student_byteify_spec: str = (
        "dllm-collection/Qwen3-0.6B-diffusion-bd3lm-v0.1:source=Qwen3"
    )

    def __post_init__(self):
        from dllm.utils.utils import resolve_with_base_env

        self.teacher_model_name_or_path = resolve_with_base_env(
            self.teacher_model_name_or_path, "BASE_MODELS_DIR"
        )


@dataclass
class DataArguments(dllm.utils.DataArguments):
    dataset_args: str = "tatsu-lab/alpaca"
    max_length: int = 512
    load_preprocessed_data: bool = False
    mask_prompt_loss: bool = field(
        default=True,
        metadata={"help": "Whether to mask the loss on the prompt tokens"},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Pad all student sequences to max_length for fixed-length attention"
        },
    )
    max_teacher_length: int = field(
        default=1024,
        metadata={"help": "Maximum teacher sequence length (cap for cutoff)"},
    )


@dataclass
class TrainingArguments(DistillBD3LMTrainer.DistillBD3LMConfig):
    output_dir: str = "models/a2d/Qwen3-0.6B-bd3lm/distill_llada2"
    remove_unused_columns: bool = False
    group_by_length: bool = True
    num_train_epochs: int = 10
    learning_rate: float = 1e-4
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    # bd3lm
    block_size: int = 32
    # distillation
    alm_weight: float = 1.0
    alm_temperature: float = 2.0
    teacher_mask_token_id: int = 156895


def train():
    # ----- Argument parsing -------------------------------------------------------
    parser = transformers.HfArgumentParser(
        (ModelArguments, TeacherModelArguments, DataArguments, TrainingArguments)
    )
    model_args, teacher_args, data_args, training_args = (
        parser.parse_args_into_dataclasses()
    )
    dllm.utils.print_args_main(model_args, data_args, training_args)
    dllm.utils.initial_training_setup(model_args, data_args, training_args)

    # ----- Student model ----------------------------------------------------------
    model = dllm.utils.get_model(model_args=model_args)

    # ----- Student tokenizer ------------------------------------------------------
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)

    # ----- Teacher model ----------------------------------------------------------
    # Import to register LLaDA2 model with AutoModel
    from dllm.pipelines.llada2 import models as _llada2_models  # noqa: F401

    teacher_dtype = getattr(torch, teacher_args.teacher_dtype, torch.bfloat16)

    device_map = dllm.utils.get_default_device_map()

    teacher_quant_config = None
    if (
        teacher_args.teacher_load_in_4bit
        and transformers.utils.is_bitsandbytes_available()
    ):
        teacher_quant_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=teacher_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    teacher_model = transformers.AutoModelForCausalLM.from_pretrained(
        teacher_args.teacher_model_name_or_path,
        dtype=teacher_dtype,
        device_map=device_map,
        quantization_config=teacher_quant_config,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    if device_map is None and torch.cuda.is_available():
        teacher_model.to(accelerate.PartialState().device)
    teacher_model.eval()
    teacher_model.requires_grad_(False)

    # ----- Teacher tokenizer ------------------------------------------------------
    teacher_tokenizer = transformers.AutoTokenizer.from_pretrained(
        teacher_args.teacher_model_name_or_path,
        padding_side="right",
        trust_remote_code=True,
    )
    if not teacher_tokenizer.pad_token:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token

    # ----- Byteify tokenizers (for alignment) ------------------------------------
    from tokenkit.byteify import load_byteify_tokenizer

    byteify_teacher = load_byteify_tokenizer(teacher_args.teacher_byteify_spec)
    byteify_student = load_byteify_tokenizer(teacher_args.student_byteify_spec)

    # ----- Dataset ----------------------------------------------------------------
    with accelerate.PartialState().local_main_process_first():
        dataset = dllm.data.load_sft_dataset(
            data_args.dataset_args,
            load_preprocessed_data=data_args.load_preprocessed_data,
        )
        if not data_args.load_preprocessed_data:
            map_fn = partial(
                distill_sft_map_fn,
                student_tokenizer=tokenizer,
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
            remove_cols = [
                c for c in dataset["train"].column_names if c not in keep_cols
            ]
            dataset = dataset.map(
                map_fn,
                num_proc=data_args.num_proc if data_args.num_proc > 1 else None,
                remove_columns=remove_cols,
                desc="Mapping dataset to distillation SFT format",
            )
        # truncate / filter long sequences if needed
            dataset = dllm.utils.post_process_dataset(dataset, data_args)

    # ----- Data collator ----------------------------------------------------------
    base_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer,
        return_tensors="pt",
        padding=True,
    )
    if data_args.pad_to_max_length:
        if data_args.max_length % training_args.block_size != 0:
            raise ValueError(
                f"max_length ({data_args.max_length}) must be divisible by "
                f"block_size ({training_args.block_size}) when pad_to_max_length=True"
            )
        base_collator = FixedLengthPadWrapper(
            base_collator,
            pad_to_length=data_args.max_length,
            label_pad_token_id=-100,
            pad_token_id=(
                tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None
                else tokenizer.eos_token_id
            ),
        )
    eos_collator = AppendEOSBlockWrapper(
        base_collator,
        block_size=training_args.block_size,
    )
    data_collator = DistillCollator(
        collator=eos_collator,
        teacher_tokenizer=teacher_tokenizer,
        teacher_pad_token_id=teacher_tokenizer.pad_token_id,
        byteify_teacher=byteify_teacher,
        byteify_student=byteify_student,
        max_teacher_length=data_args.max_teacher_length,
    )

    # ----- Training ---------------------------------------------------------------
    accelerate.PartialState().wait_for_everyone()
    logger.info("Start distillation training...")
    trainer = DistillBD3LMTrainer(
        model=model,
        teacher_model=teacher_model,
        teacher_tokenizer=teacher_tokenizer,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("test", None),
        args=training_args,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(os.path.join(training_args.output_dir, "checkpoint-final"))
    trainer.processing_class.save_pretrained(
        os.path.join(training_args.output_dir, "checkpoint-final")
    )


if __name__ == "__main__":
    train()
