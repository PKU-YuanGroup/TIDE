"""
KL distillation: WeDLM-8B-Instruct (teacher) -> Qwen3-0.6B BD3LM (student).

Supports multiple distillation modes:
- Non-aligned (single chat template): "kl", "reverse_kl", "taid"
- Aligned (native chat templates): Any *_aligned variant (e.g., "kl_aligned", "taid_aligned",
  "reverse_kl_aligned"). Character-span alignment maps content tokens between templates;
  distillation is computed on aligned positions.
- Composite modes: "taid+kl", "taid_aligned+kl_aligned", etc.

Local users
------------
- 8 GPUs (ZeRO-2, aligned):
    accelerate launch \
        --config_file scripts/accelerate_configs/zero2.yaml --num_processes 8 \
        examples/a2d/bd3lm/distill_wedlm.py \
        --teacher_model_name_or_path "tencent/WeDLM-8B-Instruct" \
        --dataset_args "tatsu-lab/alpaca" \
        --max_length 512 --distill_mode taid_aligned

Slurm users
------------
- 1 Node, 8 GPUs (ZeRO-2):
    sbatch --gres=gpu:8 scripts/train.slurm.sh \
        --accelerate_config "zero2" \
        --script_path "examples/a2d/bd3lm/distill_wedlm.py" \
        -- --teacher_model_name_or_path "tencent/WeDLM-8B-Instruct" \
           --distill_mode kl_aligned
"""

import os
from dataclasses import dataclass, field
from functools import partial

import accelerate
import torch
import transformers
from transformers.trainer_utils import get_last_checkpoint

import dllm
from dllm.core.trainers.distill_bd3lm import DistillBD3LMTrainer
from dllm.core.trainers.bd3lm import AppendEOSBlockWrapper
from dllm.utils.collators import CollatorWrapper, FixedLengthPadWrapper

logger = dllm.utils.get_default_logger(__name__)


@dataclass
class ModelArguments(dllm.utils.ModelArguments):
    model_name_or_path: str = "models/a2d/Qwen3-0.6B"


@dataclass
class TeacherModelArguments:
    teacher_model_name_or_path: str = "tencent/WeDLM-8B-Instruct"
    teacher_dtype: str = "bfloat16"
    teacher_load_in_4bit: bool = True
    teacher_attn_implementation: str = "sdpa"

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
        metadata={"help": "Pad all sequences to max_length for fixed-length attention"},
    )
    teacher_max_length: int = field(
        default=None,
        metadata={
            "help": "Maximum teacher sequence length. Truncates teacher_input_ids and "
            "filters alignment pairs beyond this length. Prevents OOM from "
            "extremely long teacher sequences. Defaults to 2 * max_length."
        },
    )


@dataclass
class TrainingArguments(DistillBD3LMTrainer.DistillBD3LMConfig):
    output_dir: str = "models/a2d/Qwen3-0.6B-bd3lm/distill_wedlm"
    remove_unused_columns: bool = False
    group_by_length: bool = True
    num_train_epochs: int = 10
    learning_rate: float = 1e-4
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    # bd3lm
    block_size: int = 32
    # distillation
    distill_mode: str = "kl"
    concat_order: str = "x0_xt"
    kl_weight: float = 1.0
    kl_temperature: float = 2.0
    teacher_mask_token_id: int = 151665  # WeDLM's mask token in Qwen3 vocab
    shared_vocab_size: int = (
        151646  # number of identical tokens shared between student and teacher tokenizers
    )


@dataclass
class AlignedKLCollator(CollatorWrapper):
    """
    Collator wrapper for kl_aligned mode.

    Wraps the student collator chain and handles teacher-side fields separately.
    Before the base collator runs (which only handles student fields), this wrapper
    extracts teacher_input_ids and alignment indices. After base collation, it pads
    teacher sequences and alignment arrays and adds them back to the batch.
    """

    block_size: int = 32
    teacher_eos_token_id: int = 0
    teacher_pad_token_id: int = 0
    teacher_max_length: int = None

    def before(self, features):
        self._teacher_ids_list = []
        self._align_s_list = []
        self._align_t_list = []

        for ex in features:
            teacher_ids = ex.pop("teacher_input_ids")
            align_s = ex.pop("align_student")
            align_t = ex.pop("align_teacher")

            # Truncate teacher to teacher_max_length, filter alignment pairs
            if self.teacher_max_length is not None and len(teacher_ids) > self.teacher_max_length:
                teacher_ids = teacher_ids[: self.teacher_max_length]
                # Filter alignment pairs where teacher position is beyond truncation
                filtered_s, filtered_t = [], []
                for s, t in zip(align_s, align_t):
                    if t < self.teacher_max_length:
                        filtered_s.append(s)
                        filtered_t.append(t)
                align_s = filtered_s
                align_t = filtered_t

            # Pad teacher sequence to block_size multiple with EOS (matching student treatment)
            L = len(teacher_ids)
            target = (L + self.block_size - 1) // self.block_size * self.block_size
            pad_len = target - L
            if pad_len > 0:
                teacher_ids = teacher_ids + [self.teacher_eos_token_id] * pad_len

            self._teacher_ids_list.append(teacher_ids)
            self._align_s_list.append(align_s)
            self._align_t_list.append(align_t)

        return features

    def after(self, outputs):
        device = outputs["input_ids"].device
        b = len(self._teacher_ids_list)

        # Pad teacher_input_ids to max length in batch
        max_l_t = max(len(ids) for ids in self._teacher_ids_list)
        teacher_input_ids = torch.full(
            (b, max_l_t), self.teacher_pad_token_id, dtype=torch.long, device=device
        )
        teacher_attention_mask = torch.zeros(
            b, max_l_t, dtype=torch.long, device=device
        )
        for i, ids in enumerate(self._teacher_ids_list):
            L = len(ids)
            teacher_input_ids[i, :L] = torch.tensor(ids, dtype=torch.long)
            teacher_attention_mask[i, :L] = 1

        # Pad alignment arrays to max alignment count in batch
        max_n = max((len(s) for s in self._align_s_list), default=0)
        # Ensure at least 1 to avoid zero-size tensors
        max_n = max(max_n, 1)
        align_student = torch.zeros(b, max_n, dtype=torch.long, device=device)
        align_teacher = torch.zeros(b, max_n, dtype=torch.long, device=device)
        n_aligned = torch.zeros(b, dtype=torch.long, device=device)

        for i in range(b):
            n = len(self._align_s_list[i])
            n_aligned[i] = n
            if n > 0:
                align_student[i, :n] = torch.tensor(
                    self._align_s_list[i], dtype=torch.long
                )
                align_teacher[i, :n] = torch.tensor(
                    self._align_t_list[i], dtype=torch.long
                )

        outputs["teacher_input_ids"] = teacher_input_ids
        outputs["teacher_attention_mask"] = teacher_attention_mask
        outputs["align_student"] = align_student
        outputs["align_teacher"] = align_teacher
        outputs["n_aligned"] = n_aligned

        # Clean up
        self._teacher_ids_list = []
        self._align_s_list = []
        self._align_t_list = []

        return outputs


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
        attn_implementation=teacher_args.teacher_attn_implementation,
        trust_remote_code=True,
    )
    if device_map is None and torch.cuda.is_available():
        teacher_model.to(accelerate.PartialState().device)
    teacher_model.eval()
    teacher_model.requires_grad_(False)

    # ----- Teacher tokenizer (for kl_aligned mode) --------------------------------
    teacher_tokenizer = None
    if "_aligned" in training_args.distill_mode:
        teacher_tokenizer = transformers.AutoTokenizer.from_pretrained(
            teacher_args.teacher_model_name_or_path, trust_remote_code=True
        )

    # ----- Dataset ----------------------------------------------------------------
    with accelerate.PartialState().local_main_process_first():
        dataset = dllm.data.load_sft_dataset(
            data_args.dataset_args,
            load_preprocessed_data=data_args.load_preprocessed_data,
        )
        if not data_args.load_preprocessed_data:
            if "_aligned" in training_args.distill_mode:
                from distill_utils import aligned_kl_sft_map_fn

                assert teacher_tokenizer is not None
                map_fn = partial(
                    aligned_kl_sft_map_fn,
                    student_tokenizer=tokenizer,
                    teacher_tokenizer=teacher_tokenizer,
                    max_length=data_args.max_length,
                    mask_prompt_loss=data_args.mask_prompt_loss,
                    align_roles=["assistant"],
                )
                keep_cols = {
                    "input_ids",
                    "labels",
                    "prompt_len",
                    "teacher_input_ids",
                    "align_student",
                    "align_teacher",
                }
            else:
                map_fn = partial(
                    dllm.utils.default_sft_map_fn,
                    tokenizer=tokenizer,
                    mask_prompt_loss=data_args.mask_prompt_loss,
                )
                keep_cols = {"input_ids", "labels", "attention_mask", "prompt_len"}

            remove_cols = [
                c for c in dataset["train"].column_names if c not in keep_cols
            ]
            dataset = dataset.map(
                map_fn,
                num_proc=data_args.num_proc if data_args.num_proc > 1 else None,
                remove_columns=remove_cols,
                desc="Mapping dataset to SFT format",
            )
            if "_aligned" not in training_args.distill_mode:
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
    data_collator = AppendEOSBlockWrapper(
        base_collator,
        block_size=training_args.block_size,
    )

    if "_aligned" in training_args.distill_mode:
        # Default teacher_max_length to 2 * max_length if not set
        teacher_max_length = data_args.teacher_max_length
        if teacher_max_length is None:
            teacher_max_length = data_args.max_length * 2
            logger.info(
                f"teacher_max_length not set, defaulting to 2 * max_length = {teacher_max_length}"
            )

        # Wrap with AlignedKLCollator to handle teacher fields + alignment
        data_collator = AlignedKLCollator(
            collator=data_collator,
            block_size=training_args.block_size,
            teacher_eos_token_id=(
                teacher_tokenizer.eos_token_id if teacher_tokenizer else 0
            ),
            teacher_pad_token_id=(
                teacher_tokenizer.pad_token_id
                if teacher_tokenizer and teacher_tokenizer.pad_token_id is not None
                else (teacher_tokenizer.eos_token_id if teacher_tokenizer else 0)
            ),
            teacher_max_length=teacher_max_length,
        )

    # ----- Training ---------------------------------------------------------------
    accelerate.PartialState().wait_for_everyone()
    logger.info("Start WeDLM KL distillation training...")
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

    # Auto-resume: detect existing checkpoint in output_dir
    resume_from_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_ckpt = get_last_checkpoint(training_args.output_dir)
        if last_ckpt is not None:
            logger.info(f"Resuming from checkpoint: {last_ckpt}")
            resume_from_checkpoint = last_ckpt

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(os.path.join(training_args.output_dir, "checkpoint-final"))
    trainer.processing_class.save_pretrained(
        os.path.join(training_args.output_dir, "checkpoint-final")
    )


if __name__ == "__main__":
    train()
