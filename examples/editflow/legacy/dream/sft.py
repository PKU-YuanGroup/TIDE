"""
Local users
------------
- 1 GPU (LoRA, useful for testing):
    PYTHONPATH=. accelerate launch \
        --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
        examples/editflow/dream/sft.py \
        --lora True

- 8 GPUs (FSDP):
    PYTHONPATH=. accelerate launch \
        --config_file scripts/accelerate_configs/fsdp.yaml \
        examples/editflow/dream/sft.py

Slurm users
# Note: run `mkdir .logs` before running sbatch; and adjust
#       `partition` and `quotatype` in `scripts/train.slurm.sh` for your cluster.
------------
- 1 Node, 8 GPUs (FSDP):
    PYTHONPATH=. sbatch --gres=gpu:8 scripts/train.slurm.sh \
        --accelerate_config "fsdp" \
        --script_path "examples/editflow/dream/sft.py"

- 2 Nodes, 16 GPUs (FSDP):
    PYTHONPATH=. sbatch --nodes=2 --gres=gpu:8 scripts/train.slurm.sh \
        --accelerate_config "fsdp" \
        --script_path "examples/editflow/dream/sft.py"
"""

from dataclasses import dataclass

import transformers

from examples.editflow import sft as editflow_sft


@dataclass
class ModelArguments(editflow_sft.ModelArguments):
    model_name_or_path: str = ".models/editflow/Dream-v0-Instruct-7B"


@dataclass
class DataArguments(editflow_sft.DataArguments):
    dataset_args: str = "tatsu-lab/alpaca"


@dataclass
class TrainingArguments(editflow_sft.TrainingArguments):
    output_dir: str = ".models/editflow/Dream-v0-Instruct-7B/alpaca"


if __name__ == "__main__":
    # ----- Argument parsing -------------------------------------------------------
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    editflow_sft.train(
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
    )
