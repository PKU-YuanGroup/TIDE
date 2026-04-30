#!/usr/bin/env bash
# One-click distillation: LLaDA2 (teacher) -> Qwen3-0.6B BD3LM (student)
#
# Supported distill modes:
#   alm            — Cross-tokenizer ALM (baseline)
#   reverse_alm    — Reverse ALM
#   alm_taid       - ALM + TAID timestep modulation
#   reverse_alm_taid — Reverse ALM + TAID
#
# Add --use_complementary_demo True for CompDemo (combinable with any mode).
#
# Usage:
#   # Basic ALM distillation (8 GPUs)
#   bash scripts/distill_llada2.sh --data_path /path/to/preprocessed_data
#
#   # ALM-TAID + CompDemo (4 GPUs)
#   bash scripts/distill_llada2.sh \
#       --data_path /path/to/data \
#       --distill_mode alm_taid \
#       --use_comp_demo True \
#       --num_gpus 4
#
#   # Use raw dataset (not preprocessed)
#   bash scripts/distill_llada2.sh \
#       --data_path tatsu-lab/alpaca \
#       --load_preprocessed_data False

set -euo pipefail

# ---- Defaults ----
STUDENT_MODEL="${STUDENT_MODEL:-dllm-collection/Qwen3-0.6B-diffusion-bd3lm-v0.1}"
TEACHER_MODEL="${TEACHER_MODEL:-inclusionAI/LLaDA2.0-mini}"
DATA_PATH="${DATA_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-output/distill_llada2}"
NUM_GPUS="${NUM_GPUS:-8}"
DISTILL_MODE="${DISTILL_MODE:-alm}"
USE_COMP_DEMO="${USE_COMP_DEMO:-False}"
DEMO_RATIO="${DEMO_RATIO:-0.5}"
MAX_LENGTH="${MAX_LENGTH:-512}"
MAX_TEACHER_LENGTH="${MAX_TEACHER_LENGTH:-1024}"
EPOCHS="${EPOCHS:-2}"
LR="${LR:-5e-5}"
BATCH_SIZE="${BATCH_SIZE:-8}"
BLOCK_SIZE="${BLOCK_SIZE:-32}"
ALM_WEIGHT="${ALM_WEIGHT:-1.0}"
ALM_TEMPERATURE="${ALM_TEMPERATURE:-5.0}"
TAID_LAMBDA_INIT="${TAID_LAMBDA_INIT:-0.1}"
TAID_LAMBDA_MAX="${TAID_LAMBDA_MAX:-0.9}"
TAID_LAMBDA_DECAY="${TAID_LAMBDA_DECAY:-cosine}"
LOAD_PREPROCESSED="${LOAD_PREPROCESSED:-True}"
TEACHER_4BIT="${TEACHER_4BIT:-False}"
PORT="${PORT:-30000}"
ATTN_IMPL="${ATTN_IMPL:-flex_attention}"
EXTRA_ARGS=""

# ---- Parse CLI args ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --student_model)       STUDENT_MODEL="$2"; shift 2 ;;
        --teacher_model)       TEACHER_MODEL="$2"; shift 2 ;;
        --data_path)           DATA_PATH="$2"; shift 2 ;;
        --output_dir)          OUTPUT_DIR="$2"; shift 2 ;;
        --num_gpus)            NUM_GPUS="$2"; shift 2 ;;
        --distill_mode)        DISTILL_MODE="$2"; shift 2 ;;
        --use_comp_demo)       USE_COMP_DEMO="$2"; shift 2 ;;
        --demo_ratio)          DEMO_RATIO="$2"; shift 2 ;;
        --max_length)          MAX_LENGTH="$2"; shift 2 ;;
        --max_teacher_length)  MAX_TEACHER_LENGTH="$2"; shift 2 ;;
        --epochs)              EPOCHS="$2"; shift 2 ;;
        --lr)                  LR="$2"; shift 2 ;;
        --batch_size)          BATCH_SIZE="$2"; shift 2 ;;
        --block_size)          BLOCK_SIZE="$2"; shift 2 ;;
        --alm_weight)          ALM_WEIGHT="$2"; shift 2 ;;
        --alm_temperature)     ALM_TEMPERATURE="$2"; shift 2 ;;
        --taid_lambda_init)    TAID_LAMBDA_INIT="$2"; shift 2 ;;
        --taid_lambda_max)     TAID_LAMBDA_MAX="$2"; shift 2 ;;
        --taid_lambda_decay)   TAID_LAMBDA_DECAY="$2"; shift 2 ;;
        --load_preprocessed)   LOAD_PREPROCESSED="$2"; shift 2 ;;
        --teacher_4bit)        TEACHER_4BIT="$2"; shift 2 ;;
        --port)                PORT="$2"; shift 2 ;;
        --attn_impl)           ATTN_IMPL="$2"; shift 2 ;;
        *)                     EXTRA_ARGS="${EXTRA_ARGS} $1"; shift ;;
    esac
done

if [[ -z "${DATA_PATH}" ]]; then
    echo "ERROR: --data_path is required (preprocessed data dir or HuggingFace dataset name)"
    exit 1
fi

# ---- Environment ----
export PYTHONPATH="${PYTHONPATH:-}:$(pwd):$(pwd)/tokenkit"
export WANDB_MODE="${WANDB_MODE:-disabled}"

echo "============================================"
echo "Distillation: LLaDA2 -> Qwen3-0.6B BD3LM"
echo "  Mode:           ${DISTILL_MODE}"
echo "  CompDemo:       ${USE_COMP_DEMO}"
echo "  Student:        ${STUDENT_MODEL}"
echo "  Teacher:        ${TEACHER_MODEL}"
echo "  Data:           ${DATA_PATH}"
echo "  Output:         ${OUTPUT_DIR}"
echo "  GPUs:           ${NUM_GPUS}"
echo "  Epochs:         ${EPOCHS}"
echo "  LR:             ${LR}"
echo "  Batch size:     ${BATCH_SIZE}"
echo "============================================"

# ---- Build TAID args if needed ----
TAID_ARGS=""
if [[ "${DISTILL_MODE}" == *"taid"* ]]; then
    TAID_ARGS="--taid_lambda_init ${TAID_LAMBDA_INIT} --taid_lambda_max ${TAID_LAMBDA_MAX} --taid_lambda_decay ${TAID_LAMBDA_DECAY}"
fi

# ---- Launch training ----
torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${PORT}" \
    examples/a2d/bd3lm/distill.py \
    --model_name_or_path "${STUDENT_MODEL}" \
    --teacher_model_name_or_path "${TEACHER_MODEL}" \
    --teacher_load_in_4bit "${TEACHER_4BIT}" \
    --dataset_args "${DATA_PATH}" \
    --load_preprocessed_data "${LOAD_PREPROCESSED}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_length "${MAX_LENGTH}" \
    --max_teacher_length "${MAX_TEACHER_LENGTH}" \
    --num_train_epochs "${EPOCHS}" \
    --learning_rate "${LR}" \
    --per_device_train_batch_size "${BATCH_SIZE}" \
    --per_device_eval_batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps 1 \
    --block_size "${BLOCK_SIZE}" \
    --distill_mode "${DISTILL_MODE}" \
    --alm_weight "${ALM_WEIGHT}" \
    --alm_temperature "${ALM_TEMPERATURE}" \
    ${TAID_ARGS} \
    --use_complementary_demo "${USE_COMP_DEMO}" \
    --demo_ratio "${DEMO_RATIO}" \
    --attn_implementation "${ATTN_IMPL}" \
    --pad_to_max_length \
    --group_by_length False \
    --logging_steps 10 \
    --save_steps 10000 \
    --bf16 True \
    --report_to wandb \
    ${EXTRA_ARGS}

echo "=== Training finished ==="
