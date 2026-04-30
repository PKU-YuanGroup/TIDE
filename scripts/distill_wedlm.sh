#!/usr/bin/env bash
# One-click distillation: WeDLM-8B-Instruct (teacher) -> Qwen3-0.6B BD3LM (student)
#
# Supported distill modes:
#   kl / kl_aligned              — Forward KL divergence
#   reverse_kl / reverse_kl_aligned — Reverse KL divergence
#   taid / taid_aligned          — TAID dual-axis interpolation
#
# Add --use_comp_demo True for CompDemo (combinable with any mode).
# Use *_aligned variants for native chat template alignment.
#
# Usage:
#   # TAID aligned + CompDemo (recommended, 8 GPUs)
#   bash scripts/distill_wedlm.sh \
#       --data_path /path/to/preprocessed_data \
#       --distill_mode taid_aligned \
#       --use_comp_demo True
#
#   # KL aligned (4 GPUs)
#   bash scripts/distill_wedlm.sh \
#       --data_path /path/to/data \
#       --distill_mode kl_aligned \
#       --num_gpus 4

set -euo pipefail

# ---- Defaults ----
STUDENT_MODEL="${STUDENT_MODEL:-dllm-collection/Qwen3-0.6B-diffusion-bd3lm-v0.1}"
TEACHER_MODEL="${TEACHER_MODEL:-tencent/WeDLM-8B-Instruct}"
DATA_PATH="${DATA_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-output/distill_wedlm}"
NUM_GPUS="${NUM_GPUS:-8}"
DISTILL_MODE="${DISTILL_MODE:-taid_aligned}"
USE_COMP_DEMO="${USE_COMP_DEMO:-False}"
DEMO_RATIO="${DEMO_RATIO:-0.5}"
MAX_LENGTH="${MAX_LENGTH:-512}"
TEACHER_MAX_LENGTH="${TEACHER_MAX_LENGTH:-768}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-5e-5}"
BATCH_SIZE="${BATCH_SIZE:-10}"
BLOCK_SIZE="${BLOCK_SIZE:-32}"
CONCAT_ORDER="${CONCAT_ORDER:-x0_xt}"
SHARED_VOCAB_SIZE="${SHARED_VOCAB_SIZE:-151646}"
TEACHER_MASK_TOKEN_ID="${TEACHER_MASK_TOKEN_ID:-151665}"
KL_TEMPERATURE="${KL_TEMPERATURE:-2.0}"
TAID_WEIGHT="${TAID_WEIGHT:-1.0}"
TAID_LAMBDA_INIT="${TAID_LAMBDA_INIT:-0.1}"
TAID_LAMBDA_MAX="${TAID_LAMBDA_MAX:-0.9}"
TAID_LAMBDA_DECAY="${TAID_LAMBDA_DECAY:-cosine}"
TAID_TIMESTEP_WEIGHT="${TAID_TIMESTEP_WEIGHT:-midrange}"
TAID_AXIS_MODE="${TAID_AXIS_MODE:-both}"
LOAD_PREPROCESSED="${LOAD_PREPROCESSED:-True}"
TEACHER_4BIT="${TEACHER_4BIT:-False}"
TEACHER_ATTN="${TEACHER_ATTN:-sdpa}"
PORT="${PORT:-30000}"
ATTN_IMPL="${ATTN_IMPL:-flex_attention}"
EXTRA_ARGS=""

# ---- Parse CLI args ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --student_model)         STUDENT_MODEL="$2"; shift 2 ;;
        --teacher_model)         TEACHER_MODEL="$2"; shift 2 ;;
        --data_path)             DATA_PATH="$2"; shift 2 ;;
        --output_dir)            OUTPUT_DIR="$2"; shift 2 ;;
        --num_gpus)              NUM_GPUS="$2"; shift 2 ;;
        --distill_mode)          DISTILL_MODE="$2"; shift 2 ;;
        --use_comp_demo)         USE_COMP_DEMO="$2"; shift 2 ;;
        --demo_ratio)            DEMO_RATIO="$2"; shift 2 ;;
        --max_length)            MAX_LENGTH="$2"; shift 2 ;;
        --teacher_max_length)    TEACHER_MAX_LENGTH="$2"; shift 2 ;;
        --epochs)                EPOCHS="$2"; shift 2 ;;
        --lr)                    LR="$2"; shift 2 ;;
        --batch_size)            BATCH_SIZE="$2"; shift 2 ;;
        --block_size)            BLOCK_SIZE="$2"; shift 2 ;;
        --concat_order)          CONCAT_ORDER="$2"; shift 2 ;;
        --shared_vocab_size)     SHARED_VOCAB_SIZE="$2"; shift 2 ;;
        --teacher_mask_token_id) TEACHER_MASK_TOKEN_ID="$2"; shift 2 ;;
        --kl_temperature)        KL_TEMPERATURE="$2"; shift 2 ;;
        --taid_weight)           TAID_WEIGHT="$2"; shift 2 ;;
        --taid_lambda_init)      TAID_LAMBDA_INIT="$2"; shift 2 ;;
        --taid_lambda_max)       TAID_LAMBDA_MAX="$2"; shift 2 ;;
        --taid_lambda_decay)     TAID_LAMBDA_DECAY="$2"; shift 2 ;;
        --taid_timestep_weight)  TAID_TIMESTEP_WEIGHT="$2"; shift 2 ;;
        --taid_axis_mode)        TAID_AXIS_MODE="$2"; shift 2 ;;
        --load_preprocessed)     LOAD_PREPROCESSED="$2"; shift 2 ;;
        --teacher_4bit)          TEACHER_4BIT="$2"; shift 2 ;;
        --teacher_attn)          TEACHER_ATTN="$2"; shift 2 ;;
        --port)                  PORT="$2"; shift 2 ;;
        --attn_impl)             ATTN_IMPL="$2"; shift 2 ;;
        *)                       EXTRA_ARGS="${EXTRA_ARGS} $1"; shift ;;
    esac
done

if [[ -z "${DATA_PATH}" ]]; then
    echo "ERROR: --data_path is required (preprocessed data dir or HuggingFace dataset name)"
    exit 1
fi

# ---- Environment ----
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export WANDB_MODE="${WANDB_MODE:-disabled}"

echo "============================================"
echo "Distillation: WeDLM -> Qwen3-0.6B BD3LM"
echo "  Mode:           ${DISTILL_MODE}"
echo "  CompDemo:       ${USE_COMP_DEMO}"
echo "  TAID axis:      ${TAID_AXIS_MODE}"
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
    TAID_ARGS="--taid_weight ${TAID_WEIGHT} --taid_lambda_init ${TAID_LAMBDA_INIT} --taid_lambda_max ${TAID_LAMBDA_MAX} --taid_lambda_decay ${TAID_LAMBDA_DECAY} --taid_timestep_weight ${TAID_TIMESTEP_WEIGHT} --taid_axis_mode ${TAID_AXIS_MODE}"
fi

# ---- Launch training ----
torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${PORT}" \
    examples/a2d/bd3lm/distill_wedlm.py \
    --model_name_or_path "${STUDENT_MODEL}" \
    --teacher_model_name_or_path "${TEACHER_MODEL}" \
    --teacher_load_in_4bit "${TEACHER_4BIT}" \
    --teacher_attn_implementation "${TEACHER_ATTN}" \
    --dataset_args "${DATA_PATH}" \
    --load_preprocessed_data "${LOAD_PREPROCESSED}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_length "${MAX_LENGTH}" \
    --teacher_max_length "${TEACHER_MAX_LENGTH}" \
    --num_train_epochs "${EPOCHS}" \
    --learning_rate "${LR}" \
    --per_device_train_batch_size "${BATCH_SIZE}" \
    --per_device_eval_batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps 1 \
    --block_size "${BLOCK_SIZE}" \
    --distill_mode "${DISTILL_MODE}" \
    --concat_order "${CONCAT_ORDER}" \
    --kl_temperature "${KL_TEMPERATURE}" \
    --teacher_mask_token_id "${TEACHER_MASK_TOKEN_ID}" \
    --shared_vocab_size "${SHARED_VOCAB_SIZE}" \
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
