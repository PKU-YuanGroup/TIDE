#!/usr/bin/env bash
# One-click evaluation: run all 8 benchmarks on a given model checkpoint.
#
# Benchmarks: mmlu_generative_dream, mmlu_pro, hellaswag_gen, gsm8k_cot,
#             bbh, minerva_math, humaneval_instruct, mbpp_instruct
#
# Usage:
#   bash scripts/eval_all.sh --model_path /path/to/checkpoint
#   bash scripts/eval_all.sh --model_path /path/to/checkpoint --num_gpus 4 --output_dir eval_results
#
# Environment variables:
#   INJECT_THINK=0  — Disable thinking prefix injection (default: inject empty think block)

set -uo pipefail

# ---- Defaults ----
MODEL_PATH="${MODEL_PATH:-}"
NUM_GPUS="${NUM_GPUS:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-eval_results}"
PORT="${PORT:-30000}"
BLOCK_SIZE="${BLOCK_SIZE:-32}"
CFG_SCALE="${CFG_SCALE:-0.0}"
MODEL_TYPE="${MODEL_TYPE:-a2d_bd3lm}"

# ---- Parse CLI args ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_path)   MODEL_PATH="$2"; shift 2 ;;
        --num_gpus)     NUM_GPUS="$2"; shift 2 ;;
        --output_dir)   OUTPUT_DIR="$2"; shift 2 ;;
        --port)         PORT="$2"; shift 2 ;;
        --block_size)   BLOCK_SIZE="$2"; shift 2 ;;
        --cfg_scale)    CFG_SCALE="$2"; shift 2 ;;
        --model_type)   MODEL_TYPE="$2"; shift 2 ;;
        *)              echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "${MODEL_PATH}" ]]; then
    echo "ERROR: --model_path is required"
    echo "Usage: bash scripts/eval_all.sh --model_path /path/to/checkpoint"
    exit 1
fi

# ---- Environment ----
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

eval_script="dllm/pipelines/a2d/eval.py"
common_args="--model ${MODEL_TYPE} --apply_chat_template"

mkdir -p "${OUTPUT_DIR}"

echo "============================================"
echo "Evaluation: ${MODEL_PATH}"
echo "  GPUs:       ${NUM_GPUS}"
echo "  Output:     ${OUTPUT_DIR}"
echo "  Model type: ${MODEL_TYPE}"
echo "============================================"

# Task configs: task_name max_new_tokens steps num_fewshot [extra_flags]
tasks=(
    "mmlu_generative_dream 3 3 0"
    "mmlu_pro 256 256 0"
    "hellaswag_gen 3 3 0"
    "gsm8k_cot 256 256 0"
    "bbh 256 256 3"
    "minerva_math 256 256 0"
    "humaneval_instruct 256 256 0 --confirm_run_unsafe_code"
    "mbpp_instruct 256 256 0 --confirm_run_unsafe_code"
)

passed=0
failed=0

for task_line in "${tasks[@]}"; do
    read -r task max_new_tokens steps num_fewshot extra <<< "$task_line"
    log_file="${OUTPUT_DIR}/${task}.log"

    # Skip if already completed
    if [[ -f "${log_file}" ]] && grep -q "^|" "${log_file}"; then
        echo ">>> SKIP ${task} (already done)"
        ((passed++))
        continue
    fi

    echo ">>> [$(date '+%H:%M:%S')] Running ${task} ..."
    torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${PORT}" \
        "${eval_script}" \
        --tasks "${task}" \
        --num_fewshot "${num_fewshot}" \
        ${common_args} \
        --model_args "pretrained=${MODEL_PATH},max_new_tokens=${max_new_tokens},steps=${steps},block_size=${BLOCK_SIZE},cfg_scale=${CFG_SCALE}" \
        --log_samples \
        --output_path "${OUTPUT_DIR}/${task}" \
        ${extra} \
        2>&1 | tee "${log_file}"

    if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
        echo ">>> [$(date '+%H:%M:%S')] PASSED ${task}"
        ((passed++))
    else
        echo ">>> [$(date '+%H:%M:%S')] FAILED ${task}"
        ((failed++))
    fi
    sleep 3
done

echo "============================================"
echo "Evaluation complete: ${passed} passed, ${failed} failed"
echo "Results saved to: ${OUTPUT_DIR}"
echo "============================================"
