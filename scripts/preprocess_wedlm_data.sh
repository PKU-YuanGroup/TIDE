#!/usr/bin/env bash
# Preprocess dataset for WeDLM same-tokenizer distillation (Pipeline B).
#
# Tokenizes data with both student and teacher tokenizers (shared vocab),
# computes character-span alignment for position-level distillation.
#
# Usage:
#   bash scripts/preprocess_wedlm_data.sh \
#       --dataset tatsu-lab/alpaca \
#       --output_dir /path/to/output \
#       --student_model dllm-collection/Qwen3-0.6B-diffusion-bd3lm-v0.1 \
#       --teacher_model tencent/WeDLM-8B-Instruct

set -euo pipefail

# ---- Defaults ----
STUDENT_MODEL="${STUDENT_MODEL:-dllm-collection/Qwen3-0.6B-diffusion-bd3lm-v0.1}"
TEACHER_MODEL="${TEACHER_MODEL:-tencent/WeDLM-8B-Instruct}"
DATASET="${DATASET:-tatsu-lab/alpaca}"
OUTPUT_DIR="${OUTPUT_DIR:-data/distill_wedlm_preprocessed}"
MAX_LENGTH="${MAX_LENGTH:-512}"
ALIGN_MODE="${ALIGN_MODE:-kl_aligned}"
NUM_PROC="${NUM_PROC:-16}"

# ---- Parse CLI args ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --student_model)      STUDENT_MODEL="$2"; shift 2 ;;
        --teacher_model)      TEACHER_MODEL="$2"; shift 2 ;;
        --dataset)            DATASET="$2"; shift 2 ;;
        --output_dir)         OUTPUT_DIR="$2"; shift 2 ;;
        --max_length)         MAX_LENGTH="$2"; shift 2 ;;
        --align_mode)         ALIGN_MODE="$2"; shift 2 ;;
        --num_proc)           NUM_PROC="$2"; shift 2 ;;
        *)                    echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ---- Environment ----
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

echo "============================================"
echo "Preprocessing: WeDLM same-tokenizer data"
echo "  Dataset:  ${DATASET}"
echo "  Student:  ${STUDENT_MODEL}"
echo "  Teacher:  ${TEACHER_MODEL}"
echo "  Align:    ${ALIGN_MODE}"
echo "  Output:   ${OUTPUT_DIR}"
echo "============================================"

python examples/a2d/bd3lm/preprocess_distill_wedlm_data.py \
    --model_name_or_path "${STUDENT_MODEL}" \
    --teacher_model_name_or_path "${TEACHER_MODEL}" \
    --dataset_args "${DATASET}" \
    --max_length "${MAX_LENGTH}" \
    --align_mode "${ALIGN_MODE}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_proc "${NUM_PROC}"

echo "=== Preprocessing finished. Output: ${OUTPUT_DIR} ==="
