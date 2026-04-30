#!/usr/bin/env bash
# Preprocess dataset for LLaDA2 cross-tokenizer distillation (Pipeline A).
#
# Tokenizes data with both student and teacher tokenizers, computes
# cross-tokenizer content ranges for ALM alignment.
#
# Usage:
#   bash scripts/preprocess_llada2_data.sh \
#       --dataset tatsu-lab/alpaca \
#       --output_dir /path/to/output \
#       --student_model dllm-collection/Qwen3-0.6B-diffusion-bd3lm-v0.1 \
#       --teacher_model inclusionAI/LLaDA2.0-mini

set -euo pipefail

# ---- Defaults ----
STUDENT_MODEL="${STUDENT_MODEL:-dllm-collection/Qwen3-0.6B-diffusion-bd3lm-v0.1}"
TEACHER_MODEL="${TEACHER_MODEL:-inclusionAI/LLaDA2.0-mini}"
DATASET="${DATASET:-tatsu-lab/alpaca}"
OUTPUT_DIR="${OUTPUT_DIR:-data/distill_llada2_preprocessed}"
MAX_LENGTH="${MAX_LENGTH:-512}"
NUM_PROC="${NUM_PROC:-16}"

# ---- Parse CLI args ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --student_model) STUDENT_MODEL="$2"; shift 2 ;;
        --teacher_model) TEACHER_MODEL="$2"; shift 2 ;;
        --dataset)       DATASET="$2"; shift 2 ;;
        --output_dir)    OUTPUT_DIR="$2"; shift 2 ;;
        --max_length)    MAX_LENGTH="$2"; shift 2 ;;
        --num_proc)      NUM_PROC="$2"; shift 2 ;;
        *)               echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ---- Environment ----
export PYTHONPATH="${PYTHONPATH:-}:$(pwd):$(pwd)/tokenkit"

echo "============================================"
echo "Preprocessing: LLaDA2 cross-tokenizer data"
echo "  Dataset:  ${DATASET}"
echo "  Student:  ${STUDENT_MODEL}"
echo "  Teacher:  ${TEACHER_MODEL}"
echo "  Output:   ${OUTPUT_DIR}"
echo "============================================"

python examples/a2d/bd3lm/preprocess_distill_data.py \
    --model_name_or_path "${STUDENT_MODEL}" \
    --teacher_model_name_or_path "${TEACHER_MODEL}" \
    --dataset_args "${DATASET}" \
    --max_length "${MAX_LENGTH}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_proc "${NUM_PROC}"

echo "=== Preprocessing finished. Output: ${OUTPUT_DIR} ==="
