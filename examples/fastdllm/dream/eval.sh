#!/usr/bin/env bash
# ===== Mandatory for proper import and evaluation =====
export PYTHONPATH=.:$PYTHONPATH             
export HF_ALLOW_CODE_EVAL=1                 # Allow code evaluation
export HF_DATASETS_TRUST_REMOTE_CODE=True   # For datasets that use remote code

# ===== Optional but recommended for stability and debugging =====
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1    # Enable async error handling for multi-GPU communication to avoid deadlocks
export NCCL_DEBUG=warn                      # Show NCCL warnings for better diagnosis without flooding logs
export TORCH_DISTRIBUTED_DEBUG=DETAIL       # Provide detailed logging for PyTorch distributed debugging

# ===== Input Arguments =====
model_name_or_path="Dream-org/Dream-v0-Base-7B"
instruct=False
num_gpu=1
max_new_tokens=256
block_size=32
threshold="0.9"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_name_or_path)
      model_name_or_path="$2"; shift 2 ;;
    --instruct)
      instruct="$2"; shift 2 ;;
    --num_gpu)
      num_gpu="$2"; shift 2 ;;
    --max_new_tokens)
      max_new_tokens="$2"; shift 2 ;;
    --block_size)
      block_size="$2"; shift 2 ;;
    --threshold)
      threshold="$2"; shift 2 ;;
    *)
      echo "Error: Unknown argument: $1"; exit 1 ;;
  esac
done

# ===== Conditional Configurations =====
# Use --model dream for dllm/pipelines/dream/eval.py; --model fastdllm_dream for dllm/pipelines/fastdllm/dream/eval.py
if [ "$instruct" = "True" ]; then
    echo ">>> Running in INSTRUCT mode"
    dream_args="--model dream --apply_chat_template"
    fastdllm_args="--model fastdllm_dream --apply_chat_template"
else
    echo ">>> Running in BASE mode"
    dream_args="--model dream"
    fastdllm_args="--model fastdllm_dream"
fi

# =======================
# GSM8K Task Evaluation
# =======================

# Baseline
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/dream/eval.py \
    --tasks gsm8k --num_fewshot 5 ${dream_args} \
    --model_args "pretrained=${model_name_or_path},max_new_tokens=${max_new_tokens},steps=${max_new_tokens},alg=entropy,dtype=bfloat16,add_bos_token=True"

# Prefix cache (fastdllm/dream/eval.py, --model fastdllm_dream)
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/fastdllm/dream/eval.py \
    --tasks gsm8k --num_fewshot 5 ${fastdllm_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=prefix,max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=${block_size},alg=entropy,dtype=bfloat16,add_bos_token=True"

# Parallel
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/fastdllm/dream/eval.py \
    --tasks gsm8k --num_fewshot 5 ${fastdllm_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=none,max_new_tokens=${max_new_tokens},steps=$((max_new_tokens/block_size)),block_size=${block_size},temperature=0.0,top_p=0.9,alg=confidence_threshold,threshold=${threshold},dtype=bfloat16,add_bos_token=True"

# Prefix cache + Parallel
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/fastdllm/dream/eval.py \
    --tasks gsm8k --num_fewshot 5 ${fastdllm_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=prefix,max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=${block_size},alg=confidence_threshold,threshold=${threshold},dtype=bfloat16,add_bos_token=True"

# Dual cache + Parallel
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/fastdllm/dream/eval.py \
    --tasks gsm8k --num_fewshot 5 ${fastdllm_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=dual,max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=${block_size},alg=confidence_threshold,threshold=${threshold},dtype=bfloat16,add_bos_token=True"

# ===========================
# Minerva-Math Task Evaluation
# ===========================

# Baseline (dream/eval.py, --model dream)
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/dream/eval.py \
    --tasks minerva_math --num_fewshot 0 ${dream_args} \
    --model_args "pretrained=${model_name_or_path},max_new_tokens=${max_new_tokens},steps=${max_new_tokens},alg=entropy,dtype=bfloat16,add_bos_token=True"

# Prefix cache
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/fastdllm/dream/eval.py \
    --tasks minerva_math --num_fewshot 0 ${fastdllm_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=prefix,max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=${block_size},alg=entropy,dtype=bfloat16,add_bos_token=True"

# Parallel
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/fastdllm/dream/eval.py \
    --tasks minerva_math --num_fewshot 0 ${fastdllm_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=none,max_new_tokens=${max_new_tokens},steps=$((max_new_tokens/block_size)),block_size=${block_size},temperature=0.0,top_p=0.9,alg=confidence_threshold,threshold=${threshold},dtype=bfloat16,add_bos_token=True"

# Prefix cache + Parallel
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/fastdllm/dream/eval.py \
    --tasks minerva_math --num_fewshot 0 ${fastdllm_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=prefix,max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=${block_size},alg=confidence_threshold,threshold=${threshold},dtype=bfloat16,add_bos_token=True"

# Dual cache + Parallel
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/fastdllm/dream/eval.py \
    --tasks minerva_math --num_fewshot 0 ${fastdllm_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=dual,max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=${block_size},alg=confidence_threshold,threshold=${threshold},dtype=bfloat16,add_bos_token=True"

# ===========================
# Humaneval Task Evaluation
# ===========================

# Baseline (dream/eval.py, --model dream)
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/dream/eval.py \
    --tasks humaneval_instruct_dream --num_fewshot 0 ${dream_args} \
    --model_args "pretrained=${model_name_or_path},max_new_tokens=${max_new_tokens},steps=${max_new_tokens},alg=entropy,dtype=bfloat16,add_bos_token=True" \
    --confirm_run_unsafe_code

# Prefix cache
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/fastdllm/dream/eval.py \
    --tasks humaneval_instruct_dream --num_fewshot 0 ${fastdllm_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=prefix,max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=${block_size},temperature=0.0,top_p=0.9,alg=entropy,dtype=bfloat16,add_bos_token=True" \
    --confirm_run_unsafe_code

# Parallel
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/fastdllm/dream/eval.py \
    --tasks humaneval_instruct_dream --num_fewshot 0 ${fastdllm_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=none,max_new_tokens=${max_new_tokens},steps=$((max_new_tokens/block_size)),block_size=${block_size},temperature=0.0,top_p=0.9,alg=confidence_threshold,threshold=${threshold},dtype=bfloat16,add_bos_token=True" \
    --confirm_run_unsafe_code

# Prefix cache + Parallel
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/fastdllm/dream/eval.py \
    --tasks humaneval_instruct_dream --num_fewshot 0 ${fastdllm_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=prefix,max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=${block_size},temperature=0.0,top_p=0.9,alg=confidence_threshold,threshold=${threshold},dtype=bfloat16,add_bos_token=True" \
    --confirm_run_unsafe_code

# Dual cache + Parallel
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/fastdllm/dream/eval.py \
    --tasks humaneval_instruct_dream --num_fewshot 0 ${fastdllm_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=dual,max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=${block_size},temperature=0.0,top_p=0.9,alg=confidence_threshold,threshold=${threshold},dtype=bfloat16,add_bos_token=True" \
    --confirm_run_unsafe_code

# ===========================
# MBPP Task Evaluation
# ===========================

# Baseline (dream/eval.py, --model dream)
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/dream/eval.py \
    --tasks mbpp --num_fewshot 3 ${dream_args} \
    --model_args "pretrained=${model_name_or_path},max_new_tokens=${max_new_tokens},steps=${max_new_tokens},alg=entropy,dtype=bfloat16,temperature=0.2,top_p=0.95,add_bos_token=True" \
    --confirm_run_unsafe_code

# Prefix cache
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/fastdllm/dream/eval.py \
    --tasks mbpp --num_fewshot 3 ${fastdllm_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=prefix,max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=${block_size},alg=entropy,dtype=bfloat16,temperature=0.2,top_p=0.95,add_bos_token=True" \
    --confirm_run_unsafe_code

# Parallel
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/fastdllm/dream/eval.py \
    --tasks mbpp --num_fewshot 3 ${fastdllm_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=none,max_new_tokens=${max_new_tokens},steps=$((max_new_tokens/block_size)),block_size=${block_size},temperature=0.0,top_p=0.9,alg=confidence_threshold,threshold=${threshold},dtype=bfloat16,add_bos_token=True" \
    --confirm_run_unsafe_code

# Prefix cache + Parallel
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/fastdllm/dream/eval.py \
    --tasks mbpp --num_fewshot 3 ${fastdllm_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=prefix,max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=${block_size},alg=confidence_threshold,threshold=${threshold},dtype=bfloat16,temperature=0.0,top_p=0.95,add_bos_token=True" \
    --confirm_run_unsafe_code

# Dual cache + Parallel
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/fastdllm/dream/eval.py \
    --tasks mbpp --num_fewshot 3 ${fastdllm_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=dual,max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=${block_size},alg=confidence_threshold,threshold=${threshold},dtype=bfloat16,temperature=0.0,top_p=0.95,add_bos_token=True" \
    --confirm_run_unsafe_code
