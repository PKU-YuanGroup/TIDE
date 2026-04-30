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
model_name_or_path="GSAI-ML/LLaDA-8B-Instruct"
instruct=True
num_gpu=1
max_new_tokens=256
threshold="0.9"
factor="1.0"

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
    --threshold)
      threshold="$2"; shift 2 ;;
    --factor)
      factor="$2"; shift 2 ;;
    *)
      echo "Error: Unknown argument: $1"; exit 1 ;;
  esac
done

# ===== Conditional Configurations =====
# Use --model llada for dllm/pipelines/llada/eval.py; --model fastdllm_llada for dllm/pipelines/fastdllm/llada/eval.py
if [ "$instruct" = "True" ]; then
    echo ">>> Running in INSTRUCT mode"
    llada_args="--model llada --apply_chat_template"
    fastdllm_args="--model fastdllm_llada --apply_chat_template"
else
    echo ">>> Running in BASE mode"
    llada_args="--model llada"
    fastdllm_args="--model fastdllm_llada"
fi

# =======================
# GSM8K Task Evaluation
# =======================

# Baseline (llada/eval.py, --model llada)
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/llada/eval.py \
    --tasks gsm8k --num_fewshot 5 ${llada_args} \
    --model_args "pretrained=${model_name_or_path},max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=32,suppress_tokens=[],begin_suppress_tokens=[126081;126348]"

# Parallel threshold
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/fastdllm/llada/eval.py \
    --tasks gsm8k --num_fewshot 5 ${fastdllm_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=none,threshold=${threshold},max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=32,suppress_tokens=[],begin_suppress_tokens=[]"

# Prefix cache
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/fastdllm/llada/eval.py \
    --tasks gsm8k --num_fewshot 5 ${fastdllm_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=prefix,max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=32,suppress_tokens=[],begin_suppress_tokens=[]"

# Dual cache
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/fastdllm/llada/eval.py \
    --tasks gsm8k --num_fewshot 5 ${fastdllm_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=dual,max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=32,suppress_tokens=[],begin_suppress_tokens=[]"

# Prefix cache + Parallel threshold
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/fastdllm/llada/eval.py \
    --tasks gsm8k --num_fewshot 5 ${fastdllm_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=prefix,threshold=${threshold},max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=32,suppress_tokens=[],begin_suppress_tokens=[]"

# Dual cache + Parallel threshold
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/fastdllm/llada/eval.py \
    --tasks gsm8k --num_fewshot 5 ${fastdllm_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=dual,threshold=${threshold},max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=32,suppress_tokens=[],begin_suppress_tokens=[]"

# Prefix cache + Parallel factor
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/fastdllm/llada/eval.py \
    --tasks gsm8k --num_fewshot 5 ${fastdllm_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=prefix,factor=${factor},max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=32,suppress_tokens=[],begin_suppress_tokens=[]"

# Dual cache + Parallel factor
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/fastdllm/llada/eval.py \
    --tasks gsm8k --num_fewshot 5 ${fastdllm_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=dual,factor=${factor},max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=32,suppress_tokens=[],begin_suppress_tokens=[]"

# ===========================
# MATH (Minerva) Task Evaluation
# ===========================

# Baseline (llada/eval.py, --model llada)
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/llada/eval.py \
    --tasks minerva_math --num_fewshot 4 ${llada_args} \
    --model_args "pretrained=${model_name_or_path},max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=32,cfg_scale=0.0,suppress_tokens=[],begin_suppress_tokens=[]"

# Parallel threshold
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/fastdllm/llada/eval.py \
    --tasks minerva_math --num_fewshot 4 ${fastdllm_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=none,threshold=${threshold},max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=32,cfg_scale=0.0,suppress_tokens=[],begin_suppress_tokens=[]"

# Prefix cache
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/fastdllm/llada/eval.py \
    --tasks minerva_math --num_fewshot 4 ${fastdllm_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=prefix,max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=32,cfg_scale=0.0,suppress_tokens=[],begin_suppress_tokens=[]"

# Dual cache
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/fastdllm/llada/eval.py \
    --tasks minerva_math --num_fewshot 4 ${fastdllm_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=dual,max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=32,cfg_scale=0.0,suppress_tokens=[],begin_suppress_tokens=[]"

# Prefix cache + Parallel threshold
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/fastdllm/llada/eval.py \
    --tasks minerva_math --num_fewshot 4 ${fastdllm_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=prefix,threshold=${threshold},max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=32,cfg_scale=0.0,suppress_tokens=[],begin_suppress_tokens=[]"

# Dual cache + Parallel threshold
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/fastdllm/llada/eval.py \
    --tasks minerva_math --num_fewshot 4 ${fastdllm_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=dual,threshold=${threshold},max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=32,cfg_scale=0.0,suppress_tokens=[],begin_suppress_tokens=[]"

# Prefix cache + Parallel factor
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/fastdllm/llada/eval.py \
    --tasks minerva_math --num_fewshot 4 ${fastdllm_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=prefix,factor=${factor},max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=32,cfg_scale=0.0,suppress_tokens=[],begin_suppress_tokens=[]"

# Dual cache + Parallel factor
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/fastdllm/llada/eval.py \
    --tasks minerva_math --num_fewshot 4 ${fastdllm_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=dual,factor=${factor},max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=32,cfg_scale=0.0,suppress_tokens=[],begin_suppress_tokens=[]"

# ===========================
# HumanEval Task Evaluation
# ===========================

# Baseline (llada/eval.py, --model llada)
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/llada/eval.py \
    --tasks humaneval_instruct_llada --num_fewshot 0 ${llada_args} \
    --model_args "pretrained=${model_name_or_path},max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=32,suppress_tokens=[],begin_suppress_tokens=[126081;126348]" \
    --confirm_run_unsafe_code

# Parallel threshold
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/fastdllm/llada/eval.py \
    --tasks humaneval_instruct_llada --num_fewshot 0 ${fastdllm_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=none,threshold=${threshold},max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=32,suppress_tokens=[],begin_suppress_tokens=[126081;126348]" \
    --confirm_run_unsafe_code

# Prefix cache + Parallel threshold
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/fastdllm/llada/eval.py \
    --tasks humaneval_instruct_llada --num_fewshot 0 ${fastdllm_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=prefix,threshold=${threshold},max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=32,suppress_tokens=[],begin_suppress_tokens=[126081;126348]" \
    --confirm_run_unsafe_code

# Prefix cache + Parallel factor
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/fastdllm/llada/eval.py \
    --tasks humaneval_instruct_llada --num_fewshot 0 ${fastdllm_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=prefix,factor=${factor},max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=32,suppress_tokens=[],begin_suppress_tokens=[126081;126348]" \
    --confirm_run_unsafe_code

# Prefix cache
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/fastdllm/llada/eval.py \
    --tasks humaneval_instruct_llada --num_fewshot 0 ${fastdllm_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=prefix,max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=32,suppress_tokens=[],begin_suppress_tokens=[126081;126348]" \
    --confirm_run_unsafe_code

# Dual cache
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/fastdllm/llada/eval.py \
    --tasks humaneval_instruct_llada --num_fewshot 0 ${fastdllm_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=dual,max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=32,suppress_tokens=[],begin_suppress_tokens=[126081;126348]" \
    --confirm_run_unsafe_code

# Dual cache + Parallel threshold
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/fastdllm/llada/eval.py \
    --tasks humaneval_instruct_llada --num_fewshot 0 ${fastdllm_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=dual,threshold=${threshold},max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=32,suppress_tokens=[],begin_suppress_tokens=[126081;126348]" \
    --confirm_run_unsafe_code

# Dual cache + Parallel factor
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/fastdllm/llada/eval.py \
    --tasks humaneval_instruct_llada --num_fewshot 0 ${fastdllm_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=dual,factor=${factor},max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=32,suppress_tokens=[],begin_suppress_tokens=[126081;126348]" \
    --confirm_run_unsafe_code

# ===========================
# MBPP Task Evaluation
# ===========================

# Baseline (llada/eval.py, --model llada)
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/llada/eval.py \
    --tasks mbpp_instruct_llada --num_fewshot 3 ${llada_args} \
    --model_args "pretrained=${model_name_or_path},max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=32,cfg_scale=0.0,suppress_tokens=[],begin_suppress_tokens=[126081;126348]" \
    --confirm_run_unsafe_code

# Parallel threshold
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/fastdllm/llada/eval.py \
    --tasks mbpp_instruct_llada --num_fewshot 3 ${fastdllm_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=none,threshold=${threshold},max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=32,cfg_scale=0.0,suppress_tokens=[],begin_suppress_tokens=[126081;126348]" \
    --confirm_run_unsafe_code

# Prefix cache
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/fastdllm/llada/eval.py \
    --tasks mbpp_instruct_llada --num_fewshot 3 ${fastdllm_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=prefix,max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=32,cfg_scale=0.0,suppress_tokens=[],begin_suppress_tokens=[126081;126348]" \
    --confirm_run_unsafe_code

# Dual cache
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/fastdllm/llada/eval.py \
    --tasks mbpp_instruct_llada --num_fewshot 3 ${fastdllm_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=dual,max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=32,cfg_scale=0.0,suppress_tokens=[],begin_suppress_tokens=[126081;126348]" \
    --confirm_run_unsafe_code

# Prefix cache + Parallel threshold
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/fastdllm/llada/eval.py \
    --tasks mbpp_instruct_llada --num_fewshot 3 ${fastdllm_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=prefix,threshold=${threshold},max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=32,cfg_scale=0.0,suppress_tokens=[],begin_suppress_tokens=[126081;126348]" \
    --confirm_run_unsafe_code

# Dual cache + Parallel threshold
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/fastdllm/llada/eval.py \
    --tasks mbpp_instruct_llada --num_fewshot 3 ${fastdllm_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=dual,threshold=${threshold},max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=32,cfg_scale=0.0,suppress_tokens=[],begin_suppress_tokens=[126081;126348]" \
    --confirm_run_unsafe_code

# Prefix cache + Parallel factor
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/fastdllm/llada/eval.py \
    --tasks mbpp_instruct_llada --num_fewshot 3 ${fastdllm_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=prefix,factor=${factor},max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=32,cfg_scale=0.0,suppress_tokens=[],begin_suppress_tokens=[126081;126348]" \
    --confirm_run_unsafe_code

# Dual cache + Parallel factor
accelerate launch --num_processes "${num_gpu}" dllm/pipelines/fastdllm/llada/eval.py \
    --tasks mbpp_instruct_llada --num_fewshot 3 ${fastdllm_args} \
    --model_args "pretrained=${model_name_or_path},use_cache=dual,factor=${factor},max_new_tokens=${max_new_tokens},steps=${max_new_tokens},block_size=32,cfg_scale=0.0,suppress_tokens=[],begin_suppress_tokens=[126081;126348]" \
    --confirm_run_unsafe_code
