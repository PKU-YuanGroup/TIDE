# A2D (AR-to-Diffusion)

[![Hugging Face Checkpoints](https://img.shields.io/badge/Hugging%20Face-Checkpoints-yellow)](https://huggingface.co/collections/dllm-hub/tiny-a2d)
[![W&B Report](https://img.shields.io/badge/W&B-Report-white?logo=weightsandbiases)](https://wandb.ai/asap-zzhou/dllm/reports/dLLM-Tiny-A2D--VmlldzoxNTI2NTEzOA)


This directory provides two key sets of resources:

- **Warmup**: Tutorials for continual pretraining and SFTing any autoregressive model on small datasets to generate text with [MDLM](#warmup-mdlm) (masked diffusion) or [BD3LM](#warmup-bd3lm) (block diffusion).
- **[`Tiny-A2D`](#tiny-a2d)**: The exact training, inference, and evaluation scripts for developing: [`Qwen3-0.6B-diffusion-mdlm-v0.1`](https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1) and [`Qwen3-0.6B-diffusion-bd3lm-v0.1`](https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1).
For detailed experimental results and reproduction instructions, please see our [![blog](https://img.shields.io/badge/W&B-white?logo=weightsandbiases) Tiny-A2D Report](https://wandb.ai/asap-zzhou/dllm/reports/dLLM-Tiny-A2D--VmlldzoxNTI2NTEzOA).

## Files
```
# example entry points for training / inference / evaluation
examples/a2d
├── bd3lm               # Block Discrete Denoising Diffusion Language Modeling (https://arxiv.org/abs/2503.09573)
│   ├── chat.py
│   ├── eval.sh
│   ├── pt.py
│   ├── sample.py
│   └── sft.py
├── mdlm                # Masked Diffusion Language Modeling (https://arxiv.org/abs/2406.07524)
│   ├── chat.py
│   ├── eval.sh
│   ├── pt.py
│   ├── sample.py
│   └── sft.py
└── README.md
```

## Setup

> (Optional) If your source AR model is **not already supported** in [`dllm/pipelines/a2d/models`](/dllm/pipelines/a2d/models):
> 
> 1. Modify the original autoregressive modeling file to support non-causal
> attention. See [`modeling_qwen3.py`](/dllm/pipelines/a2d/models/qwen3/modeling_qwen3.py#L77-L108)
> for an example.
> 
> 2. And ensure the attention behavior is correct: 
>    ```shell
>    # Run A2D attention tests
>    pytest scripts/tests/test_attention.py -k "test_a2d"
>    # (Optional; required for the bd3lm trainer)
>    pytest scripts/tests/test_attention.py -k "test_bd3lm"
>    ```
<!-- **Convert an AR model with customized attention**-->

Before training, modify and save the source autoregressive models with non-causal attention.
For example, to save [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) with its original weights but with modified non-causal attention defined in
[`modeling_qwen3.py`](/dllm/pipelines/a2d/models/qwen3/modeling_qwen3.py):
```shell
python dllm/pipelines/a2d/convert.py --model_name_or_path "Qwen/Qwen3-0.6B" --output_dir ".models/a2d/Qwen3-0.6B"
```

## Warmup: [MDLM](https://arxiv.org/abs/2406.07524)

In this section, we show toy examples of continual pretraining and SFTing [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) on small datasets to generate text with [MDLM](https://arxiv.org/abs/2406.07524).

### Continual Pretraining

To train [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) on the [`tiny-shakespeare`](https://huggingface.co/datasets/Trelis/tiny-shakespeare) dataset with [MDLM](https://arxiv.org/abs/2406.07524), run (on 1 GPU):

```shell
accelerate launch --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
    examples/a2d/mdlm/pt.py \
    --model_name_or_path ".models/a2d/Qwen3-0.6B" \
    --dataset_args "Trelis/tiny-shakespeare" \
    --text_field "Text" \
    --insert_eos False \
    --max_length 128 \
    --num_train_epochs 10 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --output_dir ".models/a2d/Qwen3-0.6B/mdlm/tiny-shakespeare"
```

To sample from the model interactively:
```shell
# Enter a prompt (e.g., "First citizen: Before we proceed any further, hear me speak."),
# or press Enter to let the model generate text from scratch.
python -u examples/a2d/mdlm/chat.py \
    --model_name_or_path ".models/a2d/Qwen3-0.6B/mdlm/tiny-shakespeare/checkpoint-final" \
    --chat_template False --remasking "random" --temperature 0.7
```

<details>
<summary>Example of pretraining on a larger dataset (OpenWebText) in streaming mode</summary>

To train [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) on the [`openwebtext`](https://huggingface.co/datasets/dylanebert/openwebtext) dataset in streaming mode with [MDLM](https://arxiv.org/abs/2406.07524), run (on 8 GPUs):
```shell
accelerate launch --config_file scripts/accelerate_configs/zero2.yaml --num_processes 8 \
    examples/a2d/mdlm/pt.py \
    --model_name_or_path ".models/a2d/Qwen3-0.6B" \
    --dataset_args "dylanebert/openwebtext" \
    --text_field "text" \
    --streaming True \
    --insert_eos True \
    --max_length 512 \
    --max_steps 20000 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --eval_strategy "no" \
    --output_dir ".models/a2d/Qwen3-0.6B/mdlm/openwebtext"
```

To sample from the model interactively:
```shell
# Enter a prompt (e.g., "Lebron James is"),
# or press Enter to let the model generate text from scratch.
python -u examples/a2d/mdlm/chat.py \
    --model_name_or_path ".models/a2d/Qwen3-0.6B/mdlm/openwebtext/checkpoint-final" \
    --chat_template False --remasking "random" --temperature 0.7
```

</details>

### SFT

To train [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) on the [`alpaca`](https://huggingface.co/datasets/tatsu-lab/alpaca) dataset with [MDLM](https://arxiv.org/abs/2406.07524), run (on 8 GPUs):

```shell
accelerate launch --config_file scripts/accelerate_configs/zero2.yaml --num_processes 8 \
    examples/a2d/mdlm/sft.py \
    --model_name_or_path ".models/a2d/Qwen3-0.6B" \
    --dataset_args "tatsu-lab/alpaca" \
    --max_length 512 \
    --num_train_epochs 10 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --output_dir ".models/a2d/Qwen3-0.6B/mdlm/alpaca"
```

To chat with the model:
```shell
python -u examples/a2d/mdlm/chat.py \
    --model_name_or_path ".models/a2d/Qwen3-0.6B/mdlm/alpaca/checkpoint-final" --block_size 32
```

## Warmup: [BD3LM](https://arxiv.org/abs/2503.09573)

In this section, we show toy examples of continual pretraining and SFTing [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) on small datasets to generate text with [BD3LM](https://arxiv.org/abs/2503.09573).

### Continual Pretraining

To train [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) on the [`tiny-shakespeare`](https://huggingface.co/datasets/Trelis/tiny-shakespeare) dataset with [BD3LM](https://arxiv.org/abs/2503.09573), run (on 1 GPU):

```shell
accelerate launch --config_file scripts/accelerate_configs/ddp.yaml --num_processes 1 \
    examples/a2d/bd3lm/pt.py \
    --model_name_or_path ".models/a2d/Qwen3-0.6B" \
    --dataset_args "Trelis/tiny-shakespeare" \
    --text_field "Text" \
    --insert_eos False \
    --max_length 128 \
    --num_train_epochs 10 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --block_size 32 \
    --output_dir ".models/a2d/Qwen3-0.6B/bd3lm/tiny-shakespeare"
```

To sample from the model interactively:
```shell
# Enter a prompt (e.g., "First citizen: Before we proceed any further, hear me speak."),
# or press Enter to let the model generate text from scratch.
python -u examples/a2d/bd3lm/chat.py \
    --model_name_or_path ".models/a2d/Qwen3-0.6B/bd3lm/tiny-shakespeare/checkpoint-final" \
    --chat_template False --block_size 32 --remasking "random" --temperature 0.7
```

<details>
<summary>Example of pretraining on a larger dataset (OpenWebText) in streaming mode</summary>

To train [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) on the [`openwebtext`](https://huggingface.co/datasets/dylanebert/openwebtext) dataset in streaming mode with [BD3LM](https://arxiv.org/abs/2503.09573), run (on 8 GPUs):
```shell
accelerate launch --config_file scripts/accelerate_configs/zero2.yaml --num_processes 8 \
    examples/a2d/bd3lm/pt.py \
    --model_name_or_path ".models/a2d/Qwen3-0.6B" \
    --dataset_args "dylanebert/openwebtext" \
    --text_field "text" \
    --streaming True \
    --insert_eos True \
    --max_length 512 \
    --max_steps 20000 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --eval_strategy "no" \
    --block_size 32 \
    --output_dir ".models/a2d/Qwen3-0.6B/bd3lm/openwebtext"
```

To sample from the model interactively:
```shell
# Enter a prompt (e.g., "Lebron James is"),
# or press Enter to let the model generate text from scratch.
python -u examples/a2d/bd3lm/chat.py \
    --model_name_or_path ".models/a2d/Qwen3-0.6B/bd3lm/openwebtext/checkpoint-final" \
    --chat_template False --block_size 32 --remasking "random" --temperature 0.7
```
</details>

### SFT

To train [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B) on the [`alpaca`](https://huggingface.co/datasets/tatsu-lab/alpaca) dataset with [BD3LM](https://arxiv.org/abs/2503.09573), run (on 8 GPUs):

```shell
accelerate launch --config_file scripts/accelerate_configs/zero2.yaml --num_processes 8 \
    examples/a2d/bd3lm/sft.py \
    --model_name_or_path ".models/a2d/Qwen3-0.6B" \
    --dataset_args "tatsu-lab/alpaca" \
    --max_length 512 \
    --num_train_epochs 10 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --block_size 32 \
    --output_dir ".models/a2d/Qwen3-0.6B/bd3lm/alpaca"
```

To chat with the model:
```shell
python -u examples/a2d/bd3lm/chat.py \
    --model_name_or_path ".models/a2d/Qwen3-0.6B/bd3lm/alpaca/checkpoint-final" --block_size 32
```

## `Tiny-A2D`

Here we show the exact commands we use to train / interact with / evaluate the [`Tiny-A2D`](https://huggingface.co/collections/dllm-hub/tiny-a2d) models:
[`Qwen3-0.6B-diffusion-mdlm-v0.1`](https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1) and [`Qwen3-0.6B-diffusion-bd3lm-v0.1`](https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1).
For training curves and other details, please see [![blog](https://img.shields.io/badge/W&B-white?logo=weightsandbiases) Tiny-A2D Report](https://wandb.ai/asap-zzhou/dllm/reports/dLLM-Tiny-A2D--VmlldzoxNTI2NTEzOA).

### Training
> Read [Useful tips for training](/README.md#useful-tips-for-training) and [(optional) Slurm setup](/README.md#optional-slurm-setup) before training.

The [`Tiny-A2D`](https://huggingface.co/collections/dllm-hub/tiny-a2d) models are trained purely with SFT.

To reproduce [`Qwen3-0.6B-diffusion-mdlm-v0.1`](https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1) (with MDLM & SFT), run the command below (about 10 hours on 64 A100s):
```shell
WANDB_MODE=online sbatch --nodes=8 --gres=gpu:8 scripts/train.slurm.sh \
    --accelerate_config "zero2" \
    --script_path "examples/a2d/mdlm/sft.py" \
    --model_name_or_path ".models/a2d/Qwen3-0.6B" \
    --dataset_args "allenai/tulu-3-sft-mixture+HuggingFaceTB/smoltalk+OpenCoder-LLM/opc-sft-stage1[lang:python]+OpenCoder-LLM/opc-sft-stage2[lang:python]" \
    --max_length 1024 \
    --num_train_epochs 10 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --output_dir ".models/a2d/Qwen3-0.6B/tulu-3-sft-mixture+smoltalk+opc-sft-stage1&2/epochs-10-bs-2048-len-1024"
```

To reproduce [`Qwen3-0.6B-diffusion-bd3lm-v0.1`](https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1) (with BD3LM & SFT), run the command below (about 10 hours on 64 A100s):
```shell
WANDB_MODE=online sbatch --nodes=8 --gres=gpu:8 scripts/train.slurm.sh \
    --accelerate_config "zero2" \
    --script_path "examples/a2d/bd3lm/sft.py" \
    --model_name_or_path ".models/a2d/Qwen3-0.6B" \
    --dataset_args "allenai/tulu-3-sft-mixture+HuggingFaceTB/smoltalk+OpenCoder-LLM/opc-sft-stage1[lang:python]+OpenCoder-LLM/opc-sft-stage2[lang:python]" \
    --max_length 512 \
    --num_train_epochs 10 \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --block_size 32 \
    --output_dir ".models/a2d/Qwen3-0.6B/tulu-3-sft-mixture+smoltalk+opc-sft-stage1&2/epochs-10-bs-2048-len-512-bls-32"
```

### Inference

To chat with the model:
```shell
python -u examples/a2d/mdlm/chat.py --model_name_or_path "dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1"
python -u examples/a2d/bd3lm/chat.py --model_name_or_path "dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1"
```

### Evaluation

> Read [(optional) Evaluation setup](/README.md#optional-evaluation-setup) before running evaluation.

To evaluate [`Qwen3-0.6B-diffusion-mdlm-v0.1`](https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1) and [`Qwen3-0.6B-diffusion-bd3lm-v0.1`](https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1) on [`gsm8k`](https://huggingface.co/datasets/openai/gsm8k) using 4 GPUs, run:
```shell
# Use model_args to adjust the sampler arguments for evaluation.
accelerate launch --num_processes 4 \
    dllm/pipelines/a2d/eval.py \
    --tasks "gsm8k_cot" \
    --model "a2d_mdlm" \
    --apply_chat_template \
    --num_fewshot 0 \
    --model_args "pretrained=dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1,max_new_tokens=256,steps=256,block_size=256,cfg_scale=0.0,temperature=0.0"

accelerate launch --num_processes 4 \
    dllm/pipelines/a2d/eval.py \
    --tasks "gsm8k_cot" \
    --model "a2d_bd3lm" \
    --apply_chat_template \
    --num_fewshot 0 \
    --model_args "pretrained=dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1,max_new_tokens=256,steps=256,block_size=32,cfg_scale=0.0,temperature=0.0"
```

To automatically evaluate [`Qwen3-0.6B-diffusion-mdlm-v0.1`](https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1) and [`Qwen3-0.6B-diffusion-bd3lm-v0.1`](https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1) on all benchmarks, run:
```shell
bash examples/a2d/mdlm/eval.sh --model_name_or_path "dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1" 
bash examples/a2d/bd3lm/eval.sh --model_name_or_path "dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1" 
```


#### Evaluation results

<table style="border-collapse: collapse; width: 100%; text-align: center; table-layout: fixed;">
  <colgroup>
    <col style="width: 32%;">   <!-- FIRST COLUMN WIDENED -->
    <col style="width: 8%;">
    <col style="width: 8%;">
    <col style="width: 8%;">
    <col style="width: 8%;">
    <col style="width: 8%;">
    <col style="width: 8%;">
    <col style="width: 8%;">
    <col style="width: 8%;">
  </colgroup>

  <thead>
    <tr style="border-bottom: 3px solid #333;">
      <th style="padding: 8px; text-align: left;">Model                        </th>
      <th style="padding: 8px;">GSM8K</th>
      <th style="padding: 8px;">BBH</th>
      <th style="padding: 8px;">MATH</th>
      <th style="padding: 8px;">MMLU</th>
      <th style="padding: 8px;">MMLU&#8209;Pro</th>
      <th style="padding: 8px;">Hellaswag</th>
      <th style="padding: 8px;">HumanEval</th>
      <th style="padding: 8px;">MBPP</th>
    </tr>
  </thead>

  <!-- mdlm v0.1 -->
  <tr>
    <td style="padding: 8px;">
      <a href="https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1"><code>Qwen3-0.6B-diffusion-mdlm-v0.1</code></a> (Reproduced)
    </td>
    <td>29.3</td><td>26.7</td><td>8.7</td>
    <td>40.0</td><td>17.3</td><td>42.1</td>
    <td>30.5</td><td>29.2</td>
  </tr>

  <!-- bd3lm v0.1 -->
  <tr>
    <td style="padding: 8px;">
      <a href="https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1"><code>Qwen3-0.6B-diffusion-bd3lm-v0.1</code></a> (Reproduced)
    </td>
    <td>46.3</td><td>26.6</td><td>12.9</td>
    <td>39.1</td><td>13.8</td><td>39.3</td>
    <td>46.3</td><td>38.2</td>
  </tr>

  <!-- Divider -->
  <tr>
    <td colspan="9" style="padding: 0; border-top: 3px double #666;"></td>
  </tr>

  <!-- AR model -->
  <tr>
    <td style="padding: 8px;"><i><a href="https://huggingface.co/Qwen/Qwen2.5-0.5B"><code>Qwen2.5-0.5B</code></a> (Official)</i></td>
    <td><i>41.6</i></td><td><i>20.3</i></td><td><i>19.5</i></td><td><i>47.5</i></td><td><i>15.7</i></td><td><i>52.1</i></td><td><i>30.5</i></td><td><i>39.3</i></td>
  </tr>

  <tr>
    <td style="padding: 8px;"><i>
      <a href="https://huggingface.co/Qwen/Qwen3-0.6B-Base"><code>Qwen3-0.6B-Base</code></a> (Official)
    </i></td>
    <td><i>59.6</i></td><td><i>41.5</i></td><td><i>32.4</i></td>
    <td><i>52.8</i></td><td><i>24.7</i></td><td><i>47.4</i></td>
    <td><i>32.3</i></td><td><i>36.6</i></td>
  </tr>

</table>



<table style="border-collapse: collapse; width: 60%; text-align: center;">
  <thead>
    <tr style="border-bottom: 3px solid #333;">
      <th style="padding: 8px; min-width: 320px; text-align: left;">Model</th>
      <th style="padding: 8px;">HumanEval</th>
      <th style="padding: 8px;">MBPP</th>
    </tr>
  </thead>

  <!-- mdlm v0.1  -->
  <tr>
    <td style="padding: 8px;">
      <a href="https://huggingface.co/dllm-hub/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1"><code>Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1</code></a> (Reproduced)
    </td>
    <td>28.1</td>
    <td>23.0</td>
  </tr>

  <!-- bd3lm v0.1  -->
  <tr>
    <td style="padding: 8px;">
      <a href="https://huggingface.co/dllm-hub/Qwen2.5-Coder-0.5B-Instruct-diffusion-bd3lm-v0.1"><code>Qwen2.5-Coder-0.5B-Instruct-diffusion-bd3lm-v0.1</code></a> (Reproduced)
    </td>
    <td>39.0</td>
    <td>33.2</td>
  </tr>

  <!-- open-dcoder -->
  <tr>
    <td style="padding: 8px;">
      <a href="https://huggingface.co/fredzzp/open-dcoder-0.5B"><code>open-dcoder-0.5B</code></a> (Official)
    </td>
    <td>20.8</td>
    <td>35.2</td>
  </tr>

  <!-- Double-line separator -->
  <tr>
    <td colspan="3" style="padding: 0; border-top: 3px double #666;"></td>
  </tr>

  <!-- AR model-->
  <tr>
    <td style="padding: 8px;"><i>
      <a href="https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct"><code>Qwen2.5-Coder-0.5B-Instruct</code></a> (Official)
    </i></td>
    <td><i>28.0</i></td>
    <td><i>52.9</i></td>
  </tr>

</table>

<p align="left" style="color: #808080; font-size: 0.9em;">
Table 1. Results (Reproduced) are obtained using our framework, while results (Official) come from the 
<a href="https://arxiv.org/pdf/2505.09388" style="color: #808080; text-decoration: none;">Qwen3 Technical Report</a>, 
<a href="https://arxiv.org/pdf/2409.12186" style="color: #808080; text-decoration: none;">Qwen2.5-Coder Technical Report</a>, 
<a href="https://qwenlm.github.io/blog/qwen2.5-llm/" style="color: #808080; text-decoration: none;">Qwen2.5 Blog</a>, and 
<a href="https://github.com/pengzhangzhi/Open-dLLM?tab=readme-ov-file#-benchmarking" style="color: #808080; text-decoration: none;">Open-dLLM</a>. 
<i>Italic rows</i> denote autoregressive models, whereas non-italic rows denote diffusion language models.
</p>
