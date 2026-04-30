# Edit Flows

> **Reference**
> 📄 Paper: [Edit Flows: Flow Matching with Edit Operations](https://arxiv.org/abs/2506.09018) 

This directory provides an educational reference for training EditFlow models. It demonstrates how to adapt open-weight DLLMs—such as [LLaDA](https://arxiv.org/abs/2502.09992) and [Dream](https://arxiv.org/abs/2508.15487)—to support *insertion*, *deletion*, beyond the standard *substitution*(`mask`->`tokens`) operations. It also includes examples for training (pretraining and finetuning) EditFlow models from scratch.

> [!NOTE]
> - Examples are available for both LLaDA and Dream. This directory contains **legacy** entry points: use [`legacy/llada/pt.py`](/examples/editflow/legacy/llada/pt.py) / [`legacy/llada/sft.py`](/examples/editflow/legacy/llada/sft.py) for LLaDA and [`legacy/dream/pt.py`](/examples/editflow/legacy/dream/pt.py) / [`legacy/dream/sft.py`](/examples/editflow/legacy/dream/sft.py) for Dream. For the current unified pipeline, see the parent [EditFlow README](/examples/editflow/README.md) and `examples/editflow/pt.py`, `examples/editflow/sft.py`.
> - While `EditFlowCollator` supports custom `x0`, this README uses a fixed-length (128) masks as `x0`. The trained model samples text by replacing masks, deleting redundant ones, and inserting tokens as needed. To change the default `x0` distribution (e.g., empty sequences for [OneFlow](https://arxiv.org/abs/2510.03506)-like insertion-only sampling), pass `--x0_sampler "empty"`.

## Table of Contents
- [Setup](#setup)
- [Files](#files)
- [Training](#training)
    - [Adapting LLaDA-8B-Instruct to support insertion and deletion](#adapting-llada-8b-instruct-to-support-insertion-and-deletion)
    - [Pretraining & Finetuning from scratch](#pretraining--finetuning-from-scratch)
- [Sampling](#sampling)
- [Acknowledgement](#acknowledgement)

## Setup
> [!IMPORTANT]  
> **Slurm users:** Update `scripts/train.slurm.sh` and `mkdir .logs`: see [(optional) Slurm setup](/README.md#optional-slurm-setup) for details.

##  Files
```
dllm/pipelines/editflow
├── __init__.py                 # Package initialization
├── models
│   ├── dream
│   │   └── modelling_dream.py  # EditFlowDream: architecture based on Dream
│   └── llada
│       └── modelling_llada.py  # EditFlowLLaDA: architecture based on LLaDA
├── trainer.py
└── utils.py

# legacy entry points (this directory)
examples/editflow/legacy
├── llada/
│   ├── pt.py                   # EditFlowLLaDA pretraining
│   └── sft.py                  # EditFlowLLaDA SFT
├── dream/
│   ├── pt.py                   # EditFlowDream pretraining
│   └── sft.py                  # EditFlowDream SFT
├── bert/
│   ├── pt.py
│   └── sft.py
├── sample.py                   # Sampling with visualization
├── viz.py
└── README.md                   # This file

# current unified entry points (parent directory)
examples/editflow
├── pt.py                        # Pretraining
├── sft.py                       # Supervised finetuning
├── sample.py                    # Sampling
├── chat.py
└── README.md
```

## Training

### Adapting [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) to support *insertion* and *deletion*

The original LLaDA model samples text by iteratively substituting the given `<mask>` tokens to real tokens. 

<p align="center">
  <img src="https://github.com/ML-GSAI/LLaDA/blob/main/imgs/example_gradio.gif" alt="LLaDA demo" width="80%">
</p>
<p align="center"><em>Figure: Example Gradio demo for LLaDA.</em></p>

However, LLaDA supports only substitution. To train an EditFlow model that also supports insertion and deletion, use the **current** pipeline (see parent [EditFlow README](/examples/editflow/README.md)) with a LLaDA-based EditFlow model, or use the legacy LLaDA SFT entry point:

```shell
# Legacy entry point: SFT with EditFlow-LLaDA (requires PYTHONPATH=. from repo root)
accelerate launch \
    --config_file scripts/accelerate_configs/zero2.yaml \
    examples/editflow/legacy/llada/sft.py \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct" \
    --dataset_args "allenai/tulu-3-sft-mixture" \
    --output_dir ".models/EditFlow-LLaDA-8B-Instruct-Adapt/tulu-3-sft-mixture" \
    --x0_sampler "masks[length:128]" \
    --max_length 1024 \
    --num_train_epochs 4 \
    --learning_rate 5e-5
```

If you are using slurm and want to train across, for example, four nodes (32 GPUs total), run:
```shell
PYTHONPATH=. sbatch --nodes=4 --gres=gpu:8 scripts/train.slurm.sh \
    --accelerate_config "fsdp" \
    --script_path "examples/editflow/legacy/llada/sft.py" \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct" \
    --dataset_args "allenai/tulu-3-sft-mixture" \
    --output_dir ".models/EditFlow-LLaDA-8B-Instruct-Adapt/tulu-3-sft-mixture" \
    --x0_sampler "masks[length:128]" \
    --max_length 1024 \
    --num_train_epochs 4 \
    --learning_rate 5e-5
```

After training, you can use [sample.py](/examples/editflow/sample.py) or [legacy/sample.py](/examples/editflow/legacy/sample.py) to get a visualized decoding trace. See [Sampling](#sampling) for details.


### Pretraining & Finetuning from scratch
You can also train an EditFlow model from scratch (pretrain → SFT) without adapting an existing DLLM.

Pretrain on a subset of [mlfoundations/dclm-baseline-1.0](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0) using 192 GPUs (24x8) and FSDP:

```shell
PYTHONPATH=. sbatch --nodes=24 --gres=gpu:8 scripts/train.slurm.sh \
    --accelerate_config "fsdp" \
    --script_path "examples/editflow/legacy/llada/pt.py" \
    --model_name_or_path "GSAI-ML/LLaDA-8B-Base" \
    --dataset_args "mlfoundations/dclm-baseline-1.0" \
    --output_dir ".models/EditFlow-LLaDA-8B-Base/dclm-baseline-1.0" \
    --x0_sampler "masks[length:128]" \
    --max_length 1024 \
    --max_steps 2000 \
    --learning_rate 3e-4
```

Finetune on a subset of [allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) using 8 GPUS and FSDP for better instruction following:

```shell
# you can also run locally with `accelerate ...`
PYTHONPATH=. sbatch --nodes=1 --gres=gpu:8 scripts/train.slurm.sh \
    --accelerate_config "fsdp" \
    --script_path "examples/editflow/legacy/llada/sft.py" \
    --model_name_or_path ".models/EditFlow-LLaDA-8B-Base/dclm-baseline-1.0/checkpoint-final" \
    --dataset_args "allenai/tulu-3-sft-mixture[train:10000,test:1000]" \
    --output_dir ".models/EditFlow-LLaDA-8B-Base/dclm-baseline-1.0" \
    --x0_sampler "masks[length:128]" \
    --max_length 1024 \
    --num_train_epochs 4 \
    --learning_rate 5e-5
```

## Sampling

After training, you can visualize how the model performs mask substitution, insertion, and deletion during sampling with [sample.py](/examples/editflow/sample.py). Inserted tokens appear <span style="color:blue; font-weight:bold">blue</span>, and tokens substituted from `<mask>` appear <span style="color:black; font-weight:bold">black</span>, and deleted tokens are shown with a strikethrough before they disappear.

```shell
# Sample a long sequence to visualize insertions after 128 <mask> tokens
python examples/editflow/sample.py \
  --model_name_or_path ".models/EditFlow-LLaDA-8B-Instruct-Adapt/tulu-3-sft-mixture/checkpoint-final" \
  --tau 0.02 --mask_length 128 --seed 7070 \
  --prompt "write a romantic story" --make_gif

# Sample a short sequence to visualize deletions after 128 <mask> tokens
python examples/editflow/sample.py \
  --model_name_or_path ".models/EditFlow-LLaDA-8B-Instruct-Adapt/tulu-3-sft-mixture/checkpoint-final" \
  --tau 0.02 --mask_length 128 --seed 7070 \
  --prompt "write a single-sentence romantic story" --make_gif
```

<p align="center">
  <img src="/examples/editflow/assets/deletion.gif" alt="EditFlow deletion demo" width="95%">
</p>
<p align="center"><em>Figure: Deletion & Substitution trace</code></em></p>

<p align="center">
  <img src="/examples/editflow/assets/insertion.gif" alt="LLaDA demo" width="95%">
</p>
<p align="center"><em>Figure: Inserction & Substitution trace</em></p>

## Acknowledgement

This Edit Flows implementation is inspired by https://github.com/TheMatrixMaster/edit-flows-demo.
