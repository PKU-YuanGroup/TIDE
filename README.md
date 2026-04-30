<div align="center">
<img src="assets/logo.gif" alt="TIDE logo" width="300px">
</div>

<h1 align="center">Turning the TIDE: Cross-Architecture Distillation for Diffusion Large Language Models</h1>

<h5 align="center">🌊 The first cross-architecture distillation framework for diffusion LLMs — 8B dense and 16B MoE teachers into a 0.6B student 🌊</h5>

<h5 align="center">

Gongbo Zhang<sup>1</sup> &nbsp;·&nbsp; Wen Wang<sup>2</sup> &nbsp;·&nbsp; Ye Tian<sup>1</sup> &nbsp;·&nbsp; Li Yuan<sup>1,*</sup>

<sup>1</sup> Peking University &nbsp;·&nbsp; <sup>2</sup> Zhejiang University &nbsp; (<sup>*</sup> corresponding author)

</h5>

<h5 align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2604.26951-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2604.26951)
[![Project Page](https://img.shields.io/badge/Project-Page-2ea44f)](https://pku-yuangroup.github.io/TIDE-Page/)
[![HF Paper](https://img.shields.io/badge/🤗-Paper-orange)](https://huggingface.co/papers/2604.26951)
[![HF Models](https://img.shields.io/badge/🤗-Models-blue)](https://huggingface.co/TIDE-dllm/models)
[![HF Datasets](https://img.shields.io/badge/🤗-Datasets-yellow)](https://huggingface.co/TIDE-dllm/datasets)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![GitHub](https://img.shields.io/github/stars/PKU-YuanGroup/TIDE?style=social)](https://github.com/PKU-YuanGroup/TIDE)

</h5>

<div align="center">
This repository is the official implementation of <strong>TIDE</strong>, the first framework for <em>cross-architecture</em> dLLM distillation. While prior work focuses on step compression within a single architecture, TIDE bridges teachers and students that differ in <strong>architecture</strong>, <strong>attention mechanism</strong>, and <strong>tokenizer</strong>, via three modular components — <strong>TIDAL</strong>, <strong>CompDemo</strong>, and <strong>Reverse CALM</strong>.
</div>

<br>

<p align="center">
  <img src="assets/Figure_teaser.png" alt="TIDE: cross-architecture distillation overview" width="100%">
</p>

## ✨ Highlights

1. **+1.53 average gain** over the non-distilled BD3LM baseline across 8 benchmarks (34.20 vs. 32.67).
2. **+16.48 on HumanEval** over the equivalent-size AR baseline (48.78 vs. 32.30) — distilled dLLMs especially excel at code generation.
3. **22× peak-memory reduction** vs. the 16B MoE LLaDA2 teacher (1.4 GB vs. 31.3 GB) and **5.2× faster inference** (6.25 s vs. 32.55 s for 256 tokens on H100), enabling commodity-hardware deployment.

> All numbers reported in the paper — see [arxiv.org/abs/2604.26951](https://arxiv.org/abs/2604.26951) for full setup and ablations.

## 🌊 The TIDE Framework

<p align="center">
  <img src="assets/Figure_main.png" alt="TIDE framework: TIDAL + CompDemo + Reverse CALM" width="100%">
</p>

| Component | Paper | Role | One-line description |
|---|:---:|---|---|
| **TIDAL** | §2.1 | Scheduling — *when* to learn | Dual-axis interpolation along training-progress AND diffusion-timestep axes; deweights the teacher at high masking ratios where it is unreliable. Generalizes prior single-axis interpolation to the diffusion setting. |
| **CompDemo** | §2.2 | Contextual — *what* to enrich | Two-pass teacher inference with complementary mask splits; every masked position sees ~50% revealed context, raising teacher signal quality at high noise. |
| **Reverse CALM** | §2.3 | Output — *how* to project | Reverse-direction chunk-level binary cross-entropy for cross-tokenizer matching. Bounded gradient coefficient (depends only on the fixed teacher) and dual-end noise filtering; equivalent to a Bernoulli-KL mode-seeking objective. |

## 🔄 Two Pipelines × Two Strategies

> **Headline finding (§3.2): each pipeline favors its native strategy.**
>
> - **Cross-Tokenizer (LLaDA2 → BD3LM): native = TIDE-Cross** = Reverse CALM. Bounded-gradient mode-seeking tolerates the alignment noise from chunk-level cross-tokenizer matching. Beats the swapped TIDE-Shared by avg **+0.37**.
> - **Shared-Tokenizer (WeDLM → BD3LM): native = TIDE-Shared** = TIDAL + CompDemo (over forward KL). Progressive scheduling and enriched signals work best when token-level alignment is exact. Beats the swapped TIDE-Cross by avg **+2.76**.

| Pipeline | Teacher | Student | Tokenizer | Native strategy | Paper avg |
|---|---|---|---|---|:---:|
| **A — Cross-Tokenizer** | LLaDA2.0-mini (16B MoE) | Qwen3-0.6B-BD3LM | Cross (chunk align via tokenkit) | **TIDE-Cross** = Reverse CALM | **34.20** |
| **B — Shared-Tokenizer** | WeDLM-8B-Instruct (8B dense) | Qwen3-0.6B-BD3LM | Shared (vocab 151646) | **TIDE-Shared** = TIDAL + CompDemo | 33.55 |

## 📊 Main Results

Main results across eight benchmarks. All distillation methods include a cross-entropy loss term. **Bold**: best among dLLM models; *italic*: second best.

<table>
  <thead>
    <tr>
      <th align="left" rowspan="2">Benchmark</th>
      <th align="center" colspan="2"><em>Qwen3-0.6B</em></th>
      <th align="center" colspan="3"><em>Shared-Tokenizer</em></th>
      <th align="center" colspan="3"><em>Cross-Tokenizer</em></th>
    </tr>
    <tr>
      <th align="center">AR</th>
      <th align="center">BD3LM</th>
      <th align="center">KL</th>
      <th align="center">TIDE-Cross</th>
      <th align="center">TIDE-Shared</th>
      <th align="center">CALM</th>
      <th align="center">TIDE-Shared</th>
      <th align="center">TIDE-Cross</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>GSM8K</td>
      <td align="center">59.60</td>
      <td align="center">45.56</td>
      <td align="center">43.97</td>
      <td align="center">45.03</td>
      <td align="center">48.98</td>
      <td align="center">48.60</td>
      <td align="center"><em>49.89</em></td>
      <td align="center"><strong>52.24</strong></td>
    </tr>
    <tr>
      <td>MATH</td>
      <td align="center">32.40</td>
      <td align="center">13.08</td>
      <td align="center">9.40</td>
      <td align="center">9.76</td>
      <td align="center">11.16</td>
      <td align="center"><em>13.14</em></td>
      <td align="center">12.98</td>
      <td align="center"><strong>13.20</strong></td>
    </tr>
    <tr>
      <td>BBH</td>
      <td align="center">41.50</td>
      <td align="center">26.32</td>
      <td align="center">25.79</td>
      <td align="center">26.00</td>
      <td align="center">26.79</td>
      <td align="center">24.21</td>
      <td align="center"><em>26.85</em></td>
      <td align="center"><strong>27.37</strong></td>
    </tr>
    <tr>
      <td>MMLU-Pro</td>
      <td align="center">24.70</td>
      <td align="center">13.80</td>
      <td align="center">13.19</td>
      <td align="center">12.88</td>
      <td align="center"><em>14.48</em></td>
      <td align="center">13.47</td>
      <td align="center">14.02</td>
      <td align="center"><strong>14.52</strong></td>
    </tr>
    <tr>
      <td>HellaSwag</td>
      <td align="center">47.40</td>
      <td align="center">39.28</td>
      <td align="center">39.78</td>
      <td align="center">39.50</td>
      <td align="center"><strong>40.50</strong></td>
      <td align="center"><em>40.42</em></td>
      <td align="center">39.57</td>
      <td align="center">39.88</td>
    </tr>
    <tr>
      <td>MMLU</td>
      <td align="center">52.80</td>
      <td align="center">39.15</td>
      <td align="center">39.57</td>
      <td align="center">39.09</td>
      <td align="center"><strong>39.92</strong></td>
      <td align="center">39.42</td>
      <td align="center">39.54</td>
      <td align="center"><em>39.59</em></td>
    </tr>
    <tr>
      <td>HumanEval</td>
      <td align="center">32.30</td>
      <td align="center">46.34</td>
      <td align="center">41.46</td>
      <td align="center">42.68</td>
      <td align="center"><em>48.78</em></td>
      <td align="center">43.90</td>
      <td align="center"><strong>49.39</strong></td>
      <td align="center">48.17</td>
    </tr>
    <tr>
      <td>MBPP</td>
      <td align="center">36.60</td>
      <td align="center">37.80</td>
      <td align="center">31.20</td>
      <td align="center">31.40</td>
      <td align="center">37.80</td>
      <td align="center">34.80</td>
      <td align="center"><em>38.40</em></td>
      <td align="center"><strong>38.60</strong></td>
    </tr>
    <tr>
      <td><strong>Avg</strong></td>
      <td align="center">40.91</td>
      <td align="center">32.67</td>
      <td align="center">30.55</td>
      <td align="center">30.79</td>
      <td align="center">33.55</td>
      <td align="center">32.25</td>
      <td align="center"><em>33.83</em></td>
      <td align="center"><strong>34.20</strong></td>
    </tr>
  </tbody>
</table>

See the paper (§3.2) at [arxiv.org/abs/2604.26951](https://arxiv.org/abs/2604.26951) for the full discussion.

## 🧭 Paper Variants ↔ Code Modes

This is the only place in the README where the legacy CLI strings `alm` / `taid` appear, because the `--distill_mode` flag values include them.

| Paper variant | Pipeline | Command | Notes |
|---|:---:|---|---|
| CALM (baseline, Cross-Tok) | A | `distill_llada2.sh --distill_mode alm` | — |
| **TIDE-Cross (native, Cross-Tok)** | A | `distill_llada2.sh --distill_mode reverse_alm` | — |
| TIDE-Shared (in Cross-Tok pipeline) | A | `distill_llada2.sh --distill_mode alm_taid --use_comp_demo True` | TIDAL + CompDemo |
| KL (baseline, Shared-Tok) | B | `distill_wedlm.sh --distill_mode kl_aligned` | — |
| **TIDE-Shared (native, Shared-Tok)** | B | `distill_wedlm.sh --distill_mode taid_aligned --use_comp_demo True` | TIDAL + CompDemo |
| TIDE-Cross (in Shared-Tok pipeline) | B | `distill_wedlm.sh --distill_mode reverse_kl_aligned` | — |

> 💡 **Note on combinations.** TIDAL is applied only to forward objectives. As discussed in the paper's gradient-analysis appendix, combining TIDAL with reverse objectives is counterproductive — the late-training $(1-\lambda_t)$ factor suppresses the self-selection mechanism of Reverse CALM.

## ⚙️ Setup

```bash
# Create environment
conda create -n dllm python=3.10 -y && conda activate dllm

# Install PyTorch (CUDA 12.4)
conda install cuda=12.4 -c nvidia
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Install dllm
pip install -e .

# Initialize submodules (lm-evaluation-harness + tokenkit)
git submodule update --init --recursive

# Install eval harness
pip install -e "lm-evaluation-harness[ifeval,math]"

# Install tokenkit (required for Pipeline A cross-tokenizer distillation)
pip install -e "tokenkit[full]"
```

## 📦 Released Models & Data

Six distilled student checkpoints (3 per pipeline) are released under [🤗 TIDE-dllm Models](https://huggingface.co/TIDE-dllm/models), and two preprocessed SFT datasets are released under [🤗 TIDE-dllm Datasets](https://huggingface.co/TIDE-dllm/datasets).

### Distilled student checkpoints

| Pipeline | Variant | 🤗 Repo |
|---|---|---|
| A — Cross-Tokenizer (LLaDA2 teacher) | **TIDE-Cross** (native) | [`distill-LLaDA2-TIDE_Cross`](https://huggingface.co/TIDE-dllm/distill-LLaDA2-TIDE_Cross) |
| A — Cross-Tokenizer (LLaDA2 teacher) | TIDE-Shared variant | [`distill-LLaDA2-TIDE_Shared`](https://huggingface.co/TIDE-dllm/distill-LLaDA2-TIDE_Shared) |
| A — Cross-Tokenizer (LLaDA2 teacher) | CALM baseline | [`distill-LLaDA2-CALM`](https://huggingface.co/TIDE-dllm/distill-LLaDA2-CALM) |
| B — Shared-Tokenizer (WeDLM teacher) | **TIDE-Shared** (native) | [`distill-WeDLM-TIDE_Shared`](https://huggingface.co/TIDE-dllm/distill-WeDLM-TIDE_Shared) |
| B — Shared-Tokenizer (WeDLM teacher) | TIDE-Cross variant | [`distill-WeDLM-TIDE_Cross`](https://huggingface.co/TIDE-dllm/distill-WeDLM-TIDE_Cross) |
| B — Shared-Tokenizer (WeDLM teacher) | KL baseline | [`distill-WeDLM-KL`](https://huggingface.co/TIDE-dllm/distill-WeDLM-KL) |

### Preprocessed SFT datasets

Both datasets share the same composition as [`dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1`](https://huggingface.co/dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1) — `tulu-3-sft-mixture` + `smoltalk` + `opc-sft-stage1` + `opc-sft-stage2` — but tokenized for each teacher in advance to avoid NCCL timeouts during distillation.

| Pipeline | 🤗 Repo |
|---|---|
| A — for the LLaDA2 teacher | [`distill_llada2_sft`](https://huggingface.co/datasets/TIDE-dllm/distill_llada2_sft) |
| B — for the WeDLM teacher | [`distill_wedlm_sft`](https://huggingface.co/datasets/TIDE-dllm/distill_wedlm_sft) |

### Download

```bash
pip install "huggingface_hub[cli]"

# Distilled checkpoint (example: native TIDE-Cross from Pipeline A)
huggingface-cli download TIDE-dllm/distill-LLaDA2-TIDE_Cross \
    --local-dir ckpts/distill-LLaDA2-TIDE_Cross

# Preprocessed datasets
huggingface-cli download TIDE-dllm/distill_llada2_sft \
    --repo-type dataset --local-dir data/distill_llada2_sft
huggingface-cli download TIDE-dllm/distill_wedlm_sft \
    --repo-type dataset --local-dir data/distill_wedlm_sft
```

Project page: [pku-yuangroup.github.io/TIDE-Page](https://pku-yuangroup.github.io/TIDE-Page/).

## 🚀 Quick Start

### 1. Data Preprocessing

Distillation requires offline-preprocessed data to avoid NCCL timeout during tokenization. **The fastest path is to download our preprocessed datasets** from `TIDE-dllm` (see [📦 Released Models & Data](#-released-models--data) above):

```bash
huggingface-cli download TIDE-dllm/distill_llada2_sft \
    --repo-type dataset --local-dir data/distill_llada2_preprocessed
huggingface-cli download TIDE-dllm/distill_wedlm_sft \
    --repo-type dataset --local-dir data/distill_wedlm_preprocessed
```

If you'd rather preprocess from scratch, the examples below use `tatsu-lab/alpaca` for a quick smoke test. To reproduce the paper, replace the `--dataset` value with:

```
allenai/tulu-3-sft-mixture+HuggingFaceTB/smoltalk+OpenCoder-LLM/opc-sft-stage1[lang:python]+OpenCoder-LLM/opc-sft-stage2[lang:python]
```

**Pipeline A (LLaDA2, cross-tokenizer):**
```bash
bash scripts/preprocess_llada2_data.sh \
    --dataset tatsu-lab/alpaca \
    --output_dir data/distill_llada2_preprocessed
```

**Pipeline B (WeDLM, same-tokenizer):**
```bash
bash scripts/preprocess_wedlm_data.sh \
    --dataset tatsu-lab/alpaca \
    --output_dir data/distill_wedlm_preprocessed
```

### 2. Distillation Training

The recommended command for each pipeline runs the **native strategy** (paper-best per §3.2).

**Pipeline A — LLaDA2 teacher, TIDE-Cross (Reverse CALM):**
```bash
bash scripts/distill_llada2.sh \
    --data_path data/distill_llada2_preprocessed \
    --distill_mode reverse_alm \
    --num_gpus 8
```

**Pipeline B — WeDLM teacher, TIDE-Shared (TIDAL + CompDemo):**
```bash
bash scripts/distill_wedlm.sh \
    --data_path data/distill_wedlm_preprocessed \
    --distill_mode taid_aligned \
    --use_comp_demo True \
    --num_gpus 8
```

<details>
<summary>📋 All training script parameters</summary>

Both `distill_llada2.sh` and `distill_wedlm.sh` support:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_path` | *required* | Preprocessed data directory or HF dataset name |
| `--output_dir` | `output/distill_*` | Checkpoint output directory |
| `--num_gpus` | `8` | Number of GPUs |
| `--distill_mode` | `alm` / `taid_aligned` | Distillation mode (see Paper Variants ↔ Code Modes table above) |
| `--use_comp_demo` | `False` | Enable CompDemo (complementary demonstration) |
| `--epochs` | `2` / `3` | Number of training epochs |
| `--lr` | `5e-5` | Learning rate |
| `--batch_size` | `8` / `10` | Per-device batch size |
| `--student_model` | `dllm-collection/Qwen3-0.6B-diffusion-bd3lm-v0.1` | Student model |
| `--teacher_model` | `inclusionAI/LLaDA2.0-mini` / `tencent/WeDLM-8B-Instruct` | Teacher model |

WeDLM-specific (TIDAL controls):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--taid_axis_mode` | `both` | TIDAL axis: `both`, `training_only`, `timestep_only` |
| `--taid_timestep_weight` | `midrange` | Timestep weighting: `uniform`, `midrange` |
| `--shared_vocab_size` | `151646` | Shared vocabulary size |
| `--teacher_mask_token_id` | `151665` | Teacher mask token ID |

</details>

### 3. Evaluation

Run all 8 benchmarks on a trained checkpoint:

```bash
bash scripts/eval_all.sh --model_path /path/to/checkpoint --num_gpus 8
```

Benchmarks: `mmlu_generative_dream`, `mmlu_pro`, `hellaswag_gen`, `gsm8k_cot`, `bbh`, `minerva_math`, `humaneval_instruct`, `mbpp_instruct`.

> Evaluation protocol: block size 32, CFG scale 0.0, sampling steps from 3 (HellaSwag/MMLU) up to 256 (everything else). Results are saved to `eval_results/` by default (override with `--output_dir`).

## 📋 Training Hyperparameters

Training settings used for the paper experiments.

| Parameter | Cross-Tokenizer (Pipeline A) | Shared-Tokenizer (Pipeline B) |
|---|---|---|
| Teacher | LLaDA2.0-mini (16B MoE) | WeDLM-8B-Instruct (8B) |
| Student init | Qwen3-0.6B-BD3LM SFT v0.1 | Qwen3-0.6B-BD3LM SFT v0.1 |
| Native method | Reverse CALM | TIDAL + CompDemo |
| Learning rate | 5e-5 | 5e-5 |
| Epochs | 10 | 10 |
| Student / teacher seq length | 512 / 1024 | 512 / 768 |
| Block size | 32 | 32 |
| Precision | bfloat16 | bfloat16 |
| TIDAL $\lambda_{\text{init}} \to \lambda_{\max}$ | — | $0.1 \to 0.9$, cosine, midrange weighting |
| CompDemo demo_ratio | — | 0.5 |
| Temperature $T$ | — | 2.0 |
| Dataset | Tulu-3 SFT + SmolTalk + OpenCoder-SFT-1/2 (Python) | (same) |

## 🛠️ Troubleshooting

<details>
<summary><strong><code>ValueError: Sequence length N exceeds pad_to_length M</code> during training</strong></summary>

For `*_aligned` modes (Pipeline B) the preprocessing script does **not** truncate samples to `--max_length` — it only filters samples whose prompt alone exceeds it. The training `--max_length` (and `--teacher_max_length`) must therefore be **at least as large as** the value used during preprocessing. The simplest rule: pass the same `--max_length` to both `preprocess_wedlm_data.sh` and `distill_wedlm.sh`.
</details>

<details>
<summary><strong>Pipeline B <code>taid_aligned</code> requires aligned preprocessed data</strong></summary>

The default `--align_mode` of `preprocess_wedlm_data.sh` is `kl_aligned`, which produces the dual-tokenizer fields (`teacher_input_ids`, `align_student`, `align_teacher`) needed by `*_aligned` training modes. If you preprocessed with `--align_mode none`, training in any `*_aligned` mode will crash with `KeyError: 'teacher_input_ids'`. Re-run preprocessing without overriding `--align_mode`.
</details>

## 📁 File Structure

```
dllm/core/trainers/
├── distill_bd3lm.py        # DistillBD3LMTrainer — all distillation modes (TIDAL, CompDemo, CALM, Reverse CALM, plus baselines)
├── distill_collator.py     # DistillCollator — chunk-level CALM alignment via tokenkit (paper §2.3)
├── bd3lm.py                # BD3LMTrainer (base block diffusion trainer)
├── mdlm.py                 # MDLMTrainer (base masked diffusion trainer)
└── losses/
    └── taid.py             # TIDAL loss implementation (paper §2.1)

examples/a2d/bd3lm/
├── distill.py              # Pipeline A entry: LLaDA2 cross-tokenizer distillation
├── distill_wedlm.py        # Pipeline B entry: WeDLM same-tokenizer distillation
├── distill_utils.py        # Shared utilities (alignment, tokenization)
├── preprocess_distill_data.py       # Data preprocessing for Pipeline A
└── preprocess_distill_wedlm_data.py # Data preprocessing for Pipeline B

scripts/
├── distill_llada2.sh       # One-click training: Pipeline A
├── distill_wedlm.sh        # One-click training: Pipeline B
├── eval_all.sh             # One-click evaluation (8 benchmarks)
├── preprocess_llada2_data.sh   # One-click preprocessing: Pipeline A
└── preprocess_wedlm_data.sh    # One-click preprocessing: Pipeline B
```

## 📝 Citation

If you find TIDE useful for your research, please consider citing:

```bibtex
@misc{zhang2026turningtidecrossarchitecturedistillation,
      title={Turning the TIDE: Cross-Architecture Distillation for Diffusion Large Language Models},
      author={Gongbo Zhang and Wen Wang and Ye Tian and Li Yuan},
      year={2026},
      eprint={2604.26951},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2604.26951},
}
```

## 🙏 Acknowledgements

Built on the [dLLM](https://github.com/ZHZisZZ/dllm) library; cross-tokenizer alignment via [tokenkit](https://github.com/Nobody-Zhang/tokenkit); evaluation through [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
