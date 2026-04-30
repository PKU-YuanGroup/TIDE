# Fast-dLLM

> 📄 Paper: [Fast-dLLM](https://arxiv.org/pdf/2505.22618) | 💻 Code: [github.com/NVlabs/Fast-dLLM](https://github.com/NVlabs/Fast-dLLM)

Resources and examples for inferencing and evaluating **LLaDA** and **Dream** with **Fast-dLLM**.

## Table of Contents
- [Files](#files)
- [Inference](#inference)
- [Evaluation](#evaluation)

## Files

```
# Pipeline modules for Fast-dLLM
dllm/pipelines/fastdllm
├── __init__.py
├── dream/
│   ├── __init__.py
│   ├── models/
│   │   ├── configuration_dream.py  # Fast-dLLM Dream model configuration
│   │   └── modeling_dream.py       # Fast-dLLM Dream model architecture
│   ├── sampler.py                  # Fast-dLLM Dream inference module
│   └── eval.py                     # Fast-dLLM Dream evaluation module
└── llada/
    ├── __init__.py
    ├── models/
    │   ├── configuration_llada.py  # Fast-dLLM LLaDA model configuration
    │   └── modeling_llada.py       # Fast-dLLM LLaDA model architecture
    ├── sampler.py                  # Fast-dLLM LLaDA inference module
    └── eval.py                     # Fast-dLLM LLaDA evaluation module

# Example entry points for inference and evaluation
examples/fastdllm
├── README.md                       # Documentation (you are here)
├── dream/
│   ├── sample.py                   # Fast-dLLM Dream inference example
│   └── eval.sh                     # Fast-dLLM Dream evaluation example
└── llada/
    ├── sample.py                   # Fast-dLLM LLaDA inference example
    └── eval.sh                     # Fast-dLLM LLaDA evaluation example
```

## Inference

> Implementation matches the original Fast-dLLM; identical hyperparameters yield identical outputs.

Sampling with the Fast-dLLM LLaDA sampler (e.g. prefix cache + confidence threshold):

```shell
# Use --use_cache (none/prefix/dual) for cache scheme, --threshold for confidence-based sampling
python examples/fastdllm/llada/sample.py --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct" --use_cache prefix --threshold 0.9
```

Sampling with the Fast-dLLM Dream sampler (e.g. prefix cache + confidence-threshold decoding):

```shell
# Use --use_cache (none/prefix/dual) for cache scheme, --alg (entropy/confidence_threshold) and --threshold for decoding
python examples/fastdllm/dream/sample.py --model_name_or_path "Dream-org/Dream-v0-Instruct-7B" --use_cache prefix --alg confidence_threshold --threshold 0.9
```

## Evaluation

> Read [(optional) Evaluation setup](/README.md#optional-evaluation-setup) before running evaluation.

For example, to evaluate [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) or [Dream-v0-Base-7B](https://huggingface.co/Dream-org/Dream-v0-Base-7B) on [GSM8K](https://huggingface.co/datasets/openai/gsm8k) with 4 GPUs, run:

```shell
# Use model_args to pass Fast-dLLM sampling options (use_cache, threshold, etc.).
accelerate launch --num_processes 4 \
    dllm/pipelines/fastdllm/llada/eval.py \
    --tasks "gsm8k" \
    --num_fewshot 5 \
    --model "fastdllm_llada" \
    --apply_chat_template \
    --model_args "pretrained=GSAI-ML/LLaDA-8B-Instruct,use_cache=prefix,threshold=0.9,max_new_tokens=256,steps=256,block_size=32,suppress_tokens=[],begin_suppress_tokens=[]"

accelerate launch --num_processes 4 \
    dllm/pipelines/fastdllm/dream/eval.py \
    --tasks "gsm8k" \
    --num_fewshot 5 \
    --model "fastdllm_dream" \
    --apply_chat_template \
    --model_args "pretrained=Dream-org/Dream-v0-Base-7B,use_cache=prefix,max_new_tokens=256,steps=256,block_size=32,alg=confidence_threshold,threshold=0.9,dtype=bfloat16,add_bos_token=True"
```

To evaluate [LLaDA-8B-Instruct](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct) and [Dream-v0-Base-7B](https://huggingface.co/Dream-org/Dream-v0-Base-7B) with the Fast-dLLM sampler across all cache settings and benchmarks, run:

```shell
bash examples/fastdllm/llada/eval.sh --model_name_or_path "GSAI-ML/LLaDA-8B-Instruct" --instruct True --num_gpu 1
bash examples/fastdllm/dream/eval.sh --model_name_or_path "Dream-org/Dream-v0-Base-7B" --instruct False --num_gpu 1
```

### Evaluation results

> Results (Reproduced) are evaluated using our framework, while results (Official) come from the original [paper](https://arxiv.org/pdf/2505.22618). All evaluation settings follow the [Fast-dLLM repository](https://github.com/NVlabs/Fast-dLLM) with minor modifications; we add support for MBPP and Minerva-Math benchmarks (not provided in the original repo).

<table align="center">
<colgroup>
  <col span="2">
  <col span="8">
  <col span="8" style="background-color: #f6f8fa;">
</colgroup>
<thead>
  <tr>
    <th rowspan="3"><b>Benchmark</b></th>
    <th rowspan="3"><b>Source</b></th>
    <th colspan="8" align="center"><b>Len = 256</b></th>
    <th colspan="8" align="center" style="background-color:#f6f8fa;"><b>Len = 512</b></th>
  </tr>
  <tr>
    <th colspan="2"><b>Baseline</b></th>
    <th colspan="2"><b>+Cache</b></th>
    <th colspan="2"><b>+Parallel</b></th>
    <th colspan="2"><b>+Cache & Parallel</b></th>
    <th colspan="2"><b>Baseline</b></th>
    <th colspan="2"><b>+Cache</b></th>
    <th colspan="2"><b>+Parallel</b></th>
    <th colspan="2"><b>+Cache & Parallel</b></th>
  </tr>
  <tr>
    <th>Acc</th><th style="color:#2563eb;">Tok/s (×)</th>
    <th>Acc</th><th style="color:#2563eb;">Tok/s (×)</th>
    <th>Acc</th><th style="color:#2563eb;">Tok/s (×)</th>
    <th>Acc</th><th style="color:#2563eb;">Tok/s (×)</th>
    <th>Acc</th><th style="color:#2563eb;">Tok/s (×)</th>
    <th>Acc</th><th style="color:#2563eb;">Tok/s (×)</th>
    <th>Acc</th><th style="color:#2563eb;">Tok/s (×)</th>
    <th>Acc</th><th style="color:#2563eb;">Tok/s (×)</th>
  </tr>
</thead>
<tbody>

<!-- GSM8K -->
<tr>
<td rowspan="2"><b>GSM8K</b></td>
<td>Official</td>
<td align="center">79.3</td><td align="center" style="color:#2563eb;">6.7&nbsp;(1.0×)</td>
<td align="center">79.5</td><td align="center" style="color:#2563eb;">21.2&nbsp;(3.2×)</td>
<td align="center">79.2</td><td align="center" style="color:#2563eb;">16.5&nbsp;(2.5×)</td>
<td align="center">78.5</td><td align="center" style="color:#2563eb;">54.4&nbsp;(8.1×)</td>
<td align="center" style="background-color:#f6f8fa;">77.5</td><td align="center" style="color:#2563eb;">3.2&nbsp;(1.0×)</td>
<td align="center">77.0</td><td align="center" style="color:#2563eb;">10.4&nbsp;(3.3×)</td>
<td align="center">77.6</td><td align="center" style="color:#2563eb;">18.6&nbsp;(5.8×)</td>
<td align="center">77.2</td><td align="center" style="color:#2563eb;">35.3&nbsp;(11.0×)</td>
</tr>

<tr>
<td>Reproduced</td>
<td align="center">78.0</td><td align="center" style="color:#2563eb;">8.1&nbsp;(1.0×)</td>
<td align="center">78.2</td><td align="center" style="color:#2563eb;">25.6&nbsp;(3.2×)</td>
<td align="center">78.9</td><td align="center" style="color:#2563eb;">18.4&nbsp;(2.3×)</td>
<td align="center">78.0</td><td align="center" style="color:#2563eb;">52.8&nbsp;(6.5×)</td>
<td align="center" style="background-color:#f6f8fa;">81.1</td><td align="center" style="color:#2563eb;">6.7&nbsp;(1.0×)</td>
<td align="center">76.0</td><td align="center" style="color:#2563eb;">19.9&nbsp;(3.0×)</td>
<td align="center">77.6</td><td align="center" style="color:#2563eb;">21.8&nbsp;(3.3×)</td>
<td align="center">76.6</td><td align="center" style="color:#2563eb;">51.7&nbsp;(7.8×)</td>
</tr>

<!-- MATH -->
<tr>
<td rowspan="2"><b>MATH</b></td>
<td>Official</td>
<td align="center">33.5</td><td align="center" style="color:#2563eb;">9.1&nbsp;(1.0×)</td>
<td align="center">33.3</td><td align="center" style="color:#2563eb;">23.7&nbsp;(2.6×)</td>
<td align="center">33.4</td><td align="center" style="color:#2563eb;">24.8&nbsp;(2.7×)</td>
<td align="center">33.2</td><td align="center" style="color:#2563eb;">51.7&nbsp;(5.7×)</td>
<td align="center" style="background-color:#f6f8fa;">37.2</td><td align="center" style="color:#2563eb;">8.0&nbsp;(1.0×)</td>
<td align="center">36.2</td><td align="center" style="color:#2563eb;">19.7&nbsp;(2.5×)</td>
<td align="center">36.8</td><td align="center" style="color:#2563eb;">23.8&nbsp;(3.0×)</td>
<td align="center">36.0</td><td align="center" style="color:#2563eb;">47.1&nbsp;(5.9×)</td>
</tr>

<tr>
<td>Reproduced</td>
<td align="center">38.3</td><td align="center" style="color:#2563eb;">9.7&nbsp;(1.0×)</td>
<td align="center">37.6</td><td align="center" style="color:#2563eb;">26.4&nbsp;(2.7×)</td>
<td align="center">38.6</td><td align="center" style="color:#2563eb;">19.6&nbsp;(2.0×)</td>
<td align="center">37.5</td><td align="center" style="color:#2563eb;">49.0&nbsp;(5.0×)</td>
<td align="center" style="background-color:#f6f8fa;">42.4</td><td align="center" style="color:#2563eb;">7.4&nbsp;(1.0×)</td>
<td align="center">41.9</td><td align="center" style="color:#2563eb;">21.1&nbsp;(2.9×)</td>
<td align="center">42.5</td><td align="center" style="color:#2563eb;">19.8&nbsp;(2.7×)</td>
<td align="center">41.8</td><td align="center" style="color:#2563eb;">44.8&nbsp;(6.1×)</td>
</tr>

<!-- HumanEval -->
<tr>
<td rowspan="2"><b>HumanEval</b></td>
<td>Official</td>
<td align="center">41.5</td><td align="center" style="color:#2563eb;">30.5&nbsp;(1.0×)</td>
<td align="center">42.7</td><td align="center" style="color:#2563eb;">40.7&nbsp;(1.3×)</td>
<td align="center">43.9</td><td align="center" style="color:#2563eb;">101.5&nbsp;(3.3×)</td>
<td align="center">43.3</td><td align="center" style="color:#2563eb;">114.1&nbsp;(3.7×)</td>
<td align="center" style="background-color:#f6f8fa;">43.9</td><td align="center" style="color:#2563eb;">18.4&nbsp;(1.0×)</td>
<td align="center">45.7</td><td align="center" style="color:#2563eb;">29.3&nbsp;(1.6×)</td>
<td align="center">43.3</td><td align="center" style="color:#2563eb;">57.1&nbsp;(3.1×)</td>
<td align="center">44.5</td><td align="center" style="color:#2563eb;">73.7&nbsp;(4.0×)</td>
</tr>

<tr>
<td>Reproduced</td>
<td align="center">38.4</td><td align="center" style="color:#2563eb;">18.8&nbsp;(1.0×)</td>
<td align="center">36.0</td><td align="center" style="color:#2563eb;">27.6&nbsp;(1.5×)</td>
<td align="center">39.6</td><td align="center" style="color:#2563eb;">53.0&nbsp;(2.8×)</td>
<td align="center">36.0</td><td align="center" style="color:#2563eb;">68.1&nbsp;(3.6×)</td>
<td align="center" style="background-color:#f6f8fa;">48.2</td><td align="center" style="color:#2563eb;">13.0&nbsp;(1.0×)</td>
<td align="center">41.5</td><td align="center" style="color:#2563eb;">23.3&nbsp;(1.8×)</td>
<td align="center">50.6</td><td align="center" style="color:#2563eb;">36.3&nbsp;(2.8×)</td>
<td align="center">41.5</td><td align="center" style="color:#2563eb;">55.8&nbsp;(4.3×)</td>
</tr>

<!-- MBPP -->
<tr>
<td rowspan="2"><b>MBPP</b></td>
<td>Official</td>
<td align="center">29.4</td><td align="center" style="color:#2563eb;">6.0&nbsp;(1.0×)</td>
<td align="center">29.6</td><td align="center" style="color:#2563eb;">17.0&nbsp;(2.8×)</td>
<td align="center">28.4</td><td align="center" style="color:#2563eb;">24.8&nbsp;(4.1×)</td>
<td align="center">28.2</td><td align="center" style="color:#2563eb;">44.8&nbsp;(7.5×)</td>
<td align="center" style="background-color:#f6f8fa;">14.8</td><td align="center" style="color:#2563eb;">4.3&nbsp;(1.0×)</td>
<td align="center">13.4</td><td align="center" style="color:#2563eb;">10.1&nbsp;(2.3×)</td>
<td align="center">15.0</td><td align="center" style="color:#2563eb;">22.3&nbsp;(5.1×)</td>
<td align="center">13.8</td><td align="center" style="color:#2563eb;">39.5&nbsp;(9.2×)</td>
</tr>

<tr>
<td>Reproduced</td>
<td align="center">36.4</td><td align="center" style="color:#2563eb;">9.3&nbsp;(1.0×)</td>
<td align="center">38.0</td><td align="center" style="color:#2563eb;">26.2&nbsp;(2.8×)</td>
<td align="center">29.0</td><td align="center" style="color:#2563eb;">17.6&nbsp;(1.9×)</td>
<td align="center">37.8</td><td align="center" style="color:#2563eb;">44.7&nbsp;(4.8×)</td>
<td align="center" style="background-color:#f6f8fa;">32.2</td><td align="center" style="color:#2563eb;">7.7&nbsp;(1.0×)</td>
<td align="center">22.0</td><td align="center" style="color:#2563eb;">20.8&nbsp;(2.7×)</td>
<td align="center">7.6</td><td align="center" style="color:#2563eb;">21.0&nbsp;(2.7×)</td>
<td align="center">21.4</td><td align="center" style="color:#2563eb;">43.9&nbsp;(5.7×)</td>
</tr>

</tbody>
</table>

<p align="center"><strong>Table 1</strong>: Evaluation results of <a href="https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct">LLaDA-8B-Instruct</a> with Fast-dLLM.</p>

<table align="center">
<colgroup>
  <col span="2">
  <col span="8">
  <col span="8" style="background-color: #f6f8fa;">
</colgroup>
<thead>
  <tr>
    <th rowspan="3"><b>Benchmark</b></th>
    <th rowspan="3"><b>Source</b></th>
    <th colspan="8" align="center" style="text-align: center;"><b>Len = 256</b></th>
    <th colspan="8" align="center" style="background-color: #f6f8fa; text-align: center;"><b>Len = 512</b></th>
  </tr>
  <tr>
    <th colspan="2" align="center"><b>Baseline</b></th>
    <th colspan="2" align="center"><b>+Cache</b></th>
    <th colspan="2" align="center"><b>+Parallel</b></th>
    <th colspan="2" align="center"><b>+Cache & Parallel</b></th>
    <th colspan="2" align="center" style="background-color: #f6f8fa;"><b>Baseline</b></th>
    <th colspan="2" align="center"><b>+Cache</b></th>
    <th colspan="2" align="center"><b>+Parallel</b></th>
    <th colspan="2" align="center"><b>+Cache & Parallel</b></th>
  </tr>
  <tr>
    <th align="center">Acc</th>
    <th align="center" style="color: #2563eb;">Tok/s (×)</th>
    <th align="center">Acc</th>
    <th align="center" style="color: #2563eb;">Tok/s (×)</th>
    <th align="center">Acc</th>
    <th align="center" style="color: #2563eb;">Tok/s (×)</th>
    <th align="center">Acc</th>
    <th align="center" style="color: #2563eb;">Tok/s (×)</th>
    <th align="center" style="background-color: #f6f8fa;">Acc</th>
    <th align="center" style="color: #2563eb;">Tok/s (×)</th>
    <th align="center">Acc</th>
    <th align="center" style="color: #2563eb;">Tok/s (×)</th>
    <th align="center">Acc</th>
    <th align="center" style="color: #2563eb;">Tok/s (×)</th>
    <th align="center">Acc</th>
    <th align="center" style="color: #2563eb;">Tok/s (×)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="2"><b>GSM8K</b></td>
    <td>Official</td>
    <td align="center">75.0</td><td align="center" style="color: #2563eb;">9.1&nbsp;(1.0×)</td>
    <td align="center">74.3</td><td align="center" style="color: #2563eb;">32.5&nbsp;(3.6×)</td>
    <td align="center">74.2</td><td align="center" style="color: #2563eb;">14.2&nbsp;(1.6×)</td>
    <td align="center">74.8</td><td align="center" style="color: #2563eb;">48.2&nbsp;(5.3×)</td>
    <td align="center" style="background-color: #f6f8fa;">76.0</td><td align="center" style="color: #2563eb;">7.7&nbsp;(1.0×)</td>
    <td align="center">74.3</td><td align="center" style="color: #2563eb;">25.6&nbsp;(3.3×)</td>
    <td align="center">73.4</td><td align="center" style="color: #2563eb;">14.6&nbsp;(1.9×)</td>
    <td align="center">74.0</td><td align="center" style="color: #2563eb;">42.9&nbsp;(5.6×)</td>
  </tr>
  <tr>
    <td>Reproduced</td>
    <td align="center">75.4</td><td align="center" style="color: #2563eb;">9.0&nbsp;(1.0×)</td>
    <td align="center">75.0</td><td align="center" style="color: #2563eb;">32.9&nbsp;(3.7×)</td>
    <td align="center">72.6</td><td align="center" style="color: #2563eb;">12.1&nbsp;(1.4×)</td>
    <td align="center">74.2</td><td align="center" style="color: #2563eb;">42.2&nbsp;(4.7×)</td>
    <td align="center" style="background-color: #f6f8fa;">75.7</td><td align="center" style="color: #2563eb;">7.6&nbsp;(1.0×)</td>
    <td align="center">73.8</td><td align="center" style="color: #2563eb;">25.6&nbsp;(3.4×)</td>
    <td align="center">72.7</td><td align="center" style="color: #2563eb;">11.8&nbsp;(1.6×)</td>
    <td align="center">74.5</td><td align="center" style="color: #2563eb;">33.7&nbsp;(4.4×)</td>
  </tr>

  <tr>
    <td rowspan="2"><b>MATH</b></td>
    <td>Official</td>
    <td align="center">38.4</td><td align="center" style="color: #2563eb;">11.4&nbsp;(1.0×)</td>
    <td align="center">36.8</td><td align="center" style="color: #2563eb;">34.3&nbsp;(3.0×)</td>
    <td align="center">37.9</td><td align="center" style="color: #2563eb;">27.3&nbsp;(2.4×)</td>
    <td align="center">37.6</td><td align="center" style="color: #2563eb;">66.8&nbsp;(5.9×)</td>
    <td align="center" style="background-color: #f6f8fa;">39.8</td><td align="center" style="color: #2563eb;">9.6&nbsp;(1.0×)</td>
    <td align="center">38.0</td><td align="center" style="color: #2563eb;">26.8&nbsp;(2.8×)</td>
    <td align="center">39.5</td><td align="center" style="color: #2563eb;">31.6&nbsp;(3.2×)</td>
    <td align="center">39.3</td><td align="center" style="color: #2563eb;">63.3&nbsp;(6.5×)</td>
  </tr>
  <tr>
    <td>Reproduced</td>
    <td align="center">31.5</td><td align="center" style="color: #2563eb;">25.1&nbsp;(1.0×)</td>
    <td align="center">33.3</td><td align="center" style="color: #2563eb;">36.8&nbsp;(1.5×)</td>
    <td align="center">23.5</td><td align="center" style="color: #2563eb;">52.1&nbsp;(2.1×)</td>
    <td align="center">31.1</td><td align="center" style="color: #2563eb;">76.7&nbsp;(3.1×)</td>
    <td align="center" style="background-color: #f6f8fa;">39.2</td><td align="center" style="color: #2563eb;">15.8&nbsp;(1.0×)</td>
    <td align="center">39.2</td><td align="center" style="color: #2563eb;">28.1&nbsp;(1.8×)</td>
    <td align="center">32.0</td><td align="center" style="color: #2563eb;">25.5&nbsp;(1.6×)</td>
    <td align="center">38.9</td><td align="center" style="color: #2563eb;">46.3&nbsp;(2.9×)</td>
  </tr>

  <tr>
    <td rowspan="2"><b>HumanEval</b></td>
    <td>Official</td>
    <td align="center">49.4</td><td align="center" style="color: #2563eb;">23.3&nbsp;(1.0×)</td>
    <td align="center">53.7</td><td align="center" style="color: #2563eb;">35.2&nbsp;(1.5×)</td>
    <td align="center">49.4</td><td align="center" style="color: #2563eb;">45.6&nbsp;(2.0×)</td>
    <td align="center">54.3</td><td align="center" style="color: #2563eb;">62.0&nbsp;(2.8×)</td>
    <td align="center" style="background-color: #f6f8fa;">54.3</td><td align="center" style="color: #2563eb;">16.3&nbsp;(1.0×)</td>
    <td align="center">54.9</td><td align="center" style="color: #2563eb;">27.8&nbsp;(1.7×)</td>
    <td align="center">51.8</td><td align="center" style="color: #2563eb;">29.8&nbsp;(1.8×)</td>
    <td align="center">54.3</td><td align="center" style="color: #2563eb;">52.8&nbsp;(3.2×)</td>
  </tr>
  <tr>
    <td>Reproduced</td>
    <td align="center">57.9</td><td align="center" style="color: #2563eb;">14.0&nbsp;(1.0×)</td>
    <td align="center">53.7</td><td align="center" style="color: #2563eb;">33.8&nbsp;(2.4×)</td>
    <td align="center">51.2</td><td align="center" style="color: #2563eb;">21.0&nbsp;(1.5×)</td>
    <td align="center">53.1</td><td align="center" style="color: #2563eb;">43.0&nbsp;(3.1×)</td>
    <td align="center" style="background-color: #f6f8fa;">54.9</td><td align="center" style="color: #2563eb;">10.4&nbsp;(1.0×)</td>
    <td align="center">54.9</td><td align="center" style="color: #2563eb;">26.3&nbsp;(2.5×)</td>
    <td align="center">50.6</td><td align="center" style="color: #2563eb;">16.4&nbsp;(1.6×)</td>
    <td align="center">54.3</td><td align="center" style="color: #2563eb;">37.4&nbsp;(3.6×)</td>
  </tr>

  <tr>
    <td rowspan="2"><b>MBPP</b></td>
    <td>Official</td>
    <td align="center">56.6</td><td align="center" style="color: #2563eb;">11.2&nbsp;(1.0×)</td>
    <td align="center">53.2</td><td align="center" style="color: #2563eb;">34.5&nbsp;(3.1×)</td>
    <td align="center">53.8</td><td align="center" style="color: #2563eb;">31.8&nbsp;(2.8×)</td>
    <td align="center">56.4</td><td align="center" style="color: #2563eb;">76.0&nbsp;(6.8×)</td>
    <td align="center" style="background-color: #f6f8fa;">55.6</td><td align="center" style="color: #2563eb;">9.4&nbsp;(1.0×)</td>
    <td align="center">53.8</td><td align="center" style="color: #2563eb;">26.7&nbsp;(2.8×)</td>
    <td align="center">55.4</td><td align="center" style="color: #2563eb;">37.6&nbsp;(4.0×)</td>
    <td align="center">55.2</td><td align="center" style="color: #2563eb;">73.6&nbsp;(7.8×)</td>
  </tr>
  <tr>
    <td>Reproduced</td>
    <td align="center">55.6</td><td align="center" style="color: #2563eb;">9.9&nbsp;(1.0×)</td>
    <td align="center">53.8</td><td align="center" style="color: #2563eb;">32.1&nbsp;(3.3×)</td>
    <td align="center">53.6</td><td align="center" style="color: #2563eb;">24.1&nbsp;(2.5×)</td>
    <td align="center">56.0</td><td align="center" style="color: #2563eb;">61.8&nbsp;(6.3×)</td>
    <td align="center" style="background-color: #f6f8fa;">56.0</td><td align="center" style="color: #2563eb;">4.6&nbsp;(1.0×)</td>
    <td align="center">52.6</td><td align="center" style="color: #2563eb;">25.0&nbsp;(5.4×)</td>
    <td align="center">52.8</td><td align="center" style="color: #2563eb;">29.5&nbsp;(6.4×)</td>
    <td align="center">54.4</td><td align="center" style="color: #2563eb;">61.1&nbsp;(13.3×)</td>
  </tr>
</tbody>
</table>

<p align="center"><strong>Table 2</strong>: Evaluation results of <a href="https://huggingface.co/Dream-org/Dream-v0-Base-7B">Dream-v0-Base-7B</a> with Fast-dLLM.</p>
