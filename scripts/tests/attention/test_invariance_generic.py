"""
LLaDA / MoE / Dream attention mask invariance test.

Run: pytest /mnt/lustrenew/mllm_aligned/dongzhichen/dllm/scripts/tests/attention/test_invariance_generic.py -v
"""

import gc

import pytest
import torch
import transformers

import dllm

from .common import (
    BASE_TOKEN_SETS,
    ERROR_THRESHOLD,
    _assert_batch_equal_to_single,
    _assert_invariance,
    _cuda_cleanup,
    _forward_batch_nopad,
    _forward_batch_padded,
    _forward_variants,
)


@pytest.mark.parametrize(
    "model_name_or_path, attn_impl, use_position_ids",
    [
        ("GSAI-ML/LLaDA-8B-Base", None, False),
        ("inclusionAI/LLaDA-MoE-7B-A1B-Base", None, False),
        ("Dream-org/Dream-v0-Base-7B", None, False),
    ],
)
def test_attention_mask_invariance(model_name_or_path, attn_impl, use_position_ids):
    """
    For each model:

      1) Single-sample invariance over all base token sets:
           no_padding, left_padding, right_padding,
           no_mask, mask_omitted.

      2) Batch without padding:
           stack all base token sets.

      3) Batch with padding:
           for each base, create right-padded + left-padded rows.

      All logits on the 4 real tokens must match single-sample "no_padding"
      for every base token set.
    """
    model_path = dllm.utils.resolve_with_base_env(model_name_or_path, "BASE_MODELS_DIR")

    # Load model. We assume it's a decoder-style model with .logits.
    if attn_impl is None:
        model = transformers.AutoModel.from_pretrained(
            model_path,
            dtype=torch.float32,
            device_map="auto",
        ).eval()
    else:
        config = transformers.AutoConfig.from_pretrained(
            model_path,
            attn_implementation=attn_impl,
        )
        model = transformers.AutoModel.from_pretrained(
            model_path,
            config=config,
            dtype=torch.float32,
            device_map="auto",
        ).eval()

    # 1) Single-sample variants over all base token sets
    outs_single = _forward_variants(model, use_position_ids=use_position_ids)
    _assert_invariance(outs_single, f"{model_name_or_path}/pos_ids={use_position_ids}")
    single_no_pad = outs_single["no_padding"]  # [N,4,H]

    # 2) Batch (no padding)
    batch_nopad = _forward_batch_nopad(
        model, use_position_ids=use_position_ids
    )  # [N,4,H]

    # 3) Batch (padded, left+right)
    batch_padded = _forward_batch_padded(
        model, use_position_ids=use_position_ids
    )  # [2N,5,H]

    _assert_batch_equal_to_single(
        single_no_pad=single_no_pad,
        batch_nopad=batch_nopad,
        batch_padded=batch_padded,
        tag=f"{model_name_or_path}/pos_ids={use_position_ids}",
    )

    print(
        f"✅ {model_name_or_path} (pos_ids={use_position_ids}) passed: "
        f"mask invariance + batch (no-pad & padded) consistency across "
        f"{len(BASE_TOKEN_SETS)} base token sets within {ERROR_THRESHOLD}."
    )

    del model
    gc.collect()
    _cuda_cleanup()
