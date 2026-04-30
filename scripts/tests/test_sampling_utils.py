"""
Unit tests for dllm sampling utilities: sample_trim, infill_trim, add_gumbel_noise, get_num_transfer_tokens.

Run from repo root:
  source ~/.zshrc && conda activate ~/miniconda3/envs/dllm
  pytest /mnt/lustrenew/mllm_aligned/dongzhichen/dllm/scripts/tests/test_sampling_utils.py -v
"""

import pytest
import torch

from dllm.utils.sampling import sample_trim, infill_trim
from dllm.core.samplers.utils import add_gumbel_noise, get_num_transfer_tokens
from dllm.core.schedulers import LinearAlphaScheduler


def _make_mock_tokenizer(
    pad_token_id=0,
    eos_token_id=1,
    eot_token_id=None,
    eos_token="</s>",
    eot_token=None,
    mask_token_id=2,
):
    """Minimal tokenizer-like object for testing trim functions."""
    from types import SimpleNamespace
    tok = SimpleNamespace()
    tok.pad_token_id = pad_token_id
    tok.eos_token_id = eos_token_id
    tok.eot_token_id = eot_token_id
    tok.eos_token = eos_token
    tok.eot_token = eot_token
    tok.mask_token_id = mask_token_id

    def decode(ids, skip_special_tokens=True):
        # Treat 0 as pad, 1 as eos; strip them if skip_special_tokens
        ids = list(ids)
        if skip_special_tokens:
            ids = [i for i in ids if i not in (0, 1)]
        return "".join(chr(ord("a") + (i % 26)) for i in ids)

    tok.decode = decode
    return tok


# ---------------------------------------------------------------------------
# sample_trim
# ---------------------------------------------------------------------------


class TestSampleTrim:
    def test_no_eos_returns_full_generation(self):
        tok = _make_mock_tokenizer(pad_token_id=0, eos_token_id=1)
        seq_ids = [10, 11, 12, 13, 14]  # prompt len 2 -> gen 3
        input_ids = [10, 11]
        out = sample_trim(tok, [seq_ids], [input_ids])
        assert len(out) == 1
        # Decoded gen part: indices 2:5 -> [12,13,14]
        assert out[0] == "mno"  # 12->m, 13->n, 14->o

    def test_stops_at_first_eos_after_prompt(self):
        tok = _make_mock_tokenizer(pad_token_id=0, eos_token_id=1)
        seq_ids = [10, 11, 12, 1, 99]  # eos at index 3
        input_ids = [10, 11]
        out = sample_trim(tok, [seq_ids], [input_ids])
        assert len(out) == 1
        # gen = 12 only (then eos); decode skips 1
        assert out[0] == "m"  # gen [12], eos at 3

    def test_left_padding_skipped(self):
        tok = _make_mock_tokenizer(pad_token_id=0, eos_token_id=1)
        seq_ids = [0, 0, 10, 11, 12]  # 2 pad, prompt 10,11 -> start at 2, gen 12
        input_ids = [10, 11]
        out = sample_trim(tok, [seq_ids], [input_ids])
        assert len(out) == 1
        # After skipping pads: [10,11,12], start=2, end=3 -> gen [12]
        assert out[0] == "m"

    def test_batch_multiple_sequences(self):
        tok = _make_mock_tokenizer(pad_token_id=0, eos_token_id=1)
        seq_list = [[1, 2, 3, 4], [5, 6, 7, 8]]
        input_list = [[1, 2], [5, 6]]
        out = sample_trim(tok, seq_list, input_list)
        assert len(out) == 2
        assert out[0] == "de"  # gen [3,4] -> d,e
        assert out[1] == "hi"  # gen [7,8] -> h,i


# ---------------------------------------------------------------------------
# infill_trim
# ---------------------------------------------------------------------------


class TestInfillTrim:
    def test_extracts_masked_positions_only(self):
        tok = _make_mock_tokenizer(pad_token_id=0, eos_token_id=1, mask_token_id=99)
        # input has mask at positions that get filled; full seq has values there
        prompt = [10, 99, 11, 99]  # two masks
        full = [10, 20, 11, 21]    # infill 20, 21
        out = infill_trim(tok, [full], [prompt])
        assert len(out) == 1
        # infill tokens = [20, 21] -> decode
        assert out[0] == "uv"  # 20->u, 21->v

    def test_infill_stops_at_eos(self):
        tok = _make_mock_tokenizer(pad_token_id=0, eos_token_id=1, mask_token_id=99)
        prompt = [99, 99, 99]
        full = [2, 3, 1]  # eos at index 2 in infill; gen before = [2,3]
        out = infill_trim(tok, [full], [prompt])
        assert len(out) == 1
        assert out[0] == "cd"  # 2->c, 3->d (chr(97+i))


# ---------------------------------------------------------------------------
# add_gumbel_noise
# ---------------------------------------------------------------------------


class TestAddGumbelNoise:
    def test_temperature_zero_returns_unchanged(self):
        logits = torch.randn(2, 10)
        out = add_gumbel_noise(logits, temperature=0.0)
        assert out is logits
        assert torch.allclose(out.float(), logits.float())

    def test_temperature_positive_changes_values(self):
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        out = add_gumbel_noise(logits, temperature=0.5)
        assert out.shape == logits.shape
        assert out.dtype == torch.float64
        # Random; at least check no nan/inf
        assert torch.isfinite(out).all()

    def test_output_dtype_float64(self):
        logits = torch.randn(3, 5, dtype=torch.float32)
        out = add_gumbel_noise(logits, temperature=0.3)
        assert out.dtype == torch.float64


# ---------------------------------------------------------------------------
# get_num_transfer_tokens
# ---------------------------------------------------------------------------


class TestGetNumTransferTokens:
    def test_deterministic_single_sample(self):
        scheduler = LinearAlphaScheduler()
        # 4 masked tokens, 3 steps
        mask_index = torch.tensor([[True, True, True, True]])
        out = get_num_transfer_tokens(mask_index, steps=3, scheduler=scheduler, stochastic=False)
        assert out.shape[0] == 1
        assert out.dtype == torch.int64
        # Sum of transfers should not exceed 4
        assert out.sum().item() <= 4

    def test_batch(self):
        scheduler = LinearAlphaScheduler()
        mask_index = torch.tensor([
            [True, True, False, False],
            [True, True, True, True],
        ])
        out = get_num_transfer_tokens(mask_index, steps=4, scheduler=scheduler, stochastic=False)
        assert out.shape[0] == 2
        assert out.dtype == torch.int64

    def test_stochastic_same_shape(self):
        scheduler = LinearAlphaScheduler()
        mask_index = torch.tensor([[True] * 10])
        out = get_num_transfer_tokens(mask_index, steps=5, scheduler=scheduler, stochastic=True)
        assert out.shape[0] == 1
        assert out.dtype == torch.int64
        assert (out >= 0).all()
