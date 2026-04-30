"""
Unit tests for dllm.utils.collators: CollatorWrapper, NoAttentionMaskWrapper,
PrependBOSWrapper.

Run from repo root:
  source ~/.zshrc && conda activate ~/miniconda3/envs/dllm
  pytest /mnt/lustrenew/mllm_aligned/dongzhichen/dllm/scripts/tests/test_collators.py -v
"""

import pytest
import torch

from dllm.utils.collators import (
    CollatorWrapper,
    NoAttentionMaskWrapper,
    PrependBOSWrapper,
)


def _identity_collator(features, return_tensors=None):
    """Minimal collator that returns a dict with input_ids and optional attention_mask.
    All input_ids must have the same length so torch.tensor stacks correctly.
    """
    input_ids = torch.tensor([f["input_ids"] for f in features])
    out = {"input_ids": input_ids}
    if any("attention_mask" in f for f in features):
        masks = [f.get("attention_mask", [1] * len(f["input_ids"])) for f in features]
        out["attention_mask"] = torch.tensor(masks)
    if any("labels" in f for f in features):
        labels = [f.get("labels", f["input_ids"]) for f in features]
        out["labels"] = torch.tensor(labels)
    return out


class TestCollatorWrapper:
    def test_passthrough(self):
        base = lambda features, return_tensors=None: {"input_ids": torch.tensor([f["input_ids"] for f in features])}
        wrapped = CollatorWrapper(collator=base)
        features = [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5, 6]}]  # same length
        out = wrapped(features)
        assert "input_ids" in out
        assert out["input_ids"].shape[0] == 2

    def test_before_after_called(self):
        log = []

        class RecordingWrapper(CollatorWrapper):
            def before(self, features):
                log.append("before")
                return features

            def after(self, outputs):
                log.append("after")
                return outputs

        base = lambda features, return_tensors=None: {"x": torch.zeros(1)}
        wrapped = RecordingWrapper(collator=base)
        wrapped([{"input_ids": [1]}])
        assert log == ["before", "after"]


class TestNoAttentionMaskWrapper:
    def test_removes_attention_mask(self):
        base = _identity_collator
        wrapped = NoAttentionMaskWrapper(collator=base)
        features = [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
            {"input_ids": [4, 5, 6], "attention_mask": [1, 1, 1]},
        ]
        out = wrapped(features)
        assert "input_ids" in out
        assert "attention_mask" not in out

    def test_no_op_when_no_mask(self):
        base = lambda features, return_tensors=None: {"input_ids": torch.tensor([f["input_ids"] for f in features])}
        wrapped = NoAttentionMaskWrapper(collator=base)
        out = wrapped([{"input_ids": [1, 2]}])
        assert "input_ids" in out


class TestPrependBOSWrapper:
    def test_prepends_bos_to_input_ids(self):
        base = _identity_collator
        wrapped = PrependBOSWrapper(collator=base, bos_token_id=99)
        features = [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5, 6]}]  # same length
        out = wrapped(features)
        assert out["input_ids"].shape[1] == 4  # 3 + 1 BOS
        assert (out["input_ids"][:, 0] == 99).all()

    def test_prepends_label_pad_for_labels(self):
        base = _identity_collator
        wrapped = PrependBOSWrapper(collator=base, bos_token_id=99, label_pad_token_id=-100)
        features = [
            {"input_ids": [1, 2], "labels": [1, 2]},
            {"input_ids": [3, 4], "labels": [3, 4]},
        ]
        out = wrapped(features)
        assert "labels" in out
        assert (out["labels"][:, 0] == -100).all()

    def test_prepends_one_to_attention_mask(self):
        base = _identity_collator
        wrapped = PrependBOSWrapper(collator=base, bos_token_id=99)
        features = [
            {"input_ids": [1, 2], "attention_mask": [1, 1]},
        ]
        out = wrapped(features)
        assert out["attention_mask"].shape[1] == 3
        assert (out["attention_mask"][:, 0] == 1).all()

    def test_bos_token_id_required(self):
        base = _identity_collator
        wrapped = PrependBOSWrapper(collator=base, bos_token_id=None)
        with pytest.raises(AssertionError):
            wrapped([{"input_ids": [1, 2]}])


class TestCollatorGetAttr:
    def test_delegates_to_inner_collator(self):
        class Inner:
            custom_attr = 42
        base = lambda features, return_tensors=None: {}
        wrapped = CollatorWrapper(collator=base)
        # CollatorWrapper doesn't have custom_attr; it delegates
        base_with_attr = Inner()
        base_with_attr.__call__ = lambda self, features, return_tensors=None: {"input_ids": torch.zeros(1)}
        wrapped2 = CollatorWrapper(collator=base_with_attr)
        assert getattr(wrapped2.collator, "custom_attr") == 42
