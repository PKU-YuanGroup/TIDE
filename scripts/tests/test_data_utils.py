"""
Unit tests for dllm.data.utils helpers: _match, _ensure_datasetdict, _truncate_split,
_merge_datasetdicts, _concat_splits.

Run from repo root:
  source ~/.zshrc && conda activate ~/miniconda3/envs/dllm
  pytest /mnt/lustrenew/mllm_aligned/dongzhichen/dllm/scripts/tests/test_data_utils.py -v
"""

import pytest
from datasets import Dataset, DatasetDict

from dllm.data.utils import (
    _match,
    _ensure_datasetdict,
    _truncate_split,
    _truncate_datasetdict,
    _concat_splits,
    _merge_datasetdicts,
)


class TestMatch:
    def test_single_needle_endswith(self):
        assert _match("path/to/tatsu-lab/alpaca", "alpaca") is True
        assert _match("tatsu-lab/alpaca", "alpaca") is True

    def test_single_needle_in(self):
        assert _match("something-alpaca-something", "alpaca") is True

    def test_no_match(self):
        assert _match("other/dataset", "alpaca") is False

    def test_list_needles(self):
        assert _match("OpenCoder-LLM/opc-sft-stage2", ["opc-sft-stage1", "opc-sft-stage2"]) is True
        assert _match("OpenCoder-LLM/opc-sft-stage1", ["opc-sft-stage1", "opc-sft-stage2"]) is True
        assert _match("other/ds", ["opc-sft-stage1"]) is False


class TestEnsureDatasetDict:
    def test_already_datasetdict(self):
        ds = DatasetDict({"train": Dataset.from_dict({"a": [1, 2]})})
        out = _ensure_datasetdict(ds)
        assert out is ds
        assert list(out.keys()) == ["train"]

    def test_dict_wrapped(self):
        d = {"train": Dataset.from_dict({"a": [1, 2]})}
        out = _ensure_datasetdict(d)
        assert isinstance(out, DatasetDict)
        assert list(out.keys()) == ["train"]

    def test_single_dataset_becomes_train(self):
        ds = Dataset.from_dict({"x": [1, 2, 3]})
        out = _ensure_datasetdict(ds)
        assert isinstance(out, DatasetDict)
        assert list(out.keys()) == ["train"]
        assert out["train"].num_rows == 3


class TestTruncateSplit:
    def test_none_limit_returns_unchanged(self):
        ds = Dataset.from_dict({"a": [1, 2, 3]})
        out = _truncate_split(ds, None)
        assert out is ds

    def test_truncate_to_n(self):
        ds = Dataset.from_dict({"a": [1, 2, 3, 4, 5]})
        out = _truncate_split(ds, 2)
        assert out.num_rows == 2

    def test_truncate_larger_than_size(self):
        ds = Dataset.from_dict({"a": [1, 2, 3]})
        out = _truncate_split(ds, 10)
        assert out.num_rows == 3


class TestTruncateDatasetDict:
    def test_single_split_with_limits(self):
        ds = DatasetDict({
            "train": Dataset.from_dict({"a": list(range(20))})
        })
        out = _truncate_datasetdict(ds, {"train": 5})
        assert "train" in out
        assert out["train"].num_rows == 5

    def test_multi_split_truncate(self):
        ds = DatasetDict({
            "train": Dataset.from_dict({"a": list(range(10))}),
            "test": Dataset.from_dict({"a": list(range(5))}),
        })
        out = _truncate_datasetdict(ds, {"train": 3, "test": 2})
        assert out["train"].num_rows == 3
        assert out["test"].num_rows == 2

    def test_empty_limits_returns_unchanged(self):
        ds = DatasetDict({"train": Dataset.from_dict({"a": [1, 2, 3]})})
        out = _truncate_datasetdict(ds, {})
        assert out["train"].num_rows == 3


class TestConcatSplits:
    def test_concat_two_datasets(self):
        a = Dataset.from_dict({"x": [1, 2]})
        b = Dataset.from_dict({"x": [3, 4]})
        out = _concat_splits(a, b)
        assert out.num_rows == 4
        assert list(out["x"]) == [1, 2, 3, 4]

    def test_none_first_returns_second(self):
        b = Dataset.from_dict({"x": [1]})
        assert _concat_splits(None, b) is b

    def test_none_second_returns_first(self):
        a = Dataset.from_dict({"x": [1]})
        assert _concat_splits(a, None) is a


class TestMergeDatasetDicts:
    def test_merge_disjoint_splits(self):
        d1 = DatasetDict({"train": Dataset.from_dict({"a": [1, 2]})})
        d2 = DatasetDict({"test": Dataset.from_dict({"a": [3, 4]})})
        out = _merge_datasetdicts(d1, d2)
        assert set(out.keys()) == {"train", "test"}
        assert out["train"].num_rows == 2
        assert out["test"].num_rows == 2

    def test_merge_overlapping_splits_concat(self):
        d1 = DatasetDict({"train": Dataset.from_dict({"a": [1, 2]})})
        d2 = DatasetDict({"train": Dataset.from_dict({"a": [3, 4]})})
        out = _merge_datasetdicts(d1, d2)
        assert out["train"].num_rows == 4
        assert list(out["train"]["a"]) == [1, 2, 3, 4]
