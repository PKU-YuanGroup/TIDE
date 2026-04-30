"""
Unit tests for dllm.utils.utils: parse_spec, resolve_with_base_env.

Run from repo root:
  source ~/.zshrc && conda activate ~/miniconda3/envs/dllm
  pytest /mnt/lustrenew/mllm_aligned/dongzhichen/dllm/scripts/tests/test_utils_parse_resolve.py -v
"""

import os
import pytest

from dllm.utils.utils import parse_spec, resolve_with_base_env


class TestParseSpec:
    def test_bare_name(self):
        name, kvs = parse_spec("foo/bar")
        assert name == "foo/bar"
        assert kvs == {}

    def test_bare_name_with_bracket_key_value(self):
        name, kvs = parse_spec("ds[name:train]")
        assert name == "ds"
        assert kvs == {"name": "train"}

    def test_bracket_multiple(self):
        name, kvs = parse_spec("x[a:b, c:d]")
        assert name == "x"
        assert "a" in kvs and kvs["a"] == "b"
        assert "c" in kvs and kvs["c"] == "d"

    def test_bracket_numeric_value(self):
        name, kvs = parse_spec("ds[train:5000]")
        assert name == "ds"
        assert kvs["train"] == 5000

    def test_bracket_numeric_with_underscores(self):
        name, kvs = parse_spec("ds[train:5_000]")
        assert name == "ds"
        assert kvs["train"] == 5000

    def test_outer_kv_pairs(self):
        name, kvs = parse_spec("a=1,b=2")
        assert name is None
        assert kvs == {"a": "1", "b": "2"}

    def test_bracket_and_outer_merged(self):
        name, kvs = parse_spec("x[key:val]")
        assert name == "x"
        assert kvs == {"key": "val"}

    def test_invalid_bracket_entry_raises(self):
        with pytest.raises(ValueError, match="expected key:value"):
            parse_spec("x[no_colon]")

    def test_stripped_whitespace(self):
        name, kvs = parse_spec("  foo/bar  ")
        assert name == "foo/bar"
        name2, kvs2 = parse_spec("x [ a : b ]")
        assert name2 == "x"
        assert kvs2["a"] == "b"


class TestResolveWithBaseEnv:
    def test_no_env_returns_path_unchanged(self):
        prev = os.environ.pop("BASE_MODELS_DIR", None)
        try:
            out = resolve_with_base_env("/abs/path", "BASE_MODELS_DIR")
            assert out == "/abs/path"
            out2 = resolve_with_base_env("relative", "BASE_MODELS_DIR")
            assert out2 == "relative"
        finally:
            if prev is not None:
                os.environ["BASE_MODELS_DIR"] = prev

    def test_absolute_path_unchanged(self):
        os.environ["BASE_X"] = "/base"
        try:
            out = resolve_with_base_env("/other/path", "BASE_X")
            assert out == "/other/path"
        finally:
            os.environ.pop("BASE_X", None)

    def test_existing_relative_unchanged(self):
        # If path exists locally, return as is (we can't assume cwd has a specific file)
        os.environ["BASE_X"] = "/tmp"
        try:
            # Path that exists: current dir or a known existing path
            out = resolve_with_base_env(".", "BASE_X")
            assert out == "."
        finally:
            os.environ.pop("BASE_X", None)

    def test_nonexistent_relative_raises_when_base_set(self):
        os.environ["BASE_X"] = "/nonexistent_base_xyz"
        try:
            with pytest.raises(FileNotFoundError, match="Path not found"):
                resolve_with_base_env("nonexistent_sub", "BASE_X")
        finally:
            os.environ.pop("BASE_X", None)
