"""
Unit tests for dllm.core.eval.base: BaseEvalHarness._build_config.

Run from repo root:
  source ~/.zshrc && conda activate ~/miniconda3/envs/dllm
  pytest /mnt/lustrenew/mllm_aligned/dongzhichen/dllm/scripts/tests/test_eval_base.py -v
"""

import pytest

from dllm.core.eval.base import BaseEvalHarness, BaseEvalConfig


class TestBuildConfig:
    def test_build_config_from_source(self):
        source = BaseEvalConfig(pretrained="my/model", device="cuda", batch_size=4)
        out = BaseEvalHarness._build_config(BaseEvalConfig, source, {})
        assert out.pretrained == "my/model"
        assert out.device == "cuda"
        assert out.batch_size == 4

    def test_kwargs_override_source(self):
        source = BaseEvalConfig(pretrained="a", device="cuda", batch_size=1)
        out = BaseEvalHarness._build_config(
            BaseEvalConfig, source, {"batch_size": 8, "device": "cpu"}
        )
        assert out.pretrained == "a"
        assert out.device == "cpu"
        assert out.batch_size == 8

    def test_partial_kwargs_fill_remaining_from_source(self):
        source = BaseEvalConfig(pretrained="base", device="cuda", batch_size=2)
        out = BaseEvalHarness._build_config(BaseEvalConfig, source, {"batch_size": 16})
        assert out.pretrained == "base"
        assert out.device == "cuda"
        assert out.batch_size == 16
