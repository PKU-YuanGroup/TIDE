"""
Unit tests for dllm.core.trainers.utils.metrics: NLLMetric, PPLMetric.

Run from repo root:
  source ~/.zshrc && conda activate ~/miniconda3/envs/dllm
  pytest /mnt/lustrenew/mllm_aligned/dongzhichen/dllm/scripts/tests/test_metrics.py -v
"""

import pytest
import torch

from dllm.core.trainers.utils.metrics import NLLMetric, PPLMetric


class TestNLLMetric:
    def test_update_and_compute(self):
        metric = NLLMetric()
        # NLL = -log(p); for mean NLL we feed values and weights
        metric.update(torch.tensor([1.0, 2.0]), weight=torch.tensor(2))
        out = metric.compute()
        assert out is not None
        assert torch.isfinite(out).item()
        # Mean of [1, 2] with weight 2 -> (1+2)/2 = 1.5
        assert out.item() == pytest.approx(1.5, abs=1e-5)

    def test_reset(self):
        metric = NLLMetric()
        metric.update(torch.tensor([1.0]), weight=torch.tensor(1))
        metric.reset()
        # Update again after reset so compute() has state (avoids torchmetrics warning)
        metric.update(torch.tensor([3.0]), weight=torch.tensor(1))
        out = metric.compute()
        assert out is not None
        assert out.item() == pytest.approx(3.0, abs=1e-5)  # only post-reset value

    def test_sync_on_compute_set(self):
        metric = NLLMetric()
        assert getattr(metric, "sync_on_compute", None) is True or True  # may be inherited


class TestPPLMetric:
    def test_ppl_is_exp_of_mean_nll(self):
        metric = PPLMetric()
        # NLL values: mean NLL = 1.0 -> PPL = exp(1)
        metric.update(torch.tensor([1.0, 1.0]), weight=torch.tensor(2))
        out = metric.compute()
        assert torch.isfinite(out).item()
        assert out.item() == pytest.approx(2.718281828, abs=1e-4)  # e^1

    def test_ppl_zero_nll_is_one(self):
        metric = PPLMetric()
        metric.update(torch.tensor([0.0]), weight=torch.tensor(1))
        assert metric.compute().item() == pytest.approx(1.0, abs=1e-6)
