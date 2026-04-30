"""
Unit tests for dllm alpha and kappa schedulers.

Run from repo root:
  source ~/.zshrc && conda activate ~/miniconda3/envs/dllm
  pytest /mnt/lustrenew/mllm_aligned/dongzhichen/dllm/scripts/tests/test_schedulers.py -v
"""

import pytest
import torch

from dllm.core.schedulers import (
    BaseAlphaScheduler,
    CosineAlphaScheduler,
    LinearAlphaScheduler,
    BaseKappaScheduler,
    CosineKappaScheduler,
    CubicKappaScheduler,
    LinearKappaScheduler,
    get_alpha_scheduler_class,
    get_kappa_scheduler_class,
    make_alpha_scheduler,
    make_kappa_scheduler,
)


# ---------------------------------------------------------------------------
# Alpha schedulers
# ---------------------------------------------------------------------------


class TestLinearAlphaScheduler:
    def test_alpha_boundaries(self):
        sched = LinearAlphaScheduler()
        assert sched.alpha(0.0) == pytest.approx(1.0, abs=1e-6)
        assert sched.alpha(1.0) == pytest.approx(0.0, abs=1e-6)

    def test_alpha_midpoint(self):
        sched = LinearAlphaScheduler()
        assert sched.alpha(0.5) == pytest.approx(0.5, abs=1e-6)

    def test_alpha_tensor(self):
        sched = LinearAlphaScheduler()
        t = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        out = sched.alpha(t)
        expected = 1.0 - t
        assert torch.allclose(out, expected)

    def test_alpha_derivative_constant(self):
        sched = LinearAlphaScheduler()
        assert sched.alpha_derivative(0.5) == pytest.approx(-1.0, abs=1e-6)
        t = torch.tensor([0.1, 0.5, 0.9])
        assert torch.allclose(sched.alpha_derivative(t), torch.full_like(t, -1.0))

    def test_weight_positive(self):
        sched = LinearAlphaScheduler()
        w = sched.weight(0.5)
        assert w > 0
        assert torch.isfinite(torch.tensor(w))

    def test_reverse_mask_prob(self):
        sched = LinearAlphaScheduler()
        # (1 - α(s)) / (1 - α(t)), s < t; linear α(t)=1-t so α(0.25)=0.75, α(0.75)=0.25
        r = sched.reverse_mask_prob(s=0.25, t=0.75)
        assert 0 <= r <= 1
        assert r == pytest.approx(1.0 / 3.0, abs=1e-5)  # (1-0.75)/(1-0.25) = 0.25/0.75

    def test_out_of_range_raises(self):
        sched = LinearAlphaScheduler()
        with pytest.raises(ValueError, match="not in"):
            sched.alpha(-0.1)
        with pytest.raises(ValueError, match="not in"):
            sched.alpha(1.1)

    def test_reverse_mask_prob_requires_s_lt_t(self):
        sched = LinearAlphaScheduler()
        with pytest.raises(ValueError, match="s < t"):
            sched.reverse_mask_prob(s=0.6, t=0.4)


class TestCosineAlphaScheduler:
    def test_alpha_boundaries(self):
        sched = CosineAlphaScheduler()
        assert sched.alpha(0.0) == pytest.approx(1.0, abs=1e-6)
        assert sched.alpha(1.0) == pytest.approx(0.0, abs=1e-6)

    def test_alpha_midpoint(self):
        sched = CosineAlphaScheduler()
        # 1 - cos(π/2 * 0.5) = 1 - cos(π/4) ≈ 1 - 0.707 ≈ 0.293
        assert sched.alpha(0.5) == pytest.approx(1.0 - (2.0 ** 0.5) / 2.0, abs=1e-5)

    def test_alpha_tensor(self):
        sched = CosineAlphaScheduler()
        t = torch.tensor([0.0, 1.0])
        out = sched.alpha(t)
        assert torch.allclose(out, torch.tensor([1.0, 0.0]))

    def test_weight_finite(self):
        sched = CosineAlphaScheduler()
        for x in [0.1, 0.5, 0.9]:
            w = sched.weight(x)
            assert torch.isfinite(torch.tensor(w)) and w >= 0


class TestAlphaSchedulerRegistry:
    def test_get_class_by_name(self):
        assert get_alpha_scheduler_class("LinearAlphaScheduler") is LinearAlphaScheduler
        assert get_alpha_scheduler_class("linearalphascheduler") is LinearAlphaScheduler
        assert get_alpha_scheduler_class("CosineAlphaScheduler") is CosineAlphaScheduler

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown scheduler"):
            get_alpha_scheduler_class("UnknownScheduler")

    def test_make_scheduler(self):
        sched = make_alpha_scheduler("LinearAlphaScheduler")
        assert isinstance(sched, LinearAlphaScheduler)
        assert sched.alpha(0.5) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Kappa schedulers
# ---------------------------------------------------------------------------


class TestLinearKappaScheduler:
    def test_kappa_boundaries(self):
        sched = LinearKappaScheduler()
        assert sched.kappa(0.0) == pytest.approx(0.0, abs=1e-6)
        assert sched.kappa(1.0) == pytest.approx(1.0, abs=1e-6)

    def test_kappa_midpoint(self):
        sched = LinearKappaScheduler()
        assert sched.kappa(0.5) == pytest.approx(0.5, abs=1e-6)

    def test_kappa_tensor(self):
        sched = LinearKappaScheduler()
        t = torch.tensor([0.0, 0.5, 1.0])
        assert torch.allclose(sched.kappa(t), t)

    def test_weight_finite(self):
        sched = LinearKappaScheduler()
        w = sched.weight(0.5)
        assert torch.isfinite(torch.tensor(w))


class TestCubicKappaScheduler:
    def test_default_ab(self):
        sched = CubicKappaScheduler()
        assert sched.kappa(0.0) == pytest.approx(0.0, abs=1e-6)
        assert sched.kappa(1.0) == pytest.approx(1.0, abs=1e-6)

    def test_custom_ab(self):
        sched = CubicKappaScheduler(a=0.5, b=0.5)
        k0 = sched.kappa(0.0)
        k1 = sched.kappa(1.0)
        assert k0 == pytest.approx(0.0, abs=1e-6)
        assert k1 == pytest.approx(1.0, abs=1e-6)


class TestCosineKappaScheduler:
    def test_kappa_boundaries(self):
        sched = CosineKappaScheduler()
        assert sched.kappa(0.0) == pytest.approx(0.0, abs=1e-6)
        assert sched.kappa(1.0) == pytest.approx(1.0, abs=1e-6)

    def test_kappa_midpoint(self):
        sched = CosineKappaScheduler()
        # 1 - cos(π/2 * 0.5) = 1 - cos(π/4)
        expected = 1.0 - (2.0 ** 0.5) / 2.0
        assert sched.kappa(0.5) == pytest.approx(expected, abs=1e-5)


class TestKappaSchedulerRegistry:
    def test_get_class_by_name(self):
        assert get_kappa_scheduler_class("LinearKappaScheduler") is LinearKappaScheduler
        assert get_kappa_scheduler_class("linearkappascheduler") is LinearKappaScheduler
        assert get_kappa_scheduler_class("CosineKappaScheduler") is CosineKappaScheduler

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown scheduler"):
            get_kappa_scheduler_class("UnknownKappa")

    def test_make_scheduler(self):
        sched = make_kappa_scheduler("LinearKappaScheduler")
        assert isinstance(sched, LinearKappaScheduler)
        assert sched.kappa(0.5) == pytest.approx(0.5)
