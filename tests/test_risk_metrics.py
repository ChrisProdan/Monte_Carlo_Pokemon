"""Tests for VaR/CVaR/breakeven-probability calculations.

These use small, fully hand-computable synthetic profit arrays specifically
to catch off-by-one errors in the k = ceil((1-alpha)*M) tail-boundary index
— exactly the kind of edge case that's easy to get subtly wrong and hard to
notice from real simulation output alone.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.risk_metrics import check_sqrt_scaling, compute_risk_metrics


def test_var_and_cvar_on_hand_computable_array():
    """20 trials, evenly spaced: -100, -90, ..., -10, 0, 10, ..., 90 (20
    values total, ascending already). At alpha=0.95: k = ceil(0.05 * 20) = 1,
    so VaR_95 is based on the single worst trial (-100) and CVaR_95 is its
    average (also -100). At alpha=0.90: k = ceil(0.10 * 20) = 2, so the tail
    is the worst two trials (-100, -90); VaR_90 = 100, CVaR_90 =
    mean(-100, -90) = -95 -> 95.
    """
    profits = np.array([-100 + 10 * i for i in range(20)], dtype=float)
    # Sanity-check the hand-picked array before trusting the test itself.
    assert profits[0] == -100
    assert profits[-1] == 90

    metrics = compute_risk_metrics(profits, y=1, alphas=(0.95, 0.90))

    m95 = metrics.by_alpha[0.95]
    assert m95.var == pytest.approx(100.0)
    assert m95.var_binding is True
    assert m95.cvar == pytest.approx(100.0)

    m90 = metrics.by_alpha[0.90]
    assert m90.var == pytest.approx(90.0)
    assert m90.cvar == pytest.approx(95.0)

    assert metrics.mean == pytest.approx(np.mean(profits))
    assert metrics.breakeven_prob == pytest.approx(np.mean(profits > 0))


def test_var_is_non_binding_when_tail_boundary_is_positive():
    """If even the worst (1 - alpha) fraction of trials was a net gain,
    VaR must be reported as 0 and flagged non-binding, not as a fabricated
    positive number derived from a profit."""
    profits = np.array([5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0])

    metrics = compute_risk_metrics(profits, y=1, alphas=(0.95,))
    m95 = metrics.by_alpha[0.95]

    assert m95.var_binding is False
    assert m95.var == 0.0
    # CVaR is still a genuine (small, or even positive-magnitude-of-gain)
    # tail average and should still be reported.
    assert m95.cvar == pytest.approx(-5.0)  # worst trial was a +5 gain -> "loss" of -5


def test_breakeven_probability_is_fraction_of_positive_trials():
    profits = np.array([-1.0, -1.0, 1.0, 1.0, 1.0, 1.0])  # 4 of 6 are profitable
    metrics = compute_risk_metrics(profits, y=1, alphas=(0.95,))
    assert metrics.breakeven_prob == pytest.approx(4 / 6)


def test_check_sqrt_scaling_accepts_matching_relationship():
    sd_y1 = 10.0
    y = 100
    sd_y = sd_y1 * (y ** 0.5)  # exact match
    assert check_sqrt_scaling(sd_y1, sd_y, y) is True


def test_check_sqrt_scaling_flags_large_deviation():
    sd_y1 = 10.0
    y = 100
    sd_y = sd_y1 * (y ** 0.5) * 3  # 3x too large -> well outside tolerance
    with pytest.warns(UserWarning):
        result = check_sqrt_scaling(sd_y1, sd_y, y)
    assert result is False
