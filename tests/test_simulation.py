"""Automated versions of the build order's hand-verification steps: does
simulated output actually match hand-calculated expectations for a toy
distribution simple enough to compute those by hand.
"""

from __future__ import annotations

import numpy as np

from backend.risk_metrics import check_sqrt_scaling
from backend.simulation import make_rng, simulate_pack_profits, simulate_y_pack_profits


def test_single_pack_mean_matches_hand_calculated_ev(toy_distribution):
    """toy_distribution: EV = 0.80*0 + 0.15*10 + 0.05*100 - 5 = 1.5 (see
    conftest.py). With enough trials, the simulated mean should land within
    Monte Carlo noise of that hand-calculated value."""
    rng = make_rng(seed=123)
    profits = simulate_pack_profits(toy_distribution, n_trials=500_000, rng=rng)

    hand_calculated_ev = 0.80 * 0 + 0.15 * 10 + 0.05 * 100 - 5.0
    assert hand_calculated_ev == 1.5

    # Standard error of the mean shrinks like 1/sqrt(n); 500k trials should
    # easily land within +/- 0.05 of the true EV for a distribution this
    # small (max single-outcome payoff is $100), so this is a generous but
    # still meaningful tolerance.
    assert abs(profits.mean() - hand_calculated_ev) < 0.05


def test_y_pack_sd_scales_with_sqrt_y(toy_distribution):
    """Independence assumption check: SD(Pi_y) should be close to
    sqrt(y) * SD(Pi_1) (see risk_metrics.check_sqrt_scaling)."""
    rng = make_rng(seed=7)
    profits_y1 = simulate_pack_profits(toy_distribution, n_trials=500_000, rng=rng)
    sd_y1 = profits_y1.std()

    for y in (5, 10, 100):
        profits_y = simulate_y_pack_profits(toy_distribution, y=y, n_trials=200_000, rng=rng)
        sd_y = profits_y.std()
        assert check_sqrt_scaling(sd_y1, sd_y, y, tolerance=0.05) is True
