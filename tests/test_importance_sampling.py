"""Tests for Esscher-tilted importance sampling: theta-solving, weighting,
ESS, and the plain-MC-vs-importance-sampling cross-check the project's
design calls for.

Uses the rare_win_distribution fixture (99.9% worthless, 0.1% a $1000 "Big
Win", $1 pack cost, EV = $0 exactly by construction — see conftest.py).
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.importance_sampling import (
    build_esscher_tilted_distribution,
    effective_sample_size,
    estimate_profit_threshold,
    estimate_upper_tail_outcome,
    simulate_y_pack_is_trials,
    solve_tilt_theta,
    weighted_conditional_expectation,
    weighted_tail_probability,
)
from backend.simulation import make_rng, simulate_pack_profits, simulate_y_pack_profits


def test_solve_tilt_theta_hits_the_target_mean_payoff(rare_win_distribution):
    """Solving for a target reachable WITHOUT exceeding
    MAX_SINGLE_OUTCOME_TILT_PROB should produce a theta whose tilted mean
    payoff matches the target closely. For this 2-outcome fixture, matching
    mean=X requires q(Big Win) = X/1000, so target=200 needs only 20%
    concentration -- comfortably under the 30% cap."""
    target = 200.0
    theta = solve_tilt_theta(rare_win_distribution, target_mean_payoff=target)

    tilted = build_esscher_tilted_distribution(rare_win_distribution, theta)
    # Recover the tilted mean payoff directly from q to check convergence.
    achieved_mean = np.sum(np.diff(tilted.q_cumulative_probs, prepend=0.0) * rare_win_distribution.payoffs)
    assert achieved_mean == pytest.approx(target, abs=0.01)


def test_solve_tilt_theta_caps_a_moderate_but_still_too_demanding_target(rare_win_distribution):
    """target=500 would need 50% concentration on Big Win (mean = 0.5*1000)
    to match exactly -- above the 30% cap -- so the achieved mean should
    stop at the capped value (0.3*1000 = 300), not reach 500."""
    theta = solve_tilt_theta(rare_win_distribution, target_mean_payoff=500.0)
    tilted = build_esscher_tilted_distribution(rare_win_distribution, theta)
    achieved_mean = np.sum(np.diff(tilted.q_cumulative_probs, prepend=0.0) * rare_win_distribution.payoffs)
    assert achieved_mean == pytest.approx(300.0, abs=0.01)


def test_solve_tilt_theta_degrades_gracefully_for_a_target_below_the_natural_mean(rare_win_distribution):
    """A target at or below the natural mean means the located threshold
    wasn't actually demanding (a real, expected case from sampling noise in
    a modest quantile estimate -- see solve_tilt_theta's docstring), not an
    error condition: theta=0 exactly reproduces the untilted distribution,
    which is the mathematically correct response."""
    natural_mean = float(np.dot(rare_win_distribution.probs, rare_win_distribution.payoffs))
    theta = solve_tilt_theta(rare_win_distribution, target_mean_payoff=natural_mean - 0.5)
    assert theta == 0.0


def test_solve_tilt_theta_caps_concentration_for_a_near_max_target(rare_win_distribution):
    """A target very close to the single highest payoff ($1000) can only be
    matched exactly by a degenerate tilt that puts ~100% probability on
    that one outcome -- MAX_SINGLE_OUTCOME_TILT_PROB should stop the search
    well short of that, accepting an imperfect mean match in exchange for a
    tilt that still produces both hit and miss samples in a finite batch
    (see solve_tilt_theta's docstring for why 100% concentration silently
    breaks the self-normalized estimator). The search should terminate
    (not loop forever) either way."""
    theta = solve_tilt_theta(rare_win_distribution, target_mean_payoff=999.999)
    tilted = build_esscher_tilted_distribution(rare_win_distribution, theta)
    q_probs = np.diff(tilted.q_cumulative_probs, prepend=0.0)
    assert q_probs[1] == pytest.approx(0.3, abs=0.01)  # capped at MAX_SINGLE_OUTCOME_TILT_PROB, not driven to ~1.0


def test_build_esscher_tilted_distribution_keeps_every_outcome_positive(rare_win_distribution):
    """The hard requirement on any valid proposal distribution -- q(x) > 0
    wherever p(x) > 0 -- should hold automatically by construction."""
    tilted = build_esscher_tilted_distribution(rare_win_distribution, theta=0.01)
    q_probs = np.diff(tilted.q_cumulative_probs, prepend=0.0)
    assert np.all(q_probs > 0)
    assert tilted.q_cumulative_probs[-1] == 1.0


def test_effective_sample_size_is_n_trials_when_weights_are_equal():
    weights = np.full(1000, 3.7)
    assert effective_sample_size(weights) == pytest.approx(1000.0)


def test_effective_sample_size_collapses_when_one_weight_dominates():
    weights = np.array([1000.0] + [0.001] * 999)
    ess = effective_sample_size(weights)
    assert 1.0 <= ess < 2.0


def test_weighted_tail_probability_matches_plain_mc_on_a_cheap_case(rare_win_distribution):
    """Cross-check required by the project's design: on a case cheap enough
    to brute-force directly, the importance-sampled estimate of
    P(profit >= threshold) should agree with a plain Monte Carlo estimate."""
    rng_plain = make_rng(seed=1)
    plain_profits = simulate_pack_profits(rare_win_distribution, n_trials=2_000_000, rng=rng_plain)
    threshold = 500.0
    plain_estimate = np.mean(plain_profits >= threshold)

    theta = solve_tilt_theta(rare_win_distribution, target_mean_payoff=threshold + rare_win_distribution.pack_cost)
    tilted = build_esscher_tilted_distribution(rare_win_distribution, theta)
    rng_is = make_rng(seed=2)
    profits, weights = simulate_y_pack_is_trials(tilted, y=1, n_trials=20_000, rng=rng_is)
    is_estimate = weighted_tail_probability(profits, weights, threshold)

    # The true probability is exactly 0.001 by construction of the fixture.
    assert is_estimate == pytest.approx(0.001, abs=0.0005)
    assert plain_estimate == pytest.approx(0.001, abs=0.0005)


def test_weighted_conditional_expectation_matches_the_known_big_win_profit(rare_win_distribution):
    """Every trial that reaches the threshold in this fixture is the exact
    same outcome (there's only one card above it), so the expected profit
    conditional on the tail should be exactly $1000 payoff - $1 cost = $999."""
    threshold = 500.0
    theta = solve_tilt_theta(rare_win_distribution, target_mean_payoff=threshold + rare_win_distribution.pack_cost)
    tilted = build_esscher_tilted_distribution(rare_win_distribution, theta)
    rng = make_rng(seed=3)
    profits, weights = simulate_y_pack_is_trials(tilted, y=1, n_trials=20_000, rng=rng)

    conditional_ev = weighted_conditional_expectation(profits, weights, threshold=threshold)
    assert conditional_ev == pytest.approx(999.0)


def test_estimate_profit_threshold_reuses_an_existing_profits_array(rare_win_distribution):
    """estimate_profit_threshold takes a profits array directly -- no RNG,
    no simulation of its own -- and returns a plausible quantile."""
    rng = make_rng(seed=4)
    profits = simulate_pack_profits(rare_win_distribution, n_trials=500_000, rng=rng)
    threshold = estimate_profit_threshold(profits, p_target=0.001)
    assert -1.0 <= threshold <= 999.0


def test_ess_stays_healthy_across_small_and_large_y(rare_win_distribution):
    """The whole point of Esscher tilting over the earlier subset-boosting
    approach: ESS should remain reasonable even at large y, where the
    old approach degenerated to ESS ~ 1."""
    rng = make_rng(seed=5)
    for y in (1, 10, 100, 1000):
        regular_profits = simulate_y_pack_profits(rare_win_distribution, y=y, n_trials=50_000, rng=rng)
        result = estimate_upper_tail_outcome(rare_win_distribution, y=y, rng=rng, regular_profits=regular_profits, n_is_trials=20_000)
        # A healthy tilt should keep ESS well above a handful of samples --
        # this is the regression check for weight degeneracy at large y.
        assert result.effective_sample_size > 100, f"ESS collapsed at y={y}: {result.effective_sample_size}"
