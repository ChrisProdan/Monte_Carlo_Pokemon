"""Computes the risk statistics reported for a batch of simulated profit trials.

This module is the "translate simulation output into the language of quant
risk analytics" layer: mean/SD are plain descriptive statistics, but VaR,
CVaR, and breakeven probability are the specific metrics this project is
built to demonstrate, so every formula below carries a comment stating it
in words, not just the code implementing it.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AlphaMetrics:
    """VaR and CVaR at one confidence level, both reported as positive loss magnitudes."""

    alpha: float
    var: float
    var_binding: bool
    cvar: float


@dataclass(frozen=True)
class RiskMetrics:
    """Everything the frontend displays for one (set, y) scenario."""

    y: int
    n_trials: int
    mean: float
    sd: float
    breakeven_prob: float
    by_alpha: dict[float, AlphaMetrics]


def compute_risk_metrics(profits: np.ndarray, y: int, alphas: tuple[float, ...]) -> RiskMetrics:
    """Computes mean, SD, VaR/CVaR per alpha, and breakeven probability for one batch of trials.

    `profits` is an array of M simulated total-profit outcomes for opening y
    packs (Π_y in the design doc's notation). Sorting once up front and
    reusing that sorted array for every alpha avoids re-sorting M values
    once per confidence level.
    """
    n_trials = len(profits)
    mean = float(np.mean(profits))
    sd = float(np.std(profits, ddof=0))

    # Breakeven probability = P(Π_y > 0): the fraction of simulated
    # scenarios where the packs' resale value exceeded what they cost.
    breakeven_prob = float(np.mean(profits > 0))

    sorted_profits = np.sort(profits)  # ascending: Π_(1) <= Π_(2) <= ... <= Π_(M)

    by_alpha: dict[float, AlphaMetrics] = {}
    for alpha in alphas:
        by_alpha[alpha] = _compute_alpha_metrics(sorted_profits, alpha)

    return RiskMetrics(
        y=y,
        n_trials=n_trials,
        mean=mean,
        sd=sd,
        breakeven_prob=breakeven_prob,
        by_alpha=by_alpha,
    )


def _compute_alpha_metrics(sorted_profits: np.ndarray, alpha: float) -> AlphaMetrics:
    """Computes VaR_alpha and CVaR_alpha from an already-ascending-sorted profits array.

    VaR_alpha: k = ceil((1 - alpha) * M) is the number of trials in the
    worst (1 - alpha) fraction of outcomes (e.g. alpha=0.95 -> the worst 5%
    of trials). VaR_alpha is defined as Π_(k), the profit value at that
    percentile boundary. It's reported as a positive LOSS magnitude, so we
    negate it — a negative Π_(k) (a real loss) becomes a positive VaR
    number. If Π_(k) is itself positive, that means even the boundary of
    the worst (1 - alpha) fraction of outcomes was a net gain — there is no
    binding loss at this confidence level, so we report VaR as 0 and flag
    var_binding=False rather than reporting a fabricated positive number
    that would misrepresent the risk.

    CVaR_alpha (Expected Shortfall): the average of Π_(1) through Π_(k) —
    i.e. the average outcome across the entire worst (1 - alpha) tail, not
    just its boundary. This is always reported (even when VaR is
    non-binding) because it's a genuine tail average and answers a
    different question than VaR: VaR says where the bad outcomes start,
    CVaR says how bad they get on average once you're in that tail.
    """
    m = len(sorted_profits)
    # k = ceil((1 - alpha) * M): the count of trials in the worst
    # (1 - alpha) fraction. Computed with a tiny epsilon subtracted before
    # ceiling because (1 - alpha) often isn't exactly representable in
    # binary floating point — e.g. (1 - 0.95) * 20 evaluates to
    # 1.0000000000000009, not exactly 1.0, which would make ceil() round up
    # to 2 instead of the mathematically intended 1. Subtracting a tiny
    # epsilon first absorbs that representation error without meaningfully
    # changing k for any case that isn't sitting exactly on an integer
    # boundary to begin with.
    k = math.ceil((1 - alpha) * m - 1e-9)
    k = max(k, 1)  # guard against alpha so close to 1 that (1-alpha)*M rounds to 0

    tail = sorted_profits[:k]
    boundary_profit = sorted_profits[k - 1]  # Π_(k), 0-indexed as k-1

    if boundary_profit > 0:
        var = 0.0
        var_binding = False
    else:
        var = -float(boundary_profit)
        var_binding = True

    cvar = -float(np.mean(tail))

    return AlphaMetrics(alpha=alpha, var=var, var_binding=var_binding, cvar=cvar)


def check_sqrt_scaling(sd_y1: float, sd_y: float, y: int, tolerance: float = 0.10) -> bool:
    """Sanity-checks that SD(Pi_y) scales roughly like sqrt(y) * SD(Pi_1).

    Formula in words: under the independence assumption used throughout
    this project (packs are iid draws — see simulation.py), the variance of
    a sum of y independent identically-distributed draws is y times the
    variance of a single draw: Var(sum) = y * Var(X). Taking square roots,
    SD(sum) = sqrt(y) * SD(X). This function is a REGRESSION CHECK, not a
    correction: if simulated results at a given y drift noticeably from
    this relationship, that's a signal to double check the sampling code
    (or that n_trials is large enough for the simulated SD to have
    converged) before trusting the numbers — it should never silently
    adjust the reported SD to force agreement.

    Returns True if within tolerance, False (with a warning) otherwise.
    """
    expected_sd_y = sd_y1 * math.sqrt(y)
    if expected_sd_y == 0:
        # A zero-variance single-pack distribution (e.g. every card in
        # scope somehow priced identically) would make relative drift
        # undefined; nothing to check in that degenerate case.
        return True

    relative_drift = abs(sd_y - expected_sd_y) / expected_sd_y
    # Cast explicitly to a Python bool: sd_y/sd_y1 are typically numpy
    # float64 values (from ndarray.std()), so the comparison above produces
    # numpy.bool_, not bool. That satisfies `if not within_tolerance:` fine,
    # but silently breaks identity checks (`is True`) and strict type
    # expectations for callers relying on the -> bool return annotation.
    within_tolerance = bool(relative_drift <= tolerance)

    if not within_tolerance:
        warnings.warn(
            f"SD(Pi_{y}) = {sd_y:.4f} deviates from the sqrt(y) prediction "
            f"({expected_sd_y:.4f}) by {relative_drift:.1%}, which exceeds the "
            f"{tolerance:.0%} tolerance. This can mean n_trials is too small for "
            "the simulated SD to have converged, or it can indicate a bug in the "
            "sampling code — check both before trusting risk metrics at this y.",
            stacklevel=2,
        )

    return within_tolerance
