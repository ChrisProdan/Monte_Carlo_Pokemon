"""Draws simulated pack-opening trials from a SetDistribution.

Everything here is vectorized: a batch of trials is drawn with a handful of
numpy array operations rather than a Python loop per pack, which is what
makes half a million (or more) trials fast enough to run interactively.
"""

from __future__ import annotations

import numpy as np

from backend.constants import DEFAULT_SEED
from backend.distribution import SetDistribution


def make_rng(seed: int | None = DEFAULT_SEED) -> np.random.Generator:
    """Creates the random number generator used for all draws in this project.

    Centralizing RNG creation in one function (instead of calling
    np.random.default_rng() ad hoc wherever a draw happens) makes it
    trivial to toggle reproducibility: pass the default fixed seed to get
    the exact same simulated numbers on every run (useful for hand-checking
    results and for unit tests), or pass seed=None (or any other integer)
    to get an independent random run.
    """
    return np.random.default_rng(seed)


def draw_outcome_indices(dist: SetDistribution, shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    """Draws outcome indices of the given shape from dist's distribution.

    Formula / method: draw uniform(0, 1) random numbers and find, for each
    one, which "bucket" of the cumulative probability array it falls into
    via searchsorted. This is mathematically identical to the two-stage
    process of first picking a rarity tier (with probability equal to its
    published pull rate, including the WORTHLESS bucket as a pseudo-tier)
    and then picking a uniform card within that tier — because
    cumulative_probs was built by laying each tier's cards end-to-end with
    the tier's probability split evenly across them (see distribution.py).
    Landing a uniform draw in a given interval of the cumulative array is
    exactly equivalent to that two-stage draw, just flattened into a single
    array lookup that numpy can vectorize instead of looping per pack.

    side="right" means a draw of exactly 0.0 lands in the first outcome with
    positive probability rather than one index too early; this only matters
    at a measure-zero boundary but costs nothing to get right.
    """
    uniforms = rng.random(shape)
    return np.searchsorted(dist.cumulative_probs, uniforms, side="right")


def simulate_pack_profits(dist: SetDistribution, n_trials: int, rng: np.random.Generator) -> np.ndarray:
    """Simulates n_trials independent single-pack openings and returns their profits.

    Profit per pack = price of the drawn card (0.0 for the WORTHLESS
    outcome) minus the pack's cost. This is the y=1 case of
    simulate_y_pack_profits, kept as its own function because it's the
    smallest unit the build order's hand-verification step checks against
    (see tests/test_distribution.py).
    """
    indices = draw_outcome_indices(dist, (n_trials,), rng)
    return dist.payoffs[indices] - dist.pack_cost


def simulate_y_pack_profits(dist: SetDistribution, y: int, n_trials: int, rng: np.random.Generator) -> np.ndarray:
    """Simulates n_trials independent scenarios of opening y packs each.

    Independence assumption (stated explicitly per the project's design
    doc, not silently assumed): packs are treated as independent draws with
    replacement from the flat distribution, even across packs within the
    same physical box. This is a reasonable approximation across separate
    boxes, but within a single sealed box, pulls aren't strictly
    independent — you're sampling without replacement from a fixed
    population of, say, 36 packs with known contents, so pulling a Hyper
    Rare from pack 3 very slightly changes the odds for the remaining packs
    in that box. That correction is not modeled here; it's a deliberate
    simplification for tractability (exact without-replacement modeling
    would require knowing box contents pack-by-pack, which the database
    doesn't provide), and its effect shrinks as box size grows relative to
    the rarity of the tail events being measured.

    Implementation: draws all n_trials * y outcome indices in a single
    (n_trials, y) shaped call, so the reshape happens once and the whole
    batch is summed along axis=1 — this is the same vectorization idea as
    simulate_pack_profits, just extended to a 2D draw instead of looping
    over y packs per trial in Python.
    """
    indices = draw_outcome_indices(dist, (n_trials, y), rng)
    total_payoff_per_trial = dist.payoffs[indices].sum(axis=1)
    return total_payoff_per_trial - (y * dist.pack_cost)
