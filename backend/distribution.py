"""Builds the flat, per-outcome probability distribution for one set.

This is the heart of the "simulate at the individual-card level, not the
rarity-tier level" requirement: if we instead simulated "which rarity was
pulled" and then plugged in that rarity's *average* card price, the
resulting profit distribution would have the correct mean but a collapsed,
artificial shape — a rarity slot that's really a $5 card or a $400 card at
equal odds would get flattened into a single $202.50 outcome, erasing
exactly the tail event (the $400 pull) that VaR and CVaR exist to
characterize. Building one flat array over every individual card (plus a
$0 "worthless" catch-all — see below) and drawing directly from it avoids
that collapse entirely.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import psycopg

from backend.constants import RARITY_RATE_COLUMNS
from backend.db import get_cards_for_rarity, get_pack_cost, get_set_rates

# Floating point tolerance used when checking that probabilities don't sum
# to more than 1.0. Rate columns are stored as percentages with two decimal
# places, so tiny rounding error (e.g. 99.999999999% instead of 100%) is
# expected and shouldn't trip an assertion meant to catch real data errors.
_PROBABILITY_EPSILON = 1e-9

# Sentinel outcome name representing "pulled a card with no modeled resale
# value" (Common, Uncommon, plain Rare, Code Card, etc. — see the module-
# level modeling note in build_set_distribution for why these are folded
# into one bucket instead of simulated individually).
WORTHLESS_OUTCOME = "WORTHLESS"


@dataclass(frozen=True)
class SetDistribution:
    """A ready-to-sample-from probability distribution over one set's pack outcomes.

    `outcomes[i]` / `payoffs[i]` are the card name and price for outcome i
    (index 0 is always the WORTHLESS_OUTCOME sentinel with payoff 0.0).
    `cumulative_probs[i]` is P(outcome index <= i) — a monotonically
    non-decreasing array ending at exactly 1.0, built once per set so that
    every simulated trial can be drawn with a single numpy.searchsorted call
    instead of looping in Python (see simulation.py).
    """

    setname: str
    groupid: int
    outcomes: list[str]
    payoffs: np.ndarray
    cumulative_probs: np.ndarray
    pack_cost: float


def build_set_distribution(conn: psycopg.Connection, setname: str) -> SetDistribution:
    """Constructs the flat per-outcome distribution for one set.

    Modeling decision (confirmed with the user, not inferred from the doc):
    only the 9 rarities named in RARITY_RATE_COLUMNS are treated as having
    resale value. Every other physical card a pack could contain (Common,
    Uncommon, plain "Rare", Code Card, energy, etc.) is worth a few cents to
    a few tens of cents and — more importantly — the database has no data
    on how many of them appear per pack, only pull rates for the "hit slot"
    rarities above. So rather than inventing a slot-count assumption the
    data can't support, every bit of probability mass not accounted for by
    the 9 named rarities is folded into one WORTHLESS_OUTCOME outcome worth
    $0. This is a deliberate, stated simplification: it slightly understates
    real-world pack value (a handful of cents per pack), which is immaterial
    next to the dollar-scale VaR/CVaR figures this project reports.

    Steps (mirrors the two-stage "pick a rarity, then a uniform card within
    it" process from the project's design doc, just flattened into one
    array so it vectorizes — see simulation.py for the draw itself):
      1. For each of the 9 rate columns that is non-null for this set,
         convert its percentage to a probability and split it uniformly
         across that rarity's cards (no finer per-card pull data exists).
      2. Whatever probability is left over becomes the WORTHLESS_OUTCOME's
         probability.
      3. Lay every outcome's probability end-to-end into a cumulative array.
    """
    rates = get_set_rates(conn, setname)
    groupid = rates["groupid"]

    outcomes: list[str] = [WORTHLESS_OUTCOME]
    payoffs: list[float] = [0.0]
    probs: list[float] = [0.0]  # placeholder for the worthless bucket; filled in last

    for column, rarity_label in RARITY_RATE_COLUMNS.items():
        rate = rates[column]
        if rate is None:
            # NULL means this rarity has no cards in this set at all —
            # confirmed with the user — so it contributes nothing, not a
            # zero-probability entry.
            continue

        # groupdata stores a PERCENTAGE (e.g. 13.76 meaning 13.76% of packs
        # contain this rarity in the hit slot), so dividing by 100 converts
        # it to a probability in [0, 1].
        tier_probability = float(rate) / 100.0

        cards = get_cards_for_rarity(conn, groupid, rarity_label)
        if not cards:
            # A rate of exactly 0 (not NULL) with zero matching cards is a
            # real, verified case in this data (e.g. a set's Hyper Rare rate
            # is listed as 0 and it indeed prints no Hyper Rares) — nothing
            # to add to the distribution for this tier.
            continue

        # Within a rarity tier, every card is equally likely — there's no
        # finer-grained per-card pull data, and the doc's own simplification
        # (section 3.1) is exactly this equal-odds-within-tier assumption.
        per_card_probability = tier_probability / len(cards)
        for cardname, price in cards:
            outcomes.append(cardname)
            payoffs.append(price)
            probs.append(per_card_probability)

    assigned_probability = sum(probs)  # includes the placeholder 0.0 at index 0
    worthless_probability = 1.0 - assigned_probability

    # If this ever goes meaningfully negative, the rate columns for this set
    # sum to more than 100%, which would mean the data itself is
    # inconsistent (e.g. two "hit slots" that were double counted) — that's
    # a data problem worth failing loudly on rather than silently clamping.
    assert worthless_probability >= -_PROBABILITY_EPSILON, (
        f"Rate columns for {setname!r} sum to more than 100% "
        f"(assigned probability = {assigned_probability:.6f}); check groupdata for this set."
    )
    probs[0] = max(worthless_probability, 0.0)

    payoffs_arr = np.array(payoffs, dtype=np.float64)
    cumulative_probs = np.cumsum(np.array(probs, dtype=np.float64))
    # Force the last entry to exactly 1.0 to eliminate floating-point drift
    # from the many small additions above — without this, a uniform draw of
    # exactly 1.0 (astronomically unlikely but not impossible) could fall
    # just past the end of the array and make searchsorted return an
    # out-of-bounds index.
    cumulative_probs[-1] = 1.0

    pack_cost = get_pack_cost(conn, groupid)

    return SetDistribution(
        setname=setname,
        groupid=groupid,
        outcomes=outcomes,
        payoffs=payoffs_arr,
        cumulative_probs=cumulative_probs,
        pack_cost=pack_cost,
    )
