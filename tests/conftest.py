"""Shared pytest fixtures.

Distribution/simulation/risk-metrics unit tests use a small synthetic
SetDistribution rather than a live database connection — that keeps them
fast, deterministic, and runnable without Postgres available (e.g. in CI).
Database-integration checks (the actual schema/query behavior) are covered
separately by the manual verification steps in the build order, since they
inherently require a live connection to the real data.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.distribution import SetDistribution, WORTHLESS_OUTCOME


@pytest.fixture
def toy_distribution() -> SetDistribution:
    """A hand-constructed, tiny distribution with numbers simple enough to
    verify every downstream calculation (EV, VaR, CVaR) by hand.

    Outcomes: 80% worthless ($0), 15% a "Common Hit" card worth $10, 5% a
    "Rare Hit" card worth $100. Pack cost is fixed at $5.
    Expected value per pack = 0.80*0 + 0.15*10 + 0.05*100 - 5 = 1.5 + 5 - 5 = 1.5.
    """
    outcomes = [WORTHLESS_OUTCOME, "Common Hit", "Rare Hit"]
    payoffs = np.array([0.0, 10.0, 100.0])
    probs = np.array([0.80, 0.15, 0.05])
    cumulative_probs = np.cumsum(probs)
    cumulative_probs[-1] = 1.0

    return SetDistribution(
        setname="Toy Set",
        groupid=1,
        outcomes=outcomes,
        payoffs=payoffs,
        cumulative_probs=cumulative_probs,
        pack_cost=5.0,
    )
