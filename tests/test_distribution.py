"""Tests for the flat per-outcome distribution builder.

build_set_distribution() itself needs a live DB connection (it calls
db.get_set_rates / get_cards_for_rarity / get_pack_cost), so these tests
exercise it against a monkeypatched database layer with small, hand-checkable
numbers instead. The live-data version of this same check (does the real
database produce a valid distribution for every in-scope set) is covered by
the manual Stage 1/2 verification steps in the build order.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend import distribution as distribution_module
from backend.constants import RARITY_RATE_COLUMNS


def test_worthless_bucket_absorbs_unassigned_probability(monkeypatch):
    """If only one rarity column is populated, the worthless bucket should
    take exactly the remaining probability mass (1 - that rarity's rate)."""
    fake_rates = {"groupid": 1}
    for column in RARITY_RATE_COLUMNS:
        fake_rates[column] = None
    fake_rates["dr_rates"] = 20.0  # 20% of packs contain a Double Rare

    def fake_get_set_rates(conn, setname):
        return fake_rates

    def fake_get_cards_for_rarity(conn, groupid, rarity):
        if rarity == "Double Rare":
            return [("Card A", 2.0), ("Card B", 6.0)]
        return []

    def fake_get_pack_cost(conn, groupid):
        return 4.0

    monkeypatch.setattr(distribution_module, "get_set_rates", fake_get_set_rates)
    monkeypatch.setattr(distribution_module, "get_cards_for_rarity", fake_get_cards_for_rarity)
    monkeypatch.setattr(distribution_module, "get_pack_cost", fake_get_pack_cost)

    dist = distribution_module.build_set_distribution(conn=None, setname="Fake Set")

    # Worthless bucket (index 0) should carry 1 - 0.20 = 0.80 probability.
    assert dist.cumulative_probs[0] == pytest.approx(0.80)
    # The two Double Rare cards should split the remaining 20% evenly.
    assert dist.outcomes[1:] == ["Card A", "Card B"]
    assert dist.payoffs[1] == pytest.approx(2.0)
    assert dist.payoffs[2] == pytest.approx(6.0)
    # Cumulative array must end at exactly 1.0 (floating-point drift guard).
    assert dist.cumulative_probs[-1] == 1.0
    # And must be non-decreasing throughout.
    assert np.all(np.diff(dist.cumulative_probs) >= 0)


def test_null_rate_column_excludes_rarity_entirely(monkeypatch):
    """A NULL rate column means that rarity has no cards in the set — it
    must not appear in the built distribution even if get_cards_for_rarity
    would otherwise return rows for it."""
    fake_rates = {"groupid": 1}
    for column in RARITY_RATE_COLUMNS:
        fake_rates[column] = None  # every rarity absent from this set

    def fake_get_set_rates(conn, setname):
        return fake_rates

    def fake_get_cards_for_rarity(conn, groupid, rarity):
        raise AssertionError("Should not be called when the rate column is NULL")

    def fake_get_pack_cost(conn, groupid):
        return 3.0

    monkeypatch.setattr(distribution_module, "get_set_rates", fake_get_set_rates)
    monkeypatch.setattr(distribution_module, "get_cards_for_rarity", fake_get_cards_for_rarity)
    monkeypatch.setattr(distribution_module, "get_pack_cost", fake_get_pack_cost)

    dist = distribution_module.build_set_distribution(conn=None, setname="All Worthless Set")

    # Every pack is worthless: the entire probability mass is the WORTHLESS bucket.
    assert dist.outcomes == [distribution_module.WORTHLESS_OUTCOME]
    assert dist.cumulative_probs[-1] == 1.0
