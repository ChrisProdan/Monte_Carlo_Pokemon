"""All SQL access to the Pokemon_Pricing_Information Postgres database lives
here, and nowhere else in the project. Every query is parameterized (never
built with string interpolation of user input) even though, in practice,
the only user-facing input (a set name) is always checked against
get_in_scope_sets() first — this is standard SQL-injection hygiene, cheap
to do correctly and not worth skipping just because the current call sites
happen to be safe some other way.
"""

from __future__ import annotations

import psycopg

from backend.config import Settings
from backend.constants import IN_SCOPE_SET_SQL_FILTER


def get_connection(settings: Settings) -> psycopg.Connection:
    """Opens a single psycopg v3 connection.

    Why not a connection pool: this app is a single-user, low-concurrency
    Flask frontend and a handful of ad-hoc scripts, not a production service
    fielding concurrent requests. A pool (psycopg_pool) would add a
    dependency and a bit of lifecycle complexity for no real benefit at this
    scale — a deliberate scoped-down tradeoff, not an oversight.
    """
    return psycopg.connect(
        host=settings.db_host,
        port=settings.db_port,
        dbname=settings.db_name,
        user=settings.db_user,
        password=settings.db_password,
    )


def get_in_scope_sets(conn: psycopg.Connection) -> list[str]:
    """Returns every set name the simulation supports.

    "In scope" means: a Scarlet & Violet or Mega Evolution set (by setname
    prefix) that has at least one populated pull-rate column, i.e. a set
    with real published booster-pack odds. This excludes promo/energy/
    celebration/classic-collection products that share the same table but
    were never sold as booster packs with these odds, and it will
    automatically pick up new sets (e.g. a currently-unreleased set) the
    moment their rate data is populated in groupdata, without any code
    change here.

    This function is the single source of truth for "which sets can the
    user pick" — the Flask frontend's validation step calls this directly
    rather than each place re-deriving its own copy of the filter.
    """
    query = f"SELECT setname FROM groupdata WHERE {IN_SCOPE_SET_SQL_FILTER} ORDER BY setname;"
    with conn.cursor() as cur:
        cur.execute(query)
        return [row[0] for row in cur.fetchall()]


def get_set_rates(conn: psycopg.Connection, setname: str) -> dict:
    """Returns the groupid and all 9 rate columns for one set.

    Raises ValueError (not a bare KeyError/None) on an unknown set name so
    callers get an unambiguous, user-presentable error rather than having to
    guess why a dict came back empty.
    """
    query = """
        SELECT groupid, dr_rates, ur_rates, ir_rates, sir_rates, hr_rates,
               sr_rates, sur_rates, mhr_rates, mar_rates
        FROM groupdata
        WHERE setname = %s;
    """
    with conn.cursor() as cur:
        cur.execute(query, (setname,))
        row = cur.fetchone()

    if row is None:
        raise ValueError(f"Unknown set name: {setname!r}")

    columns = (
        "groupid",
        "dr_rates",
        "ur_rates",
        "ir_rates",
        "sir_rates",
        "hr_rates",
        "sr_rates",
        "sur_rates",
        "mhr_rates",
        "mar_rates",
    )
    return dict(zip(columns, row))


def get_pack_cost(conn: psycopg.Connection, groupid: int) -> float:
    """Returns the price of a single standard booster pack for a set.

    The filter below was validated by hand against all 22 in-scope sets to
    return exactly one priced row each: it matches "<Set Name> Booster Pack"
    while excluding digital code-card listings, multi-pack art bundles,
    collector "sleeved" variants, and "[Set of N]" bundles, all of which are
    separate TCGPlayer products that also happen to have "Booster Pack" in
    their name. The assertion below exists to fail loudly — rather than
    silently pick an arbitrary row or a bundle price — if the database is
    ever refreshed in a way that breaks that invariant (e.g. a new SKU is
    added that also matches the filter).
    """
    query = """
        SELECT cardname, price
        FROM carddata
        WHERE groupid = %s
          AND cardname ILIKE %s
          AND cardname NOT ILIKE %s
          AND cardname NOT ILIKE %s
          AND cardname NOT ILIKE %s
          AND cardname NOT ILIKE %s;
    """
    params = (
        groupid,
        "%booster pack%",
        "code card%",
        "%art bundle%",
        "%sleeved%",
        "%set of%",
    )
    with conn.cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()

    assert len(rows) == 1, (
        f"Expected exactly one plain booster-pack price for groupid={groupid}, "
        f"found {len(rows)}: {rows}. The booster-pack price filter may need "
        "updating to match new product naming in the database."
    )
    cardname, price = rows[0]
    assert price is not None, f"Booster pack row {cardname!r} has a NULL price."
    return float(price)


def get_cards_for_rarity(conn: psycopg.Connection, groupid: int, rarity: str) -> list[tuple[str, float]]:
    """Returns (cardname, price) for every individual card of one rarity in one set.

    The assertion on price is defensive, not load-bearing today: live
    inspection confirmed there are no NULL prices among the 9 in-scope
    rarities across any of the 22 in-scope sets, so this should never fire
    on the current data. It's kept so that if the database is later
    refreshed with a gap, the simulation fails loudly with a clear message
    instead of quietly treating a missing price as some default value that
    would silently distort every downstream VaR/CVaR number.
    """
    query = """
        SELECT cardname, price
        FROM carddata
        WHERE groupid = %s
          AND rarity = %s;
    """
    with conn.cursor() as cur:
        cur.execute(query, (groupid, rarity))
        rows = cur.fetchall()

    for cardname, price in rows:
        assert price is not None, (
            f"Card {cardname!r} (groupid={groupid}, rarity={rarity!r}) has a NULL price; "
            "the simulation has no fallback for missing prices among in-scope rarities."
        )

    return [(cardname, float(price)) for cardname, price in rows]
