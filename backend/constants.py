"""Project-wide constants: the single source of truth for values that
multiple modules need to agree on.

Centralizing these avoids the classic bug where, say, db.py and
distribution.py each hardcode their own copy of the rarity-column mapping
and one gets updated without the other.
"""

# Maps each groupdata pull-rate column to the exact rarity string used in
# carddata.rarity. This mapping was verified by live inspection of the
# database: for every set where a given rate column is populated, the
# matching carddata rows for that rarity exist with counts and price ranges
# consistent with the column (e.g. Paldean Fates' sr_rates/sur_rates line up
# with its 'Shiny Rare'/'Shiny Ultra Rare' rows; Ascended Heroes' mhr_rates/
# mar_rates line up with 'Mega Hyper Rare'/'Mega Attack Rare').
#
# Modeling decision (confirmed with the user, not inferred): these 9
# rarities are the ONLY ones treated as having resale value in the
# simulation. Every other physical card in a pack (Common, Uncommon, plain
# "Rare", Code Card, energy, etc.) is folded into a single $0 "worthless"
# outcome rather than simulated individually — see distribution.py for how
# that bucket is built.
RARITY_RATE_COLUMNS: dict[str, str] = {
    "dr_rates": "Double Rare",
    "ur_rates": "Ultra Rare",
    "ir_rates": "Illustration Rare",
    "sir_rates": "Special Illustration Rare",
    "hr_rates": "Hyper Rare",
    "sr_rates": "Shiny Rare",
    "sur_rates": "Shiny Ultra Rare",
    "mhr_rates": "Mega Hyper Rare",
    "mar_rates": "Mega Attack Rare",
}

# The SQL WHERE-clause fragment defining "in scope" sets: Scarlet & Violet or
# Mega Evolution sets (by setname prefix) that have at least one non-null
# rate column, i.e. sets with real, published booster-pack odds rather than
# promo/energy/celebration products that were scraped into the same table
# but never had booster packs with these odds. Kept as one constant so db.py
# and any future script apply the exact same, already-validated filter
# (verified live to return exactly 22 sets).
#
# This is executed with no query parameters (see db.get_in_scope_sets), so
# the ILIKE wildcards below are written as a single literal '%' rather than
# the '%%' escaping psycopg would require if this string were ever
# interpolated into a call that also passes a params tuple.
IN_SCOPE_SET_SQL_FILTER = (
    "(setname ILIKE 'SV%' OR setname ILIKE 'ME%') AND ("
    + " OR ".join(f"{col} IS NOT NULL" for col in RARITY_RATE_COLUMNS)
    + ")"
)

# The pack-count scenarios the risk report is built around (doc section
# 4.1). An arbitrary user-supplied y is also supported by every function
# that takes a `y` parameter — these five are just the fixed set the
# frontend exposes as buttons.
Y_VALUES: tuple[int, ...] = (1, 5, 10, 100, 1000)

# Confidence levels for VaR/CVaR (doc section 4.2).
ALPHAS: tuple[float, ...] = (0.95, 0.99)

# Default number of Monte Carlo trials per scenario. The rarest cards in
# scope are on the order of 1-in-several-thousand, and a plain Monte Carlo
# estimate of a probability that size reaches roughly 10% relative error at
# around half a million trials (per the project's own importance-sampling
# rationale) — that's affordable but not instant. 50,000 is a deliberate
# smaller default, chosen so a y=1000 request (which draws
# n_trials * y = 50,000,000 uniforms) still returns from the Flask endpoint
# in a few seconds rather than tens of seconds. Raise this constant for
# final write-up figures where precision matters more than latency.
N_TRIALS_DEFAULT = 50_000

# Fixed default seed for the random number generator. Monte Carlo results
# are inherently random, but for a project meant to be explained line-by-line
# in an interview, being able to reproduce an exact number run-to-run (and
# in unit tests) matters more than fresh randomness on every call. Pass a
# different seed explicitly to get an independent run.
DEFAULT_SEED = 42
