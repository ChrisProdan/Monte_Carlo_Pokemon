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

# The pack-count scenarios the original design doc's risk report is built
# around (doc section 4.1). No longer used by the frontend directly -- the
# frontend now takes an arbitrary y from 1-1000 (see frontend/app.py's
# MIN_Y/MAX_Y), since every function that takes a `y` parameter already
# worked for any value, not just these five. Kept as a named reference for
# any offline reporting/notebook use that wants exactly this scenario set.
Y_VALUES: tuple[int, ...] = (1, 5, 10, 100, 1000)

# Confidence levels for VaR/CVaR (doc section 4.2).
ALPHAS: tuple[float, ...] = (0.95, 0.99)

# Default number of Monte Carlo trials per scenario. The rarest cards in
# scope are on the order of 1-in-several-thousand, and a plain Monte Carlo
# estimate of a probability that size reaches roughly 10% relative error at
# around half a million trials (per the project's own importance-sampling
# rationale) — that's affordable but not instant. 100,000 is a deliberate
# smaller default: a y=1000 request draws n_trials * y = 100,000,000 values
# in one batch, which measured at roughly 5-6 seconds and a few GB of
# transient memory on this machine — fast enough to stay responsive from
# the Flask endpoint. A larger default (e.g. 500,000) was tried and
# rejected: at y=1000 that's 500,000,000 draws, which peaks at roughly
# 12GB of transient memory (three float64-sized arrays of that length
# during the draw) — too close to this machine's 16GB of RAM to be safe
# alongside everything else normally running. Raise this constant for
# final write-up figures where precision matters more than latency, but be
# aware of that memory scaling if also raising y.
N_TRIALS_DEFAULT = 100_000

# Fixed default seed for the random number generator. Monte Carlo results
# are inherently random, but for a project meant to be explained line-by-line
# in an interview, being able to reproduce an exact number run-to-run (and
# in unit tests) matters more than fresh randomness on every call. Pass a
# different seed explicitly to get an independent run.
DEFAULT_SEED = 42

# --- Importance sampling tuning (backend/importance_sampling.py) ---
#
# The rare event importance sampling targets here is a LARGE PROFIT, not a
# loss — see importance_sampling.py's module docstring for why the loss
# tail used for VaR/CVaR in risk_metrics.py does NOT need importance
# sampling (it's the dense, common part of this bottom-heavy distribution).

# The tail probability that defines "how rare a slice of the upper tail are
# we studying." threshold(y) is defined as the profit value at the
# (1 - TAIL_PROBABILITY) quantile of Pi_y's distribution, rather than a
# fixed dollar multiple of cost — a fixed multiple would mean something
# very different at y=1 (a big win is a modest multiple of a cheap pack)
# than at y=1000 (SD(Pi_y) grows only like sqrt(y) while cost grows like y,
# so the distribution concentrates relative to its mean, and even matching
# a small multiple of cost back becomes far rarer than at y=1). Defining the
# threshold by a fixed RARITY instead of a fixed DOLLAR AMOUNT means it
# naturally lands on "a big multiple of cost" at small y and compresses
# toward (or below) breakeven at large y, tracking whatever is actually the
# extreme outcome at that y rather than a hardcoded number that means a
# different thing at every scale. 0.1% was chosen as rare enough to tell a
# genuine "big pull" story at y=1 while still being (barely) visible to the
# same batch of regular Monte Carlo samples risk_metrics.py already draws
# for VaR/CVaR/mean/SD (see estimate_profit_threshold in
# importance_sampling.py) — no separate "pilot" simulation is run just to
# locate it; that would be simulating the same thing twice.
TAIL_PROBABILITY = 0.001

# Number of importance-sampled trials to draw once a tilted distribution is
# built. Kept smaller than N_TRIALS_DEFAULT deliberately: importance
# sampling is specifically meant to get a precise answer for one narrow
# quantity (the upper-tail probability/conditional profit) from FEWER
# trials than plain Monte Carlo would need for the same precision, and the
# Flask endpoint already pays for one full N_TRIALS_DEFAULT-sized plain
# simulation per request (for the regular risk metrics) — running a second,
# equally large batch just for this would roughly double request latency
# for no real benefit.
N_IS_TRIALS_DEFAULT = 20_000

# --- Esscher (exponential) tilting ---
#
# The proposal distribution q is built by reweighting EVERY outcome
# smoothly by e^(theta * payoff), not by hand-picking a subset of "target"
# outcomes to boost (an earlier version of this code did that, and it broke
# down at large y — see importance_sampling.py's module docstring for the
# full story of why). theta is solved numerically per (set, y) so that the
# tilted distribution's mean payoff, summed over y packs, lands on the
# profit threshold — there's no fixed "how much to boost" constant to tune
# here the way TILT_TARGET_MASS used to be; the right tilt strength is
# derived fresh each time from the actual target. The two constants below
# only control the numerical root-finding procedure itself, not the model.

# Number of bisection iterations used to solve for theta. A fixed iteration
# count is used rather than stopping once the theta interval itself shrinks
# below some small epsilon, because the tilted mean payoff can be an
# extremely steep function of theta (a tiny theta can already make one
# high-payoff outcome's e^(theta*payoff) term astronomically dominant) --
# so a small remaining theta interval doesn't reliably mean a small
# remaining error in dollars. 100 iterations shrinks the bracket by a
# factor of 2^100 regardless of its starting width, far beyond
# double-precision resolution, and costs nothing extra since each iteration
# is one pass over a small, finite outcome array.
THETA_SOLVE_ITERATIONS = 100

# Upper bound on how far the bisection search will expand looking for a
# valid bracket on theta. Needed because the target mean payoff can, in an
# edge case, sit extremely close to (or at) the single highest-priced
# card's payoff -- the tilted mean approaches that value only in the limit
# as theta -> infinity, so a search cap prevents an unbounded loop; hitting
# the cap still produces a usable (very heavily concentrated) tilt, just
# not an exact match to an unreachable target.
THETA_SEARCH_MAX = 1_000_000.0

# Maximum probability the single HIGHEST-PAYOFF outcome may be assigned
# under the tilted distribution q, enforced during the theta search (not a
# cap on the max probability over ALL outcomes -- the naturally most
# probable outcome here is always WORTHLESS, often 60-99% probability even
# with no tilt at all, so a blanket "no outcome above 30%" cap would be
# violated by the untilted distribution itself and never allow any tilting
# to happen). Found necessary via live validation: a distribution whose
# MEAN equals its own MAXIMUM possible value can only be the degenerate one
# that puts ~100% probability on that single highest-payoff outcome, which
# happens whenever the target payoff sits at (or extremely close to) the
# single highest-priced card's own value -- a real case at y=1, where that
# card's own natural pull probability already exceeded the target tail
# rarity. With a finite trial batch, that extreme a concentration means
# essentially every trial draws the same outcome and the self-normalized
# weighted estimator never sees a single "miss" sample to correctly weigh
# against it -- a finite-sample failure even though the underlying math is
# still technically unbiased in the infinite-sample limit. Capping this
# specific outcome's concentration at 30% keeps a healthy mix of both hit
# and miss samples in every batch, accepting an imperfect match to the
# target mean payoff rather than a degenerate, all-eggs-in-one-basket tilt.
MAX_SINGLE_OUTCOME_TILT_PROB = 0.3
