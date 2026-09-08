# Architecture & Design Explainer

This document exists so you can explain every part of this project — down to individual functions and
formulas — without needing this session's context. It's written for interview prep: each section
doesn't just say what the code does, it says *why* it's built that way and what tradeoff or decision
sits behind it, so you can answer a follow-up question about it, not just describe it.

The backend section is exhaustive (every file, every function). The frontend section is lighter —
it focuses on the Flask mechanics and, especially, on **how the frontend and backend connect**, since
that boundary (or lack of one) is the thing most worth being able to explain clearly.

---

## Part 0: The mental model

Before the file-by-file detail, the one idea that everything else hangs off of:

**There is one Python process.** `frontend/app.py` is a Flask application. When a browser hits one of
its routes, the route handler doesn't make an HTTP call to some other "backend service" — it just
calls Python functions from the `backend` package directly, in the same process, the same way any
Python script calls functions from an imported module. `backend/` and `frontend/` are a *code
organization* boundary (math/data-access code vs. web-serving code), not a *network* boundary. There's
no API gateway, no second server, no separate deploy — one `flask run` command starts everything.

The actual computational pipeline, for "simulate opening 10 packs of SV01," is:

1. **Query Postgres** for that set's rarity pull-rates and card prices (`backend/db.py`).
2. **Build a probability distribution** over every possible pack outcome — a flat array of (card,
   price, probability) triples plus one "worthless" catch-all (`backend/distribution.py`).
3. **Draw random samples** from that distribution, vectorized with numpy — one simulated trial per
   "what if I opened 10 packs" scenario, tens of thousands of times (`backend/simulation.py`).
4. **Reduce those samples to statistics** — mean, standard deviation, VaR, CVaR, breakeven probability
   (`backend/risk_metrics.py`).
5. **Render a histogram** of the results as a PNG image (`backend/visualization.py`).
6. **Package it all as JSON** and send it back to the browser (`frontend/app.py`).

Everything below walks through exactly how each of those six steps is implemented, in the order they
run.

---

## Part 1: The Backend

### 1.1 `backend/config.py` — loading configuration

**Purpose:** read the database connection parameters (host, port, database name, user, password) out
of the `.env` file and hand them to the rest of the app as a typed object.

**Why a `.env` file at all:** credentials shouldn't be hardcoded into source code (they'd end up in
git history) or passed as command-line arguments (they'd show up in shell history and process
listings). A `.env` file that's excluded from version control (it's in `.gitignore`) is the standard
convention for local secrets.

**How loading works:**
```python
from dotenv import load_dotenv
load_dotenv()
```
`python-dotenv`'s `load_dotenv()` reads the `.env` file in the project root, parses its `KEY=value`
lines, and inserts them into `os.environ` — the same dictionary-like object Python uses for actual
operating-system environment variables. This means after calling it, `os.environ["DB_HOST"]` works
exactly as if you'd set that variable in your shell, whether it actually came from the shell or from
the file. It's called once, at *import time* (i.e., the moment any other file does
`from backend import config` or `from backend.config import ...`), so every module that touches
config shares the same populated environment without needing to remember to call it themselves.

**The `Settings` dataclass:**
```python
@dataclass(frozen=True)
class Settings:
    db_host: str
    db_port: int
    db_name: str
    db_user: str
    db_password: str
```
`@dataclass` auto-generates `__init__`, `__repr__`, and `__eq__` from the type-annotated fields, so you
get a clean constructor and printable representation for free instead of writing that boilerplate by
hand. `frozen=True` makes instances immutable — once created, you can't do `settings.db_host = "x"`,
which is appropriate for something that represents a fixed configuration loaded once at startup; if
some code tried to mutate it later, that would almost certainly be a bug, and `frozen=True` makes that
a hard error instead of a silent, confusing state change.

**Why bundle five values into one object instead of passing them around individually:** every function
elsewhere that needs a database connection (starting with `db.get_connection`) takes one `Settings`
argument instead of five loose strings. That's easier to read at call sites, and it makes testing
easier — a test can construct a fake `Settings(db_host="test", ...)` directly instead of having to
monkeypatch environment variables just to influence a function's behavior.

**`load_settings()`:**
```python
def load_settings() -> Settings:
    required = ("DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"Missing required environment variable(s): {', '.join(missing)}. ...")
    return Settings(
        db_host=os.environ["DB_HOST"],
        db_port=int(os.environ["DB_PORT"]),
        ...
    )
```
This is a **fail-loudly-at-startup** design. It would be easy to write this with `os.environ.get("DB_HOST", "localhost")`
so a missing variable just silently falls back to a default — but that's actively dangerous here: if
`DB_HOST` were missing and silently defaulted to `"localhost"`, the app might connect to a *different,
wrong* database that happens to also be running locally, and every downstream number would be
plausible-looking nonsense with no error anywhere. Explicitly checking for missing variables and
raising immediately means a misconfiguration is caught the moment the app starts, not discovered later
as mysteriously wrong simulation output.

---

### 1.2 `backend/constants.py` — the single source of truth

This file holds values that *multiple modules* need to agree on. The organizing principle: if two
files each hardcoded their own copy of the same mapping or threshold, they could silently drift apart
when one gets updated and the other doesn't. Centralizing avoids that class of bug entirely.

**`RARITY_RATE_COLUMNS`** — the most important mapping in the project:
```python
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
```
The database's `groupdata` table has nine numeric columns storing pull-rate percentages, but their
column names are abbreviations (`dr_rates`, `sir_rates`, etc.) while the actual card records in
`carddata` use full rarity name strings (`"Double Rare"`, `"Special Illustration Rare"`, etc.). This
dict is the Rosetta Stone between the two. **This mapping was not guessed — it was verified empirically**
against the live database: for sets where a given rate column had a real (non-null) value, I queried
`carddata` and confirmed cards with the mapped rarity string existed in matching counts and price
ranges. For example, `SV: Paldean Fates` has `sr_rates = 25.44` and `sur_rates = 7.72`; querying its
cards showed exactly a `"Shiny Rare"` group (120 cards) and a `"Shiny Ultra Rare"` group (12 cards),
confirming the mapping for those two columns. Similarly, `ME: Ascended Heroes` has `mhr_rates = 0.19`
and `mar_rates = 3.47`, and its cards include `"Mega Hyper Rare"` (2 cards) and `"Mega Attack Rare"`
(7 cards) groups. This kind of cross-checking against real data — rather than assuming a name implies
its abbreviation — is exactly the sort of database-integrity verification worth being able to describe
in an interview: you don't trust that a column name means what it looks like it means, you confirm it
against the data it's supposed to describe.

This mapping also encodes a **confirmed modeling decision**: only these nine rarities are treated as
having any resale value in the simulation at all. Every other card (Common, Uncommon, plain "Rare",
digital "Code Card" bonus items, energy cards) is excluded from individual modeling — see §1.4 for why.

**`IN_SCOPE_SET_SQL_FILTER`:**
```python
IN_SCOPE_SET_SQL_FILTER = (
    "(setname ILIKE 'SV%' OR setname ILIKE 'ME%') AND ("
    + " OR ".join(f"{col} IS NOT NULL" for col in RARITY_RATE_COLUMNS)
    + ")"
)
```
This builds the SQL `WHERE` clause fragment that defines "which sets does this app support" —
programmatically, from the same `RARITY_RATE_COLUMNS` dict, rather than as a second hand-typed copy
that could drift out of sync. It resolves to something like:
```sql
(setname ILIKE 'SV%' OR setname ILIKE 'ME%') AND (dr_rates IS NOT NULL OR ur_rates IS NOT NULL OR ...)
```
A worthwhile technical detail here: the `%` characters in `'SV%'` are **SQL wildcard characters** (for
`LIKE`/`ILIKE` pattern matching — "any sequence of characters"), not Python format placeholders. But
psycopg *also* uses `%s` as its parameter placeholder syntax when you pass a params tuple to
`cursor.execute()`. If this string were ever concatenated into a query executed *with* parameters,
psycopg would try to parse every `%` in the string as part of a placeholder and raise an error (or
worse, silently misinterpret it) — which is why, in a query that does take parameters, a literal `%`
has to be escaped as `%%`. This constant deliberately uses a single `%`, with a comment explaining that
it's safe specifically because `db.get_in_scope_sets()` executes it with *no* parameters at all — when
you call `cursor.execute(query)` with no second argument, psycopg sends the string through unmodified
and never scans it for placeholders. This is a subtle but real gotcha with the `%s`-style ("pyformat")
parameter substitution psycopg uses, and getting it right (or wrong) is the kind of thing that produces
a confusing runtime error if you don't understand why.

Verified live against the real database: this filter currently selects exactly **22 sets** — SV01
through SV10 plus special Scarlet & Violet releases (151, Paldean Fates, Prismatic Evolutions,
Shrouded Fable, Black Bolt, White Flare), and ME01 through ME05 plus Ascended Heroes. It automatically
excludes sets like promo collections, energy-only products, and celebration sets that share the same
tables but were never sold as booster packs with real, tracked pull odds — and it will automatically
*include* a new set (e.g. a currently-unreleased ME06) the instant its rate data is populated, with
zero code changes, because the filter is a live query condition, not a hardcoded list.

**`Y_VALUES` and `ALPHAS`:** the design doc's originally-specified scenario set `(1, 5, 10, 100, 1000)`
and confidence levels `(0.95, 0.99)` the project reports by default. Every function that takes a `y` or
`alphas` parameter also works for arbitrary values — the frontend, in fact, now takes an arbitrary `y`
from 1-1000 directly rather than exposing `Y_VALUES` as fixed buttons (see Part 2), so `Y_VALUES` itself
is currently unused by the app and kept only as a named reference for offline reporting.

**`N_TRIALS_DEFAULT` — a tuning decision with a real story behind it:**
```python
N_TRIALS_DEFAULT = 100_000
```
This went through three stages, worth being able to narrate:
1. **Started at 50,000** — a guess at "small enough to be fast, big enough to be meaningful."
2. **You asked to raise it to 500,000** for better precision (the project's own reasoning is that a
   plain Monte Carlo estimate of a roughly 1-in-several-thousand probability needs on the order of half
   a million trials to get to ~10% relative error).
3. **Measured, then dialed back to 100,000.** The simulation draws `n_trials × y` random values in a
   single batch (see §1.5). At `y = 1000` — the largest supported scenario — `500,000 × 1000 =
   500,000,000` values. Each value is a 64-bit float (8 bytes), and the draw pipeline needs *three*
   arrays of that size simultaneously at peak (the raw uniform-random draws, the looked-up outcome
   indices, and the gathered payoff values before summing) — roughly `3 × 4GB ≈ 12GB` of transient
   memory, on a machine with 16GB of RAM total. That's uncomfortably close to the edge alongside
   whatever else (browser, IDE) is normally running, so **I measured it directly** rather than guess:
   at `n_trials = 100,000`, the same `y=1000` case took about 7 seconds and comfortably fits in memory.
   500,000 was tried and consciously rejected for exactly this reason. This is a good interview
   anecdote about **measuring instead of assuming** when making a performance/precision tradeoff.

**`DEFAULT_SEED = 42`:** numpy's `default_rng(seed)` uses a specific, well-tested pseudorandom
algorithm (PCG64) that produces the exact same sequence of "random" numbers every time it's given the
same seed. A Monte Carlo simulation is inherently about randomness, but for a project meant to be
explained and re-derived later, being able to say "run it again, you'll get exactly the same number" —
for hand-verification, for unit tests, for reproducing a specific result you want to discuss — matters
more than getting a fresh random draw on every run. Any caller can still pass a different seed (or
`None`, which seeds from OS entropy) to get an independent run.

---

### 1.3 `backend/db.py` — all SQL lives here, and only here

This is the only module in the project that imports `psycopg` or writes a SQL string. That's a
deliberate isolation: if the database schema ever changes, or the project ever swapped Postgres for
something else, this is the only file that would need to change.

**`get_connection(settings) -> psycopg.Connection`:**
```python
return psycopg.connect(
    host=settings.db_host, port=settings.db_port, dbname=settings.db_name,
    user=settings.db_user, password=settings.db_password,
)
```
Two decisions worth explaining if asked:
- **Keyword arguments, not a DSN connection string** (e.g. `"postgresql://user:pass@host/db"`).
  Building a DSN string manually means you have to correctly URL-escape special characters in the
  password — this project's actual password contains a `!`, which would need escaping in a DSN string
  but needs no special handling at all when passed as a plain keyword argument. Using keyword arguments
  sidesteps that whole class of bug.
- **No connection pool.** A production web service handling concurrent users would typically use
  `psycopg_pool` to maintain a small set of reusable connections rather than opening a fresh TCP
  connection to Postgres on every request. This is a single-user, low-traffic Flask app, so a pool
  would add a dependency and lifecycle complexity (when to open it, when to close it, how many
  connections to keep) for no measurable benefit at this scale. This is called out explicitly in the
  docstring as a **deliberate, scoped-down tradeoff** — the kind of thing worth being able to say
  out loud in an interview ("I know what the production-grade version would look like, and chose not
  to build it here because the scale doesn't call for it").

**`get_in_scope_sets(conn) -> list[str]`:**
```python
query = f"SELECT setname FROM groupdata WHERE {IN_SCOPE_SET_SQL_FILTER} ORDER BY setname;"
with conn.cursor() as cur:
    cur.execute(query)
    return [row[0] for row in cur.fetchall()]
```
Returns the 22 valid set names. This is the **single source of truth** for "what can the user type in"
— the Flask frontend calls this directly (once, at startup — see §2) rather than maintaining its own
separate hardcoded list that the database filter could silently drift out of sync with.

Note the `with conn.cursor() as cur:` pattern used throughout this file: a psycopg cursor is a
context manager, so using `with` guarantees the cursor is closed (releasing its server-side resources)
even if an exception is raised inside the block — the same reason you'd use `with open(...) as f:`
for files rather than manually calling `.close()`.

**`get_set_rates(conn, setname) -> dict`:**
```python
query = """
    SELECT groupid, dr_rates, ur_rates, ir_rates, sir_rates, hr_rates,
           sr_rates, sur_rates, mhr_rates, mar_rates
    FROM groupdata WHERE setname = %s;
"""
cur.execute(query, (setname,))
row = cur.fetchone()
if row is None:
    raise ValueError(f"Unknown set name: {setname!r}")
return dict(zip(columns, row))
```
Two things worth pointing out:
- **`%s` here *is* a real psycopg placeholder** (contrast with the SQL-wildcard `%` discussed in
  §1.2) — passing `(setname,)` as a second argument to `execute()` tells psycopg to safely substitute
  it, correctly escaping any special characters in `setname` so it can never be interpreted as SQL
  syntax. This is the standard defense against SQL injection: even though in this app `setname` is
  always pre-validated against `get_in_scope_sets()` before it ever reaches this function, parameterizing
  the query costs nothing and means the function is safe to call from anywhere, not just from
  call sites that happen to validate first.
- **`dict(zip(columns, row))`** is a compact idiom for turning a database row (a plain tuple of
  values, in column order) into a dictionary keyed by column name — `zip` pairs up
  `("groupid", "dr_rates", ...)` with `(24380, 20.91, ...)` element-by-element, and `dict(...)` turns
  those pairs into `{"groupid": 24380, "dr_rates": 20.91, ...}`.
- **Raising `ValueError` on an unknown set**, rather than returning `None` or an empty dict, means
  every caller gets a clear, specific failure with a message identifying exactly what went wrong,
  instead of having to remember to check for a falsy return value and guess why.

**`get_pack_cost(conn, groupid) -> float`:**
```python
query = """
    SELECT cardname, price FROM carddata
    WHERE groupid = %s
      AND cardname ILIKE %s AND cardname NOT ILIKE %s AND cardname NOT ILIKE %s
      AND cardname NOT ILIKE %s AND cardname NOT ILIKE %s;
"""
params = (groupid, "%booster pack%", "code card%", "%art bundle%", "%sleeved%", "%set of%")
```
This is how the "what do y packs cost" number is obtained. The database doesn't have a dedicated
"pack price" column — instead, the standalone booster pack is just one more product row in `carddata`,
alongside actual card singles *and* other sealed products that also have "Booster Pack" somewhere in
their name (art bundles of 4, "sleeved" collector variants, multi-pack bundles, and digital code-card
listings tied to those bundles). The filter above was arrived at by directly inspecting real rows —
e.g. for SV01 it isolates exactly `"Scarlet & Violet Booster Pack"` at $8.98 while excluding rows like
`"Scarlet & Violet Booster Pack Art Bundle [Set of 4]"` at $38.79 — and then **validated against
all 22 in-scope sets** to confirm it returns exactly one priced row every time, not zero (missing data)
or more than one (an ambiguous match).

That validated invariant is enforced with `assert`:
```python
assert len(rows) == 1, f"Expected exactly one plain booster-pack price for groupid={groupid}, found {len(rows)}: {rows}. ..."
assert price is not None, f"Booster pack row {cardname!r} has a NULL price."
```
**Why `assert` here specifically, rather than raising a custom exception or silently picking `rows[0]`:**
these two conditions are not user-input problems (like an unknown set name, which uses `ValueError`) —
they're statements about the *shape of the data itself* that were empirically verified to hold for
every current set. An `assert` failing here means the verified invariant has broken, most likely
because the database was refreshed with new product naming that the filter no longer correctly
isolates. That's a "the code's assumptions about the data are now wrong" bug, which should fail loudly
and immediately (an `AssertionError` with a message pointing at exactly which `groupid` and which rows
confused it) rather than silently guessing (e.g. always taking `rows[0]`) and quietly reporting a wrong
pack cost, which would then quietly corrupt every downstream profit and VaR/CVaR number with no
indication anything was wrong.

**`get_cards_for_rarity(conn, groupid, rarity) -> list[tuple[str, float]]`:**
```python
query = "SELECT cardname, price FROM carddata WHERE groupid = %s AND rarity = %s;"
...
for cardname, price in rows:
    assert price is not None, f"Card {cardname!r} ... has a NULL price; the simulation has no fallback ..."
return [(cardname, float(price)) for cardname, price in rows]
```
Returns every individual card of one rarity in one set — this is the function that ultimately supplies
the "$5 card and a $400 card" granularity the whole project is built around (see §1.4). The `assert`
here is purely defensive: live inspection confirmed zero NULL prices exist among the nine in-scope
rarities across all 22 sets today, so this should never fire on current data — it exists so that *if*
the database is later refreshed with a gap, the failure is loud and specific (naming the exact card)
rather than the code silently treating a missing price as, say, `$0`, which would silently and
invisibly bias every risk statistic downward.

---

### 1.4 `backend/distribution.py` — the mathematical core of the project

This is the single most important file to be able to explain, because it's where the project's central
modeling decision lives.

**The core idea, stated precisely:** don't simulate "which rarity did I pull" and then plug in that
rarity's *average* card price. Here's why that would be wrong, concretely: suppose a rarity tier
contains exactly two cards, a $5 card and a $400 card, each equally likely. The *correct* expected
value contribution from that tier is the same either way you model it. But the *distribution* — the
actual shape of possible outcomes — is completely different. Model it as "this tier is worth $202.50"
and you get a simulation that says every pull from that tier is a certain $202.50, which has **zero
variance** from that tier and can never produce the $400 outcome or the $5 outcome individually. Model
it as "50% chance of $5, 50% chance of $400" (the correct approach) and the simulated outcomes actually
include both real possibilities. Since the entire point of this project is to characterize tail
outcomes via VaR and CVaR — literally "how bad can this get, and how bad does it get on average once
it's bad" — collapsing a two-outcome tier into its average would erase exactly the information the
project exists to report. This is why `distribution.py`'s docstring calls this "the heart of the
individual-card-level requirement."

**The `SetDistribution` dataclass:**
```python
@dataclass(frozen=True)
class SetDistribution:
    setname: str
    groupid: int
    outcomes: list[str]          # card names; index 0 is always "WORTHLESS"
    payoffs: np.ndarray          # dollar value of each outcome; payoffs[0] == 0.0
    cumulative_probs: np.ndarray # P(outcome index <= i); non-decreasing, ends at exactly 1.0
    pack_cost: float
```
This bundles everything needed to simulate a set into one immutable object, built once per set and
reused across every trial. The three parallel arrays (`outcomes`, `payoffs`, `cumulative_probs`) are
aligned by index: outcome `i` has name `outcomes[i]`, dollar value `payoffs[i]`, and its slice of
probability mass is the gap between `cumulative_probs[i-1]` and `cumulative_probs[i]`.

**The "worthless bucket" modeling decision** (this was explicitly discussed and confirmed, not assumed):
the database's `groupdata` rate columns only cover the pack's "hit slot" — the one slot in a pack that
can be upgraded to a Double Rare, Ultra Rare, etc. There is **no data at all** on how many Commons,
Uncommons, or plain "Rares" appear per pack, so there is no way to build a fully-detailed distribution
over *every physical card* the data doesn't support. Rather than inventing an unsupported assumption
(e.g. guessing "6 commons per pack" with no basis), every bit of probability mass not explained by the
nine named chase rarities is folded into one pseudo-outcome, `"WORTHLESS"`, with a fixed payoff of
`$0`. This slightly understates real pack value (a genuine Common or Uncommon does sell for a few
cents), but that's immaterial next to the dollar-scale VaR/CVaR numbers this project reports, and it's
the honest boundary of what the available data can actually support.

**`build_set_distribution(conn, setname) -> SetDistribution`, step by step:**

```python
rates = get_set_rates(conn, setname)
groupid = rates["groupid"]

outcomes: list[str] = [WORTHLESS_OUTCOME]
payoffs: list[float] = [0.0]
probs: list[float] = [0.0]  # placeholder, filled in at the end
```
Start every set's outcome list with the worthless sentinel at index 0. Its probability is a
placeholder for now because it depends on how much probability everything *else* ends up claiming —
computed last, as "whatever's left over."

```python
for column, rarity_label in RARITY_RATE_COLUMNS.items():
    rate = rates[column]
    if rate is None:
        continue  # this rarity has zero cards in this set
    tier_probability = float(rate) / 100.0  # rate is a PERCENTAGE, e.g. 13.76 -> 0.1376
    cards = get_cards_for_rarity(conn, groupid, rarity_label)
    if not cards:
        continue  # rate is exactly 0 (not null) and genuinely has no matching cards
    per_card_probability = tier_probability / len(cards)
    for cardname, price in cards:
        outcomes.append(cardname)
        payoffs.append(price)
        probs.append(per_card_probability)
```
For each of the nine rate columns: if it's `NULL`, that rarity doesn't exist in this set at all, skip
it entirely (not even a zero-probability entry — it's absent, not present-with-zero-chance). Otherwise
convert the stored percentage to a probability (divide by 100), fetch every card of that rarity, and
split the tier's total probability **evenly** across them. That even split is itself a modeling choice
worth being able to name: there's no per-card pull-rate data finer than "this rarity as a whole," so
"every card in a tier is equally likely" is the only assumption the data supports — it mirrors the same
"equal odds within a rarity" simplification used for the tier itself. (A second, separate real case is
also handled here: a rate can be exactly `0` — not `NULL` — meaning the column exists but the pull rate
is genuinely zero; this happens for a few observed sets, e.g. one set's Hyper Rare rate is listed as
`0`. That correctly produces zero matching cards, handled by the `if not cards: continue` line rather
than a division-by-zero.)

```python
assigned_probability = sum(probs)
worthless_probability = 1.0 - assigned_probability
assert worthless_probability >= -_PROBABILITY_EPSILON, f"Rate columns for {setname!r} sum to more than 100% ..."
probs[0] = max(worthless_probability, 0.0)
```
Whatever fraction of probability the nine named rarities didn't claim becomes the worthless bucket's
probability. The assertion guards against a genuine data-integrity failure: if the rate columns for a
set ever summed to *more* than 100%, that would mean the raw data is internally inconsistent (e.g. two
overlapping "hit slots" double-counted), and that should fail loudly rather than silently producing a
negative probability that would corrupt every draw. `_PROBABILITY_EPSILON = 1e-9` gives a tiny amount
of floating-point slack, since the rate columns are stored to two decimal places and floating-point
addition of many small numbers can land a hair off an exact boundary even when the underlying data is
perfectly consistent.

```python
payoffs_arr = np.array(payoffs, dtype=np.float64)
cumulative_probs = np.cumsum(np.array(probs, dtype=np.float64))
cumulative_probs[-1] = 1.0  # eliminate floating-point drift
```
`np.cumsum` turns the list of individual probabilities into a **running total** — e.g. `[0.8, 0.1,
0.1]` becomes `[0.8, 0.9, 1.0]`. This cumulative array is what makes fast sampling possible (see
§1.5): to draw an outcome, you just need to find which "bucket" of this running total a random number
falls into. The last line — forcing the final entry to be *exactly* `1.0` — matters because summing
many small floating-point numbers can accumulate tiny rounding error, landing at something like
`0.9999999999999998` instead of `1.0`. If that happened, a uniform random draw of (astronomically
unlikely, but representable) `1.0` would fall *past* the end of the array, and the lookup function used
for sampling (`numpy.searchsorted`) would return an out-of-bounds index and crash. Explicitly pinning
the last entry to `1.0` eliminates that edge case entirely, for the cost of one line.

**The mathematical equivalence this whole structure relies on:** building one flat array this way is
mathematically identical to a two-stage process — (1) pick a rarity tier with probability equal to its
published rate (treating "worthless" as a pseudo-tier with the leftover probability), then (2) pick a
uniformly-random card within that tier — but flattened into a single lookup. This is true because of
how the cumulative array is constructed: each tier's cards are laid end-to-end in the array with the
tier's total probability split evenly across them, so a random draw landing anywhere in that tier's
span of the cumulative array is exactly as likely as any other point in that span, which is precisely
what "pick a tier, then pick uniformly within it" means. The benefit of doing it this way is entirely
computational: it collapses a two-step, branching process into a single array lookup (`searchsorted`)
that numpy can vectorize across millions of draws at once, instead of writing a Python loop that
executes the two-stage decision separately for every single simulated pack.

---

### 1.5 `backend/simulation.py` — sampling, and why it's fast

**`make_rng(seed=DEFAULT_SEED) -> np.random.Generator`:** a one-line wrapper around
`np.random.default_rng(seed)`. Centralizing this in one function — rather than calling
`np.random.default_rng()` directly wherever a random draw happens — means the whole project's
reproducibility behavior is controlled from one place: pass the shared default seed (or `None` for a
fresh independent run) and every caller gets consistent behavior.

**`draw_outcome_indices(dist, shape, rng) -> np.ndarray` — the sampling method itself:**
```python
uniforms = rng.random(shape)
return np.searchsorted(dist.cumulative_probs, uniforms, side="right")
```
This is **inverse transform sampling**, a standard technique worth being able to name and explain: for
any distribution with cumulative distribution function `F`, if `U` is drawn uniformly from `(0, 1)`,
then `F⁻¹(U)` (find the value whose cumulative probability equals `U`) is a valid draw from that
distribution. Here, `dist.cumulative_probs` *is* a (discrete, step-function) CDF, and `searchsorted`
performs the "inverse" lookup: given a uniform random number `u`, it does a binary search to find the
index of the first entry in `cumulative_probs` that is `≥ u` (with `side="right"`, more precisely: the
insertion point that keeps the array sorted, treating ties as landing on the right side) — that index
*is* the drawn outcome. `rng.random(shape)` draws an entire array of uniform random numbers in one
numpy call, and `searchsorted` looks all of them up against the same cumulative array in one
vectorized call — no Python-level loop over individual draws at all, which is what makes drawing
hundreds of millions of values (see the `y=1000` case in §1.2) take single-digit seconds instead of
tying up the interpreter in a loop.

The `side="right"` detail: it decides how a random draw that lands *exactly* on a boundary between two
outcomes' probability ranges is resolved (which of the two adjacent outcomes it counts as). This is a
measure-zero edge case — the probability of a continuous uniform draw landing on an exact boundary is
effectively zero — but choosing it explicitly (rather than leaving the default and not knowing which
way it resolves) costs nothing and removes any ambiguity.

**`simulate_pack_profits(dist, n_trials, rng) -> np.ndarray`:**
```python
indices = draw_outcome_indices(dist, (n_trials,), rng)
return dist.payoffs[indices] - dist.pack_cost
```
Draws `n_trials` outcome indices, looks up each one's payoff via **numpy fancy indexing**
(`dist.payoffs[indices]` builds a new array by pulling out `payoffs[indices[0]], payoffs[indices[1]],
...` all at once), and subtracts the pack cost from every element (numpy broadcasts the scalar
`dist.pack_cost` across the whole array automatically). This is the `y = 1` case, and it's kept as its
own function specifically because it's the smallest, most hand-verifiable unit: the expected value of
its output can be checked by hand against `Σ p_i × payoff_i − pack_cost` computed directly from the
raw database values (this was done for the smallest in-scope set, `SV: Shrouded Fable`, and the
simulated mean over 1,000,000 trials matched the hand calculation to within half a cent).

**`simulate_y_pack_profits(dist, y, n_trials, rng) -> np.ndarray` — the general case:**
```python
indices = draw_outcome_indices(dist, (n_trials, y), rng)
total_payoff_per_trial = dist.payoffs[indices].sum(axis=1)
return total_payoff_per_trial - (y * dist.pack_cost)
```
Instead of looping `y` times per trial in Python, this draws a single **2D array** of shape
`(n_trials, y)` in one call — think of it as a table with `n_trials` rows (one row per "what if I
opened y packs" scenario) and `y` columns (one column per pack within that scenario). `.sum(axis=1)`
sums *across each row* (summing over the `y` packs within one scenario), collapsing the 2D array back
down to a 1D array of `n_trials` total-payoff values — one per simulated scenario. This is the same
vectorization principle as the single-pack case, just extended to two dimensions, and it's what lets
`y = 1000` with 100,000 trials — 100 million individual pack draws — run in about seven seconds rather
than looping in Python a hundred million times.

**The independence assumption, stated and explained (this is one of the project's explicitly named
simplifications):** packs are simulated as independent draws *with replacement* from the flat
distribution, even when `y` packs are meant to represent packs pulled from the same physical box. In
reality, a sealed box contains a *fixed, finite* population of packs — if you know a box has already
yielded a Hyper Rare, that (very slightly) changes the odds for the packs remaining in that same box,
because you're sampling without replacement from a population whose contents are now partially known.
This project doesn't model that correction: doing so exactly would require knowing a box's *exact*
contents pack-by-pack, which the database doesn't provide (it only has set-level aggregate pull rates).
This is called a **reasonable approximation across separate boxes** (there, packs genuinely are
independent) but a **deliberate simplification within one box** — and its distortion shrinks as box
size grows large relative to how rare the tail event being measured is, since a single rare pull has a
proportionally smaller effect on the remaining odds in a bigger box.

---

### 1.6 `backend/risk_metrics.py` — turning simulated numbers into risk statistics

**`AlphaMetrics`** and **`RiskMetrics`** are plain data-holding dataclasses — no logic, just typed
containers for the numbers `compute_risk_metrics` produces, so the rest of the app (tests, the Flask
route) can access `metrics.mean`, `metrics.by_alpha[0.95].var`, etc. by name instead of unpacking a
tuple or dict and hoping the ordering/keys are remembered correctly.

**`compute_risk_metrics(profits, y, pack_cost, alphas) -> RiskMetrics`:**
```python
n_trials = len(profits)
mean = float(np.mean(profits))
sd = float(np.std(profits, ddof=0))
total_cost = y * pack_cost
breakeven_prob = float(np.mean(profits > 0))
sorted_profits = np.sort(profits)
by_alpha = {alpha: _compute_alpha_metrics(sorted_profits, alpha) for alpha in alphas}
```
Two details worth being able to explain if asked:
- **`ddof=0`** ("delta degrees of freedom") controls whether `np.std` divides the sum of squared
  deviations by `N` (population standard deviation, `ddof=0`) or by `N-1` (sample standard deviation,
  the customary correction when you're using a sample to *estimate* the variance of some larger
  population you don't have full access to, `ddof=1`). Here, the `n_trials` simulated profits *are*
  the entire population being described — they're not a sample drawn from a larger population whose
  variance is being estimated indirectly — so the uncorrected population formula (`ddof=0`) is the
  right one, and with n_trials in the tens of thousands the difference between the two is negligible
  regardless.
- **`profits > 0` on a numpy array** produces a boolean array (`True`/`False` per element), and
  `np.mean` on a boolean array computes the fraction that are `True` — a compact way to compute
  `P(profit > 0)` (the breakeven probability) without writing an explicit counting loop.
- **`total_cost = y * pack_cost`** was added specifically so VaR/CVaR could be read *directly against*
  what was actually spent (e.g. "VaR₉₅ is 99% of the $89.80 you paid") rather than requiring separate
  mental math — it isn't itself a risk statistic, it's context that makes the risk statistics
  interpretable at a glance.
- Sorting `profits` once up front and passing the same `sorted_profits` array into
  `_compute_alpha_metrics` for every `alpha` avoids re-sorting the same tens-of-thousands-of-elements
  array once per confidence level — a small but easy efficiency to get right.

**`_compute_alpha_metrics(sorted_profits, alpha) -> AlphaMetrics` — VaR and CVaR, including a bug this project actually caught:**
```python
m = len(sorted_profits)
k = math.ceil((1 - alpha) * m - 1e-9)
k = max(k, 1)
tail = sorted_profits[:k]
boundary_profit = sorted_profits[k - 1]
if boundary_profit > 0:
    var, var_binding = 0.0, False
else:
    var, var_binding = -float(boundary_profit), True
cvar = -float(np.mean(tail))
```
The formulas, in words:
- **VaR_α** answers "how bad does it get at the boundary of the worst `(1-α)` fraction of outcomes?"
  With `M` sorted trials, `k = ⌈(1-α) × M⌉` is the count of trials in that worst slice (e.g. `α = 0.95`
  on 20,000 trials → the worst 1,000 trials). The boundary trial, `sorted_profits[k-1]` (using
  zero-based indexing, this is the "k-th worst" value), is reported as a **positive loss magnitude** —
  so a boundary of `-$88.60` (an actual loss) is reported as `VaR = $88.60`. If the boundary trial is
  itself a *gain* (positive profit), that means even the worst `(1-α)` fraction of outcomes didn't lose
  money — VaR is **non-binding** at that confidence level, and the code reports `0` with an explicit
  `var_binding = False` flag rather than a nonsensical "negative loss."
- **CVaR_α** (Conditional VaR / Expected Shortfall) answers a different question: "given that you're
  in that worst tail, how bad is it *on average*?" It's the mean of every value in the tail, not just
  the boundary — and it's always reported as a genuine tail average, even in the non-binding-VaR case
  (where it can come out negative, correctly indicating the "worst tail" was actually a mild gain on
  average).

**The bug this exact formula caught, worth telling as a specific story:** the first version of
`k = math.ceil((1 - alpha) * m)` (without the `- 1e-9`) failed a unit test with a clean, hand-picked
20-element array at `alpha = 0.95`. Mathematically, `(1 - 0.95) × 20 = 1.0` exactly, so `k` should be
`1`. But `0.95` isn't exactly representable in binary floating point — in Python, `1 - 0.95` actually
evaluates to `0.050000000000000044`, and `0.050000000000000044 × 20 = 1.0000000000000009`, not exactly
`1.0`. `math.ceil` of that rounds *up* to `2`, silently using one extra trial in the tail — a real bug
that would have quietly produced a slightly-wrong VaR/CVaR any time `(1-alpha) × M` was supposed to
land exactly on an integer. The fix subtracts a tiny epsilon (`1e-9`) before ceiling, which absorbs
that representation error without changing the result for any input that isn't sitting exactly on an
integer boundary to begin with. **This is a genuinely good interview story**: it demonstrates that the
project has real unit tests that catch real floating-point correctness bugs, not just "the code ran
without crashing."

**`check_sqrt_scaling(sd_y1, sd_y, y, tolerance=0.10) -> bool`:**
```python
expected_sd_y = sd_y1 * math.sqrt(y)
relative_drift = abs(sd_y - expected_sd_y) / expected_sd_y
within_tolerance = bool(relative_drift <= tolerance)
if not within_tolerance:
    warnings.warn(...)
return within_tolerance
```
The underlying math: if `y` packs are independent, identically-distributed draws (the independence
assumption from §1.5), then the *variance* of their sum is `y` times the variance of one draw —
`Var(ΣX) = y · Var(X)` — a standard property of variance for independent random variables. Taking the
square root of both sides gives `SD(ΣX) = √y · SD(X)`. This function is a **regression check, not a
correction**: it never adjusts the reported numbers to force agreement — it only warns when the
simulated relationship drifts meaningfully from what the math predicts, which is exactly the kind of
signal that would catch a sampling bug (e.g. if `y` packs were accidentally drawn from *correlated*
rather than independent distributions) or simply too few trials to have converged. Verified live: this
holds within 10% across `y = 5, 10, 100, 1000` for a real set.

One more small but real gotcha fixed here: `bool(relative_drift <= tolerance)` explicitly casts to a
native Python `bool`. Without the explicit cast, since `sd_y`/`sd_y1` typically come from numpy's
`.std()` (which returns `numpy.float64`, not a plain Python `float`), the comparison produces
`numpy.bool_` instead of Python's built-in `bool`. That's a subtle type-system trap: `numpy.bool_`
behaves like `bool` in almost every context (`if not result:` works fine either way), but it is *not*
the same object as Python's singleton `True`/`False`, so an identity check like `result is True` — which
a unit test used — silently fails even though the value is logically correct. This was caught by an
actual failing test during development, and it's a good example of why identity checks (`is`) on
values that might come from a numeric library are risky compared to equality checks (`==`) or just
using the value directly in a boolean context.

---

### 1.7 `backend/visualization.py` — rendering the histogram

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
```
Matplotlib's default backend assumes it's running in an environment with a display and an interactive
event loop (for popping up a window you can zoom/pan). A Flask server process run headlessly (no
monitor attached, no GUI) has neither — trying to use the default backend there would either raise an
error or hang waiting for a display that doesn't exist. `"Agg"` (Anti-Grain Geometry) is a backend that
renders directly to an image buffer instead of a window, which is exactly what's needed to produce a
PNG for an HTTP response. This is set once, at module import time, before `pyplot` is imported (the
order matters — `matplotlib.use()` must be called before `pyplot` is first imported, or the default
backend may already be locked in).

```python
def make_profit_histogram(profits, setname, y) -> bytes:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(profits, bins=60, ...)
    ax.axvline(x=0, color="#C44E52", linestyle="--", linewidth=2, label="Breakeven (profit = $0)")
    ...
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120)
    plt.close(fig)
    buffer.seek(0)
    return buffer.read()
```
The vertical line at `x=0` is the breakeven marker — by definition, `profit = (value of cards pulled) -
(cost of packs)`, so `profit = 0` is exactly the point where the packs paid for themselves. `io.BytesIO()`
is an in-memory, file-like byte buffer — `fig.savefig(buffer, format="png", ...)` writes PNG-encoded
bytes into that buffer instead of a file on disk, which means there's no temporary file to create,
track, or clean up; the bytes are read straight back out (`buffer.read()`) and returned. `plt.close(fig)`
matters because matplotlib keeps a reference to every created figure until explicitly closed — in a
long-running server process, forgetting this would leak memory a little more with every single
histogram generated.

---

### 1.8 `backend/importance_sampling.py` — a real scoping correction, then a real technique correction

Built last, after the plain Monte Carlo path (everything above) was fully working and verified, per the
project's own build order. Worth explaining is a genuine **correction to what this module is even
for**, caught before any code was written, followed by a second, deeper correction to *how* to build it
once live validation exposed a real numerical failure the first design didn't anticipate. This two-stage
story is a good thing to be able to narrate in an interview: it shows the difference between "importance
sampling is a variance-reduction technique, therefore add it somewhere" and actually reasoning about
*which* part of *this specific* distribution is rare enough to need it, and then *validating* the
chosen technique against the real data rather than trusting it because it "should" work.

**The wrong first framing:** "use importance sampling to get a more precise CVaR at the 95%/99%
confidence levels already reported in `risk_metrics.py`." This sounds plausible — CVaR involves
averaging a tail of outcomes, and tail estimation is the classic use case for importance sampling — but
it's wrong for *this particular distribution*, for a reason that only becomes obvious once you think
about the actual shape of the data rather than pattern-matching "tail statistic → importance sampling
helps."

**Why it's wrong:** this project's simulated profit distribution is **bottom-heavy**. Look at the
worthless-bucket construction in §1.4: for most sets, the single most likely outcome by far is
`"WORTHLESS"` — pulling nothing of resale value at all, which means that pack's profit is simply `-pack_cost`,
the maximum possible loss for that one pack. Recall the live-verified example from §1.5: for `SV:
Shrouded Fable`, the worthless-bucket probability alone was **66.7%** for a single pack. Extend that to
`y` packs and, especially at higher `y`, the vast majority of simulated trials cluster at or very near
the *worst possible outcome* — buying `y` packs and getting little to nothing back. That means the
**lower tail — the worst 5% of trials (used for VaR_95/CVaR_95) and the worst 1% of trials (used for
VaR_99/CVaR_99)** — sits in the single *densest, most heavily-populated* region of the whole
distribution, not a sparse one. Plain Monte Carlo, by construction, already places the most samples
exactly where the distribution has the most probability mass — so it already estimates that region
precisely, with no need for any variance-reduction trick. Building a tilted distribution to "sharpen"
CVaR_95/CVaR_99 here would be solving a problem that doesn't exist: there's no imprecision in those
numbers from insufficient sampling of that region in the first place.

**Where the real rare event actually is:** the *opposite* tail — a **large profit**. That only happens
when a low-probability, high-value card is pulled (a Special Illustration Rare, a Mega Hyper Rare,
etc. — recall from §1.2 that some of these have well under 1% pull probability per pack). *That's* the
genuinely sparse region of the distribution that plain Monte Carlo struggles to characterize precisely
without a very large trial count — exactly the situation importance sampling exists to help with.

**What this module estimates**, for whatever `y` the user picks:
- **`P(Π_y ≥ threshold)`** — the probability of landing in a large-profit tail.
- **`E[Π_y | Π_y ≥ threshold]`** — the expected profit *given* landing there, deliberately called
  "expected profit conditional on a top-tier outcome," never "CVaR." CVaR (Conditional Value at Risk) is
  a term of art in risk analytics that conventionally refers to a *downside* quantity — expected loss
  given a bad tail. Reusing it for an *upside* quantity would misrepresent what's being reported to
  anyone reading the results, even though the underlying math (a conditional tail average) is
  structurally the same operation either way. Being precise about naming only when the thing computed
  actually matches a term's convention is itself worth demonstrating in an interview.

**The threshold is dynamic, not hardcoded** — `estimate_profit_threshold(profits, p_target)` is just
`np.quantile(profits, 1 - p_target)`. Crucially, it takes an *already-computed* `profits` array rather
than running its own simulation: the app already draws a full plain-Monte-Carlo batch for the regular
`risk_metrics.py` statistics on every request, so locating this threshold from that same batch costs
nothing extra — a dedicated second "pilot" simulation would just be simulating the same thing twice.
This was a deliberate simplification made partway through building this module, not part of the
original design.

**First implementation attempt (subset-boosting), and why it was replaced:** the first working version
did exactly what the corrected framing above suggests — built a tilted `q` that boosted a hand-picked
set of high-value cards (e.g. every card priced ≥ $50) to some target combined probability, leaving
everything else proportionally scaled down. This worked well for `y = 1, 5, 10` (Effective Sample Size
of 86%, 94%, and even a reasonable 7-10% once a **second** bug — the tilt strength was applied per-pack
rather than derived for the overall trial — was fixed by solving `q_pack(y) = 1 - (1-overall_target)^(1/y)`
so the *per-trial* chance of a hit stayed constant regardless of `y`, instead of compounding into
absurdity: with a flat 15% *per-pack* boost, a `y=100` trial expects **~15 hits out of 100 packs** —
wildly unrealistic, and it compounds per-draw weight mismatches multiplicatively across every draw in a
trial, a classic failure mode called **weight degeneracy**, collapsing ESS toward 1.

Even after that fix, `y=1000` still failed outright: reaching that `y`'s threshold required **~47
simultaneous hits** on the boosted cards, not one or a few — because every card price is bounded, a
single lucky pull can only close a bounded gap, while the gap between "everything worthless" and a
demanding threshold *grows* with `y`. This is the standard distinction in large-deviations theory
between "one big jump" dominating a sum (what a subset-boost tilt is built for) and "many small
excesses adding up" (what actually happens for a bounded-payoff sum at large `y`). A tilt that can only
do the former structurally cannot reach the latter, no matter how it's tuned.

**The fix: Esscher (exponential) tilting**, `q_θ(x) = p(x)·e^(θ·payoff(x)) / M(θ)`, where
`M(θ) = Σ_x p(x)·e^(θ·payoff(x))` is the *moment generating function* of the payoff. Rather than
boosting a hand-picked subset, every outcome is reweighted smoothly by an exponential factor of its own
payoff — cards worth more get boosted more, continuously, with no arbitrary price cutoff. `θ` is solved
numerically (`solve_tilt_theta`, in `backend/importance_sampling.py`) via bisection, so that the tilted
distribution's *mean* payoff, summed over `y` packs, lands on the threshold:
`target_mean_payoff = threshold/y + pack_cost`. Because mean payoff under the tilt is a strictly
increasing, continuous function of `θ` (larger `θ` always shifts more probability toward higher
payoffs), bisection is guaranteed to converge — no derivative-based solver needed. `_log_sum_exp` (a
hand-written, four-line numerically-stable log-sum, subtracting the max term before exponentiating) is
used throughout rather than computing `e^(θ·payoff)` directly, since `θ·payoff` can be large enough to
overflow a raw `exp()` call.

This also **simplifies the weight formula** considerably. For `y` independent draws all from the same
`q_θ`, the joint probability under `q_θ` is `[Π p(xᵢ)] · e^(θ·Σxᵢ) / M(θ)^y`, so the importance weight
(joint `p` over joint `q_θ`) collapses algebraically to a single closed-form expression per trial:
```
log(weight) = y · log M(θ) − θ · (total payoff)
```
— computed directly from the same `total_payoff` already needed for profit, with no need to gather and
sum `log(p_i) − log(q_i)` over every individual draw the way the subset-boost approach required.

**A second real edge case, found via live validation, not by inspection:** at `y=1` for one real set,
the *dollar target* it solved for happened to land almost exactly on the single highest-priced card's
own value (because that card's natural pull probability, ~0.32%, already exceeded the 0.1% target
rarity). A distribution whose **mean** exactly equals its own **maximum possible value** can only be the
degenerate one that puts ~100% probability on that single outcome — so naively solving "match the mean"
pushed `θ` to make that card *near-certain* under `q`. With a finite batch of trials, that meant
essentially every trial drew the same card and *none* drew anything else, so the self-normalized
weighted estimator had zero "miss" samples to correctly weigh against the "hit" samples. The reported
probability came out as exactly `1.0` — silently, confidently wrong, not an error or a crash, which is
precisely the kind of failure worth catching before trusting a number. The fix,
`MAX_SINGLE_OUTCOME_TILT_PROB = 0.3` in `constants.py`: the `θ` search refuses to push the single
highest-*payoff* outcome's probability under `q` past 30%, accepting an imperfect match to the target
mean rather than a degenerate, all-eggs-in-one-basket tilt. (Note this tracks the highest-**payoff**
outcome specifically, not the outcome with the highest probability under `q` — the naturally most
probable outcome here is always WORTHLESS, often 60-99% probability even with *no* tilt at all, so a
blanket "no outcome above 30%" cap would have been violated by the untilted distribution itself and
never allowed any tilting to happen.)

**Live-verified results** (`SV01: Scarlet & Violet Base Set`, pack cost $8.98):

| y | threshold | θ | P(Π_y ≥ threshold) | E[Π_y \| ≥ threshold] | ESS |
|---|---|---|---|---|---|
| 1 | $62.75 | 0.077 | 0.0030 (plain MC: 0.0031) | $62.75 | 43.5% |
| 5 | $71.50 | 0.063 | 0.0009 | $85.34 | 9.8% |
| 10 | $61.63 | 0.052 | 0.0010 | $78.27 | 7.3% |
| 100 | -$356.88 | 0.026 | 0.0011 | -$323.07 | 1.3% |
| 1000 | -$5,676.83 | 0.011 | 0.0010 | -$5,595.68 | 0.4% |

Every row is now a *sensible* number, cross-checked against plain Monte Carlo at `y=1` (0.0030 vs.
0.0031 — close agreement). Compare to the subset-boost approach's `y=1000` result: an outright crash
or `P=1.0`/`ESS≈1`. ESS is honestly low at `y=1000` (0.4%) — a real, disclosed limitation of doing this
with a modest trial budget and a bounded per-outcome concentration cap, not a hidden one: every estimate
is returned alongside its ESS specifically so its reliability is never hidden from whoever reads it.
Note also the negative threshold and expected profit at `y=100`/`1000`: even the *best* 0.1% of outcomes
at that scale is still a loss, just a much smaller one than typical — exactly the "how close can you get
to breaking even" framing this project's threshold design was built around (see constants.py's
`TAIL_PROBABILITY` comment).

---

### 1.9 The test suite (`tests/`)

**`tests/conftest.py`** defines a shared pytest *fixture* (a reusable piece of test setup that pytest
automatically supplies to any test function that names it as a parameter) called `toy_distribution`: a
tiny, fully hand-computable `SetDistribution` (80% chance of $0, 15% chance of a $10 card, 5% chance of
a $100 card, $5 pack cost — hand-calculated EV: `0.80(0) + 0.15(10) + 0.05(100) - 5 = $1.50`). Using
deliberately simple, round numbers here — rather than real database values — means test failures are
immediately diagnosable by hand instead of requiring a spreadsheet to check whether the "expected"
value in a test was even computed correctly in the first place.

**`tests/test_distribution.py`** tests `build_set_distribution` using **monkeypatching** — a pytest
mechanism (via the built-in `monkeypatch` fixture) that temporarily replaces an attribute (here, the
`get_set_rates`, `get_cards_for_rarity`, and `get_pack_cost` functions as they're referenced *inside
the `distribution` module's namespace*) for the duration of one test, then automatically restores the
original afterward. This lets the test exercise the *real* `build_set_distribution` logic — the
worthless-bucket math, the cumulative-array construction — while completely avoiding a live database
connection, by substituting small, fully-controlled fake data in place of the real `db.py` functions.
This is the standard way to unit-test a function whose logic you want to verify independently of an
external dependency (a database, a network call, the current time) that would make the test slow,
flaky, or require infrastructure just to run.

**`tests/test_risk_metrics.py`** uses small, exactly hand-computable arrays (e.g. 20 evenly-spaced
values from `-100` to `90`) specifically so the tail-boundary index math (`k = ceil(...)`) can be
verified against a value worked out by hand — this is the test suite that caught the floating-point
`k` bug described in §1.6.

**`tests/test_simulation.py`** runs the actual (real, non-mocked) `simulate_pack_profits` and
`simulate_y_pack_profits` functions against the `toy_distribution` fixture with a large trial count
(500,000), and checks that the simulated mean lands within a small tolerance of the hand-calculated
$1.50 EV, and that simulated standard deviations across `y = 5, 10, 100` satisfy the `√y` scaling
relationship from §1.6 — i.e., these are automated versions of the manual "sanity check the EV by hand"
and "verify √y scaling" steps described as part of the project's build order.

---

## Part 2: The Frontend, and how it connects to the backend

### 2.1 What Flask actually is, and the key architectural point

Flask is a **WSGI micro-framework** — a lightweight Python library for mapping incoming HTTP requests
to Python functions and turning Python return values back into HTTP responses. "Micro" means it
deliberately doesn't bundle things like an ORM, form validation, or authentication out of the box the
way a framework like Django does — you add exactly what a given project needs. Here, that's just
routing, JSON handling, and Jinja2 templating (all included in Flask's core).

**The single most important architectural fact about this project, worth leading with in any
explanation:** the "frontend" and "backend" are not two separate services talking over a network.
There is one Flask process. When `frontend/app.py` needs to run a simulation, its route handler
imports and calls functions from the `backend` package **exactly like any other Python import** —
`from backend.distribution import build_set_distribution`, then later just `build_set_distribution(conn, setname)`
as a normal function call. There's no HTTP request between "frontend" and "backend," no JSON
serialization at that boundary, no separate process to deploy or keep in sync, and no network latency
between them — the entire round-trip from "user clicked a button" to "here's your histogram" happens
inside handling a single incoming HTTP request.

### 2.2 The three routes

```python
app = Flask(__name__)

_settings = load_settings()
with get_connection(_settings) as _startup_conn:
    _IN_SCOPE_SETS = get_in_scope_sets(_startup_conn)
_IN_SCOPE_SETS_SET = set(_IN_SCOPE_SETS)
```
Before any route is even defined, the app connects to the database **once**, at import/startup time,
and caches the list of 22 valid set names as a Python `set` (for O(1) membership checks — `x in a_set`
doesn't need to scan every element the way `x in a_list` would). The reasoning: this list only changes
when the database is repopulated with new sets, not on every request, so re-querying it on every single
keystroke of set-name validation would be wasted database round-trips for data that's effectively
static during the app's lifetime. The tradeoff, worth naming if asked: if the database *is* updated
with a new set while the app is running, that new set won't be recognized until the Flask process is
restarted.

**`GET /`** — `@app.route("/")` maps HTTP `GET` requests for the root URL to the `index()` function,
which calls `render_template("index.html", min_y=MIN_Y, max_y=MAX_Y)`. `render_template` invokes
Flask's built-in **Jinja2** templating engine: it reads `templates/index.html`, substitutes any
`{{ ... }}` expressions with actual Python values, and returns the resulting plain HTML string as the
response body. The pack-count field is a plain `<input type="number" min="{{ min_y }}" max="{{ max_y }}">`
rather than a row of buttons for fixed values — every backend function already accepted an arbitrary
`y`, so exposing that in the UI was just replacing the template's button loop with one input element.
`MIN_Y`/`MAX_Y` (1 and 1000) are enforced both here, client-side in `app.js` for immediate feedback, and
again, authoritatively, in `/simulate` itself — client-side validation can always be bypassed (e.g. a
direct `fetch()` call, or disabling JavaScript), so the server never trusts it alone.

**`POST /validate_set`:**
```python
payload = request.get_json(silent=True) or {}
setname = (payload.get("setname") or "").strip()
if setname in _IN_SCOPE_SETS_SET:
    return jsonify({"valid": True, "setname": setname})
return jsonify({"valid": False, "message": "..."})
```
`request` is Flask's global object representing the *current* incoming HTTP request (Flask uses a
mechanism called a "context local" so that, even though `request` looks like a plain module-level
global, it's actually correctly scoped to whichever request is currently being handled — safe even
under concurrent requests). `request.get_json(silent=True)` parses the request body as JSON, returning
`None` instead of raising an exception if the body is missing or malformed (`silent=True`) — the `or {}`
after it then guards against that `None` case so `.get("setname")` doesn't crash on it.
`jsonify(...)` is the mirror operation: it takes a Python dict, serializes it to a JSON string, and
wraps it in an HTTP response with the correct `Content-Type: application/json` header set automatically.
Matching is intentionally **exact and case-sensitive** — the project's own spec calls for a set name
that must exactly match the database, with an explicit retry message rather than any fuzzy/partial
matching, so a typo or wrong case is treated as clearly invalid input rather than silently guessing
what the user meant.

**`POST /simulate`** — the endpoint that actually invokes the whole backend pipeline described in Part 1:
```python
with get_connection(_settings) as conn:
    dist = build_set_distribution(conn, setname)
rng = make_rng()
profits = simulate_y_pack_profits(dist, y=y, n_trials=N_TRIALS_DEFAULT, rng=rng)
metrics = compute_risk_metrics(profits, y=y, pack_cost=dist.pack_cost, alphas=ALPHAS)
png_bytes = make_profit_histogram(profits, setname, y)
histogram_b64 = base64.b64encode(png_bytes).decode("ascii")
upper_tail = estimate_upper_tail_outcome(dist, y=y, rng=rng, regular_profits=profits)
return jsonify({... all the metrics ..., "top_tier_probability": upper_tail.probability, ...,
                "histogram_base64": histogram_b64})
```
This one function body *is* the entire pipeline from Part 0, plus the importance-sampled upper-tail
estimate from §1.8, called as ordinary, sequential Python function calls — no networking, no message
queue, no async job — because it's all one process. A fresh database connection is opened for this
request specifically (recall: no connection pool, by design — see §1.3) and closed automatically when
the `with` block exits. Note that `profits` — the plain Monte Carlo batch already computed for the
regular risk metrics — is passed straight into `estimate_upper_tail_outcome` as its `regular_profits`
argument: that's the "reuse existing sampling to locate the threshold, don't simulate twice" design
from §1.8, made concrete at the one call site that actually has both a fresh `profits` array and a
reason to want an upper-tail estimate.

One deliberate design choice worth explaining: the histogram PNG is **base64-encoded and embedded
directly in the same JSON response** as the numeric metrics, rather than the frontend making a second
request to a separate `/histogram.png` endpoint. Base64 is a way of encoding arbitrary binary data
(like PNG image bytes) as plain ASCII text, so it can be embedded inside a JSON string (JSON has no
native way to represent raw binary data). The tradeoff: the response payload is roughly 33% larger than
the raw PNG bytes would be (base64 always expands data by about that ratio) — but in exchange, the
metrics and the image are guaranteed to describe the *exact same* simulated batch of trials, computed
in one simulation run instead of two independent ones that could (due to randomness) show slightly
different numbers if they weren't seeded identically. For a low-traffic, single-user app, that
consistency guarantee was judged worth the modest payload-size cost.

**Input validation on `/simulate`:**
```python
if setname not in _IN_SCOPE_SETS_SET:
    return jsonify({"error": "..."}), 400
try:
    y = int(y_raw)
    if y <= 0:
        raise ValueError
except (TypeError, ValueError):
    return jsonify({"error": "y must be a positive integer."}), 400
```
Returning a tuple of `(response_body, status_code)` from a Flask route is how you set a non-default
HTTP status code — Flask's convention is that any two-item tuple returned from a view function is
interpreted as `(body, status)`. `400 Bad Request` is the standard HTTP status for "the client sent
malformed input," as distinct from `500 Internal Server Error` (the server itself failed unexpectedly).
This distinction matters for the calling JavaScript: it checks `response.ok` (`true` for any `2xx`
status, `false` otherwise) to decide whether to show the returned error message or treat the response
as a successful result.

### 2.3 The client side: `app.js`, and how it talks to the routes above

No JavaScript framework is used — just the browser's built-in `fetch()` API and direct DOM
manipulation. For a page this small (three sequential steps, no client-side routing, no complex shared
state), that's a deliberate scope-matching choice: introducing React or similar would add a build step
and a library dependency to manage roughly a hundred lines of "show this section, hide that one, update
some text" logic — complexity the actual UI doesn't call for. This is a reasonable thing to say
proactively in an interview if asked "why not React": the frontend's own spec explicitly calls for
"deliberately simple," and vanilla JS fully satisfies that scope without leaving obvious functionality
on the table.

The flow:
```javascript
const response = await fetch("/validate_set", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ setname }),
});
const data = await response.json();
```
`fetch()` returns a `Promise` — JavaScript's representation of "a value that will be available later" —
and `await` (inside an `async` function, which the event listener here is declared as) pauses execution
of *that function* until the promise resolves, without blocking the rest of the page. This is the
client-side mirror of the server's `request`/`jsonify`: the browser JSON-encodes the set name into the
request body, and decodes the server's JSON response back into a JavaScript object (`data.valid`,
`data.setname`, etc.).

Section visibility is toggled with the `hidden` HTML attribute (`stepY.hidden = false`) rather than
manipulating CSS `display` properties directly — a small nicety: `hidden` is a native HTML attribute
Elements already support, and toggling it also correctly interacts with accessibility tooling and
`:not([hidden])` CSS selectors without needing custom class names for show/hide state.

The final step, rendering the histogram:
```javascript
histogramImg.src = `data:image/png;base64,${data.histogram_base64}`;
```
A `data:` URL lets you embed a file's content directly inside a URL string instead of pointing to a
separate resource to fetch — `data:image/png;base64,<the base64 text>` tells the browser "decode this
inline base64 text as PNG image bytes and treat it exactly like an image loaded from any other URL."
Setting an `<img>` tag's `src` to a `data:` URL is what completes the round trip: the PNG bytes that
`backend/visualization.py` rendered in-memory, base64-encoded in the Flask route, arrive in the
browser as plain JSON text, and get decoded back into a displayed image with a single attribute
assignment — no separate image request ever leaves the browser.

### 2.4 How to summarize this whole boundary in one interview-ready sentence

*"The frontend is a single Flask app that serves a Jinja2-templated HTML page and two JSON API
endpoints from the same process; when a request comes in, the route handler calls the backend's
Python functions directly — database query, distribution construction, Monte Carlo simulation, risk
metrics, and chart rendering — as ordinary in-process function calls, then serializes the results to
JSON (with the chart image base64-encoded inline) for the browser's vanilla-JavaScript `fetch()` calls
to render."*
