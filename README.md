# Pokemon TCG Pack-Opening Monte Carlo Simulation

A Monte Carlo simulation of the profit/loss from opening Pokemon TCG booster packs, reported in the
language of quantitative risk analytics: mean/standard deviation, Value at Risk (VaR), Conditional
Value at Risk (CVaR), and breakeven probability. Built as a resume piece for quant/risk analytics
roles — every non-trivial function and formula in the code is commented with *why* it's built that
way, not just what it does.

Simulates all Scarlet & Violet and Mega Evolution sets, backed by real pull-rate and card-price data
in a Postgres database, with a small Flask frontend for running it interactively.

## How it works, in one paragraph

For a chosen set, the simulation builds a single flat probability distribution over every individual
card that can be pulled (not an averaged rarity-tier value), draws from it directly for each simulated
pack, and reports the resulting profit distribution's risk statistics. See "Modeling decisions" below
for exactly what "every individual card" means given what the database actually contains.

## Architecture

There is **one running process**: a Flask app. The `backend/` and `frontend/` folders are a
code-organization split, not a client/server split — `frontend/app.py` is the Flask application, and
it calls straight into the `backend/` Python modules as ordinary in-process function calls (no network
hop between them). Request flow for "open 10 packs of SV01":

```
browser → Flask route /simulate → backend.db (query Postgres)
                                 → backend.distribution (build the flat probability array)
                                 → backend.simulation (draw simulated trials)
                                 → backend.risk_metrics (VaR/CVaR/breakeven/mean/SD)
                                 → backend.visualization (render histogram PNG)
                                 → Flask serializes everything to JSON → browser
```

```
backend/
├── config.py              # loads .env into a Settings dataclass
├── db.py                  # all SQL — the only module that talks to Postgres
├── constants.py           # rarity-column mapping, y values, alphas, trial count, RNG seed
├── distribution.py        # builds the flat per-outcome probability distribution for a set
├── simulation.py          # vectorized Monte Carlo draws (numpy)
├── risk_metrics.py        # mean/SD/VaR/CVaR/breakeven probability + a sqrt(y) sanity check
├── visualization.py       # matplotlib histogram -> PNG bytes
└── importance_sampling.py # Esscher-tilted importance sampling for the upper (big-win) tail

frontend/
├── app.py                 # the Flask app: routes + static/template serving
├── templates/index.html   # the single page (set entry -> pack-count entry -> results)
└── static/                # app.js (fetch calls, DOM updates) and style.css

tests/                     # pytest unit tests (synthetic data, no live DB required)
```

## Data source

Pull-rate and card-price data live in a Postgres database (`Pokemon_Pricing_Information`), in two
tables:

- **`groupdata`** — one row per set (`groupid`, `setname`, `releasedate`), plus nine nullable
  percentage-rate columns for the pack's "hit slot": `dr_rates`, `ur_rates`, `ir_rates`, `sir_rates`,
  `hr_rates`, `sr_rates`, `sur_rates`, `mhr_rates`, `mar_rates`. A rate of e.g. `13.76` means 13.76% of
  packs contain that rarity. `NULL` means that rarity doesn't exist in that set.
- **`carddata`** — one row per TCGPlayer *product* for a set, not purely individual cards: it also
  includes sealed products (booster packs, ETBs, bundles — `rarity IS NULL`) and digital bonus codes
  (`rarity = 'Code Card'`) mixed in with real card singles.

"In-scope" sets — the ones the app will simulate — are `groupdata` rows where the set name starts with
`SV` or `ME` and at least one of the nine rate columns is populated. This currently resolves to exactly
22 sets (SV01–SV10 plus special SV releases, ME01–ME05 plus Ascended Heroes) and will pick up new sets
automatically once their rate data is populated, with no code change.

## Modeling decisions

These were deliberately made and confirmed rather than left implicit — each is also called out as a
comment at its point of implementation in the code.

- **Individual-card-level simulation, not rarity-tier averaging.** A rarity slot might contain a $5
  card and a $400 card at equal odds; averaging them first would give the correct mean but destroy the
  tail events VaR/CVaR are meant to capture. The simulation builds one flat probability array over
  every individual card and draws from it directly.
- **Only 9 named rarities carry resale value**: Double Rare, Ultra Rare, Illustration Rare, Special
  Illustration Rare, Hyper Rare, Shiny Rare, Shiny Ultra Rare, Mega Hyper Rare, Mega Attack Rare — the
  ones with a matching `groupdata` rate column. Every other card a pack could contain (Common,
  Uncommon, plain "Rare", Code Card, energy, etc.) is folded into a single implicit **worthless
  bucket** worth $0, carrying whatever probability mass the nine named rarities don't account for. The
  database has pull-rate data for the "hit slot" only, not for how many commons/uncommons appear per
  pack, so this is the boundary of what can be modeled without inventing unsupported assumptions.
- **Pack cost** is looked up from `carddata` itself — the plain "Booster Pack" product row for that
  set's `groupid` (excluding code cards, art bundles, sleeved variants, and multi-packs).
- **Independence between packs.** Packs are treated as independent draws with replacement, even across
  packs opened from the same physical box. This is a reasonable approximation across separate boxes,
  but not exactly correct within one box (a real box samples a finite population without replacement).
  Stated explicitly as a simplification, not silently assumed.
- **Fixed default RNG seed** (`backend/constants.py`) — results are reproducible run-to-run by default,
  which matters for hand-verification and for being able to explain a specific number later. Pass a
  different seed for an independent run.
- **Default trial count of 100,000** per scenario (`N_TRIALS_DEFAULT`) — a deliberate balance between
  precision and keeping a Flask request (especially at y=1000, which draws `n_trials × 1000` values)
  responsive; a larger value (500,000) was tried and rejected after measuring its ~12GB transient
  memory footprint at y=1000. Raise it for higher-precision offline figures.
- **Pack count is a free-form input, not five fixed buttons.** The frontend accepts any whole number
  from 1–1000 (enforced both client- and server-side); every backend function already worked for
  arbitrary `y`, so this just exposes that in the UI.

## Risk metrics

For a batch of M simulated total-profit trials Π_y, sorted ascending:

- **Total cost** (`y × pack_cost`) — not itself a risk statistic, but reported alongside the others so
  VaR/CVaR can be read directly against what was actually spent (the frontend shows VaR as a % of this)
  instead of requiring the reader to recompute it by hand.
- **Mean / standard deviation** of Π_y.
- **VaR_α**: with `k = ceil((1-α) × M)`, `VaR = -Π₍ₖ₎` (the k-th worst outcome), reported as a positive
  loss magnitude. If that boundary trial was actually a gain, VaR is **non-binding** at that confidence
  level — reported as 0 with an explicit flag, never a fabricated number.
- **CVaR_α**: the average of the worst k outcomes, always reported (a genuine tail average even when
  VaR is non-binding).
- **Breakeven probability**: `P(Π_y > 0)`.
- A **sqrt(y) sanity check**: under the independence assumption, `SD(Π_y) ≈ sqrt(y) × SD(Π_1)`; the
  code flags (doesn't silently correct) meaningful drift from this relationship.

Verified against the live database: hand-calculated EV matches simulated EV to within $0.005, sqrt(y)
scaling holds across y = 5/10/100/1000, and breakeven probability correctly trends toward 0 as y grows
for sets with negative per-pack expected value (which is the expected, non-buggy behavior for sealed
product priced with margin).

## Importance sampling (the upper, "big win" tail)

Its target was corrected after an initial mis-scoping: this profit distribution is bottom-heavy (most
packs land at or near the maximum loss, since "worthless" is the single most likely outcome), so the
worst 5%/1% of outcomes used for VaR/CVaR above are actually the *dense*, common part of the
distribution — plain Monte Carlo already samples that region heavily, and importance sampling would not
sharpen those numbers at any confidence level. The genuinely rare event here is the *opposite* tail: a
large profit, which only happens on a low-probability, high-value pull.

`backend/importance_sampling.py` estimates, for the same `y` the user picks:

- **`P(Π_y ≥ threshold)`** — the odds of landing in the top `TAIL_PROBABILITY` (0.1%) of outcomes.
- **`E[Π_y | Π_y ≥ threshold]`** — the expected profit conditional on landing there (deliberately called
  "expected profit conditional on a top-tier outcome," never "CVaR," which is conventionally a
  downside-risk term).

**Threshold**: dynamic, not hardcoded — the (1 − 0.1%) quantile of the *same* plain Monte Carlo batch
already computed for the regular risk metrics (no separate "pilot" simulation).

**Method**: Esscher (exponential) tilting, `q(x) ∝ p(x)·e^(θ·payoff(x))`, solved via bisection so the
tilted distribution's mean payoff (summed over `y` packs) lands on the threshold. An earlier version of
this module boosted a hand-picked subset of high-value cards instead — that works for small `y`, but
breaks down for large `y` (reaching a demanding threshold at `y=1000` needs *many* packs to each do
slightly better than average, not one lucky pull, since every card price is bounded; a subset-boost tilt
applied identically across hundreds of draws compounds multiplicatively and collapses Effective Sample
Size — a well-known failure mode called weight degeneracy). Esscher tilting reweights every outcome
smoothly instead, and the per-trial weight collapses to one clean closed form:
`log(weight) = y·log M(θ) − θ·(total payoff)`.

**A safety cap** (`MAX_SINGLE_OUTCOME_TILT_PROB`) stops the tilt from concentrating too much probability
on the single highest-payoff outcome — found necessary live at `y=1`, where the target coincided almost
exactly with that one card's own value, and "match the mean exactly" would otherwise demand a
near-100%-certain tilt that leaves zero samples of anything else in a finite trial batch.

Verified live: at `y=1`, the importance-sampled estimate (`0.00301`) matches a 2-million-trial plain
Monte Carlo cross-check (`0.00313`) closely, with 43% Effective Sample Size. At `y=1000` — where the
old subset-boost approach either crashed or returned nonsense (`P=1.0`, `ESS≈1`) — this approach returns
a sensible, if noisier (ESS ~0.4%), estimate instead of a broken one. ESS is reported alongside every
estimate so its reliability is never hidden.

## Setup

1. Install dependencies:
   ```
   py -m pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and fill in your Postgres credentials:
   ```
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=Pokemon_Pricing_Information
   DB_USER=postgres
   DB_PASSWORD=<your password>
   ```

## Running the frontend

```
py -m flask --app frontend.app run --debug
```

Open `http://127.0.0.1:5000`, enter an exact set name (e.g. `SV01: Scarlet & Violet Base Set`), enter a
pack count from 1–1000, and view the metrics table (including the importance-sampled top-tier odds) and
histogram. A `y=1000` request takes roughly 10–15 seconds — it's running two Monte Carlo batches
(regular + importance-sampled) plus a numerical root-solve, not an instant lookup.

## Running the tests

```
py -m pytest tests/ -v
```

Unit tests use small, hand-computable synthetic data (see `tests/conftest.py`) and don't require a
live database connection.
