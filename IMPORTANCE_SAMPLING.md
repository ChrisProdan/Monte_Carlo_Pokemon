# Importance Sampling & Esscher Tilting — Deep Dive

This document is a standalone tutorial on importance sampling and Esscher (exponential) tilting: the
general theory first, worked examples by hand, then how every piece maps onto this project's actual
code. `ARCHITECTURE.md` covers *what* the code does and why; this document is for building the
underlying mathematical fluency so you could derive it, not just describe it, in an interview. The
"Further reading" section at the end is curated and link-verified to take you from here to what was
earlier called **Tier 2 depth** — solid, defensible, can-derive-it-yourself understanding.

---

## Part 1: Why plain Monte Carlo fails for rare events

Suppose you want to estimate `θ = E_p[f(X)]`, where `X` is drawn from some distribution `p`. The
ordinary Monte Carlo estimator draws `X₁, ..., Xₙ ~ p` independently and averages:

```
θ̂ = (1/n) Σᵢ f(Xᵢ)
```

This is unbiased and consistent (converges to the true value as `n → ∞`), and its standard error shrinks
like `1/√n` — the single most important fact about Monte Carlo, and also its single biggest limitation.
Cutting the error in half requires **4x** the samples, no matter what you're estimating.

That's fine when `f(X)` is "active" often — most draws contribute something informative. It becomes
brutal when the quantity you care about is tied to a **rare event**: say `f(X) = 𝟙{X ∈ A}` for some
event `A` with true probability `p(A) = 0.0005`. To get even 100 samples landing in `A` — the bare
minimum for the sample average to mean anything — you need on the order of 200,000 trials. To halve the
*relative* error of that estimate, you need 4x again. Rare-event probabilities and the conditional
expectations built on top of them (like "expected profit given you landed in a top 0.1% outcome") are
exactly the situation where plain Monte Carlo becomes the bottleneck.

## Part 2: The importance sampling identity

The trick is a change of measure — rewriting the same expectation as an expectation under a *different*
distribution `q`, chosen so the event you care about isn't rare under `q` anymore:

```
θ = E_p[f(X)] = Σₓ f(x) p(x)
              = Σₓ f(x) · [p(x)/q(x)] · q(x)      (multiply and divide by q(x))
              = E_q[ f(X) · w(X) ]                 where w(x) = p(x)/q(x)
```

`w(x)` is the **importance weight** (also called the likelihood ratio, or Radon-Nikodym derivative in
the continuous/measure-theoretic version of this identity). The estimator becomes:

```
θ̂ = (1/n) Σᵢ f(Xᵢ) · w(Xᵢ),    Xᵢ ~ q
```

**Why this is still unbiased**, spelled out: `E_q[f(X)·w(X)] = Σₓ f(x)w(x)q(x) = Σₓ f(x)[p(x)/q(x)]q(x)
= Σₓ f(x)p(x) = E_p[f(X)]` — the `q(x)` in the weight and the `q(x)` from sampling under `q` cancel
exactly, leaving the original expectation under `p`. This holds for **any** `q`, as long as one hard
condition is met: **`q(x) > 0` everywhere `p(x) > 0`**. If `q` ever assigns zero probability to
something `p` says can happen, that outcome can never be drawn under `q`, so its contribution silently
vanishes — not extra noise, an actual bias.

**Why the choice of `q` matters even though every valid `q` is unbiased**: unbiasedness says the
estimator is *centered* on the right answer; it says nothing about its *spread*. The right `q` can make
the variance of `θ̂` enormously smaller than plain Monte Carlo's, for the same `n`. The wrong `q` can make
it *worse* — even infinite, in continuous settings, if `q`'s tails decay faster than `p`'s (a small
number of huge weights can dominate and blow up the variance). Importance sampling isn't automatically a
free upgrade; it's a deliberate trade of "which region gets sampling effort" that has to be aimed
correctly.

**The theoretical "zero-variance" `q`**, useful for intuition even though it can't be built directly: for
estimating `P(X ∈ A)`, the `q` that gives an estimator with *zero* variance is `q*(x) = p(x)·𝟙{x∈A} /
p(A)` — sample *only* from inside `A`, in the exact relative proportions `p` already assigns there.
Every draw under `q*` lands in `A` and gets exactly the right weight to reconstruct `p(A)` precisely.
The catch: constructing `q*` requires already knowing `p(A)` — the very thing being estimated. This is
why real importance sampling schemes are *approximations* to this ideal (get `q` reasonably close to
concentrating where `A` lives), not the ideal itself.

## Part 3: Effective Sample Size (ESS)

Since a badly-chosen `q` can produce a technically-unbiased but wildly noisy estimate, you need a way to
tell how much you can trust a given importance-sampled result. Effective Sample Size answers "how many
plain-Monte-Carlo samples is this weighted sample *worth*":

```
ESS = (Σᵢ wᵢ)² / Σᵢ wᵢ²
```

**Intuition for the formula**: if every weight is equal (say, all equal to `c`), `ESS = (nc)²/(nc²) = n`
exactly — no degradation, because equal weights mean `q` didn't distort anything relative to a plain
average. If one weight `w₁` is enormous compared to the rest, `ESS ≈ w₁²/w₁² = 1` — the estimate is
effectively being driven by a single sample, no matter how large `n` is. ESS interpolates continuously
between these extremes and is the standard diagnostic to report alongside *any* importance-sampled
number — never trust one without checking it.

## Part 4: Rare events for SUMS of independent draws — where it gets genuinely harder

Everything above covers importance sampling for a single random draw. This project's actual problem is
harder: `Π_y`, the total profit from `y` **independent** packs, is a *sum* of `y` i.i.d. draws, and the
rare event of interest is about that sum, not about any one draw.

**Two structurally different ways a sum can land in its tail:**
- **"One big jump"**: a single term in the sum is extreme; the rest are ordinary. This dominates when
  the underlying per-draw distribution is **heavy-tailed** (formally, subexponential) — extreme values
  are common enough, relative to the tail probability you're asking about, that one of them alone
  explains the whole event.
- **"Many small excesses add up"**: no single term needs to be extreme; the *whole sum* needs to be a
  bit better than typical, spread across many terms. This is the standard picture — via **Cramér's
  theorem** — for sums of i.i.d. variables whose distribution has a well-behaved moment generating
  function (this project's card payoffs are literally bounded, so this always applies here, no matter
  how "heavy" the payoff distribution feels informally).

**Why this project's naive first attempt broke at large `y`**: an earlier version of this module built
`q` by boosting a hand-picked subset of high-value outcomes — implicitly assuming the "one big jump"
picture. That's a reasonable approximation for small `y` (and it validated well there). But for `y=1000`,
reaching a demanding threshold required roughly **47 simultaneous hits**, not one — squarely in "many
small excesses" territory, which a tilt aimed at boosting one jump structurally cannot produce.
Compounding a per-step tilt identically across many independent draws also causes **weight
degeneracy**: a mismatch that's small for one draw compounds *multiplicatively* across `y` of them,
so the weight for a whole trial can vary by many orders of magnitude between trials that only differ by
a couple of hits — collapsing ESS toward 1 regardless of how the tilt strength is tuned. See
`ARCHITECTURE.md` §1.8 for the exact numbers this project measured live.

**The large-deviations picture, briefly** (this is the part worth reading the "Further reading" section
for, to go beyond a qualitative feel): Cramér's theorem says that for `S_n = X₁ + ... + Xₙ` (i.i.d.,
mean `μ`), and `a > μ`:

```
(1/n) log P(S_n/n ≥ a)  →  −I(a)     as n → ∞
```

where `I(a) = sup_θ [θa − log M(θ)]` is the **rate function** (the Legendre transform of the log moment
generating function). The probability decays *exponentially* in `n` at rate `I(a)`. The `θ` that
achieves the supremum in that formula is *exactly* the Esscher tilting parameter that makes `a` the
tilted distribution's typical (mean) outcome — which is precisely why Esscher tilting, not an ad hoc
subset boost, is the theoretically motivated choice for this kind of problem, and why it correctly
handles both the "one big jump" and "many small excesses" regimes without needing to know in advance
which one applies.

## Part 5: Esscher (exponential) tilting, derived

**Definition.** For a real-valued outcome with distribution `p` and moment generating function
`M(θ) = E_p[e^(θX)] = Σₓ p(x)·e^(θx)` (assumed finite in a neighborhood of 0 — always true here, since
payoffs are bounded), the **Esscher-tilted** (exponentially tilted) distribution is:

```
q_θ(x) = p(x) · e^(θx) / M(θ)
```

This is a valid probability distribution for any `θ` where `M(θ)` is finite: it's non-negative, and it
sums to 1 by construction (`M(θ)` is exactly the normalizing constant). Every outcome with `p(x) > 0`
also has `q_θ(x) > 0` (since `e^(θx) > 0` always) — the hard requirement on any valid proposal
distribution is satisfied automatically, with no need for a separate check.

**Why increasing `θ` shifts the mean upward.** Define the cumulant generating function `K(θ) = log
M(θ)`. Its first derivative is:

```
K'(θ) = M'(θ)/M(θ) = [Σₓ x·p(x)·e^(θx)] / M(θ) = Σₓ x · q_θ(x) = E_{q_θ}[X]
```

— the derivative of `log M` at `θ` is *exactly* the mean of the tilted distribution at that `θ`. Its
second derivative, `K''(θ)`, equals `Var_{q_θ}[X]`, which is always `≥ 0` — so `K'(θ)` (the tilted mean)
is a monotonically non-decreasing function of `θ`, strictly increasing wherever the tilted distribution
has any variance at all. This is *why* solving for `θ` by bisection is guaranteed to converge: the
target function is monotonic, so there's at most one `θ` achieving any given target mean, and a
standard bracket-and-bisect search will find it.

**Solving for `θ`** (the "saddlepoint equation"): given a target mean `a`, solve `K'(θ) = a`, i.e.
`E_{q_θ}[X] = a`. There's rarely a closed form for a general finite discrete distribution, so this
project solves it numerically via bisection (see `backend/importance_sampling.py::solve_tilt_theta`) —
compute the tilted mean at a trial `θ`, compare to the target, and halve the search interval
accordingly. This is standard practice; closed forms only exist for a few special distributions.

**The weight formula for a sum of `y` i.i.d. tilted draws** — this is the derivation behind
`simulate_y_pack_is_trials`'s single closed-form weight, worth being able to reproduce on a whiteboard:

```
Joint density under q_θ of (x₁, ..., x_y), each independently drawn from q_θ:
  ∏ᵢ q_θ(xᵢ) = ∏ᵢ [p(xᵢ)·e^(θxᵢ)/M(θ)] = [∏ᵢ p(xᵢ)] · e^(θ·Σxᵢ) / M(θ)^y

Joint density under the real distribution p (same y draws):
  ∏ᵢ p(xᵢ)

Importance weight = (joint p) / (joint q_θ) = M(θ)^y · e^(−θ·Σxᵢ)

So:  log(weight) = y·log M(θ) − θ·(total payoff)
```

Every term in the joint `p` cancels against the corresponding term in joint `q_θ`, leaving a clean
expression depending only on the **total** payoff across the `y` draws — not on the individual draws
that produced it. This is the entire reason Esscher tilting is more elegant here than the earlier
subset-boosting approach, which had no such simplification and had to track every individual draw's
`log(p) − log(q)` separately and sum them.

## Part 6: A full worked example, by hand

Take a toy 2-outcome distribution: `p(WORTHLESS) = 0.999`, payoff `0`; `p(BIG WIN) = 0.001`, payoff
`1000`. (This is exactly this project's `rare_win_distribution` test fixture.)

**1. Moment generating function:**
```
M(θ) = 0.999·e^(θ·0) + 0.001·e^(θ·1000) = 0.999 + 0.001·e^(1000θ)
```

**2. Tilted mean, as a function of θ:**
```
M'(θ) = 0.001·1000·e^(1000θ) = e^(1000θ)
K'(θ) = M'(θ)/M(θ) = e^(1000θ) / (0.999 + 0.001·e^(1000θ))
       = 1000·e^(1000θ) / (999 + e^(1000θ))          (multiplying num. and denom. by 1000)
```
At `θ=0`: `K'(0) = 1000·1/(999+1) = 1000/1000 = 1.0` — recovers the natural mean payoff exactly
(`0.999·0 + 0.001·1000 = 1.0`). ✓

**3. Solve for a target, e.g. mean payoff = 200** (want `q(BIG WIN)` such that mean = 200, i.e.
`q(BIG WIN)·1000 = 200`, so `q(BIG WIN) = 0.2`):
```
q_θ(BIG WIN) = 0.001·e^(1000θ) / (0.999 + 0.001·e^(1000θ)) = 0.2
⟹ 0.001·e^(1000θ) = 0.2·(0.999 + 0.001·e^(1000θ))
⟹ 0.001·e^(1000θ)·(1 − 0.2) = 0.2·0.999
⟹ e^(1000θ) = (0.2·0.999) / (0.0008) = 249.75
⟹ 1000θ = ln(249.75) ≈ 5.520
⟹ θ ≈ 0.00552
```
This matches (to within numerical solver tolerance) what `solve_tilt_theta` on this exact fixture with
`target_mean_payoff=200` finds — a good way to sanity-check the code's numerical answer against a
hand-solved algebraic one.

**4. Weight for a single-pack trial that draws BIG WIN under this q:**
```
e^(1000·0.00552) = e^5.52 ≈ 249.6
M(θ) ≈ 0.999 + 0.001·249.6 = 1.2486
weight = M(θ)^1 · e^(−θ·1000) = 1.2486 · e^(−5.52) = 1.2486 · 0.00401 ≈ 0.00501
```
Sanity check: for `y=1`, this closed-form weight must reduce to the plain single-outcome ratio
`p(BIG WIN)/q_θ(BIG WIN) = 0.001/0.2 = 0.005` (the algebra in Part 5 shows the closed form is exactly
`p(x)/q_θ(x)` when `y=1`) — and indeed `0.00501 ≈ 0.005`, matching to within the rounding used in this
hand calculation. This is the concrete meaning of "self-normalized importance sampling": BIG WIN now
happens 200x more often under `q` (20% vs. its true 0.1%), so each occurrence is correctly down-weighted
by that same factor of 200 (`weight = 1/200 = 0.005`), and the weighted average still recovers the true
0.1% despite `q` having pushed BIG WIN's frequency far above reality.

## Part 7: How this maps onto the actual code

| Theory | Code |
|---|---|
| `p(x)`, the real distribution | `SetDistribution.probs` (`backend/distribution.py`) |
| `M(θ) = Σ p(x)e^(θx)`, computed in log-space to avoid overflow | `_log_sum_exp`, `_tilt_stats` |
| `K'(θ) = E_{q_θ}[X]`, monotonic in θ | `_tilt_stats`'s `mean_payoff` return value |
| Solving `K'(θ) = target` by bisection | `solve_tilt_theta` |
| `q_θ(x) = p(x)e^(θx)/M(θ)` | `build_esscher_tilted_distribution` |
| `log(weight) = y·log M(θ) − θ·(total payoff)` | `simulate_y_pack_is_trials` |
| `ESS = (Σw)²/Σw²` | `effective_sample_size` |
| Self-normalized `P(A)` and `E[X\|A]` estimators | `weighted_tail_probability`, `weighted_conditional_expectation` |
| The zero-variance oracle needing to already know the answer | Why the threshold is *located* via a cheap plain-MC quantile first (`estimate_profit_threshold`), then *refined* precisely via importance sampling — you can't build the ideal `q*` without knowing the target region first |
| "One big jump" tilting (subset-boosting) failing at large `y` | The rejected first implementation, and the `y=1000` weight-degeneracy failure documented in `ARCHITECTURE.md` §1.8 |
| A distribution whose mean equals its own max forcing `q` toward a point mass | The `y=1` edge case and `MAX_SINGLE_OUTCOME_TILT_PROB` safety cap |

## Part 8: Pitfalls checklist (all found live in this project, not hypothetical)

1. **Discrete distributions and tail boundaries.** For a continuous distribution, `P(X > c)` and
   `P(X ≥ c)` are the same. For a discrete one, if `c` is itself an actual outcome with real probability
   mass, they can differ completely — including one being exactly zero when the other isn't. Always use
   `≥`/`≤` (inclusive) when the threshold itself came from an observed data point (e.g. a quantile of a
   sample), not an arbitrary continuous cutoff.
2. **A target that equals (or nearly equals) the maximum possible value of a single draw** forces the
   tilt toward a degenerate point mass — mathematically correct in the limit, but useless in a finite
   sample, since you get zero "miss" samples to correctly weigh against the "hit" samples. Cap how
   concentrated any single outcome is allowed to become.
3. **Naive per-step tilting for a sum of many draws compounds multiplicatively.** A tilt that works for
   one draw does not automatically work for a sum of a thousand of them — check ESS at every scale you
   care about, don't assume it generalizes.
4. **Always check ESS before trusting a weighted estimate.** A technically-unbiased estimator with
   ESS ≈ 1 is not a reliable number, even though the math "works."
5. **Always cross-check against plain Monte Carlo on a case cheap enough to brute-force**, whenever
   possible — an importance-sampled estimate that disagrees with a trustworthy plain-MC estimate on an
   easy case means something in the tilt or weighting is wrong, before you ever trust it on a hard case.

---

## Further reading

Ordered roughly from most approachable to most theoretical. Every link below was checked live before
being included here.

**Start here:**
- **Art B. Owen, [*Monte Carlo Theory, Methods and Examples*](https://artowen.su.domains/mc/), Chapter
  on [Importance Sampling](https://artowen.su.domains/mc/Ch-var-is.pdf) (free, online book).** The most
  approachable rigorous treatment of importance sampling itself — derivations, variance analysis, and
  concrete worked examples (including a rare-event scheduling example very similar in spirit to this
  project's "how often does a rare big outcome happen"). Read this first.
- **Wikipedia: [Importance sampling](https://en.wikipedia.org/wiki/Importance_sampling)** and
  **[Exponential tilting](https://en.wikipedia.org/wiki/Exponential_tilting)** — good for definitions,
  the Esscher-transform naming history (it comes from actuarial science — Fredrik Esscher, 1932, risk
  theory — before being adopted for rare-event simulation by David Siegmund), and a compact list of
  which standard distributions tilt into which other standard distributions.

**Tier 2 — the tutorial written specifically for this project's kind of problem:**
- **Anand Deo & Karthyek Murthy, ["Importance Sampling for Minimization of Tail Risks: A
  Tutorial"](https://arxiv.org/abs/2307.04676) (arXiv, 2023).** Directly about using importance sampling
  for tail-risk quantities like CVaR — a tutorial, not a terse research paper, and closely matches this
  project's own framing (though remember: apply its lessons to the *upper* tail here, not VaR/CVaR,
  per the scoping correction in `ARCHITECTURE.md`). This is the single best next read after Owen's
  chapter.
- **Cheng-Der Fuh, Huei-Wen Teng & Ren-Her Wang, ["Efficient Importance Sampling for Rare Event
  Simulation with Applications"](https://arxiv.org/abs/1302.0583) (arXiv).** Derives optimal exponential
  tilting parameters when a moment generating function exists (exactly this project's setup), and
  explicitly demonstrates the technique on **Value-at-Risk calculations** — worth reading for how a
  finance-flavored application of this exact math is usually written up.
- **[Stanford tutorial on Rare Event Simulation
  Techniques](https://web.stanford.edu/~jblanche/papers/Conf_Tutorial_RES_WSC_11.pdf)** (Winter
  Simulation Conference tutorial slides/notes) — a broader survey of rare-event simulation methods
  including exponential tilting, pitched at a practitioner level.

**The historical and canonical references** (worth knowing by name even if you only skim them):
- **David Siegmund, ["Importance Sampling in the Monte Carlo Study of Sequential
  Tests"](https://projecteuclid.org/journals/annals-of-statistics/volume-4/issue-4/Importance-Sampling-in-the-Monte-Carlo-Study-of-Sequential-Tests/10.1214/aos/1176343541.full),
  *Annals of Statistics* 4(4), 1976.** The paper that established exponential tilting as *the* technique
  for importance sampling on sums of i.i.d. variables (random walks) — this is the paper this whole
  project's approach traces back to.
- **Paul Glasserman, *Monte Carlo Methods in Financial Engineering* (Springer, 2003)**, chapter on
  variance reduction / importance sampling. The standard quant-finance-flavored textbook treatment,
  including VaR estimation via importance sampling specifically — extremely relevant framing for a
  risk-analytics-oriented resume project, though it's a paid textbook (check a university library) with
  no reliable free full-text link.
- **Søren Asmussen & Peter Glynn, *Stochastic Simulation: Algorithms and Analysis* (Springer, 2007)**,
  chapters on rare-event simulation and large deviations. The standard graduate-level rigorous treatment
  connecting exponential tilting to Cramér's theorem and large-deviations rate functions in full
  mathematical detail — this is where Tier 3 (fully rigorous, provable) understanding lives, if you ever
  want to go that deep. Also a paid textbook, standard at university libraries.

**For the large-deviations / Cramér's theorem background specifically** (optional — only if the "why is
this the asymptotically correct tilt" question comes up and you want to go beyond the qualitative
answer):
- Wikipedia: [Large deviations theory](https://en.wikipedia.org/wiki/Large_deviations_theory) and
  [Cramér's theorem](https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_theorem) — both give the rate
  function definition and the Legendre-transform connection to the moment generating function used in
  Part 4 above.
