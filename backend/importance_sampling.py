"""Importance sampling for estimating rare, large-profit outcomes, via
Esscher (exponential) tilting.

What this module is NOT for, and why (a correction made to an earlier draft
of this project's design): the simulated profit distribution is
bottom-heavy -- "WORTHLESS" is the single most likely outcome for any pack,
so the large majority of simulated trials land at or near the maximum
possible loss (the full pack cost). That means the LOWER tail -- the worst
5% or worst 1% of outcomes, which is what risk_metrics.compute_risk_metrics
uses for VaR_95/VaR_99/CVaR_95/CVaR_99 -- is actually the dense, common part
of this distribution, not a rare one. Plain Monte Carlo already places
plenty of samples there by default, so importance sampling would not
sharpen those loss-side VaR/CVaR numbers at all, at any confidence level.

The genuinely rare event in this distribution is the OPPOSITE tail: a large
profit, which only happens when a low-probability, high-value card is
pulled. That event is exactly what plain Monte Carlo struggles to estimate
precisely without a very large trial count, and exactly what importance
sampling is good at. So this module estimates:

  1. P(Pi_y >= threshold) for a large profit threshold, and
  2. E[Pi_y | Pi_y >= threshold] -- the expected profit conditional on
     landing in that upper tail.

Naming note: quantity (2) is deliberately NOT called "CVaR," even though
it's computed the same way a conditional tail expectation is. CVaR is
conventionally a downside-risk term -- expected loss given that a loss tail
was entered -- and reusing it for an upside quantity would misname what's
being reported. It's called "expected profit conditional on a top-tier
outcome" throughout this module instead.

The threshold itself is dynamic, not hardcoded: it's the (1 - p_target)
quantile of a batch of ordinary, already-computed Monte Carlo samples
(see estimate_profit_threshold) -- specifically the SAME batch
risk_metrics.py already draws for the regular VaR/CVaR/mean/SD numbers, not
a separate dedicated simulation run just to locate it.

Why Esscher tilting specifically, rather than boosting a chosen subset of
outcomes (the first version of this module did that): boosting a handful of
"target" cards works fine for small y, but breaks down for large y. Reaching
a demanding profit threshold at large y generally requires MANY packs to
each do a little better than average, not one single card landing --
because every card price here is bounded, a single lucky pull can only ever
close a bounded gap, while the gap between "everything worthless" and a
demanding threshold grows with y. A tilt that boosts specific outcomes and
is applied identically across many independent draws compounds per-draw
mismatches multiplicatively, which blows up importance-weight variance (a
classic failure mode called weight degeneracy) long before it can reach
that kind of threshold. Esscher tilting instead reweights EVERY outcome
smoothly by e^(theta * payoff) for a single scalar theta, chosen so the
WHOLE tilted distribution's mean shifts to make the target typical -- this
is the standard, asymptotically-motivated technique (related to Cramer's
theorem / large deviations theory) for estimating rare events involving
sums of many independent draws, and it naturally handles both "one big
jump" (small y) and "many small excesses add up" (large y) without needing
to hand-pick which outcomes matter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.constants import (
    MAX_SINGLE_OUTCOME_TILT_PROB,
    N_IS_TRIALS_DEFAULT,
    TAIL_PROBABILITY,
    THETA_SEARCH_MAX,
    THETA_SOLVE_ITERATIONS,
)
from backend.distribution import SetDistribution
from backend.simulation import draw_outcome_indices


def _log_sum_exp(log_terms: np.ndarray) -> float:
    """Numerically stable log(sum(exp(log_terms))).

    Subtracting the maximum term before exponentiating (and adding it back
    at the end) keeps every exponentiated value <= 1, avoiding the overflow
    that computing exp() of large log-terms directly could cause. Standard
    technique, implemented by hand here rather than pulling in a dependency
    (e.g. scipy.special.logsumexp) for a four-line function.
    """
    m = np.max(log_terms)
    return float(m + np.log(np.sum(np.exp(log_terms - m))))


def _tilt_stats(dist: SetDistribution, theta: float) -> tuple[float, float, float]:
    """Computes (mean payoff, log of the MGF M(theta), probability of the
    single highest-payoff outcome) under the theta-tilted distribution.

    M(theta) = sum_x p(x) * e^(theta * payoff(x)) -- the moment generating
    function of the payoff random variable evaluated at theta. The tilted
    distribution is q_theta(x) = p(x) * e^(theta * payoff(x)) / M(theta),
    and its mean payoff, E_qtheta[payoff], is exactly what needs to hit a
    target value when solving for theta (see solve_tilt_theta). All three
    values come from the same log-space sum, so computing them together
    avoids repeating it.

    The third value is specifically the probability of the single
    HIGHEST-PAYOFF outcome under q -- not the max probability over ALL
    outcomes. That distinction matters: the naturally most probable outcome
    in this project's distributions is always WORTHLESS (often 60-99%
    probability even at theta=0), so a cap on "the max probability over all
    outcomes" would be triggered by the ordinary, untilted distribution
    itself and never allow any tilting at all. What actually needs
    guarding against (see solve_tilt_theta's docstring) is the highest-PAYOFF
    outcome specifically becoming artificially dominant as theta grows.

    log(p) can be -inf for a zero-probability outcome (e.g. a rate column
    that rounds to exactly 0) -- that's mathematically valid (that outcome
    correctly contributes nothing) and is suppressed here rather than
    warned about, since it's expected, not an error condition.
    """
    payoffs = dist.payoffs
    with np.errstate(divide="ignore"):
        log_p = np.log(dist.probs)
    log_terms = log_p + theta * payoffs
    log_mgf = _log_sum_exp(log_terms)

    # log(q_theta(x)) = log_terms - log_mgf; mean payoff is the weighted
    # sum of payoffs under q_theta, computed directly rather than via a
    # symbolic derivative of M -- simpler and just as exact for a finite,
    # discrete outcome space like this one.
    log_q = log_terms - log_mgf
    q = np.exp(log_q)
    mean_payoff = float(np.sum(q * payoffs))
    top_payoff_prob = float(q[np.argmax(payoffs)])
    return mean_payoff, log_mgf, top_payoff_prob


def solve_tilt_theta(dist: SetDistribution, target_mean_payoff: float) -> float:
    """Finds theta such that the theta-tilted distribution's mean payoff equals target_mean_payoff.

    Mean payoff under the tilt is a strictly increasing function of theta
    (increasing theta shifts more probability toward higher-payoff
    outcomes), ranging from the real distribution's natural mean payoff at
    theta=0 up toward the single highest-priced outcome's payoff as
    theta -> infinity. That monotonicity is what makes bisection a safe,
    guaranteed-to-converge method here.

    Real edge case found via live validation, worth understanding: a
    distribution whose MEAN equals its own MAXIMUM possible value can only
    be the degenerate one that puts ~100% probability on that single
    outcome. If the target happens to sit at (or extremely close to) the
    single highest-priced card's payoff -- which happens whenever that
    card's own natural pull probability exceeds the target tail rarity, a
    real case found at y=1 -- naively solving "match the mean" pushes theta
    toward making that outcome near-certain under q. With a finite trial
    batch, that means essentially every trial draws the same outcome and
    none draw anything else, so the self-normalized weighted estimator has
    no "miss" samples to correctly weigh against the "hit" samples --
    a finite-sample failure (the true expectation is still recovered in the
    infinite-sample limit, but a real trial budget never gets there).
    MAX_SINGLE_OUTCOME_TILT_PROB guards against this: the search refuses to
    push theta higher once the single highest-payoff outcome's probability
    under q would exceed that cap (see _tilt_stats for why it's specifically
    that outcome being tracked, not the max over all outcomes), accepting
    an imperfect match to the target mean rather than a degenerate,
    all-eggs-in-one-basket tilt.

    Bracket-finding: start with a small theta and keep doubling it until
    the tilted mean payoff reaches or exceeds the target, or either safety
    cap (THETA_SEARCH_MAX or MAX_SINGLE_OUTCOME_TILT_PROB) is hit.
    """
    natural_mean_payoff = float(np.dot(dist.probs, dist.payoffs))
    if target_mean_payoff <= natural_mean_payoff:
        # A real, expected case, not a bug: the threshold comes from a
        # quantile of a modestly-sized regular-sampling batch (see
        # estimate_profit_threshold), which has its own sampling noise. In
        # an unlucky draw, that noise can occasionally land the located
        # "top p_target" threshold at or below the true average -- meaning
        # this particular threshold isn't actually demanding enough to be
        # worth tilting toward. theta=0 exactly reproduces the untilted
        # distribution (q = p), which is the mathematically correct
        # response: importance sampling degrades gracefully to ordinary
        # Monte Carlo (every importance weight becomes exactly 1) rather
        # than needing a special case downstream.
        return 0.0

    lo, hi = 0.0, 1.0
    hi_mean, _, hi_max_prob = _tilt_stats(dist, hi)
    while hi_mean < target_mean_payoff and hi < THETA_SEARCH_MAX and hi_max_prob < MAX_SINGLE_OUTCOME_TILT_PROB:
        hi *= 2.0
        hi_mean, _, hi_max_prob = _tilt_stats(dist, hi)

    # Bisection: at every step, mean_payoff(lo) <= target <= mean_payoff(hi)
    # is maintained (guaranteed by monotonicity), halving the bracket for a
    # fixed, generous number of iterations -- rather than stopping once the
    # THETA interval itself shrinks below some epsilon. That distinction
    # matters: mean_payoff can be an extremely steep function of theta (a
    # tiny theta can already make one outcome's e^(theta*payoff) term
    # astronomically dominant over the rest), so a small remaining theta
    # interval doesn't necessarily mean a small remaining error in the
    # achieved mean payoff. 100 iterations shrinks the bracket by a factor
    # of 2^100 regardless of its starting width -- far beyond
    # double-precision resolution -- and costs nothing extra since each
    # iteration is one pass over a small, finite outcome array.
    for _ in range(THETA_SOLVE_ITERATIONS):
        mid = (lo + hi) / 2.0
        mid_mean, _, mid_max_prob = _tilt_stats(dist, mid)
        # Treat "mean not yet at target" AND "concentration still under the
        # safety cap" as the condition for pushing theta higher -- if
        # either the mean has caught up OR the concentration cap has been
        # reached, stop growing theta further (see the docstring above for
        # why the concentration cap can bind before the mean does).
        if mid_mean < target_mean_payoff and mid_max_prob < MAX_SINGLE_OUTCOME_TILT_PROB:
            lo = mid
        else:
            hi = mid

    return hi


@dataclass(frozen=True)
class TiltedDistribution:
    """A theta-tilted proposal distribution q over the same outcomes as base.

    Keeps log_mgf (log M(theta)) alongside theta because both are needed,
    together with a trial's total payoff, to compute that trial's
    importance weight in closed form -- see simulate_y_pack_is_trials.
    """

    base: SetDistribution
    theta: float
    log_mgf: float
    q_cumulative_probs: np.ndarray


def build_esscher_tilted_distribution(dist: SetDistribution, theta: float) -> TiltedDistribution:
    """Builds the theta-tilted distribution q_theta(x) = p(x) * e^(theta * payoff(x)) / M(theta).

    Every outcome that had positive probability under p keeps positive
    probability under q_theta (e^(theta * payoff) is always > 0), so the
    hard requirement on any valid proposal distribution -- q(x) > 0
    wherever p(x) > 0 -- is satisfied automatically by construction, unlike
    the earlier subset-boosting approach which needed that checked
    explicitly.
    """
    payoffs = dist.payoffs
    with np.errstate(divide="ignore"):
        log_p = np.log(dist.probs)
    log_terms = log_p + theta * payoffs
    log_mgf = _log_sum_exp(log_terms)

    q_probs = np.exp(log_terms - log_mgf)
    q_cumulative_probs = np.cumsum(q_probs)
    # Same floating-point-drift guard used when building the original
    # distribution (distribution.py) and the previous tilting approach --
    # forces the array to end at exactly 1.0 so a uniform draw of 1.0 can
    # never fall past the end of it.
    q_cumulative_probs[-1] = 1.0

    return TiltedDistribution(base=dist, theta=theta, log_mgf=log_mgf, q_cumulative_probs=q_cumulative_probs)


def simulate_y_pack_is_trials(
    tilted: TiltedDistribution, y: int, n_trials: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Draws n_trials scenarios of y packs each from the tilted distribution q_theta,
    returning (profits, weights) for each trial.

    Weight formula, derived from the tilt's structure rather than summed
    per-draw: for y independent draws each from the same q_theta, the joint
    probability under q_theta is [product of p(x_i)] * e^(theta * total_payoff) / M(theta)^y,
    so the importance weight (joint p / joint q_theta) simplifies to:

        weight = M(theta)^y * e^(-theta * total_payoff)
        log(weight) = y * log_mgf - theta * total_payoff

    This is a single closed-form expression per trial, computed directly
    from the SAME total_payoff already needed for profit -- a genuine
    simplification over the previous subset-boosting tilt, which had to
    gather and sum log(p_i) - log(q_i) over every individual draw
    separately because there was no closed form for a product of
    differently-weighted per-outcome ratios.
    """
    dist = tilted.base
    indices = draw_outcome_indices(tilted.q_cumulative_probs, (n_trials, y), rng)

    total_payoff = dist.payoffs[indices].sum(axis=1)
    profits = total_payoff - (y * dist.pack_cost)

    log_weights = (y * tilted.log_mgf) - (tilted.theta * total_payoff)
    weights = np.exp(log_weights)

    return profits, weights


def effective_sample_size(weights: np.ndarray) -> float:
    """ESS = (sum of weights)^2 / (sum of squared weights).

    Formula in words: this estimates how many independent plain-Monte-Carlo
    samples the weighted sample is "worth." If every weight were equal,
    ESS = n_trials exactly (no degradation at all -- equal weights mean the
    tilt didn't distort anything). If a handful of trials carry enormous
    weight relative to the rest, ESS collapses toward 1, signaling that the
    estimate is effectively being driven by only a couple of samples and
    shouldn't be trusted at face value.
    """
    return float(weights.sum() ** 2 / np.sum(weights**2))


def weighted_tail_probability(profits: np.ndarray, weights: np.ndarray, threshold: float) -> float:
    """Self-normalized importance-sampling estimate of P(profit >= threshold).

    Uses >= rather than a strict > deliberately -- this profit distribution
    is discrete (a finite set of possible payoffs), and `threshold` is
    itself an actual observed profit value at a target quantile, not an
    arbitrary continuous cutoff. If a single outcome's own probability
    exceeds p_target, the located threshold can land exactly ON that
    outcome's profit value; a strict `>` would then require "exceeding the
    maximum possible profit," which is impossible by construction and
    silently returns a probability of exactly 0 no matter how many trials
    are drawn. For a continuous distribution `>` and `>=` agree (any single
    point has zero mass); for a discrete one with real mass sitting exactly
    at the boundary, only `>=` correctly includes it in the tail.

    Formula in words: a weighted average of the indicator "did this trial
    land at or above the threshold," using each trial's importance weight.
    Dividing by sum(weights) rather than n_trials (the "self-normalized"
    form) is the standard, more robust choice in practice -- it's
    insensitive to any small numerical imprecision in how exactly q was
    normalized, and generally has lower variance than dividing by the raw
    trial count.
    """
    indicator = (profits >= threshold).astype(np.float64)
    return float(np.sum(weights * indicator) / np.sum(weights))


def weighted_conditional_expectation(profits: np.ndarray, weights: np.ndarray, threshold: float) -> float:
    """Self-normalized estimate of E[profit | profit >= threshold].

    Uses >= for the same discrete-distribution reason as
    weighted_tail_probability -- see that function's docstring.

    Formula in words: a weighted average of profit, but computed only over
    the trials that landed in the tail, with weights renormalized within
    that subset (dividing by the sum of weights among just those trials,
    not all trials). This is the "expected profit conditional on a top-tier
    outcome" quantity -- deliberately not called CVaR, since CVaR is
    conventionally a downside-risk term (see the module docstring).
    """
    in_tail = profits >= threshold
    tail_weight_total = weights[in_tail].sum()
    assert tail_weight_total > 0, (
        "No (weighted) trials landed above the threshold -- n_trials is likely too small for "
        "how far this threshold sits from the tilted distribution's mean; increase n_trials."
    )
    return float(np.sum(weights[in_tail] * profits[in_tail]) / tail_weight_total)


def estimate_profit_threshold(profits: np.ndarray, p_target: float = TAIL_PROBABILITY) -> float:
    """Locates the profit value at the (1 - p_target) quantile of an existing batch of plain Monte Carlo profits.

    Deliberately takes an already-computed `profits` array rather than
    running its own simulation: the app already draws a full batch of plain
    Monte Carlo trials for the regular risk_metrics.py statistics, so
    reusing that batch to also locate this threshold costs nothing extra --
    running a second, separate "pilot" simulation just for this would
    simulate the exact same thing twice. This is a rough locator, not a
    precise estimate of the probability/expectation beyond it -- refining
    those precisely is exactly the job importance sampling does next.
    """
    return float(np.quantile(profits, 1.0 - p_target))


@dataclass(frozen=True)
class UpperTailEstimate:
    """Everything importance sampling produces for one (set, y) scenario."""

    y: int
    threshold: float
    probability: float
    expected_profit_given_tail: float
    effective_sample_size: float
    n_trials: int
    theta: float


def estimate_upper_tail_outcome(
    dist: SetDistribution,
    y: int,
    rng: np.random.Generator,
    regular_profits: np.ndarray,
    n_is_trials: int = N_IS_TRIALS_DEFAULT,
    p_target: float = TAIL_PROBABILITY,
) -> UpperTailEstimate:
    """End-to-end pipeline: locate a rare-tail threshold from existing regular
    samples, solve for the Esscher tilt that targets it, and importance-sample
    the probability and conditional expectation beyond it.

    `regular_profits` should be the same array the caller already computed
    via simulation.simulate_y_pack_profits(dist, y, ...) for the regular
    risk metrics -- see estimate_profit_threshold for why no separate
    simulation is run here to locate the threshold.

    Always inspect the returned effective_sample_size before trusting
    probability/expected_profit_given_tail at face value (see
    effective_sample_size's docstring for why).
    """
    threshold = estimate_profit_threshold(regular_profits, p_target=p_target)

    # The tilt needs the y packs' TOTAL payoff to average out to
    # (threshold + y * pack_cost) -- i.e. profit = total_payoff - y*cost
    # reaching exactly the threshold -- so the PER-PACK mean payoff target
    # is that total divided by y.
    target_mean_payoff = (threshold / y) + dist.pack_cost
    theta = solve_tilt_theta(dist, target_mean_payoff)
    tilted = build_esscher_tilted_distribution(dist, theta)

    profits, weights = simulate_y_pack_is_trials(tilted, y=y, n_trials=n_is_trials, rng=rng)

    return UpperTailEstimate(
        y=y,
        threshold=threshold,
        probability=weighted_tail_probability(profits, weights, threshold),
        expected_profit_given_tail=weighted_conditional_expectation(profits, weights, threshold),
        effective_sample_size=effective_sample_size(weights),
        n_trials=n_is_trials,
        theta=theta,
    )
