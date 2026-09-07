"""Importance sampling for rare-tail pull probability / CVaR estimation.

Deliberately left unimplemented for now. This is the demonstrated
variance-reduction technique referenced in the project's design doc
(tilted proposal distribution, log-space importance weights, effective
sample size, weighted CVaR estimates) — it's being built last, after the
plain Monte Carlo path (distribution.py, simulation.py, risk_metrics.py,
visualization.py, frontend/app.py) is working and verified, and only once
the approach itself has been discussed and confirmed in more detail.
"""
