"""Minimal Flask frontend for the pack-opening Monte Carlo simulation.

Deliberately small by design (per the project scope): one page, three
steps (enter a set name, pick a y, view results), no other pages or
persisted state. Every request is self-contained — the server holds no
per-user session, it just validates input and runs the simulation fresh
each time.
"""

from __future__ import annotations

import base64

from flask import Flask, jsonify, render_template, request

from backend.config import load_settings
from backend.constants import ALPHAS, N_TRIALS_DEFAULT, Y_VALUES
from backend.db import get_connection, get_in_scope_sets
from backend.distribution import build_set_distribution
from backend.risk_metrics import compute_risk_metrics
from backend.simulation import make_rng, simulate_y_pack_profits
from backend.visualization import make_profit_histogram

app = Flask(__name__)

# The list of valid set names changes only when the database is repopulated
# with new sets, not per-request, so it's fetched once at startup rather
# than re-querying the database on every keystroke/validation call. If the
# database is later updated with a new set, restarting the app picks it up.
_settings = load_settings()
with get_connection(_settings) as _startup_conn:
    _IN_SCOPE_SETS = get_in_scope_sets(_startup_conn)
_IN_SCOPE_SETS_SET = set(_IN_SCOPE_SETS)  # O(1) membership checks for validation


@app.route("/")
def index():
    """Step 1: renders the set-name entry form."""
    return render_template("index.html", y_values=Y_VALUES)


@app.route("/validate_set", methods=["POST"])
def validate_set():
    """Step 1 -> 2: checks a submitted set name against the in-scope list.

    Matching is exact and case-sensitive, per the project spec ("must
    exactly match a Scarlet & Violet or Mega Evolution set name as it
    appears in the database") — a near-miss should tell the user to try
    again rather than guess at a fuzzy match.
    """
    payload = request.get_json(silent=True) or {}
    setname = (payload.get("setname") or "").strip()

    if setname in _IN_SCOPE_SETS_SET:
        return jsonify({"valid": True, "setname": setname})

    return jsonify(
        {
            "valid": False,
            "message": (
                "That set name wasn't recognized. It must exactly match one of the "
                "supported Scarlet & Violet or Mega Evolution sets — please try again."
            ),
        }
    )


@app.route("/simulate", methods=["POST"])
def simulate():
    """Step 2 -> 3: runs the simulation for one (set, y) pair and returns
    the risk metrics plus a base64-encoded histogram PNG in one response.

    Rendering the histogram from the exact same simulated `profits` array
    used for the numeric metrics (rather than a separate endpoint that
    would re-simulate independently) guarantees the plot and the reported
    numbers are always describing the same underlying trial batch — a
    deliberate simplicity tradeoff, accepting a larger JSON payload in
    exchange for one simulation run per request instead of two.
    """
    payload = request.get_json(silent=True) or {}
    setname = payload.get("setname")
    y_raw = payload.get("y")

    if setname not in _IN_SCOPE_SETS_SET:
        return jsonify({"error": "Unknown or unsubmitted set name."}), 400

    try:
        y = int(y_raw)
        if y <= 0:
            raise ValueError
    except (TypeError, ValueError):
        return jsonify({"error": "y must be a positive integer."}), 400

    with get_connection(_settings) as conn:
        dist = build_set_distribution(conn, setname)

    rng = make_rng()  # fixed default seed — see constants.DEFAULT_SEED for why
    profits = simulate_y_pack_profits(dist, y=y, n_trials=N_TRIALS_DEFAULT, rng=rng)
    metrics = compute_risk_metrics(profits, y=y, alphas=ALPHAS)
    png_bytes = make_profit_histogram(profits, setname, y)
    histogram_b64 = base64.b64encode(png_bytes).decode("ascii")

    return jsonify(
        {
            "setname": setname,
            "y": y,
            "n_trials": metrics.n_trials,
            "pack_cost": dist.pack_cost,
            "mean": metrics.mean,
            "sd": metrics.sd,
            "breakeven_prob": metrics.breakeven_prob,
            "var_95": metrics.by_alpha[0.95].var,
            "var_95_binding": metrics.by_alpha[0.95].var_binding,
            "cvar_95": metrics.by_alpha[0.95].cvar,
            "var_99": metrics.by_alpha[0.99].var,
            "var_99_binding": metrics.by_alpha[0.99].var_binding,
            "cvar_99": metrics.by_alpha[0.99].cvar,
            "histogram_base64": histogram_b64,
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
