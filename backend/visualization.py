"""Renders the simulated profit distribution as a labeled histogram.

Uses matplotlib's "Agg" (Anti-Grain Geometry) backend explicitly, set once
at import time, because the default backend expects an interactive GUI
window — that hangs or errors in a headless Flask server process with no
display attached. Agg renders straight to an in-memory image buffer, which
is exactly what a web response needs.
"""

from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def make_profit_histogram(profits: np.ndarray, setname: str, y: int) -> bytes:
    """Builds a histogram of simulated profit outcomes and returns it as PNG bytes.

    The vertical line at x=0 marks the breakeven point by definition:
    profit = (value of cards pulled) - (cost of packs), so profit = 0 is
    exactly the boundary between the packs paying for themselves and not.
    Returns raw PNG bytes via an in-memory buffer rather than writing to
    disk, so the Flask route can hand them straight back in a response
    with no temp file to create or clean up.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(profits, bins=60, color="#4C72B0", edgecolor="none")
    ax.axvline(x=0, color="#C44E52", linestyle="--", linewidth=2, label="Breakeven (profit = $0)")

    ax.set_title(f"Simulated Profit Distribution — {setname} ({y} pack{'s' if y != 1 else ''})")
    ax.set_xlabel("Profit ($)")
    ax.set_ylabel("Number of simulated trials")
    ax.legend()
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120)
    plt.close(fig)  # release the figure's memory; Agg doesn't do this automatically
    buffer.seek(0)
    return buffer.read()
