#!/usr/bin/env python3
"""
inspect_tau_weights.py
──────────────────────
Visualise the distribution of  exp(-gap_days / τ)  for the current τ value
defined in elo_core.py.

Outputs
-------
• Summary statistics printed to stdout
• Histogram pop-up (matplotlib)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import elo_core                                           # local helper

# ──────────────────────────────
SCORES_DIR = Path("output/scores")
TAU        = elo_core.TAU_GAP_DAYS                        # use project τ
# ──────────────────────────────

def main() -> None:
    # Load scores
    scores = elo_core.load_scores(SCORES_DIR)

    # Build all matches once and collect weights
    weights = []
    for sk in elo_core.SKILLSETS:
        matches = elo_core.build_matches_for_skillset(scores, sk)
        if matches.empty:
            continue
        gap_days = (matches["datetime_A"] - matches["datetime_B"]).abs().dt.days
        weights.extend(np.exp(-gap_days / TAU))

    weights = np.asarray(weights)
    if weights.size == 0:
        print("No matches found – check data or filters.")
        return

    # Summary stats
    summary = pd.Series({
        "count" : weights.size,
        "mean"  : weights.mean(),
        "median": np.median(weights),
        "min"   : weights.min(),
        "max"   : weights.max(),
        "1%"    : np.percentile(weights, 1),
        "5%"    : np.percentile(weights, 5),
        "25%"   : np.percentile(weights, 25),
        "75%"   : np.percentile(weights, 75),
        "95%"   : np.percentile(weights, 95),
        "99%"   : np.percentile(weights, 99),
    }).round(4)

    print("\nexp(-gap_days / τ) summary (τ =", TAU, "days):")
    print(summary.to_string())

    # Histogram
    plt.figure()
    plt.hist(weights, bins=30, edgecolor="black")
    plt.title(f"Distribution of exp(-gap/τ)  (τ = {TAU} days)")
    plt.xlabel("weight")
    plt.ylabel("frequency")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
