#!/usr/bin/env python3
"""
elo_fairness_random_holdout.py — evaluate Elo calibration with a **random 5 %**
hold‑out, using the shared helpers from **`elo_core.py`** to avoid duplication.

*Workflow*
1. Build matches per skill‑set (`elo_core.build_matches_for_skillset`).
2. Flag `FRAC` of rows at random as **test** (seeded for reproducibility).
3. Simulate Elo chronologically:
   • For test rows: record the probability and **skip update**.
   • For train rows: update ratings normally.
4. Compute **Brier**, **log‑loss** (decisive only) and **accuracy** (decisive
   only) on the held‑out rows and print the summary table.

No files are written; results go to stdout.
"""
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score

from elo_core import (
    SKILLSETS,
    load_scores,
    build_matches_for_skillset,
    outcome_from_scores,
    RATING_INIT,
    TOLERANCE,
)

# ──────────────────────────────
# USER‑ADJUSTABLE CONSTANTS
# ──────────────────────────────
SCORES_DIR = Path("output/scores")

FRAC     = 0.05   # fraction of matches to hold out
RNG_SEED = 1     # RNG seed for repeatability

# Optionally override core constants here
K_FOR_EVAL     = 10.0    # different K to test calibration, else use K_FACTOR
TAU_FOR_EVAL   = 365 * 2 # decay constant (days); set np.inf to disable

# ──────────────────────────────
# Helper scoring functions
# ──────────────────────────────

def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    """Quadratic score that naturally supports draws (y==0.5)."""
    return float(np.mean((p - y) ** 2))


def evaluate_random_holdout(matches: pd.DataFrame, frac: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return (probabilities, outcomes) for test rows of one skill‑set."""
    n = len(matches)
    test_mask = rng.random(n) < frac

    ratings = defaultdict(lambda: RATING_INIT)
    tau = np.float64(TAU_FOR_EVAL)
    probs, outcomes = [], []

    for idx, row in matches.iterrows():
        pA, rA, wA, tA = row.player_A, row.rate_A, row.wife_A, row.datetime_A
        pB, rB, wB, tB = row.player_B, row.rate_B, row.wife_B, row.datetime_B

        RA, RB = ratings[pA], ratings[pB]
        p_winA = 1 / (1 + 10 ** ((RB - RA) / 400))
        sA = outcome_from_scores(rA, rB, wA, wB, TOLERANCE)

        if test_mask[idx]:
            probs.append(p_winA)
            outcomes.append(sA)
            continue  # no update on test rows

        # training update
        gap = abs((tA - tB).days)
        k_eff = K_FOR_EVAL if np.isinf(tau) else K_FOR_EVAL * np.exp(-gap / tau)
        ratings[pA] = RA + k_eff * (sA - p_winA)
        ratings[pB] = RB + k_eff * ((1 - sA) - (1 - p_winA))

    return np.asarray(probs), np.asarray(outcomes)


def compute_metrics(all_data: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    rows = []

    for sk in SKILLSETS:
        matches = build_matches_for_skillset(all_data, sk)
        if matches.empty:
            continue
        p, y = evaluate_random_holdout(matches, FRAC, rng)
        if len(y) == 0:
            continue

        draws = y == 0.5
        brier = brier_score(y, p)
        if (~draws).sum():
            ll  = log_loss(y[~draws], p[~draws])
            acc = accuracy_score(y[~draws], p[~draws] > 0.5)
        else:
            ll, acc = np.nan, np.nan

        rows.append({
            "skillset": sk,
            "n_test": len(y),
            "log_loss": ll,
            "brier": brier,
            "accuracy": acc,
        })

    if not rows:
        raise RuntimeError("No test rows selected — increase FRAC or check data.")

    df = pd.DataFrame(rows)
    weights = df["n_test"].to_numpy()

    def wavg(series):
        mask = ~series.isna()
        return np.average(series[mask], weights=weights[mask]) if mask.any() else np.nan

    overall = {
        "skillset": "overall",
        "n_test": weights.sum(),
        "log_loss": wavg(df["log_loss"]),
        "brier":    np.average(df["brier"], weights=weights),
        "accuracy": wavg(df["accuracy"]),
    }

    return (pd.concat([pd.DataFrame([overall]), df], ignore_index=True)
              .set_index("skillset")
              .sort_index(key=lambda s: s != "overall"))

# ──────────────────────────────
# Main
# ──────────────────────────────

def main():
    data = load_scores(SCORES_DIR)
    metrics = compute_metrics(data)
    print("Random hold‑out fairness metrics (rounded):\n")
    print(metrics.round(3))


if __name__ == "__main__":
    main()
