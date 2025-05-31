#!/usr/bin/env python3
"""
elo_fairness_random_holdout.py — evaluate Elo calibration with a random hold-out
that is consistent with the new batch-update Elo logic.

Workflow
--------
1. Build matches per skill-set (`elo_core.build_matches_for_skillset`).
2. Mark rows as *eligible* for test iff BOTH players already have at least
   `MIN_CAL_MATCHES` prior matches in that skill-set.
3. From eligible rows, draw `FRAC` at random as **test**.
4. Simulate Elo chronologically with the same batching rule used in production:
     – Within a batch (same id_A) freeze A's rating, accumulate ΔR_A,
       update each B immediately, apply ΔR_A once after batch.
     – Skip rating updates on test rows; just record p and outcome.
5. Report Brier, log-loss (decisive only) and accuracy (decisive only).
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
    outcome_dynamic,
    RATING_INIT,
    TOLERANCE,
)

# ──────────────────────────────
# USER-ADJUSTABLE CONSTANTS
# ──────────────────────────────
SCORES_DIR = Path("output/scores")

FRAC              = 0.1     # fraction of eligible rows → test
RNG_SEED          = 1
MIN_CAL_MATCHES   = 200      # min prior matches per player to be test-eligible

# Override core K / τ if you want different evaluation settings
K_FOR_EVAL   = 10.0
TAU_FOR_EVAL = 365 * 4       # days; np.inf → no decay
RATE_DIFF_SCALE_FOR_EVAL = 100.0
WIFE_DIFF_SCALE_FOR_EVAL = 1.5

# ──────────────────────────────
# Helper scoring functions
# ──────────────────────────────
def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def cross_entropy(y: np.ndarray, p: np.ndarray) -> float:
    eps = 1e-15                       
    p   = np.clip(p, eps, 1 - eps)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))
# -------------------------------------------------------------------



def evaluate_random_holdout(matches: pd.DataFrame,
                            frac: float,
                            rng:  np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Chronological Elo simulation with batch update + experience-gated hold-out.
    """
    ratings: dict[str, float] = defaultdict(lambda: RATING_INIT)
    played:  dict[str, int]   = defaultdict(int)           # matches seen so far
    tau = np.float64(TAU_FOR_EVAL)

    probs, outcomes = [], []

    # rows already chronological → group by id_A
    for id_A, grp in matches.groupby("id_A", sort=False):
        row0 = grp.iloc[0]
        pA   = row0.player_A
        RA0  = ratings[pA]           # freeze A for this batch
        delta_A_sum = 0.0            # accumulate ΔR_A

        for row in grp.itertuples(index=False):
            pB, rA, rB = row.player_B, row.rate_A, row.rate_B
            wA, wB     = row.wife_A,  row.wife_B
            tA, tB     = row.datetime_A, row.datetime_B

            # playcount gate + random test draw
            eligible = (played[pA] >= MIN_CAL_MATCHES and
                        played[pB] >= MIN_CAL_MATCHES)
            is_test  = eligible and (rng.random() < frac)

            RB   = ratings[pB]
            expA = 1.0 / (1.0 + 10.0 ** ((RB - RA0) / 400.0))
            sA = outcome_dynamic(rA, rB, wA, wB, RATE_DIFF_SCALE_FOR_EVAL, WIFE_DIFF_SCALE_FOR_EVAL)
            sB   = 1.0 - sA

            if is_test:
                probs.append(expA)
                #outcomes.append(1.0 if sA > 0.5 else 0.0)
                outcomes.append(sA)
            else:
                gap = abs((tA - tB).days)
                k_eff = K_FOR_EVAL if np.isinf(tau) else K_FOR_EVAL * np.exp(-gap / tau)
                delta_A_sum += k_eff * (sA - expA)
                ratings[pB] = RB + k_eff * (sB - (1.0 - expA))

            # update counters
            played[pA] += 1
            played[pB] += 1

        # apply A's combined update once (train rows only)
        ratings[pA] = RA0 + delta_A_sum

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

        draws = (y == 0.5)
        brier = brier_score(y, p)
        if (~draws).sum():
            ll  = cross_entropy(y[~draws], p[~draws]) #log_loss(y[~draws], p[~draws])
            #acc = accuracy_score(y[~draws], p[~draws] > 0.5)
        else:
            ll, acc = np.nan, np.nan

        rows.append({
            "skillset": sk,
            "n_test": len(y),
            "log_loss": ll,
            "brier": brier,
            #"accuracy": acc,
        })

    if not rows:
        raise RuntimeError("No eligible test rows — adjust FRAC or MIN_CAL_MATCHES.")

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
        #"accuracy": wavg(df["accuracy"]),
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
    print("Random hold-out fairness metrics (rounded):\n")
    print(metrics.round(3))


if __name__ == "__main__":
    main()
