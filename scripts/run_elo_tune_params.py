#!/usr/bin/env python3
"""
elo_tune_params.py — grid-search tuner for K-factor and time-decay τ
using the order-insensitive batch update and an experience-gated test split.
"""
from pathlib import Path
import itertools
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from elo_core import (
    SKILLSETS, load_scores, build_matches_for_skillset,
    outcome_from_scores, RATING_INIT, TOLERANCE,
)

# ──────────────────────────────
# SETTINGS
# ──────────────────────────────
SCORES_DIR = Path("output/scores")

K_GRID   = [10]
TAU_GRID = [365*3, 365 * 4, 365*4.5, 365*5, np.inf]

FRAC            = 0.10
RNG_SEED        = 1
MIN_CAL_MATCHES = 200

# ──────────────────────────────
def brier_score(y, p):
    return float(np.mean((p - y) ** 2))


def evaluate_random_holdout(    matches:  pd.DataFrame,
    frac:     float,
    rng:      np.random.Generator,
    k:        float,
    tau_days: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Chronological simulation with batch update for id_A.

    • All rows having the same id_A are processed as one batch:
        – Player A's rating is frozen for the batch.
        – ΔR_A from train rows is accumulated and applied once after batch.
        – Each opponent B is updated immediately (train rows only).

    • A row is eligible for the random test split iff BOTH players have at least
      MIN_CAL_MATCHES prior matches in this skill-set.  Eligible rows are
      sent to test with probability frac; train rows update ratings.
    """
    ratings = defaultdict(lambda: RATING_INIT)
    played  = defaultdict(int)  # prior match counts
    tau = np.float64(tau_days)

    probs, outcomes = [], []

    # rows already chronological → group by id_A in appearance order
    for id_A, grp in matches.groupby("id_A", sort=False):
        pA      = grp.iloc[0].player_A
        RA0     = ratings[pA]  # freeze A for this batch
        delta_A = 0.0 # accumulated update for A

        
        for row in grp.itertuples(index=False):
            pB, rA, rB = row.player_B, row.rate_A, row.rate_B
            wA, wB     = row.wife_A,   row.wife_B
            tA, tB     = row.datetime_A, row.datetime_B

            # eligibility & random test flag
            eligible = (played[pA] >= MIN_CAL_MATCHES and
                        played[pB] >= MIN_CAL_MATCHES)
            is_test  = eligible and (rng.random() < frac)

            RB   = ratings[pB]
            expA = 1 / (1 + 10 ** ((RB - RA0) / 400))
            sA   = outcome_from_scores(rA, rB, wA, wB, TOLERANCE)
            sB   = 1 - sA

            if is_test:
                probs.append(expA)
                outcomes.append(sA)
            else:
                gap   = abs((tA - tB).days)
                k_eff = k if np.isinf(tau) else k * np.exp(-gap / tau)
                delta_A        += k_eff * (sA - expA)
                ratings[pB]     = RB + k_eff * (sB - (1 - expA))
            
            # count this match for future eligibility
            played[pA] += 1
            played[pB] += 1

        ratings[pA] = RA0 + delta_A

    return np.asarray(probs), np.asarray(outcomes)


def score_params(match_cache, k, tau):
    rng = np.random.default_rng(RNG_SEED)
    tot_ll = tot_brier = tot_n = 0

    for sk, matches in match_cache.items():
        if matches.empty:
            continue
        p, y = evaluate_random_holdout(matches, FRAC, rng, k, tau)
        if y.size == 0:
            continue

        draws  = (y == 0.5)
        brier  = brier_score(y, p)
        ll     = log_loss(y[~draws], p[~draws]) if (~draws).any() else np.nan
        n      = y.size

        tot_n     += n
        tot_brier += brier * n
        if not np.isnan(ll):
            tot_ll += ll * n

    if tot_n == 0:
        return np.inf, np.inf
    return tot_ll / tot_n, tot_brier / tot_n


def main():
    data = load_scores(SCORES_DIR)

    # ---------- build matches once per skill-set -------------------- #
    match_cache = {
        sk: build_matches_for_skillset(data, sk) for sk in SKILLSETS
    }
    # ---------------------------------------------------------------- #

    results = []
    for k, tau in itertools.product(K_GRID, TAU_GRID):
        ll, br = score_params(match_cache, k, tau)
        tau_lbl = "inf" if np.isinf(tau) else int(tau)
        results.append({"K": k, "tau": tau_lbl, "log_loss": ll, "brier": br})
        print(f"K={k:>2}, τ={tau_lbl:>4} → log_loss={ll:.4f}  brier={br:.4f}")

    results.sort(key=lambda d: d["log_loss"])
    best = results[0]
    print("\n===== Best parameters (by log-loss) =====")
    print(f"K = {best['K']}, τ = {best['tau']}  ⇒  "
          f"log_loss = {best['log_loss']:.4f},  brier = {best['brier']:.4f}")


if __name__ == "__main__":
    main()
