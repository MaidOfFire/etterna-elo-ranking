#!/usr/bin/env python3
"""
elo_tune_params.py — grid-search tuner for K-factor and time-decay τ

New requirement
---------------
A match can be sampled into the *test* split only if **both players already have
at least `MIN_CAL_MATCHES` total matches** in that skill-set (across the whole
data-set).  This avoids evaluating the model on players whose ratings are still
in the volatile “cold-start” phase.
"""
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from collections import defaultdict
from elo_core import (                       # ← shared helpers & constants
    SKILLSETS, load_scores, build_matches_for_skillset,
    outcome_from_scores, RATING_INIT, TOLERANCE,
)

# ──────────────────────────────
# SEARCH SPACE & EVAL SETTINGS
# ──────────────────────────────
SCORES_DIR = Path("output/scores")

K_GRID   = [10, 20, 30, 40, 50, 60, 70]
TAU_GRID = [365]  # days; np.inf → no decay

FRAC      = 0.1
RNG_SEED  = 1
MIN_CAL_MATCHES = 200
# ────────────────────────────

def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def evaluate_random_holdout(
        matches: pd.DataFrame,
        frac: float,
        rng:  np.random.Generator,
        k:    float,
        tau_days: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Chronological simulation **with per-player experience gating**.

    ─────────────
    Eligibility
    ─────────────
    Let `played[p]` be the number of matches player *p* has completed *before*
    the current row.  A row is *eligible for test* iff

        played[player_A] ≥ MIN_CAL_MATCHES   AND
        played[player_B] ≥ MIN_CAL_MATCHES

    Only eligible rows are subjected to the random-`frac` draw.  
    Non-eligible rows are always used for training.
    """
    # ratings and per-player match counters
    ratings = {}
    played  = defaultdict(int)

    def r(pid):       # current rating
        return ratings.get(pid, RATING_INIT)

    tau   = np.float64(tau_days)
    probs, outcomes = [], []

    for idx, row in matches.iterrows():          # already chronological
        pA, rA, wA, tA = row.player_A, row.rate_A, row.wife_A, row.datetime_A
        pB, rB, wB, tB = row.player_B, row.rate_B, row.wife_B, row.datetime_B

        RA, RB = r(pA), r(pB)
        p_winA = 1 / (1 + 10 ** ((RB - RA) / 400))
        sA     = outcome_from_scores(rA, rB, wA, wB)

        # ---------- experience gate ----------------------------------- #
        eligible = (
            played[pA] >= MIN_CAL_MATCHES and
            played[pB] >= MIN_CAL_MATCHES
        )
        is_test_row = eligible and (rng.random() < frac)
        # ----------------------------------------------------------------

        if is_test_row:                      # ------ held-out row
            probs.append(p_winA)
            outcomes.append(sA)
        else:                                # ------ training row
            gap_days = abs((tA - tB).days)
            k_eff = k if np.isinf(tau) else k * np.exp(-gap_days / tau)
            ratings[pA] = RA + k_eff * (sA - p_winA)
            ratings[pB] = RB + k_eff * ((1 - sA) - (1 - p_winA))

        # update match counters *after* processing the row
        played[pA] += 1
        played[pB] += 1

    return np.asarray(probs), np.asarray(outcomes)


def score_params(data: pd.DataFrame, k: float, tau: float):
    rng = np.random.default_rng(RNG_SEED)
    tot_ll, tot_brier, tot_n = 0.0, 0.0, 0

    for sk in SKILLSETS:
        matches = build_matches_for_skillset(data, sk)
        if matches.empty:
            continue

        p, y = evaluate_random_holdout(matches, FRAC, rng, k, tau)
        if len(y) == 0:
            continue

        draws = y == 0.5
        brier = brier_score(y, p)
        ll = (log_loss(y[~draws], p[~draws]) if (~draws).sum() else np.nan)

        n = len(y)
        tot_n     += n
        tot_brier += brier * n
        if not np.isnan(ll):
            tot_ll += ll * n

    if tot_n == 0:
        return np.inf, np.inf
    return tot_ll / tot_n, tot_brier / tot_n


def main():
    data = load_scores(SCORES_DIR)

    results = []
    for k, tau in itertools.product(K_GRID, TAU_GRID):
        ll, br = score_params(data, k, tau)
        tau_lbl = "inf" if np.isinf(tau) else int(tau)
        results.append({"K": k, "tau": tau_lbl, "log_loss": ll, "brier": br})
        print(f"K={k:>2}, τ={tau_lbl:>4}  →  log_loss={ll:.4f}  brier={br:.4f}")

    results.sort(key=lambda d: d["log_loss"])
    best = results[0]
    print("\n===== Best parameters (by log-loss) =====")
    print(f"K = {best['K']}, τ = {best['tau']}  ⇒  log_loss = "
          f"{best['log_loss']:.4f},  brier = {best['brier']:.4f}")


if __name__ == "__main__":
    main()
