#!/usr/bin/env python3
"""
elo_tune_params_parallel.py — grid-search tuner for K-factor and time-decay τ
running combinations in parallel with concurrent.futures.
"""
from pathlib import Path
import itertools, os, math
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from sklearn.metrics import log_loss                          # optional
from elo_core import (                                         # unchanged
    SKILLSETS, load_scores, build_matches_for_skillset,
    outcome_from_scores, outcome_dynamic, RATING_INIT, TOLERANCE,
)

# ───────────────────────── settings ───────────────────────── #
SCORES_DIR = Path("output/scores")

K_GRID              = [8]
TAU_GRID            = [365 * 4]
RATE_DIFF_SCALE_GRID = [100,]
WIFE_DIFF_SCALE_GRID = [  0.75,0.8,0.9,1.0,1.1]
WIFE_LINERIZER_GRID  = [ 6, 6.5,7,7.5,8]
WIFE_DENOM = 110.0

SKILLSETS = ["technical"]

FRAC, RNG_SEED, MIN_CAL_MATCHES = 0.05, 1, 200
# ──────────────────────────────────────────────────────────── #

# -------------------------------------------------------------------
def brier_score(y, p):      return float(np.mean((p - y) ** 2))
def cross_entropy(y, p):
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))
# -------------------------------------------------------------------

def evaluate_random_holdout(matches: pd.DataFrame, frac: float, rng,
                            k: float, tau_days: float,
                            w_scale: float, r_scale: float, theta: float):
    """Identical to original, kept verbatim for brevity."""
    ratings = defaultdict(lambda: RATING_INIT)
    played  = defaultdict(int)
    tau = np.float64(tau_days)
    probs, outcomes = [], []

    for id_A, grp in matches.groupby("id_A", sort=False):
        pA = grp.iloc[0].player_A
        RA0, delta_A = ratings[pA], 0.0

        for row in grp.itertuples(index=False):
            pB, rA, rB = row.player_B, row.rate_A, row.rate_B
            wA, wB     = row.wife_A,   row.wife_B
            tA, tB     = row.datetime_A, row.datetime_B

            eligible = (played[pA] >= MIN_CAL_MATCHES
                        and played[pB] >= MIN_CAL_MATCHES)
            is_test  = eligible and (rng.random() < frac)

            RB   = ratings[pB]
            expA = 1 / (1 + 10 ** ((RB - RA0) / 400))
            sA   = outcome_dynamic(rA, rB, wA, wB, r_scale, w_scale, theta, denom=WIFE_DENOM)
            sB   = 1 - sA

            if is_test:
                probs.append(expA)
                outcomes.append(sA)
            else:
                gap   = abs((tA - tB).days)
                k_eff = k if math.isinf(tau) else k * math.exp(-gap / tau)
                delta_A        += k_eff * (sA - expA)
                ratings[pB]     = RB + k_eff * (sB - (1 - expA))

            played[pA] += 1
            played[pB] += 1
        ratings[pA] = RA0 + delta_A

    return np.asarray(probs), np.asarray(outcomes)

# -------------------------------------------------------------------
def score_params(match_cache, k, tau, w_scale, r_scale, theta):
    rng = np.random.default_rng(RNG_SEED)
    tot_ll = tot_brier = tot_n = 0

    for sk, matches in match_cache.items():
        if matches.empty: continue
        p, y = evaluate_random_holdout(matches, FRAC, rng,
                                       k, tau, w_scale, r_scale, theta)
        if y.size == 0: continue
        draws  = (y == 0.5)
        brier  = brier_score(y, p)
        ll     = cross_entropy(y[~draws], p[~draws]) if (~draws).any() else np.nan
        n      = y.size
        tot_n     += n
        tot_brier += brier * n
        if not np.isnan(ll): tot_ll += ll * n

    return (np.inf, np.inf) if tot_n == 0 else (tot_ll / tot_n, tot_brier / tot_n)
# -------------------------------------------------------------------

# ===============   PARALLEL SECTION   =============== #
def _worker(args):
    """Run one grid-point; loading data inside each worker avoids
    large pickles and lets processes work independently."""
    k, tau, w_scale, r_scale, theta = args

    data = load_scores(SCORES_DIR)
    match_cache = {sk: build_matches_for_skillset(data, sk) for sk in SKILLSETS}
    ll, br = score_params(match_cache, k, tau, w_scale, r_scale, theta)
    tau_lbl = "inf" if np.isinf(tau) else int(tau)
    return {"K": k, "tau": tau_lbl, "wife_scale": w_scale,
            "rate_scale": r_scale, "theta": theta,
            "log_loss": ll, "brier": br}

def main():
    param_grid = list(itertools.product(
        K_GRID, TAU_GRID, WIFE_DIFF_SCALE_GRID,
        RATE_DIFF_SCALE_GRID, WIFE_LINERIZER_GRID))

    results = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        futures = {pool.submit(_worker, p): p for p in param_grid}
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            print(f"K={res['K']:>2}, τ={res['tau']:>4}, w_s={res['wife_scale']}, "
                  f"r_s={res['rate_scale']}, θ={res['theta']:.2f} → "
                  f"log_loss={res['log_loss']:.4f}  brier={res['brier']:.4f}")

    results.sort(key=lambda d: d["log_loss"])
    best = results[0]
    print("\n===== Best parameters (by log-loss) =====")
    print(f"K = {best['K']}, τ = {best['tau']}, "
          f"w_scale = {best['wife_scale']}, r_scale = {best['rate_scale']}, "
          f"θ = {best['theta']}  ⇒  "
          f"log_loss = {best['log_loss']:.4f},  brier = {best['brier']:.4f}")

if __name__ == "__main__":
    main()