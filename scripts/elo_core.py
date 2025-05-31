#!/usr/bin/env python3
"""
Elo constants and helper functions.

Example:

```python
from elo_core import (
    SKILLSETS, load_scores, build_matches_for_skillset,
    outcome_from_scores, run_elo,
)
```

"""
from __future__ import annotations
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from typing import List, Tuple

# ──────────────────────────────
# CONSTANTS (edit here → everywhere)
# ──────────────────────────────
RATING_INIT: float  = 1500.0
K_FACTOR: float     = 10.0
TOLERANCE: float    = 1e-3
TAU_GAP_DAYS: float = 4*365   # np.inf → no time decay

WIFE_DIFF_SCALE = 1.5
RATE_DIFF_SCALE = 100.0

WIFE_RANGE: Tuple[float,float] = (89.0, 99.0)

SKILLSETS: List[str] = [
    "stream", "jumpstream", "handstream",
    "chordjacks", "technical",
]

__all__ = [
    "RATING_INIT", "K_FACTOR", "TOLERANCE", "TAU_GAP_DAYS",
    "WIFE_RANGE", "SKILLSETS",
    "load_scores", "build_matches_for_skillset",
    "outcome_from_scores", "run_elo",
]

# ──────────────────────────────
# Data‑loading utilities
# ──────────────────────────────

def load_scores(scores_dir: Path) -> pd.DataFrame:
    """Read all `*score_data*.parquet` in *scores_dir* and pre‑clean them.

    Adds columns: chart_key, chart_id, dominant *skillset*, datetime⇢Timestamp.
    Filters scores outside *WIFE_RANGE*.
    """
    files = list(scores_dir.glob("*score_data*.parquet"))
    if not files:
        raise FileNotFoundError(f"No '*score_data*.parquet' found in {scores_dir}")

    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    df["chart_key"] = df["chart"].apply(lambda x: x["key"])
    df["chart_id"]  = df["chart"].apply(lambda x: x["id"])
    df = df[(df["wife"] > WIFE_RANGE[0]) & (df["wife"] < WIFE_RANGE[1])].copy()
    df["skillset"] = df[SKILLSETS].idxmax(axis=1)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df

# ──────────────────────────────
# Match‑construction logic (top‑rate pairs)
# ──────────────────────────────

def build_matches_for_skillset(df: pd.DataFrame, sk: str) -> pd.DataFrame:
    """Return chronological DataFrame of pairwise matches for *sk*."""
    sdf = df[df["skillset"] == sk].copy()

    # group per chart into small arrays for fast Python iteration
    grp = sdf.groupby("chart_key")[["id", "datetime", "player", "rate"]]
    chart_arrays = grp.apply(np.array)
    multi_player = sdf.groupby("chart_key")["player"].nunique() > 1
    chart_arrays = chart_arrays[multi_player]

    def toprate_pairs(arr: np.ndarray) -> np.ndarray:
        cols = ["id", "datetime", "player", "rate"]
        tmp = pd.DataFrame(arr, columns=cols).sort_values("datetime")
        best, pairs = {}, []
        for row in tmp.itertuples(index=False):
            sid, ts, pid, r = row.id, row.datetime, row.player, row.rate
            if pid not in best or r > best[pid][0]:
                for opp, (opp_r, opp_id, opp_ts) in best.items():
                    if opp == pid:
                        continue
                    pairs.append((sid, opp_id))
                best[pid] = (r, sid, ts)
        return np.array(pairs, dtype=object)

    if chart_arrays.empty:
        return pd.DataFrame()

    match_ids = np.concatenate([toprate_pairs(a) for a in chart_arrays.values])
    lookup = sdf.set_index("id")[["player", "wife", "rate", "datetime"]]
    matches = (pd.DataFrame(match_ids, columns=["id_A", "id_B"])
               .join(lookup, on="id_A")
               .join(lookup, on="id_B", rsuffix="_B")
               .rename(columns={
                   "player": "player_A", "wife": "wife_A",
                   "rate":   "rate_A",   "datetime": "datetime_A"}))
    matches["latest"] = matches[["datetime_A", "datetime_B"]].max(axis=1)
    return matches.sort_values("latest").drop(columns="latest").reset_index(drop=True)

# ──────────────────────────────
# Core Elo helpers
# ──────────────────────────────

def outcome_dynamic(
    rA: float, rB: float,
    wA: float, wB: float,
    alpha: float = RATE_DIFF_SCALE,     # weight for log(rate ratio)
    beta:  float = WIFE_DIFF_SCALE      # weight for WIFE diff
) -> float:
    """
    Return a value in (0,1) giving the probability‐like ‘score’ for player A.

        z = alpha * log(rate_A / rate_B) + beta * (wife_A - wife_B)
        S_A = 1 / (1 + exp(-z))

    • Symmetric: swapping A/B makes z → –z →  S_A ↔ 1–S_A
    • If rates and WIFEs equal → z = 0 → 0.5 (draw).
    """
    # avoid divide-by-zero: rate values are positive by construction
    z = alpha * np.log(rA / rB) + beta * (wA - wB)

    return 1.0 / (1.0 + np.exp(-z))


def outcome_from_scores(rA: float, rB: float, wA: float, wB: float,
                        tol: float = TOLERANCE) -> float:
    """Return 1 if A beats B, 0 if B beats A, 0.5 for draw."""
    if rA > rB + tol:
        return 1.0
    if rB > rA + tol:
        return 0.0
    if wA > wB + tol:
        return 1.0
    if wB > wA + tol:
        return 0.0
    return 0.5

def run_elo(
    matches: pd.DataFrame,
    *,
    rating_init: float = RATING_INIT,
    k:          float  = K_FACTOR,
    tau_gap_days: float = TAU_GAP_DAYS,
    tol:        float  = TOLERANCE,
) -> pd.DataFrame:
    """
    Compute final and peak Elo ratings for a given series of matches.

    For every *new* personal-best score **id_A** (player A):

      • Compare A against each opponent B (id_B) using A’s current rating.
      • Accumulate ΔR_A from all comparisons.
      • Update each opponent B right away.
      • When all matches for this id_A are done, apply the summed ΔR_A once.

    Returns
    -------
    pd.DataFrame
        index = player,
        columns = ["elo", "peak"]
        where “elo” is the rating after the final match and “peak” is the
        highest value that player reached at any point in the simulation.
    """
    rating: dict[str, float] = defaultdict(lambda: rating_init)
    peak:   dict[str, float] = defaultdict(lambda: rating_init)
    tau = np.float64(tau_gap_days)

    for id_A, grp in matches.groupby("id_A", sort=False):
        row0 = grp.iloc[0]
        pA, RA0 = row0.player_A, rating[row0.player_A]
        delta_A_sum = 0.0

        for row in grp.itertuples(index=False):
            pB, rA, rB = row.player_B, row.rate_A, row.rate_B
            wA, wB     = row.wife_A, row.wife_B
            tA, tB     = row.datetime_A, row.datetime_B

            RB = rating[pB]
            gap = abs((tA - tB).days)
            k_eff = k if np.isinf(tau) else k * np.exp(-gap / tau)

            #sA = outcome_from_scores(rA, rB, wA, wB, tol)
            sA = outcome_dynamic(rA, rB, wA, wB)
            sB = 1.0 - sA
            expA = 1.0 / (1.0 + 10.0 ** ((RB - RA0) / 400.0))

            # accumulate A's delta
            delta_A_sum += k_eff * (sA - expA)

            # update B immediately
            rating[pB] = RB + k_eff * (sB - (1.0 - expA))
            peak[pB]   = max(peak[pB], rating[pB])

        # batch-apply A's combined update once
        rating[pA] = RA0 + delta_A_sum
        peak[pA]   = max(peak[pA], rating[pA])

    df = (pd.DataFrame({"elo": rating})
            .join(pd.Series(peak, name="peak"))
            .sort_values("elo", ascending=False))
    return df
