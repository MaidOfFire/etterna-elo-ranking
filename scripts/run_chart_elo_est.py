#!/usr/bin/env python3
"""Compute chart's Elo difficulty."""

from pathlib import Path

import numpy as np
import pandas as pd

from elo_core import load_scores, RATE_DIFF_SCALE, WIFE_DIFF_SCALE


# ──────────────────────────────
# CONFIG
# ──────────────────────────────
PLAYCOUNT_THRESHOLD = 15

SCORES_DIR = Path("output/scores")
HISTORY_CSV = Path("output/elo_by_score.csv")
OUT_CSV = Path("output/chart_elo_diff.csv")
OUT_MD = Path("output/chart_elo_diff.md")

# ──────────────────────────────
# LOAD & PREPARE SCORE DATA
# ──────────────────────────────
scores_full = load_scores(SCORES_DIR)

scores_full["pseudo_rate"] = (
    scores_full["rate"]
    * np.exp((WIFE_DIFF_SCALE / RATE_DIFF_SCALE) * (scores_full["wife"] - 93)) #Based on the outcome formula
)

scores = scores_full[["id", "chart_id", "rate", "wife", "pseudo_rate"]].copy()

# keep only charts with enough plays
play_counts = scores.groupby("chart_id")["id"].count()
valid_chart_ids = play_counts[play_counts > PLAYCOUNT_THRESHOLD].index
scores = scores[scores["chart_id"].isin(valid_chart_ids)]

# ──────────────────────────────
# MERGE IN ELO HISTORY
# ──────────────────────────────
history = pd.read_csv(HISTORY_CSV)  

scores_with_rating = scores.merge(
    history,
    left_on="id",
    right_on="score_id",
    how="inner", 
)

# adjust Elo by pseudo-rate
avg_chart_rate = (
    scores_with_rating[["chart_id", "rate", "pseudo_rate", "elo_at_score"]]
    .sort_values("chart_id")
    .copy()
)
avg_chart_rate["adj_elo_at_score"] = (
    avg_chart_rate["elo_at_score"] / avg_chart_rate["pseudo_rate"]
)

# ──────────────────────────────
# CHART AGGREGATES
# ──────────────────────────────
# dominant skillset per chart
chart_skillset = scores_full.groupby("chart_id")["skillset"].apply(
    lambda s: np.unique(s, return_counts=True)[0][
        np.unique(s, return_counts=True)[1].argmax()
    ]
)

skill_cols = ["stream", "jumpstream", "handstream", "chordjacks", "technical"]

scores_full["msd"] = scores_full[skill_cols].max(axis=1)

cap = np.exp((WIFE_DIFF_SCALE / RATE_DIFF_SCALE) * (96.5 - 93)) # Msd doesn't scale linearly so this pseudo-rate approximation is very rough
scores_full["pseudo_rate_msd"] = np.minimum(
    scores_full["pseudo_rate"],
    scores_full["rate"] * cap,
)

# the approximation. we want to know 93% wife msds 
chart_msd = (
    scores_full.groupby("chart_id")
    .apply(lambda df: (df["msd"] / df["pseudo_rate_msd"]).mean())
    .loc[valid_chart_ids]
)

# readable chart name: (song, first-pack)
chart_name = {
    cid: (
        scores_full.loc[scores_full["chart_id"] == cid, "song"].iloc[0]["name"],
        scores_full.loc[scores_full["chart_id"] == cid, "song"].iloc[0]["packs"][0][
            "name"
        ],
    )
    for cid in valid_chart_ids.astype(int)
}

# Elo diff
chart_diff = (
    avg_chart_rate.groupby("chart_id")["adj_elo_at_score"]
    .agg(["mean"])
    .rename(columns={"mean": "elo_diff"})
    .round(2)
)

# MSD-overrated metric
farm_potential = chart_msd[chart_diff.index] / chart_diff["elo_diff"]
farm_potential /= farm_potential.median()

chart_diff["msd_overrated"] = farm_potential.round(4)
chart_diff["skillset"] = chart_diff.index.map(chart_skillset)
chart_diff["chart_name"] = chart_diff.index.map(chart_name)

# ──────────────────────────────
# OUTPUT
# ──────────────────────────────
chart_diff = chart_diff.sort_values(["skillset", "elo_diff"],ascending=False)
chart_diff.to_csv(OUT_CSV)
chart_diff.to_markdown(OUT_MD)

print(f"Wrote {OUT_CSV} / {OUT_MD}")
