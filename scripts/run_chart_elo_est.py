#!/usr/bin/env python3
"""Compute chart's Elo difficulty."""

from pathlib import Path

import numpy as np
import pandas as pd

from elo_core import load_scores, RATE_DIFF_SCALE, WIFE_DIFF_SCALE


# ──────────────────────────────
# CONFIG
# ──────────────────────────────
CHART_PLAYCOUNT_THRESHOLD = 15
PLAYER_PLAYCOUNT_THRESHOLD = 15

SCORES_DIR = Path("output/scores")
HISTORY_CSV = Path("output/elo_by_score.csv")
OUT_CSV = Path("output/chart_elo_diff.csv")
OUT_MD = Path("output/chart_elo_diff.md")

# ──────────────────────────────
# LOAD & PREPARE SCORE DATA
# ──────────────────────────────
scores_full = load_scores(SCORES_DIR)
scores_full = scores_full[~scores_full["id"].duplicated()]

scores_full["pseudo_rate"] = (
    scores_full["rate"]
    * np.exp((WIFE_DIFF_SCALE / RATE_DIFF_SCALE) * (scores_full["wife"] - 93)) #Based on the outcome formula
)

scores = scores_full[["id", "chart_id", "rate", "wife", "pseudo_rate"]].copy()

history = pd.read_csv(HISTORY_CSV)         

history['score_number'] = (history
    .sort_values(['player', 'skillset', 'datetime'])  
    .groupby(['player', 'skillset'])
    .cumcount()                                       
    .add(1)                                            
    .sort_index()                                      
)
scores_with_rating = scores.merge(
    history,
    left_on="id",
    right_on="score_id",
    how="inner"       
)

valid_scores = scores_with_rating[scores_with_rating["score_number"]>PLAYER_PLAYCOUNT_THRESHOLD].id.to_numpy()
scores = scores[scores.id.isin(valid_scores)]
scores_full = scores_full[scores_full.id.isin(valid_scores)]


# keep only charts with enough plays
play_counts = scores.groupby("chart_id")["id"].count()
valid_chart_ids = play_counts[play_counts > CHART_PLAYCOUNT_THRESHOLD].index


scores = scores[scores["chart_id"].isin(valid_chart_ids)]
scores_full = scores_full[scores_full["chart_id"].isin(valid_chart_ids)]

scores_with_rating = scores_with_rating[scores_with_rating.id.isin(scores.id)]

# adjust Elo by pseudo-rate
rating_df = (
    scores_with_rating[["id","chart_id", "rate", "pseudo_rate", "elo_after_score"]]
    .sort_values("chart_id")
    .copy()
)
rating_df["adj_elo_after_score"] = (
    rating_df["elo_after_score"] / rating_df["pseudo_rate"]
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

score_msd = scores_full.set_index("id").loc[rating_df["id"]]["msd"]

rating_df = rating_df.set_index("id")

farm_potential = np.log(score_msd/rating_df["elo_after_score"])
farm_potential -= (farm_potential).median()
rating_df["overrated"] = farm_potential


# readable chart name: (song, first-pack)
chart_name = {
    cid: (
        scores_full.loc[scores_full["chart_id"] == cid, "song"].iloc[0]["name"],
        scores_full.loc[scores_full["chart_id"] == cid, "song"].iloc[0]["packs"][0]["name"],
    )
    for cid in valid_chart_ids.astype(int)
}

# Elo diff
chart_diff = (
    rating_df.groupby("chart_id")["adj_elo_after_score"]
    .mean()
    .round(2)
    .rename("elo_diff")
    .to_frame()
)

# MSD-overrated metric
chart_diff["msd_overrated"] = np.exp(rating_df.groupby("chart_id")["overrated"].mean()).round(4)

chart_diff["skillset"] = chart_diff.index.map(chart_skillset)
chart_diff["chart_name"] = chart_diff.index.map(chart_name)

# ──────────────────────────────
# OUTPUT
# ──────────────────────────────
chart_diff = chart_diff.sort_values(["skillset", "elo_diff"],ascending=False)
chart_diff.to_csv(OUT_CSV)
chart_diff.to_markdown(OUT_MD)

print(f"Wrote {OUT_CSV} / {OUT_MD}")
