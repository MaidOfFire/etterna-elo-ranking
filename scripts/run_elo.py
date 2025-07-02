#!/usr/bin/env python3
"""
run_elo.py — builds two tables:

1.  Current Elo per skill-set  →  output/elo_dtw_ord_skillsets.{csv,md}
2.  Peak   Elo per skill-set  →  output/elo_dtw_ord_peak_skillsets.{csv,md}

"""
from pathlib import Path
import pandas as pd

from elo_core import (
    SKILLSETS,
    load_scores,
    build_matches_for_skillset,
    run_elo,                
)

# ──────────────────────────────
# I/O PATHS
# ──────────────────────────────
SCORES_DIR          = Path("output/scores")
OUT_CURR_CSV        = Path("output/elo_dtw_ord_skillsets.csv")
OUT_CURR_MD         = Path("output/elo_dtw_ord_skillsets.md")
OUT_PEAK_CSV        = Path("output/elo_dtw_ord_peak_skillsets.csv")
OUT_PEAK_MD         = Path("output/elo_dtw_ord_peak_skillsets.md")

# ──────────────────────────────
def compute_tables_and_history(data: pd.DataFrame):
    curr_cols, peak_cols, hist_frames = [], [], []

    for sk in SKILLSETS:
        matches = build_matches_for_skillset(data, sk)
        print(f"Found {len(matches)} matches for skillset '{sk}'")

        final_df, hist_df = run_elo(matches, return_history=True)
        hist_df["skillset"] = sk        # tag for later merging
        hist_frames.append(hist_df)

        curr_cols.append(final_df["elo"].rename(sk))
        peak_cols.append(final_df["peak"].rename(sk))

    # rating tables (unchanged)
    curr_df = pd.concat(curr_cols, axis=1).fillna(0)
    peak_df = pd.concat(peak_cols, axis=1).fillna(0)
    curr_df["overall"] = curr_df[SKILLSETS].apply(lambda r: r.nlargest(3).mean(), axis=1)
    peak_df["overall"] = peak_df[SKILLSETS].apply(lambda r: r.nlargest(3).mean(), axis=1)

    # NEW: per-score ratings across all skill-sets
    history_df = pd.concat(hist_frames).reset_index()  # index == score_id
    return (
        curr_df[["overall"] + SKILLSETS].sort_values("overall", ascending=False),
        peak_df[["overall"] + SKILLSETS].sort_values("overall", ascending=False),
        history_df,
    )
# ──────────────────────────────
def main() -> None:
    data = load_scores(SCORES_DIR)
    curr_df, peak_df, history_df = compute_tables_and_history(data)

    # ensure target folder exists
    OUT_CURR_CSV.parent.mkdir(parents=True, exist_ok=True)

    # save current ratings
    curr_df.round(0).astype(int).to_csv(OUT_CURR_CSV)
    curr_df.round(0).astype(int).to_markdown(OUT_CURR_MD)

    # save peak ratings
    peak_df.round(0).astype(int).to_csv(OUT_PEAK_CSV)
    peak_df.round(0).astype(int).to_markdown(OUT_PEAK_MD)

    history_df.to_csv("output/elo_by_score.csv")

    print(f"Current ratings  → {OUT_CURR_CSV} / {OUT_CURR_MD}")
    print(f"Peak    ratings  → {OUT_PEAK_CSV} / {OUT_PEAK_MD}")

# ──────────────────────────────
if __name__ == "__main__":
    main()
