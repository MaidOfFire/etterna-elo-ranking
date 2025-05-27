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
def compute_tables(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (current_df, peak_df) with per-skill-set columns."""
    curr_cols, peak_cols = [], []

    for sk in SKILLSETS:
        matches = build_matches_for_skillset(data, sk)
        print(f"Found {len(matches)} matches for skillset '{sk}'")
        res = run_elo(matches)           # DataFrame with 'elo' and 'peak'
        curr = res["elo"].rename(sk)
        peak = res["peak"].rename(sk)
        curr_cols.append(curr)
        peak_cols.append(peak)

    curr_df = pd.concat(curr_cols, axis=1).fillna(0)
    peak_df = pd.concat(peak_cols, axis=1).fillna(0)

    curr_df["overall"] = curr_df[SKILLSETS].apply(lambda r: r.nlargest(3).mean(), axis=1)
    peak_df["overall"] = peak_df[SKILLSETS].apply(lambda r: r.nlargest(3).mean(), axis=1)

    order = ["overall"] + [c for c in curr_df.columns if c != "overall"]
    return (curr_df[order].sort_values("overall", ascending=False),
            peak_df[order].sort_values("overall", ascending=False))

# ──────────────────────────────
def main() -> None:
    data = load_scores(SCORES_DIR)
    curr_df, peak_df = compute_tables(data)

    # ensure target folder exists
    OUT_CURR_CSV.parent.mkdir(parents=True, exist_ok=True)

    # save current ratings
    curr_df.round(0).astype(int).to_csv(OUT_CURR_CSV)
    curr_df.round(0).astype(int).to_markdown(OUT_CURR_MD)

    # save peak ratings
    peak_df.round(0).astype(int).to_csv(OUT_PEAK_CSV)
    peak_df.round(0).astype(int).to_markdown(OUT_PEAK_MD)

    print(f"Current ratings  → {OUT_CURR_CSV} / {OUT_CURR_MD}")
    print(f"Peak    ratings  → {OUT_PEAK_CSV} / {OUT_PEAK_MD}")

# ──────────────────────────────
if __name__ == "__main__":
    main()
