#!/usr/bin/env python3
"""
calculate_elo_dtw_ord.py — thin wrapper that relies on **elo_core.py**
for all shared logic and constants.  This removes duplication and guarantees
consistency with the evaluation / tuning scripts.
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
# I/O PATHS (edit as needed)
# ──────────────────────────────
SCORES_DIR  = Path("output/scores")
OUTFILE_CSV = Path("output/elo_dtw_ord_skillsets.csv")
OUTFILE_MD  = Path("output/elo_dtw_ord_skillsets.md")

# ──────────────────────────────
# Main workflow
# ──────────────────────────────

def compute_elo_all(data: pd.DataFrame) -> pd.DataFrame:
    """Return rating table with per‑skill‑set columns and overall score."""
    cols = []
    for sk in SKILLSETS:
        matches = build_matches_for_skillset(data, sk)
        print(f"Found {len(matches)} matches for skillset '{sk}'")
        elo_sk = run_elo(matches)
        elo_sk.name = sk
        cols.append(elo_sk)

    df = pd.concat(cols, axis=1).fillna(0)
    df["overall"] = df[SKILLSETS].apply(lambda r: r.nlargest(3).mean(), axis=1)
    ordered = ["overall"] + [c for c in df.columns if c != "overall"]
    return df[ordered].sort_values("overall", ascending=False)


def main() -> None:
    data = load_scores(SCORES_DIR)
    elo_df = compute_elo_all(data)

    # save outputs
    OUTFILE_CSV.parent.mkdir(parents=True, exist_ok=True)
    elo_df.round(0).astype(int).to_csv(OUTFILE_CSV)
    elo_df.round(0).astype(int).to_markdown(OUTFILE_MD)
    print(f"Wrote Elo table to {OUTFILE_CSV} and {OUTFILE_MD}")


if __name__ == "__main__":
    main()
