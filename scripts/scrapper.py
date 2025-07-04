#!/usr/bin/env python3
"""
Etterna Scores Scraper & Updater

Collects global‑leaderboard usernames in a chosen rank range,
downloads their scores via the EtternaOnline API, and stores everything
as compressed Parquet files in a local folder.

If run again, it only fetches scores newer than the most recent already
stored per‑player, so subsequent runs are fast.

Tested on Python 3.10+
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# 0. LOCAL OUTPUT SET‑UP
# ---------------------------------------------------------------------------
OUTDIR = Path("output/scores")  # destination folder for all Parquet files
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------
START_RANK = 1  # inclusive, 1‑based
END_RANK = 270  # inclusive, 1‑based
assert 1 <= START_RANK <= END_RANK, "invalid rank range"

PARQUET_COMPRESSION = "zstd"  # "zstd", "snappy", "gzip", …
PARQUET_LEVEL = 5  # zstd: 1‑22. 5‑7 ≈ best size/speed trade‑off
MAX_RETRIES_PER_PAGE = 20  # very high; effectively “try until it works”
PAGE_SIZE_LB = 25  # leaderboard API rows per page

# ---------------------------------------------------------------------------
# 2. HTTP SESSION
# ---------------------------------------------------------------------------
session = requests.Session()
session.headers.update({"Accept": "application/json"})
LEADER_URL = "https://api.etternaonline.com/api/leaderboards/global"

# ---------------------------------------------------------------------------
# 3. HELPER – ROBUST GET
# ---------------------------------------------------------------------------

def safe_get(url: str, **kwargs) -> requests.Response:
    """GET with retries + exponential back‑off."""
    attempt = 0
    while True:
        try:
            r = session.get(url, timeout=30, **kwargs)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            attempt += 1
            if attempt >= MAX_RETRIES_PER_PAGE:
                raise RuntimeError(f"failed after {attempt} attempts: {e}") from e
            sleep_s = min(60, 2 ** min(10, attempt))  # cap wait at 60 s
            print(f"· (retry {attempt})", end="", flush=True)
            time.sleep(sleep_s)


# ---------------------------------------------------------------------------
# 4. COLLECT USERNAMES IN DESIRED RANK RANGE
# ---------------------------------------------------------------------------

def collect_usernames(start_rank: int, end_rank: int) -> list[str]:
    page_from = (start_rank - 1) // PAGE_SIZE_LB + 1
    page_to = (end_rank - 1) // PAGE_SIZE_LB + 1
    rows_needed = end_rank - start_rank + 1

    names: list[str] = []
    with tqdm(total=rows_needed, desc="Leaderboard") as pbar:
        for page in range(page_from, page_to + 1):
            data = safe_get(LEADER_URL, params={"page": page}).json()["data"]
            for i, entry in enumerate(data, start=1):
                rank = (page - 1) * PAGE_SIZE_LB + i
                if start_rank <= rank <= end_rank:
                    names.append(entry["username"])
                    pbar.update(1)
                if len(names) == rows_needed:
                    break
            if len(names) == rows_needed:
                break
    return names


# ---------------------------------------------------------------------------
# 5. FETCH SCORES (INCREMENTAL)
# ---------------------------------------------------------------------------
TRASH = {
    "background",
    "backgroundTinyThumb",
    "backgroundSrcSet",
    "banner",
    "bannerTinyThumb",
    "bannerSrcSet",
}


def most_recent_dt(path: Path) -> datetime | None:
    """Return most recent datetime inside existing parquet (UTC‑naïve)."""
    if not path.exists():
        return None
    col = pd.read_parquet(path, columns=["datetime"])["datetime"]
    return (
        pd.to_datetime(col, utc=True).max().tz_convert(None).to_pydatetime()  # type: ignore[arg-type]
    )


def fetch_scores(username: str, since: datetime | None = None) -> pd.DataFrame:
    """Download only scores newer than *since* (UTC‑naïve)."""
    base = f"https://api.etternaonline.com/api/users/{username}/scores"
    params = {"limit": 25, "sort": "-datetime", "filter[valid]": 1}

    def get(page: int):
        return safe_get(base, params={**params, "page": page}).json()

    rows: list[dict] = []
    page = 1
    while True:
        chunk = get(page)
        stop = False
        for entry in chunk["data"]:
            ts = (
                pd.to_datetime(entry["datetime"], utc=True)
                .tz_convert(None)
                .to_pydatetime()
            )
            if since is not None and ts <= since:
                stop = True
                break
            rows.append(entry)
        if stop or page >= chunk["meta"]["last_page"]:
            break
        page += 1

    df = pd.DataFrame.from_records(rows)
    if df.empty:
        return df

    if "song" in df.columns:
        df["song"] = df["song"].apply(lambda d: {k: v for k, v in d.items() if k not in TRASH})
    df.insert(0, "player", username)
    return df


# ---------------------------------------------------------------------------
# 6. MAIN PIPELINE
# ---------------------------------------------------------------------------

def main() -> None:
    usernames = collect_usernames(START_RANK, END_RANK)
    print(f"Collected {len(usernames)} usernames (ranks {START_RANK}-{END_RANK})")

    combined: list[pd.DataFrame] = []

    for name in usernames:
        path = OUTDIR / f"score_data_{name}.parquet"
        since_dt = most_recent_dt(path)
        new_df = fetch_scores(name, since=since_dt)

        if new_df.empty:
            print(f"{name:>12}: up-to-date")
            if path.exists():
                combined.append(pd.read_parquet(path))
            continue

        if path.exists():
            old_df = pd.read_parquet(path)
            full = pd.concat([old_df, new_df], ignore_index=True)
            full = full.drop_duplicates(subset="id")
        else:
            full = new_df

        full.to_parquet(
            path,
            index=False,
            compression=PARQUET_COMPRESSION,
            compression_level=PARQUET_LEVEL,
        )
        combined.append(full)
        print(f"{name:>12}: +{len(new_df)} new scores")

    if not combined:
        print("Nothing to update.")
        return

    big_df = pd.concat(combined, ignore_index=True)
    big_df.to_parquet(
        OUTDIR / f"scores_{START_RANK:03d}_{END_RANK:03d}.parquet",
        index=False,
        compression=PARQUET_COMPRESSION,
        compression_level=PARQUET_LEVEL,
    )
    print(f"\nCombined rows: {len(big_df):,}")
    print(f"Files saved under: {OUTDIR.resolve()}")


# ---------------------------------------------------------------------------
# 7. SCRIPT ENTRY‑POINT
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
