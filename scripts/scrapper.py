#!/usr/bin/env python3
"""
Etterna Scores Scraper & *Incremental Updater*
================================================
Downloads new scores for each player in a chosen rank range and stores
**one Parquet per player**.  It *no longer* rebuilds the combined
rank‑range Parquet, eliminating the heavy CPU / disk phase the original
script triggered.

Run this from a cron/systemd timer or with `nohup` — subsequent runs are
fast when nothing new is available.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# 0.  LOCAL OUTPUT SET‑UP
# ---------------------------------------------------------------------------
from pathlib import Path
OUTDIR = Path("output/scores")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1.  CONFIGURATION
# ---------------------------------------------------------------------------
START_RANK = 1         # inclusive, 1‑based
END_RANK   = 270       # inclusive, 1‑based
assert 1 <= START_RANK <= END_RANK

PARQUET_COMPRESSION = "zstd"   # "zstd", "snappy", "gzip", …
PARQUET_LEVEL       = 5        # zstd: 1‑22
MAX_RETRIES_PER_PAGE = 20      # capped exponential back‑off

# ---------------------------------------------------------------------------
# 2.  LIBRARIES  &  HTTP SESSION
# ---------------------------------------------------------------------------
import time, requests, pandas as pd
from tqdm.auto import tqdm

session = requests.Session()
session.headers.update({"Accept": "application/json"})

LEADER_URL   = "https://api.etternaonline.com/api/leaderboards/global"
PAGE_SIZE_LB = 25

# ---------------------------------------------------------------------------
# 3.  COLLECT USERNAMES IN DESIRED RANK RANGE
# ---------------------------------------------------------------------------
page_from   = (START_RANK - 1) // PAGE_SIZE_LB + 1
page_to     = (END_RANK   - 1) // PAGE_SIZE_LB + 1
rows_needed = END_RANK - START_RANK + 1


def safe_get(url: str, **kwargs):
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
            sleep_s = min(60, 2 ** min(10, attempt))
            print(".", end="", flush=True)
            time.sleep(sleep_s)

usernames: list[str] = []
with tqdm(total=rows_needed, desc="Leaderboard") as pbar:
    for page in range(page_from, page_to + 1):
        data = safe_get(LEADER_URL, params={"page": page}).json()["data"]
        for i, entry in enumerate(data, start=1):
            rank = (page - 1) * PAGE_SIZE_LB + i
            if START_RANK <= rank <= END_RANK:
                usernames.append(entry["username"])
                pbar.update(1)
            if len(usernames) == rows_needed:
                break
        if len(usernames) == rows_needed:
            break
print(f"Collected {len(usernames)} usernames (ranks {START_RANK}-{END_RANK})")

# ---------------------------------------------------------------------------
# 4.  INCREMENTAL SCORE FETCHING
# ---------------------------------------------------------------------------
TRASH = {
    "background", "backgroundTinyThumb", "backgroundSrcSet",
    "banner",      "bannerTinyThumb",      "bannerSrcSet",
}

from datetime import datetime, timezone

def most_recent_dt(path: Path) -> datetime | None:
    """Return latest datetime in an existing parquet (UTC‑naïve)."""
    if not path.exists():
        return None
    col = pd.read_parquet(path, columns=["datetime"])["datetime"]
    return pd.to_datetime(col, utc=True).max().tz_convert(None)


def fetch_scores(username: str, since: datetime | None = None) -> pd.DataFrame:
    """Download *only* scores newer than `since`."""
    base   = f"https://api.etternaonline.com/api/users/{username}/scores"
    params = {"limit": 25, "sort": "-datetime", "filter[valid]": 1}

    def get(page: int):
        return safe_get(base, params={**params, "page": page}).json()

    rows, page = [], 1
    while True:
        chunk = get(page)
        stop_early = False
        for entry in chunk["data"]:
            ts = pd.to_datetime(entry["datetime"], utc=True).tz_convert(None)
            if since is not None and ts <= since:
                stop_early = True
                break
            rows.append(entry)
        if stop_early or page >= chunk["meta"]["last_page"]:
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
# 5.  UPDATE / APPEND PER‑PLAYER PARQUETS
# ---------------------------------------------------------------------------
frames: list[pd.DataFrame] = []
new_rows_total = 0
for name in usernames:
    path = OUTDIR / f"score_data_{name}.parquet"
    since_dt = most_recent_dt(path)
    new_df = fetch_scores(name, since=since_dt)

    if new_df.empty:
        print(f"{name:>12}: up‑to‑date")
        if path.exists():
            frames.append(pd.read_parquet(path))
        continue

    new_rows_total += len(new_df)

    if path.exists():
        old_df = pd.read_parquet(path)
        full = pd.concat([old_df, new_df], ignore_index=True).drop_duplicates(subset="id")
    else:
        full = new_df

    full.to_parquet(
        path,
        index=False,
        compression=PARQUET_COMPRESSION,
        compression_level=PARQUET_LEVEL,
    )
    frames.append(full)
    print(f"{name:>12}: +{len(new_df)} new scores")

# ---------------------------------------------------------------------------
# 6.  SUMMARY — *no combined parquet is created*
# ---------------------------------------------------------------------------
print("\nDone.")
print(f"Players processed: {len(frames)}")
print(f"New rows added  : {new_rows_total:,}")
print("Combined parquet was intentionally skipped to save CPU and disk I/O.")
