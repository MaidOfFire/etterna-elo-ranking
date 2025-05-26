# ============================================================================
# Etterna Scores Scraper
# Collects global-leaderboard usernames in a chosen rank range, downloads all
# their scores via the EtternaOnline API, and stores everything as compressed
# Parquet files in a local folder.
# Tested in Colab and plain-Python environments.
# ============================================================================

# ---------------------------------------------------------------------------
# 0.  LOCAL OUTPUT SET-UP  
# ---------------------------------------------------------------------------
from pathlib import Path

OUTDIR = Path("output/scores")          # local destination for all Parquet files
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1.  CONFIGURATION
# ---------------------------------------------------------------------------
START_RANK = 1           # inclusive, 1-based
END_RANK   = 270         # inclusive, 1-based
assert 1 <= START_RANK <= END_RANK

PARQUET_COMPRESSION = "zstd"   # options: "zstd", "snappy", "gzip", …
PARQUET_LEVEL       = 5        # zstd: 1-22. 5-7 ≈ best size/speed trade-off
MAX_RETRIES_PER_PAGE = 20      # very high; effectively “try until it works”

# ---------------------------------------------------------------------------
# 2.  LIBRARIES  &  HTTP SESSION
# ---------------------------------------------------------------------------
import time, requests, math, pandas as pd
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

def safe_get(url, **kwargs):
    """GET with infinite (or capped) retries + exponential back-off."""
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
            print(".", end="", flush=True)
            time.sleep(sleep_s)

usernames = []
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
# 4.  FETCH SCORES FOR ONE USER
# ---------------------------------------------------------------------------
TRASH = {
    "background", "backgroundTinyThumb", "backgroundSrcSet",
    "banner",      "bannerTinyThumb",      "bannerSrcSet",
}

def fetch_scores(username: str) -> pd.DataFrame:
    base   = f"https://api.etternaonline.com/api/users/{username}/scores"
    params = {"limit": 25, "sort": "-datetime", "filter[valid]": 1}

    def get(page: int):
        return safe_get(base, params={**params, "page": page}).json()

    first   = get(1)
    rows    = first["data"]
    last_pg = first["meta"]["last_page"]

    for p in tqdm(range(2, last_pg + 1), desc=f"{username:>12}", leave=False):
        rows.extend(get(p)["data"])

    df = pd.DataFrame.from_records(rows)
    if "song" in df.columns:
        df["song"] = df["song"].apply(
            lambda d: {k: v for k, v in d.items() if k not in TRASH}
        )
    df.insert(0, "player", username)
    return df

# ---------------------------------------------------------------------------
# 5.  DOWNLOAD EACH PLAYER  &  SAVE AS COMPRESSED PARQUET
# ---------------------------------------------------------------------------
frames = []
for name in usernames:
    try:
        df = fetch_scores(name)
    except Exception as e:
        print(f"\n!! Skipping {name} – {e}")
        continue

    df.to_parquet(
        OUTDIR / f"score_data_{name}.parquet",
        index=False,
        compression=PARQUET_COMPRESSION,
        compression_level=PARQUET_LEVEL,
    )
    frames.append(df)

# combined file
big_df = pd.concat(frames, ignore_index=True)
big_df.to_parquet(
    OUTDIR / f"scores_{START_RANK:03d}_{END_RANK:03d}.parquet",
    index=False,
    compression=PARQUET_COMPRESSION,
    compression_level=PARQUET_LEVEL,
)

print(f"\nCombined rows: {len(big_df):,}")
print(f"Files saved under: {OUTDIR.resolve()}")
