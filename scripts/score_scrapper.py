import requests
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from time import sleep

# Configuration
OUTDIR = Path("output/replays")
OUTDIR.mkdir(parents=True, exist_ok=True)

def get_play_data(player_name: str, score_id: int, auth_token: str, retries: int = 5, delay: int = 5):
    url = f"https://api.etternaonline.com/api/users/{player_name}/scores/{score_id}/replay"

    headers = {
        "Authorization": auth_token,
        "Accept": "application/json",
        "Accept-Encoding": "gzip, deflate, br",
        "Origin": "https://etternaonline.com",
        "Referer": "https://etternaonline.com/",
        "User-Agent": "Mozilla/5.0"
    }

    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            json_data = response.json()
            return np.array(json_data)

        except requests.RequestException as e:
            print(f"Request error for score {score_id} on attempt {attempt}/{retries}: {e}")
            if attempt < retries:
                sleep(delay)
            else:
                print(f"All retries failed for score {score_id}")

        except json.JSONDecodeError as e:
            print(f"JSON decoding failed for score {score_id}: {e}")
            if 'response' in locals():
                print(f"Response content: {response.content[:200]}")
            break

    return None


def download_replays_from_csvs(score_csv_path: str, player_csv_path: str, auth_token: str):
    scores_df = pd.read_csv(score_csv_path, index_col=0)
    players_df = pd.read_csv(player_csv_path, index_col=0)

    for (idx1, score_ids), (idx2, player_names) in tqdm(zip(scores_df.itertuples(), players_df.itertuples()), total=len(scores_df), desc="Downloading replays"):
        chart_id = idx1
        score_ids = np.fromstring(score_ids.strip("[]"), sep=" ", dtype=int)
        player_names = np.array(player_names.strip("[]").replace("'", "").split(), dtype=str)

        for score_id, player_name in zip(score_ids, player_names):
            replay_data = get_play_data(player_name, score_id, auth_token)
            if replay_data is not None:
                np.save(OUTDIR / f"replay_{chart_id}_{player_name}_{score_id}.npy", replay_data)
                print(f"Saved replay for chart {chart_id}, player {player_name}, score {score_id}")
                break
        else:
            print(f"Failed to download replay for chart {chart_id}")


# Example usage
score_csv_file_path = "output/chart_id_score_ids.csv"
player_csv_file_path = "output/chart_id_score_playernames.csv"
auth_token = "your_auth_token_here"

download_replays_from_csvs(score_csv_file_path, player_csv_file_path, auth_token)
