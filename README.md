# Etterna Elo Ranking

Uses player data to approximate players' **Elo rating**.

## What is Elo rating?

The **Elo rating system** is a method for calculating the relative skill levels of players. Elo assigns players numeric ratings that increase or decrease based on wins, losses, and draws against opponents. 

Higher Elo indicates higher skill, and differences in ratings predict the outcome probabilities of matches between players.

## Rating Table
_Last updated: Jul 5, 2025_

Full table (400 players) [here](/output/elo_dtw_ord_skillsets.md). Peak rating table [here](/output/elo_dtw_ord_peak_skillsets.md). 

Note that because the player pool was much smaller in earlier years, Elo inflation occurs quickly.
Ratings achieved in different eras are therefore generally not comparable.

---

## How matches are constructed:

Player scores are filtered leaving only scores with accuracy between **89.0% and 99.0%**.

For each chart and skillset (e.g., `stream`, `jumpstream`, `handstream` (defined as largest MSD)), player scores are matched as follows:

- When a player achieves a **new score** on a chart, it is paired against the **current personal best scores** of all other players on the same chart.
- Each pairing becomes one "match". 

All pairs from the same new personal best are processed as a single batch: player's rating update is accumulated and applied once after all comparisons.

The outcome of a match is a continuous score $S_A \in [0,1]$ for player A
computed from two differences:

```math
x_1 = \log\!\Bigl(\tfrac{\text{rate}_A}{\text{rate}_B}\Bigr), 
\qquad
x_2 = \text{WIFE}_A - \text{WIFE}_B
```

These inputs feed a logistic curve:

```math
S_A = \sigma\!\bigl(\alpha \, x_1 + \beta \, x_2\bigr),
\quad \sigma(z)=\frac{1}{1 + e^{-z}}
```

where $\alpha$ and $\beta$ are tunable parameters, determining equivalence of different (rate, wife) pairs (see [this](/scripts/explore_params.ipynb)).

## Elo Update formula:

The Elo ratings are updated after each match using the standard Elo formula, with an additional **time-decay parameter ($\tau$)** that gives less weight to matches where player scores are far apart in time:

```math
R_{A,\text{new}} = R_{A,\text{old}} + K_{\text{eff}} \cdot (S_A - E_A)
```

where:

- $R_{A,\text{old}}$ and $R_{A,\text{new}}$ are player A’s Elo ratings before and after the match.
- $K_{\text{eff}}$ is the effective K-factor adjusted by time-decay:
  
```math
K_{\text{eff}} = K \cdot e^{-\Delta / \tau}
```

  Here:
  - $K$ is the base update step (typically 10–30).
  - $\Delta$ is the absolute difference in days between the two matched scores.
  - $\tau$ (**tau parameter**) controls how quickly the weight decays with time (measured in days). Larger values of $\tau$ mean slower decay; set $\tau = \infty$ to disable time decay entirely.

- $S_A$ is the actual score from the match.
- $E_A$ is the expected score, calculated as:
  
```math
E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}}
```

A large Elo difference reduces the rating changes for expected outcomes, while close-in-time matches have a stronger effect on ratings compared to matches separated by long periods.


## Running the code:

Install [UV](https://docs.astral.sh/uv/).

Initialize the project environment:
```bash
uv init
uv venv
uv sync
```

Download scores from the EtternaOnline:
```bash
uv run scripts/scrapper.py
```
Or from [releases](https://github.com/MaidOfFire/etterna-elo-ranking/releases)

Run Elo rating script:
```bash
uv run scripts/run_elo.py
```

Results will appear under the output/ directory

## Charts Elo difficulty

In addition to player ratings, the pipeline also assigns each chart a difficulty score by estimating the Elo a player would need to achieve 93% WIFE at 1.0× rate, assuming a linear relationship between rate and Elo (e.g. 1.0× ~ 1000 elo, 1.2× ~ 1200 elo, etc.).


**_msd_overrated_**, a very rough estimation of whether the chart overrated (> 1.0) or underrated (< 1.0) in terms of MDS.

Sample of chart difficulty table:

               
Full table (5307 charts) [here](/output/chart_elo_diff.md).


