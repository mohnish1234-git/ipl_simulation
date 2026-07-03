"""
Data Cleaning Pipeline
Cleans raw ipl_final.csv and produces a standardized dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path


RAW_PATH = Path("data/raw/ipl_final.csv")
PROCESSED_DIR = Path("data/processed")


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows × {df.shape[1]} columns")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── Parse date → proper datetime ────────────────────────────────────────
    # Raw format is "18,Apr,2008"
    df["date"] = pd.to_datetime(df["date"], format="%d,%b,%Y", errors="coerce")

    # ── Derive ball number within over ──────────────────────────────────────
    # 'over' column is float like 0.1, 0.2 … 0.6, 1.1, 1.2 …
    df["over_num"] = df["over"].astype(int)               # 0-indexed over (0–19)
    df["ball_num"] = (df["over"].round(1) * 10 % 10).astype(int)  # 1–6

    # ── Wicket flag ──────────────────────────────────────────────────────────
    df["is_wicket"] = df["wicket_type"].notna().astype(int)

    # ── Legal delivery flag (wides & no-balls don't count toward over) ───────
    df["is_legal"] = ((df["wide"] == 0) & (df["noballs"] == 0)).astype(int)

    # ── Boundary / six flags ─────────────────────────────────────────────────
    df["is_four"] = (df["runs_of_bat"] == 4).astype(int)
    df["is_six"]  = (df["runs_of_bat"] == 6).astype(int)
    df["is_dot"]  = ((df["runs_of_bat"] == 0) & (df["is_wicket"] == 0)).astype(int)

    # ── Total runs per ball ──────────────────────────────────────────────────
    df["total_runs"] = df["runs_of_bat"] + df["extras"]

    # ── Outcome label ────────────────────────────────────────────────────────
    # Values: 0, 1, 2, 3, 4, 6, W
    def ball_outcome(row):
        if row["is_wicket"] and row["runs_of_bat"] == 0:
            return "W"
        return str(int(row["runs_of_bat"]))

    df["outcome"] = df.apply(ball_outcome, axis=1)

    # ── Phase label ──────────────────────────────────────────────────────────
    df["phase"] = df["over_num"].apply(_phase_label)

    # ── Season weight for recency (exponential decay, half-life = 3 seasons) ─
    # Most recent season gets weight 1.0; each prior season is discounted.
    max_season = df["season"].max()
    HALF_LIFE  = 3.0
    decay_rate = np.log(2) / HALF_LIFE
    df["season_weight"] = np.exp(-decay_rate * (max_season - df["season"]))

    # ── Fill missing strings ─────────────────────────────────────────────────
    for col in ["venue", "batting_team", "bowling_team", "striker", "bowler"]:
        df[col] = df[col].fillna("Unknown").str.strip()

    # ── Drop deliveries with missing critical fields ─────────────────────────
    df.dropna(subset=["striker", "bowler", "innings", "over"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"After cleaning: {len(df):,} rows")
    print(f"Season weight range: {df['season_weight'].min():.4f} – {df['season_weight'].max():.4f}")
    return df


def _phase_label(over: int) -> str:
    if over < 6:
        return "powerplay"
    if over < 15:
        return "middle"
    return "death"


if __name__ == "__main__":
    df = load_raw()
    df = clean(df)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "cleaned.csv"
    df.to_csv(out, index=False)
    print(f"Saved → {out}")