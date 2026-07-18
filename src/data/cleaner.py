"""
Data Cleaning Pipeline
Cleans raw ipl_final.csv and produces a standardized dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from .team_mapping import canonicalize_team, print_unmapped_teams

RAW_PATH = Path("data/raw/ipl_final.csv")
PROCESSED_DIR = Path("data/processed")

# Set True once, run prepare_data.py, read the printout, then set back to
# False. Lets you see every raw team string that didn't match a known
# alias before you silently misgroup that data.
DIAGNOSE_UNMAPPED_VENUES = False

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
    # Standard 7-class outcome space used throughout this whole pipeline:
    # {0,1,2,3,4,6,W}. predictor.py, calibration.py, and Colab's num_class
    # all hardcode exactly these 7 classes.
    #
    # A wicket ALWAYS labels the ball "W", regardless of runs completed
    # before the dismissal. The previous version only did this when
    # runs_of_bat == 0, so a run-out after the batters had already crossed
    # for 1+ runs got labeled with the run count instead of "W" — silently
    # dropping that wicket from the classification target entirely. This
    # matters: run-outs are a real, non-trivial share of dismissals, and
    # they were being taught to the model as "just a run", not a wicket.
    #
    # Any run value outside {0,1,2,3,4,6} (a rare "5", or an even rarer 7+
    # from extreme overthrows) is folded into the nearest standard class —
    # simpler than expanding every downstream file to handle an 8th class
    # for what is a genuine but very rare scoring edge case.
    _STANDARD_RUNS = (0, 1, 2, 3, 4, 6)

    def ball_outcome(row):
        if row["is_wicket"]:
            return "W"
        runs = int(row["runs_of_bat"])
        if runs not in _STANDARD_RUNS:
            runs = min(_STANDARD_RUNS, key=lambda x: abs(x - runs))
        return str(runs)

    df["outcome"] = df.apply(ball_outcome, axis=1)

    # ── Phase label ──────────────────────────────────────────────────────────
    df["phase"] = df["over_num"].apply(_phase_label)

    # ── Season weight for recency (exponential decay, half-life = 3 seasons) ─
    # Most recent season gets weight 1.0; each prior season is discounted.
    max_season = df["season"].max()
    HALF_LIFE  = 3.0
    decay_rate = np.log(2) / HALF_LIFE
    df["season_weight"] = np.exp(-decay_rate * (max_season - df["season"]))

    # ── Fill missing strings (except venue/team — canonicalized next) ────────
    for col in ["batting_team", "bowling_team", "striker", "bowler"]:
        df[col] = df[col].fillna("Unknown").str.strip()

    # ── Canonicalize team names ───────────────────────────────────────────────
    # Same rationale as venue canonicalization below: without this, franchise
    # renames (Delhi Daredevils/Delhi Capitals, Kings XI Punjab/Punjab Kings,
    # RCB's city-spelling change, short codes like "CSK" vs full names, etc.)
    # silently fragment one team's history into multiple buckets — for
    # batting_team/bowling_team encoding, team-level features, AND
    # prepare_data.py's batters_by_team/bowlers_by_team roster grouping.

    for col in ["batting_team", "bowling_team"]:
        canon = df[col].map(canonicalize_team)
        # Rows whose team string isn't a known franchise/alias keep their
        # original ("Unknown" or the raw string) rather than becoming NaN —
        # unlike venues, we don't drop rows here, since a batting_team we
        # can't canonicalize doesn't invalidate the delivery the way an
        # off-allowlist venue does.
        df[col] = canon.where(canon.notna(), df[col])

    if DIAGNOSE_UNMAPPED_VENUES:
        print_unmapped_teams(df)

    ALLOWED_VENUES = [
        "MA Chidambaram Stadium, Chennai",
        "Wankhede Stadium, Mumbai",
        "Arun Jaitley Stadium, Delhi",
        "Eden Gardens, Kolkata",
        "Narendra Modi Stadium, Ahmedabad",
        "Shaheed Veer Narayan Singh International Stadium, Raipur",
        "Barsapara Cricket Stadium, Guwahati",
        "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam",
        "Rajiv Gandhi International Stadium, Hyderabad",
        "M Chinnaswamy Stadium, Bengaluru",
        "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow",
        "Sawai Mansingh Stadium, Jaipur",
        "Himachal Pradesh Cricket Association Stadium, Dharamsala",
        "Maharaja Yadavindra Singh International Cricket Stadium, New Chandigarh",
    ]

    before = len(df)
    df = df[df["venue"].isin(ALLOWED_VENUES)].reset_index(drop=True)

    print(
        f"Venue filter: {before:,} rows → {len(df):,} rows "
        f"({before - len(df):,} dropped — venue not in the allowed list)"
    )

    for venue in ALLOWED_VENUES:
        print(f"    {venue:<70} {(df['venue'] == venue).sum():,} deliveries")

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