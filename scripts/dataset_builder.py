"""
Fantasy Dataset Builder
========================
Builds the training dataset for the Dream11 fantasy-points regression model.

Pipeline:
  1. Load raw ball-by-ball data and clean it with the EXISTING cleaning
     pipeline (cleaner.clean) — same cleaning the match simulator uses.
  2. Run the EXISTING feature_engineer.build_features() over the full
     cleaned history. That function is already strictly anti-leakage (every
     recency-weighted stat only ever looks at seasons STRICTLY BEFORE the
     row's own season — see feature_engineer.py's module docstring), so its
     bat_/bowl_/bvb_/venue_ columns are safe to reuse here verbatim.
  3. For every (match_id, player) pair, take exactly ONE pre-match feature
     snapshot — these historical columns are already constant across an
     entire match (they're keyed by season, not by ball), so any one row
     for that player in that match carries the correct value. LIVE
     match-state columns (cumulative_runs, crr, momentum, partnership,
     pressure_index, etc.) are explicitly dropped here — they don't exist
     before a ball has been bowled and must never leak into a pre-match
     prediction target.
  4. Independently compute each player's ACTUAL Dream11 fantasy points for
     that match from the cleaned ball-by-ball data (batting + bowling +
     fielding, standard Dream11 T20 scoring). This is the label, and is
     computed from the match's own outcome — it is never used as a feature.
  5. Join pre-match features (X) with actual points (y) on (match_id,
     player), validate, dedupe, fill any residual missing values, and
     export data/processed/fantasy_dataset.csv.

Usage:
    python -m scripts.dataset_builder
"""

from pathlib import Path
import numpy as np
import pandas as pd

from src.data.cleaner import load_raw, clean
import src.data.feature_engineer as fe

PROCESSED_DIR = Path("data/processed")
OUTPUT_PATH = PROCESSED_DIR / "fantasy_dataset.csv"

# ── Columns that describe LIVE, in-match state. These must NEVER be used as
# pre-match prediction features (a fantasy team is picked before a ball is
# bowled), so they're explicitly excluded when building the per-player
# feature snapshot, even though build_features() computes them.
LIVE_MATCH_STATE_COLS = {
    "over_num", "ball_num",
    "cumulative_runs", "cumulative_wickets",
    "balls_remaining", "wickets_remaining", "crr",
    "batter_balls_faced", "batter_runs_scored", "batter_innings_sr",
    "balls_vs_bowler", "runs_vs_bowler",
    "runs_last6", "runs_last12", "runs_last18", "runs_last_over",
    "consec_dots", "consec_boundaries",
    "partnership_runs", "partnership_balls", "partnership_run_rate",
    "current_matchup_sr",
    "prev_ball_outcome", "prev2_ball_outcome", "prev3_ball_outcome",
    "target", "runs_needed", "rrr", "pressure_index",
    "required_runs_per_wicket", "balls_per_required_run",
    "pressure_weighted_rrr", "pressure_weighted_aggression",
    "sample_weight", "outcome",
}

# Columns that are the same for every player in a match (context, not a
# per-player skill signal) — kept, since venue/team/phase-agnostic-season
# info is legitimate pre-match context, just not "live state".
MATCH_CONTEXT_COLS = ["match_id", "season", "date", "venue", "is_batting_first"]


# ═══════════════════════════════════════════════════════════════════════════
# Dream11-style fantasy points (standard T20 scoring)
# ═══════════════════════════════════════════════════════════════════════════

def _batting_points(runs: int, balls: int, fours: int, sixes: int, dismissed: bool) -> float:
    pts = runs * 1.0
    pts += fours * 1.0
    pts += sixes * 2.0
    if runs >= 100:
        pts += 16
    elif runs >= 50:
        pts += 8
    elif runs >= 30:
        pts += 4
    if dismissed and runs == 0:
        pts -= 2
    if balls >= 10:
        sr = runs / balls * 100
        if sr > 170:
            pts += 6
        elif sr >= 150.01:
            pts += 4
        elif sr >= 130:
            pts += 2
        elif sr <= 50 - 1e-9:
            pts -= 6
        elif sr <= 59.99:
            pts -= 4
        elif sr < 70:
            pts -= 2
    return pts


def _bowling_points(wickets: int, balls: int, runs_conceded: int, maidens: int) -> float:
    pts = wickets * 25.0
    if wickets >= 5:
        pts += 16
    elif wickets == 4:
        pts += 8
    pts += maidens * 12.0
    overs = balls / 6.0
    if overs >= 2:
        econ = runs_conceded / overs
        if econ < 5:
            pts += 6
        elif econ <= 5.99:
            pts += 4
        elif econ <= 7:
            pts += 2
        elif econ <= 10:
            pass
        elif econ <= 11:
            pts -= 2
        elif econ <= 12:
            pts -= 4
        else:
            pts -= 6
    return pts


def _compute_actual_fantasy_points(cleaned: pd.DataFrame) -> pd.DataFrame:
    """One row per (match_id, player) with the ACTUAL fantasy points earned
    in that match (batting + bowling [+ fielding if raw data supports it]).
    """
    rows = []

    # ── Batting ────────────────────────────────────────────────────────────
    bat = (
        cleaned[cleaned["is_legal"] == 1]
        .groupby(["match_id", "striker"])
        .agg(
            runs=("runs_of_bat", "sum"),
            balls=("is_legal", "sum"),
            fours=("is_four", "sum"),
            sixes=("is_six", "sum"),
        )
        .reset_index()
    )
    # dismissed = the striker was the one out on a ball charged against them.
    # wicket_type-based is_wicket doesn't tell us WHO was dismissed for
    # non-bowled-style dismissals (e.g. run-outs of the non-striker), but the
    # vast majority of wicket_type dismissals are of the striker on strike;
    # this is a reasonable approximation given the columns available upstream.
    dismissed = (
        cleaned[cleaned["is_wicket"] == 1]
        .groupby(["match_id", "striker"])
        .size()
        .rename("dismissed_count")
        .reset_index()
    )
    bat = bat.merge(dismissed, on=["match_id", "striker"], how="left")
    bat["dismissed_count"] = bat["dismissed_count"].fillna(0)

    for r in bat.itertuples(index=False):
        pts = _batting_points(r.runs, r.balls, r.fours, r.sixes, r.dismissed_count > 0)
        rows.append({"match_id": r.match_id, "player": r.striker, "batting_points": pts})
    bat_pts = pd.DataFrame(rows)

    # ── Bowling ────────────────────────────────────────────────────────────
    bowl_rows = []
    legal = cleaned[cleaned["is_legal"] == 1]
    bowl = legal.groupby(["match_id", "bowler"]).agg(
        balls=("is_legal", "sum"),
        runs_conceded=("total_runs", "sum"),
        wickets=("is_wicket", "sum"),
    ).reset_index()
    # Maidens: overs (6 legal balls bowled by the same bowler, same
    # match/innings/over_num) with zero runs conceded off the bat + extras.
    over_group = legal.groupby(["match_id", "innings", "bowler", "over_num"]).agg(
        legal_balls=("is_legal", "sum"),
        over_runs=("total_runs", "sum"),
    ).reset_index()
    maidens = (
        over_group[(over_group["legal_balls"] == 6) & (over_group["over_runs"] == 0)]
        .groupby(["match_id", "bowler"])
        .size()
        .rename("maidens")
        .reset_index()
    )
    bowl = bowl.merge(maidens, on=["match_id", "bowler"], how="left")
    bowl["maidens"] = bowl["maidens"].fillna(0)

    for r in bowl.itertuples(index=False):
        pts = _bowling_points(r.wickets, r.balls, r.runs_conceded, r.maidens)
        bowl_rows.append({"match_id": r.match_id, "player": r.bowler, "bowling_points": pts})
    bowl_pts = pd.DataFrame(bowl_rows)

    # ── Fielding ───────────────────────────────────────────────────────────
    # The upstream raw schema (see cleaner.py) exposes only `wicket_type`,
    # with no `fielder` / `player_dismissed` columns to attribute a catch,
    # stumping, or run-out to a specific fielder. Fielding points are
    # therefore NOT computable from this data and are set to 0 for every
    # player rather than guessed. If a `fielder` column is added upstream in
    # future, wire it in here (catch=+8, 3-catch bonus=+4, stumping=+12,
    # run-out direct=+12 / thrower+catcher=+6 each).
    print("  ⚠ Fielding points set to 0 for all players — raw data has no "
          "fielder/player_dismissed column to attribute catches/stumpings/"
          "run-outs to a specific player.")

    all_players = pd.concat([
        bat[["match_id", "striker"]].rename(columns={"striker": "player"}),
        bowl[["match_id", "bowler"]].rename(columns={"bowler": "player"}),
    ]).drop_duplicates()

    out = all_players.merge(bat_pts, on=["match_id", "player"], how="left")
    out = out.merge(bowl_pts, on=["match_id", "player"], how="left")
    out["batting_points"] = out["batting_points"].fillna(0)
    out["bowling_points"] = out["bowling_points"].fillna(0)
    out["fielding_points"] = 0.0
    out["fantasy_points"] = out["batting_points"] + out["bowling_points"] + out["fielding_points"]
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Pre-match feature snapshot (one row per player per match)
# ═══════════════════════════════════════════════════════════════════════════

def _build_pre_match_features(features: pd.DataFrame) -> pd.DataFrame:
    """features = full output of feature_engineer.build_features(). Collapses
    it to one row per (match_id, player) WITHOUT splitting into separate
    batter/bowler tables and merging them back together — that split
    approach threw away every specialist's other-side columns (a pure
    batter has no bowler row -> all bowl_* NaN, and vice versa), which both
    inflated NaN counts and quietly dropped ~half the engineered columns
    from the final dataset.

    Instead: a player's involvement in a match — as striker OR as bowler —
    is unioned into one long table, sorted chronologically, and the FIRST
    row for that (match_id, player) is kept as-is, with only the live
    match-state columns stripped out. Since every non-live engineered
    column here (bat_/bowl_/bvb_/venue_) is causal and keyed by season (see
    feature_engineer.py), it's already constant across the whole match, so
    taking the first appearance loses nothing and needs no merge.
    """
    # Sort chronologically so "first row" == the player's earliest
    # involvement in the match, before any live-state column is dropped.
    sort_cols = [c for c in ["match_id", "innings", "over_num", "ball_num"] if c in features.columns]
    feat_sorted = features.sort_values(sort_cols).reset_index(drop=True)

    as_striker = feat_sorted.assign(player=feat_sorted["striker"])
    as_bowler = feat_sorted.assign(player=feat_sorted["bowler"])
    long = pd.concat([as_striker, as_bowler], ignore_index=True)

    pre_match = long.drop_duplicates(subset=["match_id", "player"], keep="first")

    keep_cols = ["player"] + [c for c in pre_match.columns
                               if c not in LIVE_MATCH_STATE_COLS and c not in ("striker", "bowler", "player")]
    pre_match = pre_match[keep_cols].reset_index(drop=True)
    return pre_match


# ═══════════════════════════════════════════════════════════════════════════
# Validation / cleanup
# ═══════════════════════════════════════════════════════════════════════════

def _validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["match_id", "player"]).reset_index(drop=True)
    print(f"  Deduped: {before:,} -> {len(df):,} rows")

    # A player row with no batting AND no bowling history at all (both sides
    # of the outer-merge were empty, e.g. a fielder-only substitute with no
    # recorded ball-by-ball involvement) can't be featurized -> drop.
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in
                    ("match_id", "season", "batting_points", "bowling_points",
                     "fielding_points", "fantasy_points")]

    all_na_mask = df[numeric_cols].isna().all(axis=1) if numeric_cols else pd.Series(False, index=df.index)
    dropped = int(all_na_mask.sum())
    if dropped:
        print(f"  Dropping {dropped:,} rows with zero engineered feature history")
    df = df[~all_na_mask].reset_index(drop=True)

    # Median-fill any remaining sparse NaNs in NUMERIC columns (a player's
    # first-ever match, a brand-new venue, etc.) — same philosophy as
    # feature_engineer's own final NaN safety net.
    for col in numeric_cols:
        n_missing = int(df[col].isna().sum())
        if n_missing:
            median = df[col].median()
            if pd.isna(median):
                median = 0.0
            df[col] = df[col].fillna(median)

    # The numeric fill above skips object/string columns entirely (venue,
    # batting_team, bowling_team, phase, date, etc.) — select_dtypes(number)
    # never selects them, so any NaN in those columns previously survived
    # untouched all the way to the exported CSV. Fill those here too.
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols and c != "fantasy_points"]
    for col in non_numeric_cols:
        n_missing = int(df[col].isna().sum())
        if not n_missing:
            continue
        if col == "date":
            df[col] = df[col].fillna(method="ffill")
        else:
            df[col] = df[col].fillna("Unknown")

    df["fantasy_points"] = df["fantasy_points"].fillna(0)

    total_na = int(df.isna().sum().sum())
    print(f"  Total remaining NaN after cleanup: {total_na:,}")
    if total_na:
        print(f"  ⚠ NaN still present in: \n{df.isna().sum()[df.isna().sum() > 0]}")

    return df


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def build_dataset() -> pd.DataFrame:
    print("Step 1: loading + cleaning raw data …")
    raw = load_raw()
    cleaned = clean(raw)

    print("\nStep 2: engineering historical features (reusing feature_engineer.py) …")
    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
    features = fe.build_features(cleaned)

    print("\nStep 3: building pre-match player feature snapshots …")
    pre_match = _build_pre_match_features(features)
    print(f"  {len(pre_match):,} (match, player) pre-match feature rows")

    print("\nStep 4: computing actual Dream11 fantasy points per player per match …")
    actual_pts = _compute_actual_fantasy_points(cleaned)
    print(f"  {len(actual_pts):,} (match, player) actual-points rows")

    print("\nStep 5: joining features (X) with actual points (y) …")
    dataset = pre_match.merge(actual_pts, on=["match_id", "player"], how="inner")
    print(f"  {len(dataset):,} joined rows")

    print("\nStep 6: validating + cleaning …")
    dataset = _validate_and_clean(dataset)

    print(f"\nFinal fantasy dataset shape: {dataset.shape}")
    print(f"Fantasy points distribution:\n{dataset['fantasy_points'].describe()}")
    return dataset


if __name__ == "__main__":
    dataset = build_dataset()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved → {OUTPUT_PATH}")
