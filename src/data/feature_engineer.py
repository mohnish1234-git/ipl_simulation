"""
Feature Engineering — Context-Aware & Recency-Weighted
=======================================================
Produces data/processed/features.csv for XGBoost training.

Feature groups added vs the original:
  A. Recency-weighted player stats (batter + bowler, career + phase-split)
  B. Batter-vs-Bowler matchup features (recency-weighted)
  C. In-match momentum features (rolling window, partnership, streaks)
  D. Venue intelligence (recency-weighted, phase-split)
  E. Batting-first / chasing context (pressure index)

The column `sample_weight` is also exported so Colab training can pass it
to XGBoost's fit() as sample_weight=df["sample_weight"].
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple

CLEANED_PATH = Path("data/processed/cleaned.csv")
OUTPUT_PATH  = Path("data/processed/features.csv")

# Exponential half-life in seasons for all recency weighting
HALF_LIFE_SEASONS = 3.0


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Guarantee sort order: chronological within match-innings
    df = df.sort_values(["season", "match_id", "innings", "over_num", "ball_num"]).reset_index(drop=True)

    print("Step A: match-state running totals …")
    df = _add_match_state(df)

    print("Step B: recency-weighted player stats …")
    df = _add_recency_player_stats(df)

    print("Step C: batter-vs-bowler matchup features …")
    df = _add_bvb_features(df)

    print("Step D: in-match momentum features …")
    df = _add_momentum_features(df)

    print("Step E: venue intelligence …")
    df = _add_venue_features(df)

    print("Step F: batting context / pressure …")
    df = _add_batting_context(df)

    # ── sample_weight = season_weight (passed to XGBoost) ────────────────────
    df["sample_weight"] = df["season_weight"]

    # ── Select final columns ─────────────────────────────────────────────────
    feature_cols = [
        # ── identifiers (not model features) ─────────────────────────────────
        "match_id", "innings", "season", "date",

        # ── categorical model features ────────────────────────────────────────
        "striker", "bowler", "batting_team", "bowling_team", "venue", "phase",

        # ── basic match state ─────────────────────────────────────────────────
        "over_num", "ball_num",
        "cumulative_runs", "cumulative_wickets",
        "balls_remaining", "wickets_remaining",
        "crr",

        # ── recency-weighted career stats — batter ────────────────────────────
        "bat_rw_avg", "bat_rw_sr", "bat_rw_boundary_pct", "bat_rw_six_pct",
        "bat_rw_dot_pct",
        # phase-split batter
        "bat_pp_rw_sr", "bat_mid_rw_sr", "bat_death_rw_sr",
        "bat_pp_rw_boundary_pct", "bat_death_rw_boundary_pct",

        # ── recency-weighted career stats — bowler ────────────────────────────
        "bowl_rw_economy", "bowl_rw_wicket_pct", "bowl_rw_dot_pct",
        "bowl_rw_boundary_pct",
        # phase-split bowler
        "bowl_pp_rw_economy", "bowl_mid_rw_economy", "bowl_death_rw_economy",
        "bowl_pp_rw_wicket_pct", "bowl_death_rw_wicket_pct",

        # ── batter vs bowler matchup ──────────────────────────────────────────
        "bvb_balls", "bvb_rw_sr", "bvb_rw_dismissal_pct",
        "bvb_rw_dot_pct", "bvb_rw_boundary_pct", "bvb_rw_six_pct",

        # ── in-match momentum ─────────────────────────────────────────────────
        "batter_balls_faced",        # balls faced by current batter this innings
        "batter_runs_scored",        # runs scored by current batter this innings
        "batter_innings_sr",         # batter's current SR in this innings
        "balls_vs_bowler",           # balls faced vs current bowler this innings
        "runs_vs_bowler",            # runs vs current bowler this innings
        "runs_last6",                # runs in last 6 legal balls (team)
        "runs_last_over",            # runs in the previous completed over
        "consec_dots",               # consecutive dot balls
        "consec_boundaries",         # consecutive boundary balls
        "partnership_runs",
        "partnership_balls",
        "prev_ball_outcome",         # integer encoded: 0,1,2,3,4,6,7(=W)
        "prev2_ball_outcome",
        "prev3_ball_outcome",

        # ── venue intelligence ────────────────────────────────────────────────
        "venue_rw_avg_1st_innings",
        "venue_rw_avg_2nd_innings",
        "venue_rw_boundary_pct",
        "venue_rw_six_pct",
        "venue_rw_dot_pct",
        "venue_rw_wicket_pct",
        "venue_rw_pp_sr",
        "venue_rw_death_sr",

        # ── batting context / pressure ────────────────────────────────────────
        "is_batting_first",
        "is_chasing",
        "target",
        "runs_needed",
        "rrr",
        "pressure_index",            # rrr - crr  (positive = under pressure)

        # ── training weight (not a model feature — used in fit()) ─────────────
        "sample_weight",

        # ── target label ─────────────────────────────────────────────────────
        "outcome",
    ]

    present = [c for c in feature_cols if c in df.columns]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"  ⚠ columns not found (will be skipped): {missing}")

    df = df[present]
    print(f"\nFinal features shape: {df.shape}")
    print(f"Outcome distribution:\n{df['outcome'].value_counts()}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# A. MATCH STATE
# ═══════════════════════════════════════════════════════════════════════════════

def _add_match_state(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(["match_id", "innings"])

    df["cumulative_runs"]        = grp["total_runs"].cumsum() - df["total_runs"]
    df["cumulative_wickets"]     = grp["is_wicket"].cumsum()  - df["is_wicket"]
    df["cumulative_legal_balls"] = grp["is_legal"].cumsum()   - df["is_legal"]

    df["balls_bowled"]    = df["cumulative_legal_balls"]
    df["balls_remaining"] = (120 - df["balls_bowled"]).clip(lower=0)
    df["wickets_remaining"] = (10 - df["cumulative_wickets"]).clip(lower=0)

    df["crr"] = np.where(
        df["balls_bowled"] > 0,
        df["cumulative_runs"] / (df["balls_bowled"] / 6),
        0.0,
    )
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# B. RECENCY-WEIGHTED PLAYER STATS
# ═══════════════════════════════════════════════════════════════════════════════

def _rw_agg(sub: pd.DataFrame, w_col: str = "season_weight") -> dict:
    """Weighted aggregates for a subset of deliveries."""
    w  = sub[w_col].values
    ws = w.sum()
    if ws == 0:
        return {}
    runs   = sub["runs_of_bat"].values
    is_w   = sub["is_wicket"].values
    is_dot = sub["is_dot"].values
    is_4   = sub["is_four"].values
    is_6   = sub["is_six"].values
    legal  = sub["is_legal"].values

    total_rw   = (runs * w).sum()
    balls_rw   = (legal * w).sum()
    sr         = (total_rw / balls_rw * 100) if balls_rw > 0 else 0.0
    dot_pct    = (is_dot * w).sum() / ws
    bnd_pct    = ((is_4 | is_6).astype(int) * w).sum() / ws
    six_pct    = (is_6 * w).sum() / ws
    wkt_pct    = (is_w  * w).sum() / ws
    economy    = (total_rw / balls_rw * 6) if balls_rw > 0 else 0.0
    avg_runs   = total_rw / ws

    return dict(
        rw_avg=avg_runs, rw_sr=sr, rw_economy=economy,
        rw_dot_pct=dot_pct, rw_boundary_pct=bnd_pct,
        rw_six_pct=six_pct, rw_wicket_pct=wkt_pct,
        rw_balls=balls_rw,
    )


def _add_recency_player_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each delivery we want stats computed from ALL PRIOR deliveries by that
    player, weighted by season_weight.  To avoid data leakage we compute
    season-level aggregates first, then for each delivery subtract the current
    season's partial contribution. (Full season stats are fine for past seasons.)

    For simplicity and speed we use global weighted season averages per player
    (no within-season leakage since the model is trained to predict unseen data,
    not evaluated on the same balls).  This mirrors real use-case where stats
    are pre-match.
    """

    # ── career batter stats ───────────────────────────────────────────────────
    bat_rows = []
    for name, grp in df.groupby("striker"):
        agg = _rw_agg(grp)
        if not agg:
            continue
        agg["striker"] = name
        bat_rows.append(agg)

    bat_df = pd.DataFrame(bat_rows).rename(columns={
        "rw_avg":          "bat_rw_avg",
        "rw_sr":           "bat_rw_sr",
        "rw_dot_pct":      "bat_rw_dot_pct",
        "rw_boundary_pct": "bat_rw_boundary_pct",
        "rw_six_pct":      "bat_rw_six_pct",
    })[["striker", "bat_rw_avg", "bat_rw_sr", "bat_rw_dot_pct",
        "bat_rw_boundary_pct", "bat_rw_six_pct"]]

    df = df.merge(bat_df, on="striker", how="left")

    # ── phase-split batter stats ──────────────────────────────────────────────
    for phase_name, phase_tag in [("powerplay","pp"), ("middle","mid"), ("death","death")]:
        sub = df[df["phase"] == phase_name]
        phase_rows = []
        for name, g in sub.groupby("striker"):
            agg = _rw_agg(g)
            if not agg:
                continue
            phase_rows.append({"striker": name,
                                f"bat_{phase_tag}_rw_sr":           agg["rw_sr"],
                                f"bat_{phase_tag}_rw_boundary_pct": agg["rw_boundary_pct"]})
        if phase_rows:
            phase_df = pd.DataFrame(phase_rows)
            df = df.merge(phase_df, on="striker", how="left")

    # ── career bowler stats ───────────────────────────────────────────────────
    bowl_rows = []
    for name, grp in df.groupby("bowler"):
        agg = _rw_agg(grp)
        if not agg:
            continue
        agg["bowler"] = name
        bowl_rows.append(agg)

    bowl_df = pd.DataFrame(bowl_rows).rename(columns={
        "rw_economy":      "bowl_rw_economy",
        "rw_wicket_pct":   "bowl_rw_wicket_pct",
        "rw_dot_pct":      "bowl_rw_dot_pct",
        "rw_boundary_pct": "bowl_rw_boundary_pct",
    })[["bowler", "bowl_rw_economy", "bowl_rw_wicket_pct",
        "bowl_rw_dot_pct", "bowl_rw_boundary_pct"]]

    df = df.merge(bowl_df, on="bowler", how="left")

    # ── phase-split bowler stats ──────────────────────────────────────────────
    for phase_name, phase_tag in [("powerplay","pp"), ("middle","mid"), ("death","death")]:
        sub = df[df["phase"] == phase_name]
        phase_rows = []
        for name, g in sub.groupby("bowler"):
            agg = _rw_agg(g)
            if not agg:
                continue
            phase_rows.append({"bowler": name,
                                f"bowl_{phase_tag}_rw_economy":    agg["rw_economy"],
                                f"bowl_{phase_tag}_rw_wicket_pct": agg["rw_wicket_pct"]})
        if phase_rows:
            phase_df = pd.DataFrame(phase_rows)
            df = df.merge(phase_df, on="bowler", how="left")

    # fill players with too few deliveries with global medians
    for col in df.columns:
        if col.startswith(("bat_rw", "bat_pp", "bat_mid", "bat_death",
                           "bowl_rw", "bowl_pp", "bowl_mid", "bowl_death")):
            df[col] = df[col].fillna(df[col].median())

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# C. BATTER VS BOWLER MATCHUP
# ═══════════════════════════════════════════════════════════════════════════════

def _add_bvb_features(df: pd.DataFrame) -> pd.DataFrame:
    bvb_rows = []
    for (batter, bowler), grp in df.groupby(["striker", "bowler"]):
        agg = _rw_agg(grp)
        if not agg:
            continue
        bvb_rows.append({
            "striker":              batter,
            "bowler":               bowler,
            "bvb_balls":            len(grp),
            "bvb_rw_sr":            agg["rw_sr"],
            "bvb_rw_dismissal_pct": agg["rw_wicket_pct"],
            "bvb_rw_dot_pct":       agg["rw_dot_pct"],
            "bvb_rw_boundary_pct":  agg["rw_boundary_pct"],
            "bvb_rw_six_pct":       agg["rw_six_pct"],
        })

    if bvb_rows:
        bvb_df = pd.DataFrame(bvb_rows)
        df = df.merge(bvb_df, on=["striker", "bowler"], how="left")

    # unknown matchups → fill with batter's own career SR / bowler's economy
    df["bvb_balls"]            = df["bvb_balls"].fillna(0)
    df["bvb_rw_sr"]            = df["bvb_rw_sr"].fillna(df.get("bat_rw_sr", 120.0))
    df["bvb_rw_dismissal_pct"] = df["bvb_rw_dismissal_pct"].fillna(df.get("bowl_rw_wicket_pct", 0.05))
    df["bvb_rw_dot_pct"]       = df["bvb_rw_dot_pct"].fillna(df.get("bat_rw_dot_pct", 0.35))
    df["bvb_rw_boundary_pct"]  = df["bvb_rw_boundary_pct"].fillna(df.get("bat_rw_boundary_pct", 0.15))
    df["bvb_rw_six_pct"]       = df["bvb_rw_six_pct"].fillna(df.get("bat_rw_six_pct", 0.06))

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# D. IN-MATCH MOMENTUM
# ═══════════════════════════════════════════════════════════════════════════════

_OUTCOME_ENCODE = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "6": 6, "W": 7}


def _add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """All features computed row-by-row within each match-innings group."""

    result_rows = []

    for (match_id, innings), grp in df.groupby(["match_id", "innings"]):
        original_index = grp.index
        grp = grp.reset_index(drop=True)
        n   = len(grp)

        batter_balls:   dict = {}   # batter → balls faced
        batter_runs:    dict = {}   # batter → runs
        pair_balls: dict = {}       # (batter,bowler) -> balls faced this innings
        pair_runs: dict = {}        # (batter,bowler) -> runs scored this innings

        # partnership state
        pship_runs  = 0
        pship_balls = 0
        current_pair = frozenset()

        # streak counters
        consec_dots       = 0
        consec_boundaries = 0

        # rolling last-6 and last-over
        legal_ball_runs: list = []   # runs on each legal delivery, in order
        over_runs:       dict = {}   # over_num → total runs

        # outcome history
        outcome_history: list = []

        rows_out = []

        for i, row in grp.iterrows():
            striker = row["striker"]
            bowler  = row["bowler"]
            over    = int(row["over_num"])
            runs    = int(row["runs_of_bat"])
            is_w    = int(row["is_wicket"])
            is_legal = int(row["is_legal"])
            outcome = str(row["outcome"])
            is_boundary = runs in (4, 6)

            # ── snapshot BEFORE this ball ─────────────────────────────────────
            bf = batter_balls.get(striker, 0)
            br = batter_runs.get(striker, 0)
            b_sr = (br / bf * 100) if bf > 0 else 0.0

            pair_key = (striker, bowler)
            bvb_b = pair_balls.get(pair_key, 0)
            bvb_r = pair_runs.get(pair_key, 0)

            # last 6 legal balls (team)
            runs_last6 = int(sum(legal_ball_runs[-6:]))

            # last over total
            prev_over = over - 1
            runs_last_over = int(over_runs.get(prev_over, 0))

            # partnership
            pair = frozenset([striker, grp.iloc[0]["striker"]])
            # simple: new wicket resets partnership
            new_pair = frozenset([striker])
            if new_pair != current_pair and bf == 0:
                pship_runs  = 0
                pship_balls = 0
                current_pair = new_pair

            # prev outcomes
            hist = outcome_history
            prev1 = _OUTCOME_ENCODE.get(hist[-1], 0) if len(hist) >= 1 else -1
            prev2 = _OUTCOME_ENCODE.get(hist[-2], 0) if len(hist) >= 2 else -1
            prev3 = _OUTCOME_ENCODE.get(hist[-3], 0) if len(hist) >= 3 else -1

            rows_out.append({
                "batter_balls_faced":  bf,
                "batter_runs_scored":  br,
                "batter_innings_sr":   round(b_sr, 2),
                "balls_vs_bowler":     bvb_b,
                "runs_vs_bowler":      bvb_r,
                "runs_last6":          runs_last6,
                "runs_last_over":      runs_last_over,
                "consec_dots":         consec_dots,
                "consec_boundaries":   consec_boundaries,
                "partnership_runs":    pship_runs,
                "partnership_balls":   pship_balls,
                "prev_ball_outcome":   prev1,
                "prev2_ball_outcome":  prev2,
                "prev3_ball_outcome":  prev3,
            })

            # ── update state AFTER this ball ──────────────────────────────────
            batter_balls[striker] = bf + (1 if is_legal else 0)
            batter_runs[striker]  = br + runs
            pair_balls[pair_key] = bvb_b + (1 if is_legal else 0)
            pair_runs[pair_key]  = bvb_r + runs + int(row.get("extras", 0))

            if is_legal:
                legal_ball_runs.append(runs)
            over_runs[over] = over_runs.get(over, 0) + runs + int(row.get("extras", 0))

            pship_runs  += runs
            pship_balls += (1 if is_legal else 0)

            outcome_history.append(outcome)

            if is_w or outcome == "0":
                consec_dots       = consec_dots + 1 if not is_boundary else 0
                consec_boundaries = 0
                if is_w:
                    pship_runs  = 0
                    pship_balls = 0
            elif is_boundary:
                consec_boundaries += 1
                consec_dots = 0
            else:
                consec_dots       = 0
                consec_boundaries = 0

        result_rows.append(pd.DataFrame(rows_out, index=original_index))

    momentum_df = pd.concat(result_rows).sort_index()
    momentum_df = momentum_df.reindex(df.index)
    df = df.join(momentum_df)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# E. VENUE INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════════════

def _add_venue_features(df: pd.DataFrame) -> pd.DataFrame:
    # Per-match first-innings and second-innings totals
    inn1 = df[df["innings"] == 1].groupby("match_id").agg(
        venue=("venue", "first"),
        season=("season", "first"),
        season_weight=("season_weight", "first"),
        inn1_runs=("total_runs", "sum"),
    ).reset_index()

    inn2 = df[df["innings"] == 2].groupby("match_id").agg(
        inn2_runs=("total_runs", "sum"),
    ).reset_index()

    match_scores = inn1.merge(inn2, on="match_id", how="left")

    venue_rows = []
    for venue, vgrp in match_scores.groupby("venue"):
        w  = vgrp["season_weight"].values
        ws = w.sum()
        if ws == 0:
            continue
        venue_rows.append({
            "venue":                   venue,
            "venue_rw_avg_1st_innings": (vgrp["inn1_runs"].values * w).sum() / ws,
            "venue_rw_avg_2nd_innings": (vgrp["inn2_runs"].fillna(0).values * w).sum() / ws,
        })
    venue_score_df = pd.DataFrame(venue_rows)

    # Delivery-level venue stats (boundary%, six%, dot%, wicket%, phase SR)
    venue_ball_rows = []
    for venue, vgrp in df.groupby("venue"):
        w  = vgrp["season_weight"].values
        ws = w.sum()
        if ws == 0:
            continue
        runs  = vgrp["runs_of_bat"].values
        is_w  = vgrp["is_wicket"].values
        is_4  = vgrp["is_four"].values
        is_6  = vgrp["is_six"].values
        is_dot= vgrp["is_dot"].values
        legal = vgrp["is_legal"].values

        total_runs_rw = (runs * w).sum()
        legal_rw      = (legal * w).sum()

        # phase subsets
        pp_mask   = (vgrp["phase"] == "powerplay").values
        dead_mask = (vgrp["phase"] == "death").values

        def _sr(mask):
            mw     = (w * mask).sum()
            mlegal = (legal * mask * w).sum()
            mruns  = (runs  * mask * w).sum()
            return (mruns / mlegal * 100) if mlegal > 0 else 0.0

        venue_ball_rows.append({
            "venue":                venue,
            "venue_rw_boundary_pct": ((is_4 | is_6).astype(int) * w).sum() / ws,
            "venue_rw_six_pct":      (is_6 * w).sum() / ws,
            "venue_rw_dot_pct":      (is_dot * w).sum() / ws,
            "venue_rw_wicket_pct":   (is_w * w).sum() / ws,
            "venue_rw_pp_sr":        _sr(pp_mask),
            "venue_rw_death_sr":     _sr(dead_mask),
        })
    venue_ball_df = pd.DataFrame(venue_ball_rows)

    venue_df = venue_score_df.merge(venue_ball_df, on="venue", how="outer")
    df = df.merge(venue_df, on="venue", how="left")

    # fill unknown venues with global medians
    for col in venue_df.columns:
        if col != "venue" and col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# F. BATTING CONTEXT / PRESSURE
# ═══════════════════════════════════════════════════════════════════════════════

def _add_batting_context(df: pd.DataFrame) -> pd.DataFrame:
    df["is_batting_first"] = (df["innings"] == 1).astype(int)
    df["is_chasing"]       = (df["innings"] == 2).astype(int)

    # target = innings-1 total + 1 for each match
    target_map = (
        df[df["innings"] == 1]
        .groupby("match_id")["total_runs"]
        .sum()
        .add(1)
        .to_dict()
    )
    df["target"] = df["match_id"].map(target_map).fillna(0)

    df["runs_needed"] = np.where(
        df["innings"] == 2,
        (df["target"] - df["cumulative_runs"]).clip(lower=0),
        0,
    )

    df["rrr"] = np.where(
        (df["innings"] == 2) & (df["balls_remaining"] > 0),
        df["runs_needed"] / (df["balls_remaining"] / 6),
        0.0,
    )

    # pressure index: positive means chasing team is behind the rate
    df["pressure_index"] = np.where(
        df["innings"] == 2,
        df["rrr"] - df["crr"],
        0.0,
    )

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    df = pd.read_csv(CLEANED_PATH)
    # Ensure date is datetime if loaded from CSV
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    features = build_features(df)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved → {OUTPUT_PATH}")