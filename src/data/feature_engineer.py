"""
Feature Engineering — Context-Aware & Recency-Weighted
=======================================================
Produces data/processed/features.csv for XGBoost training.

Feature groups:
  A. Match-state running totals (score, wickets, balls remaining, CRR)
  B. Recency-weighted player stats (batter + bowler, career + phase-split)
  C. Batter-at-venue / Bowler-at-venue interaction stats (NEW)
  D. Batter-vs-Bowler matchup features (recency-weighted)
  E. In-match momentum features (rolling window, partnership, streaks)
  F. Venue intelligence (recency-weighted, phase-split)
  G. Batting-first / chasing context (pressure index)
  H. Final NaN safety net (see _fillna_engineered_columns)

Anti-leakage guarantee: every recency-weighted stat (B, C, D, F) is computed
via _causal_season_stats(), which only ever uses seasons STRICTLY BEFORE the
season of the row being featurized. A player's/venue's/matchup's stat for a
ball in season S can never depend on anything from season S itself (the
current match included) or any later season. All of B/C/D/F use the SAME
recency half-life (HALF_LIFE_SEASONS), so a match from 2018 still
contributes to every player's/venue's current-day numbers — just discounted
relative to more recent seasons, not excluded outright.

The column `sample_weight` is also exported so Colab training can pass it
to XGBoost's fit() as sample_weight=df["sample_weight"].
"""

import pandas as pd
import numpy as np
from pathlib import Path

CLEANED_PATH = Path("data/processed/cleaned.csv")
OUTPUT_PATH  = Path("data/processed/features.csv")

# Exponential half-life in seasons for all recency weighting. A match from
# `half_life` seasons ago carries half the weight of a match from this
# season; one from `2 * half_life` seasons ago carries a quarter, etc.
# Nothing is ever hard-excluded by age — the 2018 season always contributes
# something, just proportionally less as more recent data accumulates.
HALF_LIFE_SEASONS = 3.0

# Shrinkage constant (in weighted balls) for player-at-venue blending.
# With PLAYER_VENUE_SHRINK_K weighted balls of venue-specific history, the
# blended stat sits exactly halfway between the venue-specific number and
# the player's global career number. Below that it leans toward the global
# number (protects against a handful of lucky/unlucky balls at one ground
# swamping a much larger career sample); above it, it leans toward the
# venue-specific number.
PLAYER_VENUE_SHRINK_K = 30.0

# Over-band split for finer-grained situational stats than the 3-phase split
# above — separates the early-middle overs (7-10) from the late-middle overs
# (11-15), since batting approach genuinely differs between them even though
# both fall under the single "middle" phase bucket.
BVB_DISMISSAL_SHRINK_K = 25.0
BVB_OTHER_SHRINK_K = 45.0
OVER_BANDS = [
    ("1_6",   0,  5),
    ("7_10",  6,  9),
    ("11_15", 10, 14),
    ("16_20", 15, 19),
]
# Shrinkage constants (in weighted balls) for the over-band hierarchy.
# Small K at the finest level (player x venue x band, or matchup x band) —
# these are inherently thin, so lean on the broader fallback quickly.
OVERBAND_SHRINK_K_FINE   = 15.0
OVERBAND_SHRINK_K_COARSE = 25.0

# Sane, non-zero fallback values used ONLY if a column's own median is
# itself unavailable (e.g. an entirely-empty column) — this must never
# happen in normal operation given the causal design below, but guards
# against silently emitting NaN (which XGBoost would otherwise accept
# without complaint) or a lazy 0-fill (which reads as "worst possible
# player/bowler ever" rather than "no data", both bugs we've hit before).
_SENSIBLE_DEFAULTS = {
    "sr": 128.0, "economy": 8.5, "avg": 26.0,
    "dot_pct": 0.35, "boundary_pct": 0.15, "six_pct": 0.06,
    "wicket_pct": 0.05, "dismissal_pct": 0.05, "balls": 0.0,
}


def _default_for(col: str) -> float:
    for suffix, val in _SENSIBLE_DEFAULTS.items():
        if col.endswith(suffix):
            return val
    return 0.0


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

    print("Step C: player-at-venue interaction stats …")
    df = _add_player_venue_features(df)

    print("Step D: batter-vs-bowler matchup features …")
    df = _add_bvb_features(df)

    print("Step D2: over-band stats (venue / player-venue / BvB, hierarchically shrunk) …")
    df = _add_overband_features(df)

    print("Step E: in-match momentum features …")
    df = _add_momentum_features(df)

    print("Step F: venue intelligence …")
    df = _add_venue_features(df)

    print("Step G: batting context / pressure …")
    df = _add_batting_context(df)

    print("Step H: final NaN safety net …")
    df = _fillna_engineered_columns(df)

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
        # phase-split batter (SR + boundary% + dot% for each phase)
        "bat_pp_rw_sr", "bat_mid_rw_sr", "bat_death_rw_sr",
        "bat_pp_rw_boundary_pct", "bat_mid_rw_boundary_pct", "bat_death_rw_boundary_pct",
        "bat_pp_rw_dot_pct", "bat_mid_rw_dot_pct", "bat_death_rw_dot_pct",

        # ── recency-weighted career stats — bowler ────────────────────────────
        "bowl_rw_economy", "bowl_rw_wicket_pct", "bowl_rw_dot_pct",
        "bowl_rw_boundary_pct",
        # phase-split bowler (economy + wicket% + dot% + boundary% for each phase)
        "bowl_pp_rw_economy", "bowl_mid_rw_economy", "bowl_death_rw_economy",
        "bowl_pp_rw_wicket_pct", "bowl_mid_rw_wicket_pct", "bowl_death_rw_wicket_pct",
        "bowl_pp_rw_dot_pct", "bowl_mid_rw_dot_pct", "bowl_death_rw_dot_pct",
        "bowl_pp_rw_boundary_pct", "bowl_mid_rw_boundary_pct", "bowl_death_rw_boundary_pct",

        # ── player-at-venue interaction (NEW) ─────────────────────────────────
        "bat_venue_adj_sr", "bat_venue_adj_boundary_pct", "bat_venue_rw_balls",
        "bowl_venue_adj_economy", "bowl_venue_adj_wicket_pct", "bowl_venue_rw_balls",

        # ── batter vs bowler matchup ──────────────────────────────────────────
        "bvb_balls", "bvb_rw_sr", "bvb_rw_dismissal_pct",
        "bvb_rw_dot_pct", "bvb_rw_boundary_pct", "bvb_rw_six_pct",

        # ── in-match momentum ─────────────────────────────────────────────────
        "batter_balls_faced", "batter_runs_scored", "batter_innings_sr",
        "balls_vs_bowler", "runs_vs_bowler",
        "runs_last6", "runs_last12", "runs_last18", "runs_last_over",
        "consec_dots", "consec_boundaries",
        "partnership_runs", "partnership_balls", "partnership_run_rate",
        "current_matchup_sr",
        "prev_ball_outcome", "prev2_ball_outcome", "prev3_ball_outcome",

        # ── venue intelligence ────────────────────────────────────────────────
        "venue_rw_avg_1st_innings",
        "venue_rw_avg_2nd_innings",
        "venue_rw_boundary_pct",
        "venue_rw_six_pct",
        "venue_rw_dot_pct",
        "venue_rw_wicket_pct",
        "venue_rw_pp_sr", "venue_rw_mid_sr", "venue_rw_death_sr",
        "venue_rw_pp_boundary_pct", "venue_rw_mid_boundary_pct", "venue_rw_death_boundary_pct",
        "venue_rw_pp_wicket_pct", "venue_rw_mid_wicket_pct", "venue_rw_death_wicket_pct",

        # ── over-band stats (1-6 / 7-10 / 11-15 / 16-20), hierarchically shrunk ─
        "venue_rw_1_6_rr", "venue_rw_7_10_rr", "venue_rw_11_15_rr", "venue_rw_16_20_rr",
        "venue_rw_1_6_wicket_pct", "venue_rw_7_10_wicket_pct",
        "venue_rw_11_15_wicket_pct", "venue_rw_16_20_wicket_pct",
        "bat_venue_1_6_sr", "bat_venue_7_10_sr", "bat_venue_11_15_sr", "bat_venue_16_20_sr",
        "bat_venue_1_6_avg", "bat_venue_7_10_avg", "bat_venue_11_15_avg", "bat_venue_16_20_avg",
        "bowl_venue_1_6_economy", "bowl_venue_7_10_economy",
        "bowl_venue_11_15_economy", "bowl_venue_16_20_economy",
        "bowl_venue_1_6_wicket_pct", "bowl_venue_7_10_wicket_pct",
        "bowl_venue_11_15_wicket_pct", "bowl_venue_16_20_wicket_pct",
        "bvb_1_6_sr", "bvb_7_10_sr", "bvb_11_15_sr", "bvb_16_20_sr",
        "bvb_1_6_avg", "bvb_7_10_avg", "bvb_11_15_avg", "bvb_16_20_avg",

        # ── batting context / pressure ────────────────────────────────────────
        "is_batting_first", "is_chasing", "target", "runs_needed", "rrr",
        "pressure_index", "required_runs_per_wicket", "balls_per_required_run",
        "pressure_weighted_rrr", "pressure_weighted_aggression",

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

    # Belt-and-braces: this should be a no-op after Step H, but confirm it —
    # if this ever prints anything, Step H itself has a bug and needs fixing
    # before training, not after.
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    remaining_na = df[numeric_cols].isna().sum()
    remaining_na = remaining_na[remaining_na > 0]
    if len(remaining_na):
        print(f"  ⚠ WARNING: NaN survived to the final feature set: \n{remaining_na}")
    else:
        print("  ✓ No NaN in any numeric feature column.")

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

    df["balls_bowled"]      = df["cumulative_legal_balls"]
    df["balls_remaining"]   = (120 - df["balls_bowled"]).clip(lower=0)
    df["wickets_remaining"] = (10 - df["cumulative_wickets"]).clip(lower=0)

    df["crr"] = np.where(
        df["balls_bowled"] > 0,
        df["cumulative_runs"] / (df["balls_bowled"] / 6),
        0.0,
    )
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# B. RECENCY-WEIGHTED PLAYER STATS  (+ shared causal-aggregation primitive)
# ═══════════════════════════════════════════════════════════════════════════════

def _causal_season_stats(df: pd.DataFrame, group_cols, half_life: float = HALF_LIFE_SEASONS) -> pd.DataFrame:
    """
    THE CORE ANTI-LEAKAGE PRIMITIVE.

    Returns one row per (group_cols..., season) with recency-weighted stats
    computed using ONLY seasons STRICTLY BEFORE that season. This guarantees
    a player's/matchup's/venue's stat for anything happening in season S can
    never depend on data from season S itself (let alone later seasons).

    "Recency-weighted" here means every prior season contributes — including
    the very first season in the dataset — but with exponentially decaying
    weight the further back it is. Nothing is excluded by a hard age cutoff;
    a 2018 match always contributes some (shrinking) amount to a player's
    current numbers, exactly as intended.

    Merge the result back onto df using group_cols + ["season"] (NOT just
    group_cols) so each row only ever sees its own season's causal snapshot.
    """
    group_cols = list(group_cols)
    per_season = df.groupby(group_cols + ["season"]).agg(
        runs=("runs_of_bat", "sum"),
        legal=("is_legal", "sum"),
        wkts=("is_wicket", "sum"),
        dots=("is_dot", "sum"),
        fours=("is_four", "sum"),
        sixes=("is_six", "sum"),
    ).reset_index()

    all_seasons = np.sort(df["season"].unique())
    out_rows = []

    for keys, egrp in per_season.groupby(group_cols):
        egrp = egrp.set_index("season").reindex(all_seasons, fill_value=0)
        seasons_arr = egrp.index.values

        for target_season in all_seasons:
            prior_mask = seasons_arr < target_season
            if not prior_mask.any():
                continue
            legal = egrp["legal"].values[prior_mask]
            if legal.sum() == 0:
                continue

            weights = 0.5 ** ((target_season - seasons_arr[prior_mask]) / half_life)
            legal_w = (legal * weights).sum()
            if legal_w == 0:
                continue
            runs_w  = (egrp["runs"].values[prior_mask]  * weights).sum()
            wkts_w  = (egrp["wkts"].values[prior_mask]  * weights).sum()
            dots_w  = (egrp["dots"].values[prior_mask]  * weights).sum()
            fours_w = (egrp["fours"].values[prior_mask] * weights).sum()
            sixes_w = (egrp["sixes"].values[prior_mask] * weights).sum()

            row = {"season": target_season}

            if not isinstance(keys, tuple):
                keys = (keys,)

            for c, v in zip(group_cols, keys):
                row[c] = v
            row.update(
                rw_sr=runs_w / legal_w * 100,
                rw_economy=runs_w / legal_w * 6,
                # batting average = runs per dismissal, not runs per weighted
                # ball. Falls back to runs_w when never dismissed in this
                # window (genuinely "not out" across the sample).
                rw_avg=(runs_w / wkts_w) if wkts_w > 0 else runs_w,
                rw_dot_pct=dots_w / legal_w,
                rw_boundary_pct=(fours_w + sixes_w) / legal_w,
                rw_six_pct=sixes_w / legal_w,
                rw_wicket_pct=wkts_w / legal_w,
                rw_balls=legal_w,
            )
            out_rows.append(row)

    return pd.DataFrame(out_rows)


def _rw_agg(sub: pd.DataFrame, w_col: str = "season_weight") -> dict:
    """
    Weighted aggregate for a subset of deliveries, used by prepare_data.py to
    build the "as-of-today" JSON stat snapshots the live simulator reads
    (StatsStore) — NOT used for per-row training features anymore (those go
    through _causal_season_stats above to avoid leakage). This is fine here
    because prepare_data.py's snapshots are explicitly meant to reflect a
    player's full history as of the most recent data, for simulating NEW,
    never-before-seen matches.

    Rate stats (dot/boundary/six/wicket %) are divided by the weighted LEGAL
    BALL COUNT, not by the raw sum of weights — dividing by raw weight sum
    was a real bug in an earlier version of this function that silently
    inflated every rate stat for players whose career spanned many seasons.
    """
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

    total_rw = (runs * w).sum()
    balls_rw = (legal * w).sum()
    sr       = (total_rw / balls_rw * 100) if balls_rw > 0 else 0.0
    economy  = (total_rw / balls_rw * 6) if balls_rw > 0 else 0.0
    dot_pct  = ((is_dot * legal) * w).sum() / balls_rw if balls_rw > 0 else 0.0
    bnd_pct  = (((is_4 | is_6).astype(int) * legal) * w).sum() / balls_rw if balls_rw > 0 else 0.0
    six_pct  = ((is_6 * legal) * w).sum() / balls_rw if balls_rw > 0 else 0.0
    wkt_pct  = ((is_w * legal) * w).sum() / balls_rw if balls_rw > 0 else 0.0
    wkts_rw  = (is_w * w).sum()
    avg_runs = (total_rw / wkts_rw) if wkts_rw > 0 else total_rw

    return dict(
        rw_avg=avg_runs, rw_sr=sr, rw_economy=economy,
        rw_dot_pct=dot_pct, rw_boundary_pct=bnd_pct,
        rw_six_pct=six_pct, rw_wicket_pct=wkt_pct,
        rw_balls=balls_rw,
    )


def _add_recency_player_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each delivery, stats reflect ONLY seasons strictly before that
    delivery's season — see _causal_season_stats docstring for why.
    """

    # ── career batter stats (causal by season) ────────────────────────────────
    bat_df = _causal_season_stats(df, ["striker"]).rename(columns={
        "rw_avg":          "bat_rw_avg",
        "rw_sr":           "bat_rw_sr",
        "rw_dot_pct":      "bat_rw_dot_pct",
        "rw_boundary_pct": "bat_rw_boundary_pct",
        "rw_six_pct":      "bat_rw_six_pct",
    })[["striker", "season", "bat_rw_avg", "bat_rw_sr", "bat_rw_dot_pct",
        "bat_rw_boundary_pct", "bat_rw_six_pct"]]
    df = df.merge(bat_df, on=["striker", "season"], how="left")

    # ── phase-split batter stats (causal by season, computed on phase subset) ─
    for phase_name, phase_tag in [("powerplay", "pp"), ("middle", "mid"), ("death", "death")]:
        sub = df[df["phase"] == phase_name]
        phase_df = _causal_season_stats(sub, ["striker"])
        if len(phase_df):
            phase_df = phase_df.rename(columns={
                "rw_sr":           f"bat_{phase_tag}_rw_sr",
                "rw_boundary_pct": f"bat_{phase_tag}_rw_boundary_pct",
                "rw_dot_pct":      f"bat_{phase_tag}_rw_dot_pct",
            })[["striker", "season", f"bat_{phase_tag}_rw_sr",
                f"bat_{phase_tag}_rw_boundary_pct", f"bat_{phase_tag}_rw_dot_pct"]]
            df = df.merge(phase_df, on=["striker", "season"], how="left")

    # ── career bowler stats (causal by season) ────────────────────────────────
    bowl_df = _causal_season_stats(df, ["bowler"]).rename(columns={
        "rw_economy":      "bowl_rw_economy",
        "rw_wicket_pct":   "bowl_rw_wicket_pct",
        "rw_dot_pct":      "bowl_rw_dot_pct",
        "rw_boundary_pct": "bowl_rw_boundary_pct",
    })[["bowler", "season", "bowl_rw_economy", "bowl_rw_wicket_pct",
        "bowl_rw_dot_pct", "bowl_rw_boundary_pct"]]
    df = df.merge(bowl_df, on=["bowler", "season"], how="left")

    # ── phase-split bowler stats (causal by season) ───────────────────────────
    for phase_name, phase_tag in [("powerplay", "pp"), ("middle", "mid"), ("death", "death")]:
        sub = df[df["phase"] == phase_name]
        phase_df = _causal_season_stats(sub, ["bowler"])
        if len(phase_df):
            phase_df = phase_df.rename(columns={
                "rw_economy":      f"bowl_{phase_tag}_rw_economy",
                "rw_wicket_pct":   f"bowl_{phase_tag}_rw_wicket_pct",
                "rw_dot_pct":      f"bowl_{phase_tag}_rw_dot_pct",
                "rw_boundary_pct": f"bowl_{phase_tag}_rw_boundary_pct",
            })[["bowler", "season", f"bowl_{phase_tag}_rw_economy", f"bowl_{phase_tag}_rw_wicket_pct",
                f"bowl_{phase_tag}_rw_dot_pct", f"bowl_{phase_tag}_rw_boundary_pct"]]
            df = df.merge(phase_df, on=["bowler", "season"], how="left")

    # Rookies / first-ever-appearance rows have no PRIOR season yet, so the
    # merges above leave NaN — the blanket safety net in Step H catches
    # these (and anything else) at the very end, so no fillna here.
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# C. PLAYER-AT-VENUE INTERACTION  (NEW)
# ═══════════════════════════════════════════════════════════════════════════════

def _add_player_venue_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    How has THIS player performed at THIS specific ground, historically —
    distinct from both their venue-blind career stats (Step B) and the
    venue's player-blind overall character (Step F). Causal-by-season and
    recency-weighted like everything else, using the SAME half-life, so a
    player's 2018 innings at a ground still counts today, just discounted.

    Any one venue is a small sample for any one player, so the raw
    venue-specific number is shrunk toward that player's global career
    number (from Step B) using simple balls-based shrinkage — this prevents
    a handful of lucky/unlucky balls at one ground from swamping a much
    larger, more reliable career sample. Must run AFTER _add_recency_player_stats
    since it blends against bat_rw_sr / bowl_rw_economy etc.
    """
    K = PLAYER_VENUE_SHRINK_K

    # ── batter at venue ────────────────────────────────────────────────────
    bv_df = _causal_season_stats(df, ["striker", "venue"]).rename(columns={
        "rw_sr":           "bat_venue_rw_sr",
        "rw_boundary_pct": "bat_venue_rw_boundary_pct",
        "rw_balls":        "bat_venue_rw_balls",
    })[["striker", "venue", "season", "bat_venue_rw_sr", "bat_venue_rw_boundary_pct", "bat_venue_rw_balls"]]
    df = df.merge(bv_df, on=["striker", "venue", "season"], how="left")

    df["bat_venue_rw_balls"] = df["bat_venue_rw_balls"].fillna(0.0)
    shrink = df["bat_venue_rw_balls"] / (df["bat_venue_rw_balls"] + K)
    df["bat_venue_adj_sr"] = (
        shrink * df["bat_venue_rw_sr"].fillna(df["bat_rw_sr"])
        + (1 - shrink) * df["bat_rw_sr"]
    )
    df["bat_venue_adj_boundary_pct"] = (
        shrink * df["bat_venue_rw_boundary_pct"].fillna(df["bat_rw_boundary_pct"])
        + (1 - shrink) * df["bat_rw_boundary_pct"]
    )

    # ── bowler at venue ────────────────────────────────────────────────────
    bwv_df = _causal_season_stats(df, ["bowler", "venue"]).rename(columns={
        "rw_economy":    "bowl_venue_rw_economy",
        "rw_wicket_pct": "bowl_venue_rw_wicket_pct",
        "rw_balls":      "bowl_venue_rw_balls",
    })[["bowler", "venue", "season", "bowl_venue_rw_economy", "bowl_venue_rw_wicket_pct", "bowl_venue_rw_balls"]]
    df = df.merge(bwv_df, on=["bowler", "venue", "season"], how="left")

    df["bowl_venue_rw_balls"] = df["bowl_venue_rw_balls"].fillna(0.0)
    shrink_b = df["bowl_venue_rw_balls"] / (df["bowl_venue_rw_balls"] + K)
    df["bowl_venue_adj_economy"] = (
        shrink_b * df["bowl_venue_rw_economy"].fillna(df["bowl_rw_economy"])
        + (1 - shrink_b) * df["bowl_rw_economy"]
    )
    df["bowl_venue_adj_wicket_pct"] = (
        shrink_b * df["bowl_venue_rw_wicket_pct"].fillna(df["bowl_rw_wicket_pct"])
        + (1 - shrink_b) * df["bowl_rw_wicket_pct"]
    )

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# C2. OVER-BAND STATS  (venue / player-at-venue / BvB, each split into
#     overs 1-6, 7-10, 11-15, 16-20, with hierarchical shrinkage)
# ═══════════════════════════════════════════════════════════════════════════════

def _band_stats(df: pd.DataFrame, group_cols, lo: int, hi: int) -> pd.DataFrame:
    """_causal_season_stats restricted to a single over-band."""
    sub = df[(df["over_num"] >= lo) & (df["over_num"] <= hi)]
    return _causal_season_stats(sub, group_cols)


def _add_overband_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds over-band-specific (1-6 / 7-10 / 11-15 / 16-20) stats for:
      - venue alone           (venue_rw_{band}_rr, venue_rw_{band}_wicket_pct)
      - batter at venue       (bat_venue_{band}_sr, bat_venue_{band}_avg)
      - bowler at venue       (bowl_venue_{band}_economy, bowl_venue_{band}_wicket_pct)
      - batter vs bowler      (bvb_{band}_sr, bvb_{band}_avg)

    MUST run after _add_player_venue_features and _add_bvb_features — every
    band-specific number here is hierarchically shrunk toward a broader,
    already-computed fallback so a near-empty (player, venue, band) or
    (batter, bowler, band) bucket can't collapse to a degenerate constant
    the way plain (player, venue) and (batter, bowler) stats did before
    fallback blending was added. Shrinkage chain, broadest to narrowest:

      career (Step B)  ->  venue-adjusted / bvb (Step C / D)  ->  this band

    i.e. the band-specific number leans on the ALREADY-shrunk venue/matchup
    number as its fallback, not straight on the raw global career number —
    so band granularity only pulls the estimate away from the broader
    number when there's real band-specific data to justify it.
    """

    # ── venue over-bands (single-entity, no shrinkage needed — same pattern
    #    as the phase-split venue stats, just finer bins) ─────────────────────
    for band_tag, lo, hi in OVER_BANDS:
        band_df = _band_stats(df, ["venue"], lo, hi)
        if len(band_df):
            band_df = band_df.assign(**{
                f"venue_rw_{band_tag}_rr":         band_df["rw_sr"] * 0.06,  # SR (runs/100 balls) -> runs/over
                f"venue_rw_{band_tag}_wicket_pct":  band_df["rw_wicket_pct"],
            })[["venue", "season", f"venue_rw_{band_tag}_rr", f"venue_rw_{band_tag}_wicket_pct"]]
            df = df.merge(band_df, on=["venue", "season"], how="left")
        else:
            df[f"venue_rw_{band_tag}_rr"] = np.nan
            df[f"venue_rw_{band_tag}_wicket_pct"] = np.nan

    # ── batter-at-venue over-bands, shrunk toward the already-shrunk
    #    venue-adjusted career number (bat_venue_adj_sr) ───────────────────────
    for band_tag, lo, hi in OVER_BANDS:
        band_df = _band_stats(df, ["striker", "venue"], lo, hi)
        if len(band_df):
            band_df = band_df.rename(columns={
                "rw_sr": f"_band_sr_{band_tag}", "rw_avg": f"_band_avg_{band_tag}",
                "rw_balls": f"_band_balls_{band_tag}",
            })[["striker", "venue", "season", f"_band_sr_{band_tag}",
                f"_band_avg_{band_tag}", f"_band_balls_{band_tag}"]]
            df = df.merge(band_df, on=["striker", "venue", "season"], how="left")
        else:
            df[f"_band_sr_{band_tag}"]    = np.nan
            df[f"_band_avg_{band_tag}"]   = np.nan
            df[f"_band_balls_{band_tag}"] = 0.0

        balls = df[f"_band_balls_{band_tag}"].fillna(0.0)
        shrink = balls / (balls + OVERBAND_SHRINK_K_FINE)
        df[f"bat_venue_{band_tag}_sr"] = (
            shrink * df[f"_band_sr_{band_tag}"].fillna(df["bat_venue_adj_sr"])
            + (1 - shrink) * df["bat_venue_adj_sr"]
        )
        df[f"bat_venue_{band_tag}_avg"] = (
            shrink * df[f"_band_avg_{band_tag}"].fillna(df["bat_rw_avg"])
            + (1 - shrink) * df["bat_rw_avg"]
        )
        df.drop(columns=[f"_band_sr_{band_tag}", f"_band_avg_{band_tag}", f"_band_balls_{band_tag}"], inplace=True)

    # ── bowler-at-venue over-bands, shrunk toward bowl_venue_adj_economy ──────
    for band_tag, lo, hi in OVER_BANDS:
        band_df = _band_stats(df, ["bowler", "venue"], lo, hi)
        if len(band_df):
            band_df = band_df.rename(columns={
                "rw_economy": f"_band_econ_{band_tag}", "rw_wicket_pct": f"_band_wkt_{band_tag}",
                "rw_balls": f"_band_balls_{band_tag}",
            })[["bowler", "venue", "season", f"_band_econ_{band_tag}",
                f"_band_wkt_{band_tag}", f"_band_balls_{band_tag}"]]
            df = df.merge(band_df, on=["bowler", "venue", "season"], how="left")
        else:
            df[f"_band_econ_{band_tag}"]  = np.nan
            df[f"_band_wkt_{band_tag}"]   = np.nan
            df[f"_band_balls_{band_tag}"] = 0.0

        balls = df[f"_band_balls_{band_tag}"].fillna(0.0)
        shrink = balls / (balls + OVERBAND_SHRINK_K_FINE)
        df[f"bowl_venue_{band_tag}_economy"] = (
            shrink * df[f"_band_econ_{band_tag}"].fillna(df["bowl_venue_adj_economy"])
            + (1 - shrink) * df["bowl_venue_adj_economy"]
        )
        df[f"bowl_venue_{band_tag}_wicket_pct"] = (
            shrink * df[f"_band_wkt_{band_tag}"].fillna(df["bowl_venue_adj_wicket_pct"])
            + (1 - shrink) * df["bowl_venue_adj_wicket_pct"]
        )
        df.drop(columns=[f"_band_econ_{band_tag}", f"_band_wkt_{band_tag}", f"_band_balls_{band_tag}"], inplace=True)

    # ── BvB over-bands, shrunk toward the already-shrunk bvb_rw_sr ────────────
    for band_tag, lo, hi in OVER_BANDS:
        band_df = _band_stats(df, ["striker", "bowler"], lo, hi)
        if len(band_df):
            band_df = band_df.rename(columns={
                "rw_sr": f"_band_sr_{band_tag}", "rw_avg": f"_band_avg_{band_tag}",
                "rw_balls": f"_band_balls_{band_tag}",
            })[["striker", "bowler", "season", f"_band_sr_{band_tag}",
                f"_band_avg_{band_tag}", f"_band_balls_{band_tag}"]]
            df = df.merge(band_df, on=["striker", "bowler", "season"], how="left")
        else:
            df[f"_band_sr_{band_tag}"]    = np.nan
            df[f"_band_avg_{band_tag}"]   = np.nan
            df[f"_band_balls_{band_tag}"] = 0.0

        balls = df[f"_band_balls_{band_tag}"].fillna(0.0)
        shrink = balls / (balls + OVERBAND_SHRINK_K_FINE)
        df[f"bvb_{band_tag}_sr"] = (
            shrink * df[f"_band_sr_{band_tag}"].fillna(df["bvb_rw_sr"])
            + (1 - shrink) * df["bvb_rw_sr"]
        )
        df[f"bvb_{band_tag}_avg"] = (
            shrink * df[f"_band_avg_{band_tag}"].fillna(df["bat_rw_avg"])
            + (1 - shrink) * df["bat_rw_avg"]
        )
        df.drop(columns=[f"_band_sr_{band_tag}", f"_band_avg_{band_tag}", f"_band_balls_{band_tag}"], inplace=True)

    return df

def _add_bvb_features(df: pd.DataFrame) -> pd.DataFrame:
    """Same causal-by-season approach as everything else — a matchup's
    history for a given season only includes seasons strictly before it.
    All prior seasons contribute (a 2018 head-to-head still counts), just
    recency-discounted like every other stat here. Falls back to the
    batter's/bowler's own career numbers (Step B) when there's no matchup
    history yet; Step H's blanket safety net catches anything left over."""
    bvb_df = _causal_season_stats(df, ["striker", "bowler"]).rename(columns={
        "rw_balls":        "bvb_balls",
        "rw_sr":           "bvb_rw_sr",
        "rw_wicket_pct":   "bvb_rw_dismissal_pct",
        "rw_dot_pct":      "bvb_rw_dot_pct",
        "rw_boundary_pct": "bvb_rw_boundary_pct",
        "rw_six_pct":      "bvb_rw_six_pct",
    })[["striker", "bowler", "season", "bvb_balls", "bvb_rw_sr",
        "bvb_rw_dismissal_pct", "bvb_rw_dot_pct", "bvb_rw_boundary_pct", "bvb_rw_six_pct"]]

    if len(bvb_df):
        df = df.merge(bvb_df, on=["striker", "bowler", "season"], how="left")
    else:
        for col in ["bvb_balls", "bvb_rw_sr", "bvb_rw_dismissal_pct",
                    "bvb_rw_dot_pct", "bvb_rw_boundary_pct", "bvb_rw_six_pct"]:
            df[col] = np.nan

    df["bvb_balls"] = df["bvb_balls"].fillna(0.0)
    df["bvb_rw_sr"]            = df["bvb_rw_sr"].fillna(df["bat_rw_sr"])
    df["bvb_rw_dismissal_pct"] = df["bvb_rw_dismissal_pct"].fillna(df["bowl_rw_wicket_pct"])
    df["bvb_rw_dot_pct"]       = df["bvb_rw_dot_pct"].fillna(df["bat_rw_dot_pct"])
    df["bvb_rw_boundary_pct"]  = df["bvb_rw_boundary_pct"].fillna(df["bat_rw_boundary_pct"])
    df["bvb_rw_six_pct"]       = df["bvb_rw_six_pct"].fillna(df["bat_rw_six_pct"])

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# E. IN-MATCH MOMENTUM
# ═══════════════════════════════════════════════════════════════════════════════

_OUTCOME_ENCODE = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "6": 6, "W": 7}


def _add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """All features computed row-by-row within each match-innings group.

    consec_dots and partnership tracking reset on a wicket EXACTLY the way
    match_simulator.py does at inference time — a mismatch here previously
    meant the model trained on a different distribution of these features
    than it saw during simulation."""

    result_rows = []

    for (match_id, innings), grp in df.groupby(["match_id", "innings"]):
        original_index = grp.index
        grp = grp.reset_index(drop=True)

        batter_balls: dict = {}   # batter → balls faced
        batter_runs:  dict = {}   # batter → runs
        pair_balls:   dict = {}   # (batter,bowler) -> balls faced this innings
        pair_runs:    dict = {}   # (batter,bowler) -> runs scored this innings

        # partnership state
        pship_runs  = 0
        pship_balls = 0

        # streak counters
        consec_dots       = 0
        consec_boundaries = 0

        # rolling last-6 and last-over
        legal_ball_runs: list = []
        over_runs: dict = {}

        # outcome history
        outcome_history: list = []

        rows_out = []

        for i, row in grp.iterrows():
            striker  = row["striker"]
            bowler   = row["bowler"]
            over     = int(row["over_num"])
            runs     = int(row["runs_of_bat"])
            is_w     = int(row["is_wicket"])
            is_legal = int(row["is_legal"])
            outcome  = str(row["outcome"])
            is_boundary = runs in (4, 6)

            # ── snapshot BEFORE this ball ─────────────────────────────────────
            bf = batter_balls.get(striker, 0)
            br = batter_runs.get(striker, 0)
            b_sr = (br / bf * 100) if bf > 0 else 0.0

            pair_key = (striker, bowler)
            bvb_b = pair_balls.get(pair_key, 0)
            bvb_r = pair_runs.get(pair_key, 0)

            runs_last6  = int(sum(legal_ball_runs[-6:]))
            runs_last12 = int(sum(legal_ball_runs[-12:]))
            runs_last18 = int(sum(legal_ball_runs[-18:]))
            prev_over = over - 1
            runs_last_over = int(over_runs.get(prev_over, 0))

            # partnership resets when a NEW batter arrives (first ball faced)
            if bf == 0:
                pship_runs  = 0
                pship_balls = 0

            partnership_run_rate = (pship_runs / pship_balls * 6) if pship_balls > 0 else 0.0
            current_matchup_sr   = (bvb_r / bvb_b * 100) if bvb_b > 0 else 0.0

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
                "runs_last12":         runs_last12,
                "runs_last18":         runs_last18,
                "runs_last_over":      runs_last_over,
                "consec_dots":         consec_dots,
                "consec_boundaries":   consec_boundaries,
                "partnership_runs":    pship_runs,
                "partnership_balls":   pship_balls,
                "partnership_run_rate": round(partnership_run_rate, 2),
                "current_matchup_sr":   round(current_matchup_sr, 2),
                "prev_ball_outcome":   prev1,
                "prev2_ball_outcome":  prev2,
                "prev3_ball_outcome":  prev3,
            })

            # ── update state AFTER this ball ──────────────────────────────────
            batter_balls[striker] = bf + (1 if is_legal else 0)
            batter_runs[striker]  = br + runs
            pair_balls[pair_key]  = bvb_b + (1 if is_legal else 0)
            pair_runs[pair_key]   = bvb_r + runs + int(row.get("extras", 0))

            if is_legal:
                legal_ball_runs.append(runs)
            over_runs[over] = over_runs.get(over, 0) + runs + int(row.get("extras", 0))

            outcome_history.append(outcome)

            # Streaks and partnership: a wicket cleanly resets both,
            # matching match_simulator.py's inference-time behavior exactly.
            if is_w:
                consec_dots       = 0
                consec_boundaries = 0
                pship_runs        = 0
                pship_balls       = 0
            elif outcome == "0":
                consec_dots       += 1
                consec_boundaries  = 0
                pship_runs         += runs
                pship_balls        += (1 if is_legal else 0)
            elif is_boundary:
                consec_boundaries += 1
                consec_dots        = 0
                pship_runs         += runs
                pship_balls        += (1 if is_legal else 0)
            else:
                consec_dots       = 0
                consec_boundaries = 0
                pship_runs        += runs
                pship_balls        += (1 if is_legal else 0)

        result_rows.append(pd.DataFrame(rows_out, index=original_index))

    momentum_df = pd.concat(result_rows).sort_index()
    momentum_df = momentum_df.reindex(df.index)
    df = df.join(momentum_df)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# F. VENUE INTELLIGENCE
# ═══════════════════════════════════════════════════════════════════════════════

def _add_venue_features(df: pd.DataFrame) -> pd.DataFrame:
    """Venue's overall (player-blind) character — causal-by-season, all
    prior seasons contribute with recency decay, same as everywhere else."""

    inn1 = df[df["innings"] == 1].groupby("match_id").agg(
        venue=("venue", "first"),
        season=("season", "first"),
        inn1_runs=("total_runs", "sum"),
    ).reset_index()
    inn2 = df[df["innings"] == 2].groupby("match_id").agg(
        inn2_runs=("total_runs", "sum"),
    ).reset_index()
    match_scores = inn1.merge(inn2, on="match_id", how="left")

    all_seasons = np.sort(df["season"].unique())
    score_rows = []
    for venue, vgrp in match_scores.groupby("venue"):
        by_season = vgrp.groupby("season").agg(
            inn1_sum=("inn1_runs", "sum"), inn1_n=("inn1_runs", "count"),
            inn2_sum=("inn2_runs", "sum"), inn2_n=("inn2_runs", "count"),
        ).reindex(all_seasons, fill_value=0)
        for target_season in all_seasons:
            prior = by_season[by_season.index < target_season]
            if prior["inn1_n"].sum() == 0:
                continue
            w = 0.5 ** ((target_season - prior.index.values) / HALF_LIFE_SEASONS)
            inn1_n_w = (prior["inn1_n"].values * w).sum()
            inn2_n_w = (prior["inn2_n"].values * w).sum()
            score_rows.append({
                "venue": venue, "season": target_season,
                "venue_rw_avg_1st_innings": (prior["inn1_sum"].values * w).sum() / inn1_n_w if inn1_n_w > 0 else np.nan,
                "venue_rw_avg_2nd_innings": (prior["inn2_sum"].values * w).sum() / inn2_n_w if inn2_n_w > 0 else np.nan,
            })
    venue_score_df = pd.DataFrame(score_rows)

    venue_ball_df = _causal_season_stats(df, ["venue"]).rename(columns={
        "rw_boundary_pct": "venue_rw_boundary_pct",
        "rw_six_pct":      "venue_rw_six_pct",
        "rw_dot_pct":      "venue_rw_dot_pct",
        "rw_wicket_pct":   "venue_rw_wicket_pct",
    })[["venue", "season", "venue_rw_boundary_pct", "venue_rw_six_pct",
        "venue_rw_dot_pct", "venue_rw_wicket_pct"]]

    # Full phase-split venue character (previously only had pp/death SR —
    # missing the middle overs entirely, and missing boundary/wicket rate
    # per phase, which matters a lot for telling a "flat, boundary-friendly
    # ground" apart from a "hard to score on but wicket-prone" one in the
    # same phase).
    for phase_name, phase_tag in [("powerplay", "pp"), ("middle", "mid"), ("death", "death")]:
        phase_df = _causal_season_stats(df[df["phase"] == phase_name], ["venue"])
        if len(phase_df):
            phase_df = phase_df.rename(columns={
                "rw_sr":           f"venue_rw_{phase_tag}_sr",
                "rw_boundary_pct": f"venue_rw_{phase_tag}_boundary_pct",
                "rw_wicket_pct":   f"venue_rw_{phase_tag}_wicket_pct",
            })[["venue", "season", f"venue_rw_{phase_tag}_sr",
                f"venue_rw_{phase_tag}_boundary_pct", f"venue_rw_{phase_tag}_wicket_pct"]]
            venue_ball_df = venue_ball_df.merge(phase_df, on=["venue", "season"], how="left")

    venue_df = venue_score_df.merge(venue_ball_df, on=["venue", "season"], how="outer")
    df = df.merge(venue_df, on=["venue", "season"], how="left")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# G. BATTING CONTEXT / PRESSURE
# ═══════════════════════════════════════════════════════════════════════════════

def _add_batting_context(df: pd.DataFrame) -> pd.DataFrame:
    df["is_batting_first"] = (df["innings"] == 1).astype(int)
    df["is_chasing"]       = (df["innings"] == 2).astype(int)

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

    df["pressure_index"] = np.where(
        df["innings"] == 2,
        df["rrr"] - df["crr"],
        0.0,
    )

    # ── advanced pressure features (innings 2 only; 0 elsewhere) ──────────────
    df["required_runs_per_wicket"] = np.where(
        (df["innings"] == 2) & (df["wickets_remaining"] > 0),
        df["runs_needed"] / df["wickets_remaining"],
        0.0,
    )
    df["balls_per_required_run"] = np.where(
        (df["innings"] == 2) & (df["runs_needed"] > 0),
        df["balls_remaining"] / df["runs_needed"],
        df["balls_remaining"],  # target already reached — "surplus" balls, not a rate
    )
    # RRR scaled up as wickets fall — chasing 9/run with 8 wickets in hand is
    # very different pressure than the same rate with 2 wickets in hand.
    df["pressure_weighted_rrr"] = np.where(
        df["innings"] == 2,
        df["rrr"] * (1 + (10 - df["wickets_remaining"]) / 10),
        0.0,
    )
    # How much of the batting side's raw pressure differential is actually
    # "live" given wickets still in hand — same pressure_index means less if
    # the side is already 8 down.
    df["pressure_weighted_aggression"] = np.where(
        df["innings"] == 2,
        df["pressure_index"] * (df["wickets_remaining"] / 10),
        0.0,
    )
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# H. FINAL NaN SAFETY NET
# ═══════════════════════════════════════════════════════════════════════════════

def _fillna_engineered_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final catch-all, run LAST before column selection. Any engineered
    numeric column that still has NaN at this point — a player's first-ever
    season, a venue with no prior history, a bug in an upstream merge we
    haven't caught yet — gets filled with the column's own median. If the
    median itself is NaN (the column is entirely empty, which should never
    happen given the causal design above but is checked anyway), fall back
    to a hardcoded sane default rather than emitting 0 or NaN — both of
    which have caused real, hard-to-diagnose bugs in this pipeline before
    (0 reads as "worst possible player ever", NaN silently poisons whatever
    XGBoost's internal missing-value handling decides to do with it).
    """
    engineered_prefixes = ("bat_", "bowl_", "bvb_", "venue_")
    for col in df.columns:
        if not col.startswith(engineered_prefixes):
            continue
        if df[col].dtype.kind not in "fi":
            continue
        n_missing = int(df[col].isna().sum())
        if n_missing == 0:
            continue
        median = df[col].median()
        if pd.isna(median):
            median = _default_for(col)
            print(f"    '{col}' had NO valid values at all — using hardcoded "
                  f"default {median} instead of median")
        df[col] = df[col].fillna(median)
        print(f"    filled {n_missing:,} NaN in '{col}' with {median:.4f}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    df = pd.read_csv(CLEANED_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    features = build_features(df)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved → {OUTPUT_PATH}")