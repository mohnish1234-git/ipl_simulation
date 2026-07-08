"""
scripts/prepare_data.py
Run this ONCE before training or simulating.

Usage:
    python scripts/prepare_data.py

Outputs:
    data/processed/cleaned.csv
    data/processed/features.csv          ← upload to Colab for training
    data/processed/meta.json             ← players / teams / venues for UI
    data/processed/player_batter_stats.json
    data/processed/player_bowler_stats.json
    data/processed/bvb_stats.json
    data/processed/venue_stats.json
"""

import sys, json
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from src.data.cleaner        import load_raw, clean
from src.data.feature_engineer import (
    build_features, _rw_agg,
    HALF_LIFE_SEASONS,
    PLAYER_VENUE_SHRINK_K, BVB_DISMISSAL_SHRINK_K, BVB_OTHER_SHRINK_K,
)
from src.model.retirement_filter import (
    compute_active_players,
    ACTIVE_WINDOW_SEASONS,
)

RAW_PATH      = Path("data/raw/ipl_final.csv")
PROCESSED_DIR = Path("data/processed")


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("Step 1: Loading raw data …")
    df_raw = load_raw(RAW_PATH)

    print("\nStep 2: Cleaning …")
    df_clean = clean(df_raw)
    df_clean.to_csv(PROCESSED_DIR / "cleaned.csv", index=False)
    print(f"  Saved → data/processed/cleaned.csv")

    print("\nStep 3: Feature engineering …")
    df_feat = build_features(df_clean)
    df_feat.to_csv(PROCESSED_DIR / "features.csv", index=False)
    print(f"  Saved → data/processed/features.csv")

    # NOTE: active_players is used only to scope which names the SIMULATOR/UI
    # can select — it does NOT drop rows from training data. The model still
    # trains on the full history (recency weighting already handles staleness
    # there); we just stop retired/unaffiliated players from being pickable
    # in a lineup at simulation time, which is what was producing Frankenstein
    # XIs (e.g. a CSK legend or a specialist bowler slotted into an MI top 7).
    active_players = compute_active_players(df_clean, window_seasons=ACTIVE_WINDOW_SEASONS)
    print(f"\n  {len(active_players)} players considered active "
          f"(appeared in the last {ACTIVE_WINDOW_SEASONS} seasons)")

    print("\nStep 4: Exporting lookup tables for the simulator …")
    _export_lookup_tables(df_clean, active_players)

    print("\nStep 5: Building meta.json …")
    _export_meta(df_clean, active_players)

    print("\n" + "=" * 55)
    print("Data preparation complete.")
    print(f"\nNext → upload  data/processed/features.csv  to Google Colab.")


# ─────────────────────────────────────────────────────────────────────────────

def _export_lookup_tables(df: pd.DataFrame, active_players: set):
    """
    Export pre-computed recency-weighted stats as JSON files.
    The MatchSimulator StatsStore.load_from_csv() reads these at runtime.

    Only ACTIVE players are exported here — this is what the simulator's
    dropdowns/rosters draw from, so retired players (and one-off names with
    no recent appearances) simply can't be selected into a lineup anymore.
    Training itself (features.csv above) still uses the full history.
    """

    # ── batter career stats ───────────────────────────────────────────────────
    batter_stats = {}
    for name, grp in df.groupby("striker"):
        if name not in active_players:
            continue
        agg = _rw_agg(grp)
        if not agg:
            continue
        # phase splits
        pp    = _rw_agg(grp[grp["phase"] == "powerplay"])
        mid   = _rw_agg(grp[grp["phase"] == "middle"])
        death = _rw_agg(grp[grp["phase"] == "death"])

        batter_stats[name] = {
            "bat_rw_avg":              round(agg.get("rw_avg", 0), 4),
            "bat_rw_sr":               round(agg.get("rw_sr", 120), 4),
            "bat_rw_boundary_pct":     round(agg.get("rw_boundary_pct", 0.15), 4),
            "bat_rw_six_pct":          round(agg.get("rw_six_pct", 0.06), 4),
            "bat_rw_dot_pct":          round(agg.get("rw_dot_pct", 0.33), 4),
            "bat_pp_rw_sr":            round(pp.get("rw_sr", 120) if pp else 120, 4),
            "bat_mid_rw_sr":           round(mid.get("rw_sr", 120) if mid else 120, 4),
            "bat_death_rw_sr":         round(death.get("rw_sr", 140) if death else 140, 4),
            "bat_pp_rw_boundary_pct":  round(pp.get("rw_boundary_pct", 0.14) if pp else 0.14, 4),
            "bat_death_rw_boundary_pct": round(death.get("rw_boundary_pct", 0.22) if death else 0.22, 4),
        }

    with open(PROCESSED_DIR / "player_batter_stats.json", "w") as f:
        json.dump(batter_stats, f)
    print(f"  player_batter_stats.json ({len(batter_stats)} batters)")

    # ── bowler career stats ───────────────────────────────────────────────────
    bowler_stats = {}
    for name, grp in df.groupby("bowler"):
        if name not in active_players:
            continue
        agg   = _rw_agg(grp)
        if not agg:
            continue
        pp    = _rw_agg(grp[grp["phase"] == "powerplay"])
        mid   = _rw_agg(grp[grp["phase"] == "middle"])
        death = _rw_agg(grp[grp["phase"] == "death"])

        bowler_stats[name] = {
            "bowl_rw_economy":          round(agg.get("rw_economy", 8.5), 4),
            "bowl_rw_wicket_pct":       round(agg.get("rw_wicket_pct", 0.055), 4),
            "bowl_rw_dot_pct":          round(agg.get("rw_dot_pct", 0.33), 4),
            "bowl_rw_boundary_pct":     round(agg.get("rw_boundary_pct", 0.15), 4),
            "bowl_pp_rw_economy":       round(pp.get("rw_economy", 7.5) if pp else 7.5, 4),
            "bowl_mid_rw_economy":      round(mid.get("rw_economy", 8.0) if mid else 8.0, 4),
            "bowl_death_rw_economy":    round(death.get("rw_economy", 9.5) if death else 9.5, 4),
            "bowl_pp_rw_wicket_pct":    round(pp.get("rw_wicket_pct", 0.07) if pp else 0.07, 4),
            "bowl_death_rw_wicket_pct": round(death.get("rw_wicket_pct", 0.055) if death else 0.055, 4),
        }

    with open(PROCESSED_DIR / "player_bowler_stats.json", "w") as f:
        json.dump(bowler_stats, f)
    print(f"  player_bowler_stats.json ({len(bowler_stats)} bowlers)")

    # ── batter-vs-bowler matchup stats ───────────────────────────────────────
    # Precedence: dismissal/wicket column shrinks toward the bowler's own
    # career wicket_pct at BVB_DISMISSAL_SHRINK_K (some real weight, sooner);
    # everything else shrinks toward the batter's own career number at the
    # slower BVB_OTHER_SHRINK_K (lowest precedence of the three tiers).
    # Previously matchups under 6 balls were dropped outright, which meant
    # the simulator fell back to StatsStore.DEFAULT_BVB — a global league
    # constant — instead of THIS batter's own real career numbers. Now every
    # matchup with at least 1 ball is exported, correctly shrunk, so a thin
    # matchup still reflects the specific player, just mostly via their
    # career stat rather than the noisy 1-2 ball matchup number.
    bvb_stats = {}
    for (batter, bowler), grp in df.groupby(["striker", "bowler"]):
        if batter not in active_players or bowler not in active_players:
            continue
        agg = _rw_agg(grp)
        if not agg:
            continue
        balls = agg["rw_balls"]
        batter_career = batter_stats.get(batter, {})
        bowler_career = bowler_stats.get(bowler, {})

        shrink_dismissal = balls / (balls + BVB_DISMISSAL_SHRINK_K)
        shrink_other      = balls / (balls + BVB_OTHER_SHRINK_K)

        key = f"{batter}|||{bowler}"
        bvb_stats[key] = {
            "bvb_balls": int(len(grp)),
            "bvb_rw_dismissal_pct": round(
                shrink_dismissal * agg.get("rw_wicket_pct", 0.055)
                + (1 - shrink_dismissal) * bowler_career.get("bowl_rw_wicket_pct", 0.055), 4),
            "bvb_rw_sr": round(
                shrink_other * agg.get("rw_sr", 120)
                + (1 - shrink_other) * batter_career.get("bat_rw_sr", 120), 4),
            "bvb_rw_dot_pct": round(
                shrink_other * agg.get("rw_dot_pct", 0.33)
                + (1 - shrink_other) * batter_career.get("bat_rw_dot_pct", 0.33), 4),
            "bvb_rw_boundary_pct": round(
                shrink_other * agg.get("rw_boundary_pct", 0.15)
                + (1 - shrink_other) * batter_career.get("bat_rw_boundary_pct", 0.15), 4),
            "bvb_rw_six_pct": round(
                shrink_other * agg.get("rw_six_pct", 0.06)
                + (1 - shrink_other) * batter_career.get("bat_rw_six_pct", 0.06), 4),
        }

    with open(PROCESSED_DIR / "bvb_stats.json", "w") as f:
        json.dump(bvb_stats, f)
    print(f"  bvb_stats.json ({len(bvb_stats)} matchups, all shrunk toward career stats)")

    # ── batter-at-venue / bowler-at-venue interaction stats ──────────────────
    # Same shrinkage-toward-career-number idea as feature_engineer.py's
    # _add_player_venue_features — kept as-of-today snapshots (not causal by
    # season, since this is what's used to simulate a brand new future match).
    # Uses the SAME K as training (PLAYER_VENUE_SHRINK_K) so live simulation
    # trusts venue-specific data at exactly the rate the model was trained to
    # expect — a mismatch here would quietly reintroduce train/serve skew.
    K = PLAYER_VENUE_SHRINK_K
    batter_venue_stats = {}
    for (name, venue), grp in df.groupby(["striker", "venue"]):
        if name not in active_players or len(grp) < 3:
            continue
        agg = _rw_agg(grp)
        if not agg:
            continue
        career = batter_stats.get(name, {})
        balls = agg["rw_balls"]
        shrink = balls / (balls + K)
        key = f"{name}|||{venue}"
        batter_venue_stats[key] = {
            "bat_venue_rw_balls": round(balls, 2),
            "bat_venue_adj_sr": round(
                shrink * agg["rw_sr"] + (1 - shrink) * career.get("bat_rw_sr", 120.0), 4),
            "bat_venue_adj_boundary_pct": round(
                shrink * agg["rw_boundary_pct"] + (1 - shrink) * career.get("bat_rw_boundary_pct", 0.15), 4),
        }
    with open(PROCESSED_DIR / "batter_venue_stats.json", "w") as f:
        json.dump(batter_venue_stats, f)
    print(f"  batter_venue_stats.json ({len(batter_venue_stats)} player-venue pairs)")

    bowler_venue_stats = {}
    for (name, venue), grp in df.groupby(["bowler", "venue"]):
        if name not in active_players or len(grp) < 3:
            continue
        agg = _rw_agg(grp)
        if not agg:
            continue
        career = bowler_stats.get(name, {})
        balls = agg["rw_balls"]
        shrink = balls / (balls + K)
        key = f"{name}|||{venue}"
        bowler_venue_stats[key] = {
            "bowl_venue_rw_balls": round(balls, 2),
            "bowl_venue_adj_economy": round(
                shrink * agg["rw_economy"] + (1 - shrink) * career.get("bowl_rw_economy", 8.5), 4),
            "bowl_venue_adj_wicket_pct": round(
                shrink * agg["rw_wicket_pct"] + (1 - shrink) * career.get("bowl_rw_wicket_pct", 0.05), 4),
        }
    with open(PROCESSED_DIR / "bowler_venue_stats.json", "w") as f:
        json.dump(bowler_venue_stats, f)
    print(f"  bowler_venue_stats.json ({len(bowler_venue_stats)} player-venue pairs)")

    # ── venue stats ───────────────────────────────────────────────────────────
    venue_stats = {}
    inn1_scores = (
        df[df["innings"] == 1]
        .groupby(["match_id", "venue", "season"])
        .agg(total=("total_runs", "sum"), sw=("season_weight", "first"))
        .reset_index()
    )
    inn2_scores = (
        df[df["innings"] == 2]
        .groupby("match_id")
        .agg(total=("total_runs", "sum"))
        .reset_index()
        .rename(columns={"total": "inn2_total"})
    )
    match_scores = inn1_scores.merge(inn2_scores, on="match_id", how="left")

    for venue, vg in match_scores.groupby("venue"):
        w   = vg["sw"].values
        ws  = w.sum()
        if ws == 0:
            continue
        rw_1st = (vg["total"].values   * w).sum() / ws
        rw_2nd = (vg["inn2_total"].fillna(0).values * w).sum() / ws

        dg = df[df["venue"] == venue]
        dagg = _rw_agg(dg)
        if not dagg:
            continue

        pp_mask   = dg["phase"] == "powerplay"
        death_mask= dg["phase"] == "death"

        def _phase_sr(mask):
            sg = dg[mask]
            if len(sg) == 0:
                return 0.0
            a = _rw_agg(sg)
            return round(a.get("rw_sr", 0), 4) if a else 0.0

        venue_stats[venue] = {
            "venue_rw_avg_1st_innings": round(rw_1st, 2),
            "venue_rw_avg_2nd_innings": round(rw_2nd, 2),
            "venue_rw_boundary_pct":    round(dagg.get("rw_boundary_pct", 0.17), 4),
            "venue_rw_six_pct":         round(dagg.get("rw_six_pct", 0.08), 4),
            "venue_rw_dot_pct":         round(dagg.get("rw_dot_pct", 0.31), 4),
            "venue_rw_wicket_pct":      round(dagg.get("rw_wicket_pct", 0.054), 4),
            "venue_rw_pp_sr":           _phase_sr(pp_mask),
            "venue_rw_death_sr":        _phase_sr(death_mask),
        }

    with open(PROCESSED_DIR / "venue_stats.json", "w") as f:
        json.dump(venue_stats, f)
    print(f"  venue_stats.json ({len(venue_stats)} venues)")


def _export_meta(df: pd.DataFrame, active_players: set):
    # Team-scoped rosters: which ACTIVE players have actually batted/bowled
    # for each team historically. This is what fixes lineups like a CSK
    # legend or a non-MI bowler showing up in an "MI" XI — the old
    # meta.batters/meta.bowlers were a single unscoped historical pool.
    batters_by_team: Dict[str, list] = {}
    for team, grp in df.groupby("batting_team"):
        batters_by_team[team] = sorted(
            p for p in grp["striker"].dropna().unique() if p in active_players
        )

    bowlers_by_team: Dict[str, list] = {}
    for team, grp in df.groupby("bowling_team"):
        bowlers_by_team[team] = sorted(
            p for p in grp["bowler"].dropna().unique() if p in active_players
        )

    meta = {
        # kept for backwards compatibility — DO NOT use these for lineup
        # dropdowns anymore, they mix every team and every era. Use
        # batters_by_team / bowlers_by_team instead.
        "batters":  sorted(df["striker"].dropna().unique().tolist()),
        "bowlers":  sorted(df["bowler"].dropna().unique().tolist()),

        "active_batters":  sorted(p for p in df["striker"].dropna().unique() if p in active_players),
        "active_bowlers":  sorted(p for p in df["bowler"].dropna().unique() if p in active_players),
        "batters_by_team": batters_by_team,
        "bowlers_by_team": bowlers_by_team,

        "teams":    sorted(df["batting_team"].dropna().unique().tolist()),
        "venues":   sorted(df["venue"].dropna().unique().tolist()),
        "seasons":  sorted(int(x) for x in df["season"].dropna().unique().tolist()),
        "outcomes": ["0", "1", "2", "3", "4", "6", "W"],
    }
    with open(PROCESSED_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  meta.json  ({len(meta['active_batters'])} active batters | "
          f"{len(meta['active_bowlers'])} active bowlers | "
          f"{len(meta['venues'])} venues)")


if __name__ == "__main__":
    main()