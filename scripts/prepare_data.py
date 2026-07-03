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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from src.data.cleaner        import load_raw, clean
from src.data.feature_engineer import (
    build_features, _rw_agg,
    HALF_LIFE_SEASONS,
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

    print("\nStep 4: Exporting lookup tables for the simulator …")
    _export_lookup_tables(df_clean)

    print("\nStep 5: Building meta.json …")
    _export_meta(df_clean)

    print("\n" + "=" * 55)
    print("Data preparation complete.")
    print(f"\nNext → upload  data/processed/features.csv  to Google Colab.")


# ─────────────────────────────────────────────────────────────────────────────

def _export_lookup_tables(df: pd.DataFrame):
    """
    Export pre-computed recency-weighted stats as JSON files.
    The MatchSimulator StatsStore.load_from_csv() reads these at runtime.
    """

    # ── batter career stats ───────────────────────────────────────────────────
    batter_stats = {}
    for name, grp in df.groupby("striker"):
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
    bvb_stats = {}
    for (batter, bowler), grp in df.groupby(["striker", "bowler"]):
        if len(grp) < 6:
            continue
        agg = _rw_agg(grp)
        if not agg:
            continue
        key = f"{batter}|||{bowler}"
        bvb_stats[key] = {
            "bvb_balls":            int(len(grp)),
            "bvb_rw_sr":            round(agg.get("rw_sr", 120), 4),
            "bvb_rw_dismissal_pct": round(agg.get("rw_wicket_pct", 0.055), 4),
            "bvb_rw_dot_pct":       round(agg.get("rw_dot_pct", 0.33), 4),
            "bvb_rw_boundary_pct":  round(agg.get("rw_boundary_pct", 0.15), 4),
            "bvb_rw_six_pct":       round(agg.get("rw_six_pct", 0.06), 4),
        }

    with open(PROCESSED_DIR / "bvb_stats.json", "w") as f:
        json.dump(bvb_stats, f)
    print(f"  bvb_stats.json ({len(bvb_stats)} matchups with ≥6 balls)")

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


def _export_meta(df: pd.DataFrame):
    meta = {
        "batters":  sorted(df["striker"].dropna().unique().tolist()),
        "bowlers":  sorted(df["bowler"].dropna().unique().tolist()),
        "teams":    sorted(df["batting_team"].dropna().unique().tolist()),
        "venues":   sorted(df["venue"].dropna().unique().tolist()),
        "seasons":  sorted(int(x) for x in df["season"].dropna().unique().tolist()),
        "outcomes": ["0", "1", "2", "3", "4", "6", "W"],
    }
    with open(PROCESSED_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  meta.json  ({len(meta['batters'])} batters | "
          f"{len(meta['bowlers'])} bowlers | "
          f"{len(meta['venues'])} venues)")


if __name__ == "__main__":
    main()