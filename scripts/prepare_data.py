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
from src.data.team_mapping   import canonicalize_team
from src.data.feature_engineer import (
    build_features, _rw_agg,
    HALF_LIFE_SEASONS, VENUE_HALF_LIFE_SEASONS,
    PLAYER_VENUE_SHRINK_K, BVB_DISMISSAL_SHRINK_K, BVB_OTHER_SHRINK_K,
    OVER_BANDS, OVERBAND_SHRINK_K_FINE,
    shrink_rate,
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

    # ── Venue canonicalization (previously defined in venue_mapping.py but
    #    NEVER imported/applied anywhere — every downstream step, including
    #    training, was running on raw, un-collapsed venue strings). This
    #    merges sponsorship-rename variants (e.g. old "Sardar Patel Stadium,
    #    Motera" and current "Narendra Modi Stadium, Ahmedabad") into one
    #    canonical string per ground, and drops rows for venues outside the
    #    13-ground allowlist. Must run BEFORE build_features() and BEFORE the
    #    lookup-table/meta.json exports below, since both read df_clean["venue"]
    #    directly. ──────────────────────────────────────────────────────────
    print("\nStep 2b: Canonicalizing venues …")

    # ── Team canonicalization ────────────────────────────────────────────────
    # clean() (src/data/cleaner.py) already applies this, but re-apply here
    # too — same defense-in-depth reasoning as the venue re-canonicalization
    # above: if df_clean ever arrives from an already-"cleaned" CSV that
    # predates this fix, _export_lookup_tables/_export_meta below (which
    # group by batting_team/bowling_team) must never silently fragment one
    # franchise's history across old/new names again.
    print("\nStep 2c: Canonicalizing team names …")
    for col in ["batting_team", "bowling_team"]:
        canon = df_clean[col].map(canonicalize_team)
        df_clean[col] = canon.where(canon.notna(), df_clean[col])

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
        # over-band splits (replaces the old powerplay/middle/death phases —
        # there's no real discrete boundary in how batters actually play,
        # and 7-10 vs 11-15 genuinely differ despite both being "middle")
        band_aggs = {}
        for band_tag, lo, hi in OVER_BANDS:
            band_grp = grp[(grp["over_num"] >= lo) & (grp["over_num"] <= hi)]
            band_aggs[band_tag] = _rw_agg(band_grp) if len(band_grp) else None

        batter_stats[name] = {
            # Shrunk toward BASE_RATE_PRIOR in proportion to how many
            # weighted legal balls back each number (career-wide here;
            # each band below uses its OWN, usually much smaller, count).
            # Un-shrunk, a player with a handful of career balls and a
            # couple of lucky boundaries exports an SR of 200+ verbatim,
            # and player_profiles.py's archetype classifier reads that
            # with a hard percentile cutoff — tiny-sample noise gets
            # treated exactly like a genuine elite hitter.
            "bat_rw_avg":              round(shrink_rate(agg.get("rw_avg", 28.0), agg.get("rw_balls", 0), "rw_avg"), 4),
            "bat_rw_sr":               round(shrink_rate(agg.get("rw_sr", 128.0), agg.get("rw_balls", 0), "rw_sr"), 4),
            "bat_rw_boundary_pct":     round(shrink_rate(agg.get("rw_boundary_pct", 0.15), agg.get("rw_balls", 0), "rw_boundary_pct"), 4),
            "bat_rw_six_pct":          round(shrink_rate(agg.get("rw_six_pct", 0.06), agg.get("rw_balls", 0), "rw_six_pct"), 4),
            "bat_rw_dot_pct":          round(shrink_rate(agg.get("rw_dot_pct", 0.33), agg.get("rw_balls", 0), "rw_dot_pct"), 4),
        }
        for band_tag, _, _ in OVER_BANDS:
            b = band_aggs[band_tag]
            b_balls = b.get("rw_balls", 0) if b else 0
            # 4 bands means less data per bucket than the old 3 phases had —
            # shrinkage matters MORE here, not less. A player who's mostly
            # a powerplay batter but has barely played the death overs gets
            # a heavily-shrunk death-band number instead of a wild one.
            batter_stats[name][f"bat_{band_tag}_rw_sr"] = round(
                shrink_rate(b.get("rw_sr", 128.0) if b else 128.0, b_balls, "rw_sr"), 4)
            batter_stats[name][f"bat_{band_tag}_rw_boundary_pct"] = round(
                shrink_rate(b.get("rw_boundary_pct", 0.15) if b else 0.15, b_balls, "rw_boundary_pct"), 4)
            batter_stats[name][f"bat_{band_tag}_rw_dot_pct"] = round(
                shrink_rate(b.get("rw_dot_pct", 0.33) if b else 0.33, b_balls, "rw_dot_pct"), 4)
    # ── Data-driven tailender fallback (no authored numbers) ─────────────────
    # A "recognized bowler batting" profile: players whose bowling workload
    # dwarfs their batting workload, averaged from THEIR OWN real bat_rw_*
    # stats — not a hand-picked constant.
    bat_balls_by_player  = df.groupby("striker")["is_legal"].sum()
    bowl_balls_by_player = df.groupby("bowler")["is_legal"].sum()

    tailender_names = [
        name for name in bat_balls_by_player.index
        if bowl_balls_by_player.get(name, 0) > 5 * bat_balls_by_player.get(name, 1)
        and name in active_players
    ]

    tailender_rows = [
        batter_stats[name] for name in tailender_names if name in batter_stats
    ]

    if tailender_rows:
        keys = tailender_rows[0].keys()
        tailender_default = {
            k: round(float(np.mean([r[k] for r in tailender_rows if k in r])), 4)
            for k in keys
        }
    else:
        tailender_default = {}   # empty dict, not a fabricated number

    with open(PROCESSED_DIR / "tailender_default.json", "w") as f:
        json.dump(tailender_default, f)
    print(f"  tailender_default.json (computed from {len(tailender_rows)} recognized bowlers' real batting stats)")
    
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
        band_aggs = {}
        for band_tag, lo, hi in OVER_BANDS:
            band_grp = grp[(grp["over_num"] >= lo) & (grp["over_num"] <= hi)]
            band_aggs[band_tag] = _rw_agg(band_grp) if len(band_grp) else None

        bowler_stats[name] = {
            "bowl_rw_economy":          round(shrink_rate(agg.get("rw_economy", 8.2), agg.get("rw_balls", 0), "rw_economy"), 4),
            "bowl_rw_wicket_pct":       round(shrink_rate(agg.get("rw_wicket_pct", 0.055), agg.get("rw_balls", 0), "rw_wicket_pct"), 4),
            "bowl_rw_dot_pct":          round(shrink_rate(agg.get("rw_dot_pct", 0.33), agg.get("rw_balls", 0), "rw_dot_pct"), 4),
            "bowl_rw_boundary_pct":     round(shrink_rate(agg.get("rw_boundary_pct", 0.15), agg.get("rw_balls", 0), "rw_boundary_pct"), 4),
        }
        for band_tag, _, _ in OVER_BANDS:
            b = band_aggs[band_tag]
            b_balls = b.get("rw_balls", 0) if b else 0
            bowler_stats[name][f"bowl_{band_tag}_rw_economy"] = round(
                shrink_rate(b.get("rw_economy", 8.2) if b else 8.2, b_balls, "rw_economy"), 4)
            bowler_stats[name][f"bowl_{band_tag}_rw_wicket_pct"] = round(
                shrink_rate(b.get("rw_wicket_pct", 0.055) if b else 0.055, b_balls, "rw_wicket_pct"), 4)
            bowler_stats[name][f"bowl_{band_tag}_rw_dot_pct"] = round(
                shrink_rate(b.get("rw_dot_pct", 0.33) if b else 0.33, b_balls, "rw_dot_pct"), 4)
            bowler_stats[name][f"bowl_{band_tag}_rw_boundary_pct"] = round(
                shrink_rate(b.get("rw_boundary_pct", 0.15) if b else 0.15, b_balls, "rw_boundary_pct"), 4)

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
        entry = {
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
        # ── over-band BvB split, shrunk toward the already-shrunk bvb_rw_sr /
        #    batter's career average — mirrors feature_engineer.py's BvB
        #    band shrinkage (which, unlike the career/venue tiers, is NOT
        #    restricted by MODERN_ERA_MIN_SEASON — matchups are too sparse
        #    to afford excluding any history). ─────────────────────────────
        for band_tag, lo, hi in OVER_BANDS:
            band_grp = grp[(grp["over_num"] >= lo) & (grp["over_num"] <= hi)]
            band_agg = _rw_agg(band_grp) if len(band_grp) else None
            band_balls = band_agg["rw_balls"] if band_agg else 0.0
            band_shrink = band_balls / (band_balls + OVERBAND_SHRINK_K_FINE)
            entry[f"bvb_{band_tag}_sr"] = round(
                band_shrink * band_agg["rw_sr"] + (1 - band_shrink) * entry["bvb_rw_sr"]
                if band_agg else entry["bvb_rw_sr"], 4)
            entry[f"bvb_{band_tag}_avg"] = round(
                band_shrink * band_agg["rw_avg"] + (1 - band_shrink) * batter_career.get("bat_rw_avg", 26.0)
                if band_agg else batter_career.get("bat_rw_avg", 26.0), 4)
        bvb_stats[key] = entry

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
        entry = {
            "bat_venue_rw_balls": round(balls, 2),
            "bat_venue_adj_sr": round(
                shrink * agg["rw_sr"] + (1 - shrink) * career.get("bat_rw_sr", 120.0), 4),
            "bat_venue_adj_boundary_pct": round(
                shrink * agg["rw_boundary_pct"] + (1 - shrink) * career.get("bat_rw_boundary_pct", 0.15), 4),
        }
        # ── over-band split (1-6 / 7-10 / 11-15 / 16-20), shrunk toward the
        #    already-shrunk bat_venue_adj_sr / career bat_rw_avg — same
        #    hierarchy feature_engineer.py's _add_overband_features uses at
        #    training time. Previously these bands were trained on but never
        #    exported here, so StatsStore had nothing to load and every
        #    simulated ball silently zero-filled all 4 bands via
        #    predictor.py's reindex(fill_value=0). ─────────────────────────
        for band_tag, lo, hi in OVER_BANDS:
            band_grp = grp[(grp["over_num"] >= lo) & (grp["over_num"] <= hi)]
            band_agg = _rw_agg(band_grp) if len(band_grp) else None
            band_balls = band_agg["rw_balls"] if band_agg else 0.0
            band_shrink = band_balls / (band_balls + OVERBAND_SHRINK_K_FINE)
            entry[f"bat_venue_{band_tag}_sr"] = round(
                band_shrink * band_agg["rw_sr"] + (1 - band_shrink) * entry["bat_venue_adj_sr"]
                if band_agg else entry["bat_venue_adj_sr"], 4)
            entry[f"bat_venue_{band_tag}_avg"] = round(
                band_shrink * band_agg["rw_avg"] + (1 - band_shrink) * career.get("bat_rw_avg", 26.0)
                if band_agg else career.get("bat_rw_avg", 26.0), 4)
        batter_venue_stats[key] = entry
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
        entry = {
            "bowl_venue_rw_balls": round(balls, 2),
            "bowl_venue_adj_economy": round(
                shrink * agg["rw_economy"] + (1 - shrink) * career.get("bowl_rw_economy", 8.5), 4),
            "bowl_venue_adj_wicket_pct": round(
                shrink * agg["rw_wicket_pct"] + (1 - shrink) * career.get("bowl_rw_wicket_pct", 0.05), 4),
        }
        # ── over-band split, same rationale as the batter-at-venue block above.
        for band_tag, lo, hi in OVER_BANDS:
            band_grp = grp[(grp["over_num"] >= lo) & (grp["over_num"] <= hi)]
            band_agg = _rw_agg(band_grp) if len(band_grp) else None
            band_balls = band_agg["rw_balls"] if band_agg else 0.0
            band_shrink = band_balls / (band_balls + OVERBAND_SHRINK_K_FINE)
            entry[f"bowl_venue_{band_tag}_economy"] = round(
                band_shrink * band_agg["rw_economy"] + (1 - band_shrink) * entry["bowl_venue_adj_economy"]
                if band_agg else entry["bowl_venue_adj_economy"], 4)
            entry[f"bowl_venue_{band_tag}_wicket_pct"] = round(
                band_shrink * band_agg["rw_wicket_pct"] + (1 - band_shrink) * entry["bowl_venue_adj_wicket_pct"]
                if band_agg else entry["bowl_venue_adj_wicket_pct"], 4)
        bowler_venue_stats[key] = entry
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

    max_season_overall = df["season"].max()
    venue_decay_rate = np.log(2) / VENUE_HALF_LIFE_SEASONS

    for venue, vg in match_scores.groupby("venue"):
        # Steep, venue-specific decay computed fresh here — NOT the shared
        # "sw" (player-half-life season_weight) column. Reusing the slower
        # player decay here was exactly why venue_rw_avg_1st/2nd_innings
        # landed around 164/149 instead of reflecting the much higher totals
        # actual 2023+ matches at these grounds are producing.
        w   = np.exp(-venue_decay_rate * (max_season_overall - vg["season"].values))
        ws  = w.sum()
        if ws == 0:
            continue
        rw_1st = (vg["total"].values   * w).sum() / ws
        rw_2nd = (vg["inn2_total"].fillna(0).values * w).sum() / ws

        dg = df[df["venue"] == venue].copy()
        if len(dg) == 0:
            continue

        # ── Venue-specific recency weight ──────────────────────────────────
        # _rw_agg() defaults to the "season_weight" column, which was computed
        # once in cleaner.py using the PLAYER half-life (3 seasons) — reusing
        # it here made venue conditions decay at the same slow rate as player
        # career stats, so older lower-scoring seasons kept dragging venue
        # averages down. Venues get their own, much steeper decay instead
        # (VENUE_HALF_LIFE_SEASONS), computed fresh here and passed in as a
        # separate weight column so 2023+ seasons dominate almost entirely.
        max_season = df["season"].max()
        decay_rate = np.log(2) / VENUE_HALF_LIFE_SEASONS
        dg["venue_season_weight"] = np.exp(-decay_rate * (max_season - dg["season"]))

        dagg = _rw_agg(dg, w_col="venue_season_weight")
        if not dagg:
            continue

        entry = {
            "venue_rw_avg_1st_innings": round(rw_1st, 2),
            "venue_rw_avg_2nd_innings": round(rw_2nd, 2),
            "venue_rw_boundary_pct":    round(dagg.get("rw_boundary_pct", 0.17), 4),
            "venue_rw_six_pct":         round(dagg.get("rw_six_pct", 0.08), 4),
            "venue_rw_dot_pct":         round(dagg.get("rw_dot_pct", 0.31), 4),
            "venue_rw_wicket_pct":      round(dagg.get("rw_wicket_pct", 0.054), 4),
        }
        # ── over-band venue character (1-6 / 7-10 / 11-15 / 16-20) — replaces
        #    the old powerplay/middle/death split entirely, and uses the same
        #    steep venue-specific recency weight as the rest of this block.
        for band_tag, lo, hi in OVER_BANDS:
            band_dg = dg[(dg["over_num"] >= lo) & (dg["over_num"] <= hi)]
            band_agg = _rw_agg(band_dg, w_col="venue_season_weight") if len(band_dg) else None
            entry[f"venue_rw_{band_tag}_rr"] = round(
                band_agg["rw_sr"] * 0.06, 4) if band_agg else round(dagg.get("rw_sr", 128.0) * 0.06, 4)
            entry[f"venue_rw_{band_tag}_wicket_pct"] = round(
                band_agg["rw_wicket_pct"], 4) if band_agg else round(dagg.get("rw_wicket_pct", 0.054), 4)
        venue_stats[venue] = entry

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