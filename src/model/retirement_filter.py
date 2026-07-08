"""
Retired-player filtering for training data.

Two independent uses:
  1. TRAINING: down-weight/drop deliveries so the model isn't spending
     capacity learning fine-grained patterns for players who will never
     appear in a future simulation.
  2. SIMULATION-TIME VALIDATION: refuse to simulate a match with a player
     who hasn't appeared in ACTIVE_WINDOW_SEASONS seasons, so a stale
     default-stat fallback never silently produces a bad simulated player.

Definition of "active": appeared (batting OR bowling) in at least one match
within the last ACTIVE_WINDOW_SEASONS completed seasons, as of the most
recent season in the dataset. This is intentionally simple and transparent —
swap in a real retirement-announcement list if you have one, it will always
beat an appearance-based heuristic.
"""

import pandas as pd
from typing import Set

ACTIVE_WINDOW_SEASONS = 3   # tune: 2 = stricter/current squads only, 4 = more lenient


def compute_active_players(df: pd.DataFrame, window_seasons: int = ACTIVE_WINDOW_SEASONS) -> Set[str]:
    """
    df must have a 'season' column plus 'striker' and 'bowler' columns
    (the raw ball-by-ball frame, BEFORE label-encoding).
    Returns the set of player names considered active.
    """
    if "season" not in df.columns:
        raise ValueError("df needs a 'season' column to determine player activity")

    latest_season = df["season"].max()
    cutoff_season = latest_season - window_seasons + 1
    recent = df[df["season"] >= cutoff_season]

    active = set(recent["striker"].dropna().unique()) | set(recent["bowler"].dropna().unique())
    return active


def filter_to_active_players(df: pd.DataFrame, active_players: Set[str],
                              mode: str = "either") -> pd.DataFrame:
    """
    mode="either": drop a row only if BOTH striker and bowler are inactive
                   (keeps deliveries where at least one side is a current
                   player — usually what you want, since batting/bowling
                   patterns for the active side are still informative).
    mode="both":   drop a row if EITHER striker or bowler is inactive
                   (stricter — only keep fully "current era" match-ups).
    """
    striker_active = df["striker"].isin(active_players)
    bowler_active  = df["bowler"].isin(active_players)

    if mode == "either":
        keep = striker_active | bowler_active
    elif mode == "both":
        keep = striker_active & bowler_active
    else:
        raise ValueError("mode must be 'either' or 'both'")

    before = len(df)
    out = df[keep].copy()
    print(f"Retirement filter ({mode}, last {ACTIVE_WINDOW_SEASONS} seasons): "
          f"{before - len(out):,} rows removed, {len(out):,} remaining "
          f"({len(active_players)} active players)")
    return out


def validate_squad_is_active(player_names, active_players: Set[str]) -> None:
    """Call this from MatchSimulator.simulate() before running an innings —
    fail loudly instead of silently falling back to DEFAULT_BATTER/DEFAULT_BOWLER
    for a name that shouldn't realistically appear in a new match at all."""
    stale = [p for p in player_names if p not in active_players]
    if stale:
        raise ValueError(
            f"These players are not in the active-player set (last "
            f"{ACTIVE_WINDOW_SEASONS} seasons) and will fall back to league "
            f"defaults, which usually understates realism: {stale}"
        )


# ── Usage in colab_training.py, inserted right after Cell 3 (load features.csv),
#    BEFORE Cell 4's cleaning steps ─────────────────────────────────────────────
#
# from retirement_filter import compute_active_players, filter_to_active_players
#
# active_players = compute_active_players(df, window_seasons=3)
# df = filter_to_active_players(df, active_players, mode="either")
#
# Save `active_players` alongside the other artifacts (e.g.
# joblib.dump(sorted(active_players), "active_players.pkl")) so the simulator
# can load it and call validate_squad_is_active() before every match.