"""
Fantasy Classifier Dataset Builder
====================================
Builds the training dataset for the Dream11 "high fantasy performer" binary
classification model, by reusing the already-built regression dataset
(data/processed/fantasy_dataset.csv, produced by scripts/dataset_builder.py)
instead of re-deriving features from raw ball-by-ball data.

Why reuse fantasy_dataset.csv rather than re-run the feature pipeline:
  - dataset_builder.py already does the hard, leakage-sensitive work (causal
    pre-match snapshots, LIVE_MATCH_STATE_COLS stripped, opportunity/team/
    matchup features joined, NaN cleanup). Re-deriving any of that here
    would risk silently drifting out of sync with the regression dataset —
    two pipelines producing two different feature definitions for what
    should be the exact same 24-player-pool inputs downstream.
  - The only new thing a classifier needs is a LABEL. Everything else
    (X) is identical to the regression dataset, by construction.

Label — WITHIN-MATCH TOP-N RANKING:
  is_high_performer = 1 for a player if their ACTUAL fantasy_points in that
  match ranks among the top TOP_N_PER_MATCH players in that SAME match
  (ties broken deterministically, see _add_top_n_per_match_label), else 0.

  This directly mirrors the real decision this model feeds: "would this
  player have been one of the best 11 picks in this specific match's
  player pool?" — not "did they clear some absolute or season-relative
  points bar in isolation". A player scoring 55 points in a low-scoring
  match where they'd have made the top 11 is exactly as much a "high
  performer" as a player scoring 85 in a high-scoring match where 85
  barely cracks the top 11 — the label now reflects that directly instead
  of via an indirect, per-season percentile proxy.

  TOP_N_PER_MATCH defaults to 11 (a full Dream11 XI's worth of "would have
  been selected") rather than a fixed fraction of the match's player pool,
  since 11 is the actual number of picks the downstream optimizer needs to
  make regardless of how many total players appear in a given match's
  dataset rows.

  fantasy_points is the label's source and must NEVER be included as a
  model input — it is dropped from the feature set below, exactly as
  train_fantasy_model.ipynb already excludes it via TARGET_COL.

Usage:
    python -m scripts.fantasy_classifier_dataset
"""

from pathlib import Path
import numpy as np
import pandas as pd

PROCESSED_DIR = Path("data/processed")
REGRESSION_DATASET_PATH = PROCESSED_DIR / "fantasy_dataset.csv"
OUTPUT_PATH = PROCESSED_DIR / "fantasy_classifier_dataset.csv"

# ── Label configuration ──────────────────────────────────────────────────
# Within each match_id, rank players by fantasy_points (descending) and
# label the top TOP_N_PER_MATCH as 1, everyone else in that match as 0.
TOP_N_PER_MATCH = 11

LABEL_COL = "is_high_performer"
TARGET_SOURCE_COL = "fantasy_points"  # never becomes a feature


def _load_regression_dataset(path: Path = REGRESSION_DATASET_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    print(f"Loaded {len(df):,} rows x {df.shape[1]} columns from {path}")
    if TARGET_SOURCE_COL not in df.columns:
        raise ValueError(
            f"'{TARGET_SOURCE_COL}' column not found in {path} — "
            "fantasy_classifier_dataset.py must run after dataset_builder.py."
        )
    if "match_id" not in df.columns:
        raise ValueError(
            f"'match_id' column not found in {path} — required for "
            "within-match top-N labeling. Re-run dataset_builder.py to regenerate it."
        )
    return df


def _add_top_n_per_match_label(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Rank players within each match_id by fantasy_points (descending) and
    label the top `top_n` as 1. Ties at the cutoff are broken deterministically
    (stable sort on fantasy_points, ties keep the original row order — i.e.
    whichever tied player already appears first in the dataset for that
    match) so re-running this script on the same input always reproduces
    the exact same label, rather than an arbitrary tie-break shifting who
    lands inside vs. just outside the top 11 from one run to the next.
    """
    # rank(method="first"): ties broken by row order, not fractional/average
    # ranks — gives a clean integer 1..N rank per match with no ambiguity
    # about whether a tied player is "in" or "out" of the top_n cutoff.
    df["_match_rank"] = (
        df.groupby("match_id")[TARGET_SOURCE_COL]
        .rank(method="first", ascending=False)
    )
    df[LABEL_COL] = (df["_match_rank"] <= top_n).astype(int)

    n_matches = df["match_id"].nunique()
    pool_sizes = df.groupby("match_id").size()
    n_small_pools = int((pool_sizes < top_n).sum())
    if n_small_pools:
        print(
            f"  Note: {n_small_pools:,}/{n_matches:,} matches have fewer than "
            f"{top_n} player rows — every player in those matches is labeled 1 "
            "(there's no 'bottom' to separate them from)."
        )

    df = df.drop(columns=["_match_rank"])
    return df


def build_classifier_dataset() -> pd.DataFrame:
    print("Step 1: loading regression dataset (features + actual fantasy points) …")
    df = _load_regression_dataset()

    print(f"\nStep 2: assigning '{LABEL_COL}' label "
          f"(within-match top {TOP_N_PER_MATCH} by fantasy_points) …")
    df = _add_top_n_per_match_label(df, TOP_N_PER_MATCH)

    n_pos = int(df[LABEL_COL].sum())
    n_tot = len(df)
    print(f"\nOverall class balance: {n_pos:,} positive / {n_tot:,} total "
          f"({n_pos / n_tot:.1%} high performers)")

    print(f"\nFinal classifier dataset shape: {df.shape}")
    return df


if __name__ == "__main__":
    dataset = build_classifier_dataset()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved -> {OUTPUT_PATH}")