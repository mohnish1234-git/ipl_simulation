"""
Model Predictor
Loads the trained XGBoost model exported from Google Colab and exposes
a clean predict_proba() interface used by the simulation engine.

Expected files in models/:
  - ipl_ball_model.pkl
  - label_encoders.pkl
  - feature_columns.pkl
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List

MODELS_DIR = Path("models")
OUTCOMES   = ["0", "1", "2", "3", "4", "6", "W"]


class BallOutcomePredictor:
    """Thin wrapper around the trained model."""

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.model:        object    = joblib.load(models_dir / "ipl_ball_model.pkl")
        self.encoders:     dict      = joblib.load(models_dir / "label_encoders.pkl")
        self.feature_cols: List[str] = joblib.load(models_dir / "feature_columns.pkl")
        self.label_encoder           = self.encoders.get("outcome")
        print("Model loaded ✓")

    def predict_proba(self, ball_context: dict) -> Dict[str, float]:
        """
        Parameters
        ----------
        ball_context : dict with keys matching ALL feature columns produced by
                       the new feature_engineer.py pipeline.  All 7 groups:

          Categorical:
            striker, bowler, batting_team, bowling_team, venue, phase

          Match state:
            over_num, ball_num, cumulative_runs, cumulative_wickets,
            balls_remaining, wickets_remaining, crr

          Recency-weighted batter stats:
            bat_rw_avg, bat_rw_sr, bat_rw_boundary_pct, bat_rw_six_pct,
            bat_rw_dot_pct,
            bat_pp_rw_sr, bat_mid_rw_sr, bat_death_rw_sr,
            bat_pp_rw_boundary_pct, bat_death_rw_boundary_pct

          Recency-weighted bowler stats:
            bowl_rw_economy, bowl_rw_wicket_pct, bowl_rw_dot_pct,
            bowl_rw_boundary_pct,
            bowl_pp_rw_economy, bowl_mid_rw_economy, bowl_death_rw_economy,
            bowl_pp_rw_wicket_pct, bowl_death_rw_wicket_pct

          Batter vs Bowler matchup:
            bvb_balls, bvb_rw_sr, bvb_rw_dismissal_pct,
            bvb_rw_dot_pct, bvb_rw_boundary_pct, bvb_rw_six_pct

          In-match momentum:
            batter_balls_faced, batter_runs_scored, batter_innings_sr,
            balls_vs_bowler, runs_vs_bowler,
            runs_last6, runs_last_over,
            consec_dots, consec_boundaries,
            partnership_runs, partnership_balls,
            prev_ball_outcome, prev2_ball_outcome, prev3_ball_outcome

          Venue intelligence:
            venue_rw_avg_1st_innings, venue_rw_avg_2nd_innings,
            venue_rw_boundary_pct, venue_rw_six_pct, venue_rw_dot_pct,
            venue_rw_wicket_pct, venue_rw_pp_sr, venue_rw_death_sr

          Batting context / pressure:
            is_batting_first, is_chasing,
            target, runs_needed, rrr, pressure_index

        Returns
        -------
        dict  e.g. {"0": 0.28, "1": 0.30, "2": 0.08, "3": 0.01,
                     "4": 0.17, "6": 0.10, "W": 0.06}
        """
        row = self._encode_row(ball_context)
        X   = pd.DataFrame([row]).reindex(columns=self.feature_cols, fill_value=0)
        probs  = self.model.predict_proba(X)[0]
        classes = self.label_encoder.classes_
        result  = {cls: float(p) for cls, p in zip(classes, probs)}
        for o in OUTCOMES:
            result.setdefault(o, 0.0)
        return result

    def _encode_row(self, ctx: dict) -> dict:
        row = dict(ctx)
        cat_cols = ["striker", "bowler", "batting_team", "bowling_team", "venue", "phase"]
        for col in cat_cols:
            le = self.encoders.get(col)
            if le is None:
                row[col] = -1
                continue
            val = str(row.get(col, "Unknown"))
            if val not in le.classes_:
                val = "Unknown" if "Unknown" in le.classes_ else le.classes_[0]
            row[col] = int(le.transform([val])[0])
        return row


# ── Fallback / mock predictor ──────────────────────────────────────────────────

class MockPredictor:
    """
    Returns heuristic probabilities tuned to produce realistic T20 scores
    (~160–175 in powerplay/death, ~140–150 middle).
    Used automatically when model pkl files are absent.
    """

    # Probabilities calibrated so E[runs/ball] ≈ 1.35–1.55 in PP/death
    #   and ≈ 1.15–1.25 in middle, with wicket rates matching IPL norms.
    PHASE_WEIGHTS = {
        "powerplay": {"0": 0.25, "1": 0.32, "2": 0.09, "3": 0.01,
                      "4": 0.18, "6": 0.10, "W": 0.05},
        "middle":    {"0": 0.30, "1": 0.34, "2": 0.09, "3": 0.01,
                      "4": 0.12, "6": 0.06, "W": 0.08},
        "death":     {"0": 0.22, "1": 0.26, "2": 0.08, "3": 0.01,
                      "4": 0.20, "6": 0.14, "W": 0.09},
    }

    def predict_proba(self, ball_context: dict) -> Dict[str, float]:
        phase = ball_context.get("phase", "middle")
        base  = dict(self.PHASE_WEIGHTS.get(phase, self.PHASE_WEIGHTS["middle"]))

        # Simple momentum adjustment using context if available
        pressure = float(ball_context.get("pressure_index", 0.0))
        if pressure > 3:                          # chasing team desperate
            base["4"] = min(base["4"] + 0.03, 0.28)
            base["6"] = min(base["6"] + 0.03, 0.20)
            base["0"] = max(base["0"] - 0.03, 0.12)
            base["W"] = min(base["W"] + 0.01, 0.12)

        consec_dots = int(ball_context.get("consec_dots", 0))
        if consec_dots >= 3:                      # batsman under pressure, takes risk
            base["4"] = min(base["4"] + 0.02, 0.28)
            base["6"] = min(base["6"] + 0.02, 0.20)
            base["W"] = min(base["W"] + 0.02, 0.14)
            base["0"] = max(base["0"] - 0.04, 0.10)

        # normalise to sum to 1
        total = sum(base.values())
        return {k: v / total for k, v in base.items()}


def load_predictor(use_mock: bool = False):
    if use_mock:
        print("Using MockPredictor (no model file needed)")
        return MockPredictor()
    try:
        return BallOutcomePredictor()
    except FileNotFoundError:
        print("⚠  Model files not found in models/ — falling back to MockPredictor")
        return MockPredictor()