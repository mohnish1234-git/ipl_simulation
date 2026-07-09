"""
Model Predictor
Loads the trained XGBoost model exported from Google Colab and exposes
a clean predict_proba() interface used by the simulation engine.

Expected files in models/:
  - ipl_ball_model.pkl
  - label_encoders.pkl
  - feature_columns.pkl
  - outcome_calibrator.pkl   (optional — produced by colab_training.py's
                              "Calibration" cell; if absent, raw XGBoost
                              probabilities are used unchanged)
"""

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from typing import Dict, List

from .calibration import load_calibrator, OutcomeCalibrator

MODELS_DIR = Path("models")
OUTCOMES   = ["0", "1", "2", "3", "4", "6", "W"]


class BallOutcomePredictor:
    """Thin wrapper around the trained XGBoost model.

    Loads the Booster from a version-portable JSON/UBJSON export
    (models/ipl_ball_model.json) rather than the pickled XGBClassifier —
    XGBoost's internal binary Booster format is NOT guaranteed compatible
    across major versions (1.x/2.x/3.x), so a model trained in Colab on one
    version and unpickled locally on another can fail deep inside the C++
    deserializer with an opaque XGBoostError. The JSON/UBJSON model format
    is explicitly documented as the portable, forward-compatible option.

    Applies post-hoc isotonic calibration (if a fitted calibrator is present
    in models/) to the raw softmax output before handing probabilities back
    to the simulator — the XGBoost model and feature pipeline are untouched.
    """

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.encoders:     dict      = joblib.load(models_dir / "label_encoders.pkl")
        self.feature_cols: List[str] = joblib.load(models_dir / "feature_columns.pkl")
        self.label_encoder           = self.encoders.get("outcome")

        json_path = models_dir / "ipl_ball_model.json"
        pkl_path  = models_dir / "ipl_ball_model.pkl"
        if json_path.exists():
            self.booster = xgb.Booster()
            self.booster.load_model(str(json_path))
            self._num_classes = len(self.label_encoder.classes_)
        else:
            # Legacy path — only works if local xgboost matches the version
            # the model was trained with. Re-export from Colab as JSON
            # (booster.save_model("ipl_ball_model.json")) to avoid this.
            print("⚠  models/ipl_ball_model.json not found — falling back to "
                  "the pickled model. This WILL break if your local xgboost "
                  "version differs from the one used to train it. Re-export "
                  "from Colab with booster.save_model('ipl_ball_model.json').")
            self.model = joblib.load(pkl_path)
            self.booster = None

        self.calibrator: OutcomeCalibrator = load_calibrator(models_dir)
        print("Model loaded ✓")

    def predict_proba(self, ball_context: dict) -> Dict[str, float]:
        row = self._encode_row(ball_context)
        X   = pd.DataFrame([row]).reindex(columns=self.feature_cols, fill_value=0)

        if self.booster is not None:
            dmat  = xgb.DMatrix(X, feature_names=self.feature_cols)
            probs = self.booster.predict(dmat)[0]
        else:
            probs = self.model.predict_proba(X)[0]

        classes = self.label_encoder.classes_
        result  = {cls: float(p) for cls, p in zip(classes, probs)}
        for o in OUTCOMES:
            result.setdefault(o, 0.0)
        if self.calibrator is not None:
            result = self.calibrator.calibrate(result)
        return result

    def _encode_row(self, ctx: dict) -> dict:
        row = dict(ctx)
        for col in ["striker", "bowler", "batting_team", "bowling_team", "venue", "phase"]:
            le = self.encoders.get(col)
            if le is None:
                row[col] = -1
                continue
            val = str(row.get(col, "Unknown"))
            if val not in le.classes_:
                val = "Unknown" if "Unknown" in le.classes_ else le.classes_[0]
            row[col] = int(le.transform([val])[0])
        return row


# ── MockPredictor ──────────────────────────────────────────────────────────────
# Weights calibrated from real IPL data (2008-2026, 295K deliveries):
#
#  Phase       W%      0%      1%      2%      4%      6%    E[runs/ball]
#  Powerplay   3.87%  46.09%  25.84%   4.12%  15.55%   4.07%   1.22
#  Middle      4.13%  30.97%  44.36%   6.47%   9.05%   4.78%   1.23
#  Death       7.56%  26.22%  37.71%   8.54%  11.51%   8.20%   1.51
#
# These produce realistic T20 innings totals of 150-175 on average.

class MockPredictor:
    """
    Returns data-calibrated probabilities matching real IPL outcome distributions.
    Used automatically when model pkl files are absent.
    """

    # Each phase maps to {outcome: probability}.
    # Derived directly from 295K IPL deliveries — NOT hand-tuned.
    PHASE_WEIGHTS = {
        "powerplay": {
            "0": 0.461, "1": 0.258, "2": 0.041, "3": 0.004,
            "4": 0.156, "6": 0.041, "W": 0.039,
        },
        "middle": {
            "0": 0.310, "1": 0.444, "2": 0.065, "3": 0.002,
            "4": 0.091, "6": 0.048, "W": 0.041,
        },
        "death": {
            "0": 0.262, "1": 0.377, "2": 0.085, "3": 0.002,
            "4": 0.115, "6": 0.082, "W": 0.076,
        },
    }

    def predict_proba(self, ball_context: dict) -> Dict[str, float]:
        phase = ball_context.get("phase", "middle")
        base  = dict(self.PHASE_WEIGHTS.get(phase, self.PHASE_WEIGHTS["middle"]))

        # ── Pressure adjustment: chasing team behind the rate ─────────────────
        pressure = float(ball_context.get("pressure_index", 0.0))
        if pressure > 4:          # very high pressure — more sixes/fours, more wickets
            shift = min(pressure * 0.003, 0.025)
            base["4"] = min(base["4"] + shift, 0.22)
            base["6"] = min(base["6"] + shift, 0.15)
            base["0"] = max(base["0"] - shift, 0.15)
            base["W"] = min(base["W"] + shift * 0.5, 0.12)

        # ── Momentum: batter on strike < 10 balls — more cautious ─────────────
        bf = int(ball_context.get("batter_balls_faced", 20))
        if bf < 5:
            base["1"] = min(base["1"] + 0.04, 0.50)
            base["4"] = max(base["4"] - 0.02, 0.05)
            base["6"] = max(base["6"] - 0.02, 0.02)

        # ── Consecutive dots — batsman takes risk ─────────────────────────────
        dots = int(ball_context.get("consec_dots", 0))
        if dots >= 4:
            base["4"] = min(base["4"] + 0.03, 0.22)
            base["6"] = min(base["6"] + 0.03, 0.15)
            base["W"] = min(base["W"] + 0.02, 0.12)
            base["0"] = max(base["0"] - 0.04, 0.15)

        # ── Normalise to sum to exactly 1.0 ───────────────────────────────────
        total = sum(base.values())
        return {k: v / total for k, v in base.items()}


def load_predictor(use_mock: bool = False, models_dir: Path = MODELS_DIR):
    if use_mock:
        print("Using MockPredictor (calibrated from real IPL data)")
        return MockPredictor()
    try:
        return BallOutcomePredictor(models_dir)
    except Exception as e:
        # Deliberately broad: a corrupt pickle, an XGBoost version mismatch,
        # or any other load-time failure should degrade to MockPredictor
        # rather than crashing the whole API on startup — but it prints
        # loudly so this can't silently persist unnoticed like it did before.
        print(f"⚠  Model load failed ({type(e).__name__}: {e}) "
              f"— using data-calibrated MockPredictor")
        return MockPredictor()