"""
Post-hoc probability calibration for the multiclass ball-outcome model.

XGBoost's predict_proba() is optimised for log-loss / accuracy, not for
producing well-calibrated probabilities — this matters a lot here because
we SAMPLE from the distribution every ball, so a systematically over- or
under-confident model (especially for the rare 'W' class) directly distorts
run-rates and wicket clustering over a full simulated innings.

Approach: one-vs-rest isotonic regression per outcome class, fit on
predictions from a held-out validation set the XGBoost model never trained
on (colab_training.py already keeps a chronological last-15%-seasons split
for exactly this reason), followed by renormalisation so the 7 calibrated
probabilities still sum to 1. This is the standard recipe for calibrating a
multiclass model when you don't want to touch the underlying classifier —
it composes cleanly on top of BallOutcomePredictor.predict_proba() without
changing the XGBoost pipeline at all.

Produced by: colab_training.py ("Calibration" cell) -> outcome_calibrator.pkl
Consumed by: predictor.py (BallOutcomePredictor)
"""

import joblib
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

try:
    from sklearn.isotonic import IsotonicRegression
except ImportError:  # calibration is optional at runtime if sklearn is unavailable
    IsotonicRegression = None

OUTCOMES = ["0", "1", "2", "3", "4", "6", "W"]


class OutcomeCalibrator:
    """Holds one fitted IsotonicRegression per outcome class."""

    def __init__(self):
        self.calibrators: Dict[str, "IsotonicRegression"] = {}
        self.fitted = False

    def fit(self, raw_probs: np.ndarray, y_true_labels: List[str], classes: List[str]):
        """
        raw_probs     : (n_samples, n_classes) UNCALIBRATED probabilities from
                        model.predict_proba() run on a held-out validation set
                        (never seen during XGBoost training).
        y_true_labels : ground-truth outcome string for each row.
        classes       : class label for each column of raw_probs, in order
                        (i.e. le_target.classes_).
        """
        if IsotonicRegression is None:
            raise ImportError("scikit-learn is required to fit calibration")

        y_true_labels = np.asarray(y_true_labels)
        for i, cls in enumerate(classes):
            binary_target = (y_true_labels == cls).astype(float)
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(raw_probs[:, i], binary_target)
            self.calibrators[cls] = iso
        self.fitted = True
        return self

    def calibrate(self, probs: Dict[str, float]) -> Dict[str, float]:
        """Apply calibration + renormalise. Falls back to the raw probs
        untouched if calibration hasn't been fitted/loaded."""
        if not self.fitted:
            return probs

        out = {}
        for cls in OUTCOMES:
            p = probs.get(cls, 0.0)
            iso = self.calibrators.get(cls)
            out[cls] = float(iso.predict([p])[0]) if iso is not None else p

        total = sum(out.values())
        if total <= 0:
            return probs  # degenerate — don't hand back all-zero probabilities
        return {k: v / total for k, v in out.items()}

    def save(self, path: Path):
        joblib.dump(self.calibrators, path)

    def load(self, path: Path) -> "OutcomeCalibrator":
        path = Path(path)
        self.calibrators = joblib.load(path)
        self.fitted = True
        return self


def load_calibrator(models_dir: Path) -> Optional[OutcomeCalibrator]:
    """Loads outcome_calibrator.pkl if present; returns None otherwise so
    predictor.py can transparently skip calibration (raw probabilities)."""
    path = Path(models_dir) / "outcome_calibrator.pkl"
    if path.exists():
        cal = OutcomeCalibrator()
        try:
            cal.load(path)
            print("Calibrator loaded ✓")
            return cal
        except Exception as e:
            print(f"⚠  Failed to load calibrator ({e}) — using raw model probabilities")
            return None
    print("⚠  No calibrator found in models/ — using raw model probabilities")
    return None