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

Two calibrator shapes are supported, auto-detected from outcome_calibrator.pkl's
top-level dict keys:

  - OutcomeCalibrator : one global isotonic curve per class. Simple, but a
    single curve can't correct different phases of an innings differently —
    e.g. if the raw model is more overconfident about wickets in the middle
    overs than at the death (which measurably happens here), one curve has
    to compromise between the two rather than fixing either properly.

  - PhaseCalibrator   : one OutcomeCalibrator per match phase (powerplay /
    middle / death). Preferred when the miscalibration itself varies by
    phase, which is the case for this model.

Produced by: colab_training.py ("Calibration" cell) -> outcome_calibrator.pkl
Consumed by: predictor.py (BallOutcomePredictor)
"""

import joblib
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union

try:
    from sklearn.isotonic import IsotonicRegression
except ImportError:  # calibration is optional at runtime if sklearn is unavailable
    IsotonicRegression = None

OUTCOMES = ["0", "1", "2", "3", "4", "6", "W"]
PHASES   = ["powerplay", "middle", "death"]

# Isotonic regression is a monotone STEP function fit to whatever it saw in
# validation — if a rare outcome (e.g. "W", or "1" against a specific
# bowler/phase combination) never happened at some raw-probability level in
# that held-out sample, it legitimately learns and returns an exact 0.0
# there. That's correct for "matches my sample" but wrong for a match
# simulator: no real delivery has a truly impossible outcome — even the
# best bowler in the tightest death over still concedes the occasional
# single. A literal 0% makes that outcome structurally unsampleable for
# every ball that lands in that raw-probability region, which is a
# stronger (and false) claim than "rare." MIN_PROB re-establishes a floor
# so every one of the 7 outcomes always keeps some sampling weight.
MIN_PROB = 0.003


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

    def calibrate(self, probs: Dict[str, float], phase: Optional[str] = None) -> Dict[str, float]:
        """Apply calibration + renormalise. Falls back to the raw probs
        untouched if calibration hasn't been fitted/loaded.

        `phase` is accepted (and ignored) purely so callers can pass it
        unconditionally regardless of whether they got an OutcomeCalibrator
        or a PhaseCalibrator back from load_calibrator() — see PhaseCalibrator
        below, which actually uses it.
        """
        if not self.fitted:
            return probs

        out = {}
        for cls in OUTCOMES:
            p = probs.get(cls, 0.0)
            iso = self.calibrators.get(cls)
            out[cls] = float(iso.predict([p])[0]) if iso is not None else p

        # Never let a real outcome hit a literal 0% — see MIN_PROB above.
        out = {k: max(v, MIN_PROB) for k, v in out.items()}

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


class PhaseCalibrator:
    """Fits one OutcomeCalibrator per match phase (powerplay/middle/death).

    A single global isotonic curve per class can't correct different phases
    of an innings differently. Checked directly against this model: raw
    predicted P(W) runs ~1.7x too high in the middle overs but is roughly
    correct at the death — a single OutcomeCalibrator has to settle for one
    compromise curve between those two, which is exactly why noticeable
    over-prediction of wickets remained in the middle overs even after
    global calibration. This fits three independent calibrators instead,
    reusing OutcomeCalibrator's fit/calibrate machinery for each phase.
    """

    def __init__(self):
        self.by_phase: Dict[str, OutcomeCalibrator] = {}
        self.fitted = False

    def fit(self, raw_probs: np.ndarray, y_true_labels: List[str], classes: List[str],
            phase_labels: List[str]):
        phase_labels = np.asarray(phase_labels)
        y_true_labels = np.asarray(y_true_labels)
        for phase in PHASES:
            mask = phase_labels == phase
            if mask.sum() == 0:
                print(f"[WARN] No validation rows for phase='{phase}' - skipping "
                      f"(that phase will fall back to raw, uncalibrated probabilities).")
                continue
            cal = OutcomeCalibrator()
            cal.fit(raw_probs[mask], y_true_labels[mask], classes)
            self.by_phase[phase] = cal
        self.fitted = True
        return self

    def calibrate(self, probs: Dict[str, float], phase: Optional[str] = None) -> Dict[str, float]:
        if not self.fitted:
            return probs
        cal = self.by_phase.get(phase)
        if cal is None:
            # Unknown/unfitted phase — return uncorrected rather than guess
            # with the wrong phase's curve.
            return probs
        return cal.calibrate(probs)

    def save(self, path: Path):
        # Saved as {phase: {outcome: fitted IsotonicRegression}} — this is
        # how load_calibrator() tells a phase-conditional file apart from a
        # flat OutcomeCalibrator file (whose top-level keys are OUTCOMES,
        # not PHASES).
        joblib.dump(
            {phase: cal.calibrators for phase, cal in self.by_phase.items()},
            path,
        )

    def load(self, path: Path) -> "PhaseCalibrator":
        raw = joblib.load(Path(path))
        self.by_phase = {}
        for phase, calibrators in raw.items():
            cal = OutcomeCalibrator()
            cal.calibrators = calibrators
            cal.fitted = True
            self.by_phase[phase] = cal
        self.fitted = True
        return self


def load_calibrator(models_dir: Path) -> Optional[Union[OutcomeCalibrator, PhaseCalibrator]]:
    """Loads outcome_calibrator.pkl if present; returns None otherwise so
    predictor.py can transparently skip calibration (raw probabilities).

    Auto-detects flat vs. phase-conditional format from the pickle's
    top-level dict keys, so predictor.py doesn't need to know or care which
    one it got back — both expose the same `.calibrate(probs, phase=...)`
    interface.
    """
    path = Path(models_dir) / "outcome_calibrator.pkl"
    if not path.exists():
        print("[WARN] No calibrator found in models/ - using raw model probabilities")
        return None

    try:
        raw = joblib.load(path)
    except Exception as e:
        print(f"[WARN] Failed to load calibrator ({e}) - using raw model probabilities")
        return None

    top_keys = set(raw.keys())
    if top_keys and top_keys.issubset(set(PHASES)):
        cal = PhaseCalibrator()
        cal.by_phase = {}
        for phase, calibrators in raw.items():
            oc = OutcomeCalibrator()
            oc.calibrators = calibrators
            oc.fitted = True
            cal.by_phase[phase] = oc
        cal.fitted = True
        print(f"Calibrator loaded [OK] (phase-conditional: {sorted(cal.by_phase.keys())})")
        return cal

    if top_keys.issubset(set(OUTCOMES)):
        cal = OutcomeCalibrator()
        cal.calibrators = raw
        cal.fitted = True
        print("Calibrator loaded [OK] (flat, single global calibrator)")
        return cal

    print(f"[WARN] Unrecognized calibrator file format (top-level keys={top_keys}) "
          f"- using raw model probabilities")
    return None