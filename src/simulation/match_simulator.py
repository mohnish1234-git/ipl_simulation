"""
Match Simulation Engine
Simulates a full T20 match ball-by-ball using probabilities from the ML model.

Context dict passed to predictor now includes all 7 feature groups:
  - Match state
  - Recency-weighted player stats  (looked up from stats_store)
  - Batter-vs-bowler matchup       (looked up from stats_store)
  - In-match momentum              (tracked live)
  - Venue intelligence             (looked up from stats_store)
  - Batting context / pressure     (computed live)
"""

import contextlib
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.model.predictor import load_predictor
from src.simulation.player_profiles import apply_collapse_adjustment

# ── Outcome encoding (for prev_ball_outcome features) ─────────────────────────
_OUTCOME_ENCODE = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "6": 6, "W": 7}

# Must match the SAME normalization feature_engineer.py's build_features()
# divides pressure_index by (see _modern_par_run_rate_by_season / modern_par_rr)
# — a recency-weighted average 1st-innings run rate (runs/over) computed from
# PRIOR completed seasons. At live-simulation time there's no "season" to
# look this up per-row the way training does; this is the most recent
# season's value from that same computation, i.e. today's scoring
# environment. If you regenerate models/ after a season with materially
# different scoring, recompute this from prepare_data.py's
# _modern_par_run_rate_by_season() output (its latest entry) rather than
# hand-editing it, or the pressure_index scale will drift from what the
# model was actually trained on again.
MODERN_PAR_RR = 8.6

# Must match feature_engineer.py's BVB_OTHER_SHRINK_K / BVB_DISMISSAL_SHRINK_K
# exactly — this is the LIVE-inference counterpart of the same shrinkage the
# training pipeline applies to BvB (batter-vs-bowler) stats. Duplicated here
# rather than imported because match_simulator.py and feature_engineer.py
# live in different packages (src.simulation vs the offline data-prep code)
# with no shared dependency between them; if either K value changes, update
# both places.
BVB_OTHER_SHRINK_K     = 15.0   # sr / dot% / boundary% / six%
BVB_DISMISSAL_SHRINK_K = 40.0   # dismissal% only

# ── Extras sampling — fixed empirical rates computed from the user's own
#    cleaned.csv (raw["wide"], raw["noballs"], raw["byes"], raw["legbyes"]).
#    Not retrained into the model; applied as a pre-model layer so the
#    XGBoost model only ever sees genuine legal deliveries, exactly as
#    it was trained. ────────────────────────────────────────────────────────
EXTRA_RATES = {
    "wide":   0.0332,
    "noball": 0.00415,
    "bye":    0.00251,
    "legbye": 0.01486,
}
WIDE_BOUNDARY_RATE = 0.0   # data showed this never happens — kept at 0, no nested check needed
BYE_RUN_DIST = {1: 0.70, 2: 0.20, 4: 0.08, 5: 0.02}


def _sample_extra() -> str:
    r = random.random()
    cum = 0.0
    for kind, p in EXTRA_RATES.items():
        cum += p
        if r < cum:
            return kind
    return "none"


# ═══════════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BatterState:
    name:      str
    runs:      int = 0
    balls:     int = 0
    fours:     int = 0
    sixes:     int = 0
    dismissed: bool = False

    @property
    def strike_rate(self) -> float:
        return round(self.runs / self.balls * 100, 1) if self.balls else 0.0


@dataclass
class BowlerState:
    name:    str
    runs:    int = 0
    balls:   int = 0
    wickets: int = 0
    wides:   int = 0
    noballs: int = 0

    @property
    def economy(self) -> float:
        overs = self.balls / 6
        return round(self.runs / overs, 2) if overs else 0.0


@dataclass
class InningsState:
    batting_team:    str
    bowling_team:    str
    batting_order:   List[str]
    bowling_rotation: List[str]       # one entry per over (20 entries)
    venue:           str
    innings_num:     int = 1
    target:          Optional[int] = None

    # Par run rate used to normalize pressure_index, matching training's
    # modern_par_rr — sourced from meta.json when available (see
    # StatsStore.modern_par_rr), falling back to MODERN_PAR_RR otherwise.
    # Kept per-innings-state (not just the global constant) so a future
    # multi-era/multi-competition setup could vary it per match without
    # touching this class.
    par_rr: float = MODERN_PAR_RR

    score:       int = 0
    wickets:     int = 0
    legal_balls: int = 0
    extras:      int = 0

    batter_states: Dict[str, BatterState] = field(default_factory=dict)
    bowler_states: Dict[str, BowlerState] = field(default_factory=dict)

    striker_idx:     int = 0
    non_striker_idx: int = 1
    next_batter_idx: int = 2

    ball_log: List[dict] = field(default_factory=list)

    # ── momentum tracking ─────────────────────────────────────────────────────
    legal_ball_runs: List[int] = field(default_factory=list)   # team rolling window
    over_runs:       Dict[int, int] = field(default_factory=dict)
    outcome_history: List[str] = field(default_factory=list)
    consec_dots:     int = 0
    consec_boundaries: int = 0

    # per-batter in-match stats
    batter_innings_balls: Dict[str, int] = field(default_factory=dict)
    batter_innings_runs:  Dict[str, int] = field(default_factory=dict)

    # per-bowler in-match stats (this innings) — bowler's OWN whole-innings
    # figures, aggregated across every batter they've faced. Kept for the
    # bowler's own state/economy tracking (BowlerState); NOT the same thing
    # as the batter-specific "vs this bowler, this innings" feature below.
    bowler_innings_balls: Dict[str, int] = field(default_factory=dict)
    bowler_innings_runs:  Dict[str, int] = field(default_factory=dict)

    # per-(batter, bowler) PAIR in-match stats — this is what feature_engineer.py
    # actually means by "balls_vs_bowler" / "runs_vs_bowler" / "current_matchup_sr"
    # (see its per-row loop: pair_key = (striker, bowler)). Previously this was
    # read off bowler_innings_balls/runs instead — the bowler's aggregate across
    # ALL batters — so every batter facing the same bowler in an innings got an
    # IDENTICAL value for what was supposed to be a batter-specific signal.
    pair_balls: Dict[Tuple[str, str], int] = field(default_factory=dict)
    pair_runs:  Dict[Tuple[str, str], int] = field(default_factory=dict)

    # partnership
    partnership_runs:  int = 0
    partnership_balls: int = 0

    @property
    def overs_bowled(self) -> float:
        return self.legal_balls / 6

    @property
    def crr(self) -> float:
        overs = self.overs_bowled
        return round(self.score / overs, 2) if overs else 0.0

    @property
    def balls_remaining(self) -> int:
        return max(0, 120 - self.legal_balls)

    @property
    def wickets_remaining(self) -> int:
        return 10 - self.wickets

    @property
    def runs_needed(self) -> int:
        return max(0, (self.target - self.score)) if self.target else 0

    @property
    def rrr(self) -> float:
        br = self.balls_remaining
        if self.innings_num == 2 and br > 0 and self.target:
            return round(self.runs_needed / (br / 6), 2)
        return 0.0

    @property
    def pressure_index(self) -> float:
        # MUST match feature_engineer.py's training formula exactly:
        #   pressure_index = (rrr - crr) / modern_par_rr
        # This division was added to training when modern_par_rr was
        # introduced, but was never carried over here — pressure_index (and
        # everything downstream of it, like pressure_weighted_aggression)
        # was being fed to the model RAW, at roughly 8-9x the scale it was
        # ever trained on (the model only ever saw values normalized into
        # roughly [-2, +3]; raw (rrr-crr) routinely swings [-20, +20] in a
        # real chase). That's a heavily-weighted numeric feature landing
        # far outside the model's trained range on every single ball of
        # every second-innings simulation — exactly the kind of thing that
        # produces wild, unrealistic ball-outcome predictions in either
        # direction (no-boundary 2s-and-3s streaks, rapid collapses).
        return round((self.rrr - self.crr) / self.par_rr, 2) if self.innings_num == 2 else 0.0

    def striker(self) -> str:
        return self.batting_order[self.striker_idx]

    def non_striker(self) -> str:
        return self.batting_order[self.non_striker_idx]

    def is_complete(self) -> bool:
        if self.wickets >= 10 or self.legal_balls >= 120:
            return True
        if self.innings_num == 2 and self.target and self.score >= self.target:
            return True
        return False

    def runs_last6(self) -> int:
        return int(sum(self.legal_ball_runs[-6:]))

    def runs_last12(self) -> int:
        return int(sum(self.legal_ball_runs[-12:]))

    def runs_last18(self) -> int:
        return int(sum(self.legal_ball_runs[-18:]))

    def runs_last_over(self, over: int) -> int:
        return int(self.over_runs.get(over - 1, 0))

    @property
    def partnership_run_rate(self) -> float:
        return round(self.partnership_runs / self.partnership_balls * 6, 2) if self.partnership_balls else 0.0

    def prev_outcome(self, n: int) -> int:
        hist = self.outcome_history
        return _OUTCOME_ENCODE.get(hist[-n], 0) if len(hist) >= n else -1


@dataclass
class MatchResult:
    batting_team_1:  str
    batting_team_2:  str
    score_1:         int
    score_2:         int
    wickets_1:       int
    wickets_2:       int
    winner:          str
    win_margin:      int
    win_type:        str
    innings_1_log:   List[dict]
    innings_2_log:   List[dict]
    batter_stats_1:  Dict
    batter_stats_2:  Dict
    bowler_stats_1:  Dict
    bowler_stats_2:  Dict


# ═══════════════════════════════════════════════════════════════════════════════
# Stats Store — pre-computed player / venue lookup tables
# ═══════════════════════════════════════════════════════════════════════════════

class StatsStore:
    """
    Holds pre-computed recency-weighted statistics loaded from the processed
    dataset or defaulting to IPL-typical values.

    In production: load from data/processed/player_stats.json (generated by
    prepare_data.py).  For simulation without that file, sensible defaults
    are used — the model will still produce realistic scores because momentum
    and context features dominate for known players.
    """

    DEFAULT_BATTER = dict(
        bat_rw_avg=1.28, bat_rw_sr=128.0,
        bat_rw_boundary_pct=0.15, bat_rw_six_pct=0.06, bat_rw_dot_pct=0.33,
        bat_pp_rw_sr=125.0, bat_mid_rw_sr=130.0, bat_death_rw_sr=145.0,
        bat_pp_rw_boundary_pct=0.14, bat_mid_rw_boundary_pct=0.15,
        bat_death_rw_boundary_pct=0.22,
        # These three were missing here too — trained-on features that must
        # exist on every batter dict (real or default), or they zero-fill
        # for whichever players fall back to DEFAULT_BATTER.
        bat_pp_rw_dot_pct=0.33, bat_mid_rw_dot_pct=0.33, bat_death_rw_dot_pct=0.30,
    )
    DEFAULT_BOWLER = dict(
        bowl_rw_economy=8.2, bowl_rw_wicket_pct=0.055, bowl_rw_dot_pct=0.33,
        bowl_rw_boundary_pct=0.15,
        bowl_pp_rw_economy=7.5, bowl_mid_rw_economy=8.0, bowl_death_rw_economy=9.2,
        bowl_pp_rw_wicket_pct=0.07, bowl_mid_rw_wicket_pct=0.05,
        bowl_death_rw_wicket_pct=0.055,
        bowl_pp_rw_dot_pct=0.35, bowl_mid_rw_dot_pct=0.33, bowl_death_rw_dot_pct=0.28,
        bowl_pp_rw_boundary_pct=0.14, bowl_mid_rw_boundary_pct=0.15,
        bowl_death_rw_boundary_pct=0.20,
    )
    DEFAULT_BVB = dict(
        bvb_balls=0, bvb_rw_sr=128.0, bvb_rw_dismissal_pct=0.05,
        bvb_rw_dot_pct=0.33, bvb_rw_boundary_pct=0.15, bvb_rw_six_pct=0.06,
        # Over-band split (1-6/7-10/11-15/16-20) — trained on but, before
        # prepare_data.py's over-band export existed, never supplied here
        # either, so every ball for an unmatched pair zero-filled all 8 of
        # these via predictor.py's reindex(fill_value=0). Falls back to the
        # same flat career-ish numbers as the rest of DEFAULT_BVB.
        bvb_1_6_sr=128.0, bvb_1_6_avg=26.0,
        bvb_7_10_sr=128.0, bvb_7_10_avg=26.0,
        bvb_11_15_sr=128.0, bvb_11_15_avg=26.0,
        bvb_16_20_sr=145.0, bvb_16_20_avg=22.0,
    )
    DEFAULT_VENUE = dict(
        venue_rw_avg_1st_innings=165.0, venue_rw_avg_2nd_innings=155.0,
        venue_rw_boundary_pct=0.17, venue_rw_six_pct=0.08,
        venue_rw_dot_pct=0.31, venue_rw_wicket_pct=0.054,
        venue_rw_pp_sr=140.0, venue_rw_death_sr=165.0,
        # Over-band venue character — same missing-column issue as above,
        # for the 8 venue_rw_{band}_rr / venue_rw_{band}_wicket_pct columns.
        # rr values interpolate roughly powerplay -> death; wicket_pct held
        # at the flat league-average default.
        venue_rw_1_6_rr=8.2,   venue_rw_1_6_wicket_pct=0.054,
        venue_rw_7_10_rr=8.4,  venue_rw_7_10_wicket_pct=0.054,
        venue_rw_11_15_rr=8.8, venue_rw_11_15_wicket_pct=0.054,
        venue_rw_16_20_rr=9.9, venue_rw_16_20_wicket_pct=0.054,
    )

    def __init__(self):
        self._batter: Dict[str, dict] = {}
        self._bowler: Dict[str, dict] = {}
        self._bvb:    Dict[Tuple, dict] = {}
        self._venue:  Dict[str, dict] = {}
        self._batter_venue: Dict[Tuple, dict] = {}   # (batter, venue) -> stats
        self._bowler_venue: Dict[Tuple, dict] = {}   # (bowler, venue) -> stats
        self._tailender_default: dict = {}            # ← was missing; batter() reads
                                                        #   this even when load_from_csv()
                                                        #   is never called or the file
                                                        #   is absent, so it must always exist
        # Falls back to the MODERN_PAR_RR module constant unless meta.json
        # (written by prepare_data.py) supplies the real recency-weighted
        # value for the latest season — see load_from_csv() below.
        self.modern_par_rr: float = MODERN_PAR_RR

    def load_from_csv(self, processed_dir: str = "data/processed"):
        """
        Attempt to load pre-computed stats from CSVs saved by prepare_data.py.
        Silently falls back to defaults if files are not present.

        batter_venue_stats.json / bowler_venue_stats.json were previously
        generated by prepare_data.py but never loaded here — meaning the
        highest-precedence feature tier (player-at-venue) was always
        zero-filled by predictor.py's reindex(fill_value=0) at inference,
        no matter how well the model was trained on it. That's fixed below.
        """
        import json
        from pathlib import Path
        p = Path(processed_dir)

        for fname, attr in [
            ("player_batter_stats.json", "_batter"),
            ("player_bowler_stats.json", "_bowler"),
            ("bvb_stats.json",           "_bvb"),
            ("venue_stats.json",         "_venue"),
            ("batter_venue_stats.json",  "_batter_venue"),
            ("bowler_venue_stats.json",  "_bowler_venue"),
        ]:
            fpath = p / fname
            if fpath.exists():
                with open(fpath) as f:
                    data = json.load(f)
                if attr in ("_bvb", "_batter_venue", "_bowler_venue"):
                    setattr(self, attr, {tuple(k.split("|||")): v for k, v in data.items()})
                else:
                    setattr(self, attr, data)
                print(f"  Loaded {fname}")
            else:
                print(f"  ⚠ {fname} not found — {attr} will fall back to career "
                      f"stats / defaults for every player")

        tpath = p / "tailender_default.json"
        if tpath.exists():
            with open(tpath) as f:
                self._tailender_default = json.load(f)
            print(f"  Loaded tailender_default.json ({len(self._tailender_default)} stats)")
        else:
            print("  ⚠ tailender_default.json not found — bowlers with no batting "
                  "record will fall back to DEFAULT_BATTER (league average)")

        mpath = p / "meta.json"
        if mpath.exists():
            with open(mpath) as f:
                meta = json.load(f)
            if "modern_par_rr" in meta:
                self.modern_par_rr = float(meta["modern_par_rr"])
                print(f"  Loaded modern_par_rr from meta.json: {self.modern_par_rr:.3f} "
                      f"(overrides the {MODERN_PAR_RR} module default)")
            else:
                print(f"  ⚠ meta.json has no modern_par_rr — pressure_index will "
                      f"normalize against the {MODERN_PAR_RR} module default instead "
                      f"of the actual recency-weighted current-season value")
        else:
            print(f"  ⚠ meta.json not found — pressure_index will normalize against "
                  f"the {MODERN_PAR_RR} module default instead of the actual "
                  f"recency-weighted current-season value")

    def batter(self, name: str) -> dict:
        hit = self._batter.get(name)
        if hit is not None:
            return hit
        if name in self._bowler and self._tailender_default:
            return self._tailender_default
        return self.DEFAULT_BATTER

    def bowler(self, name: str) -> dict:
        return self._bowler.get(name, self.DEFAULT_BOWLER)

    def bvb(self, batter: str, bowler: str) -> dict:
        """Falls back to blending toward THIS batter's/bowler's own career
        stats — not a hardcoded generic default — matching feature_engineer.py's
        BVB_OTHER_SHRINK_K / BVB_DISMISSAL_SHRINK_K blend, and batter_venue()/
        bowler_venue()'s "fall back to the real player, not a league constant"
        pattern above.

        Previously this did a bare dict lookup: DEFAULT_BVB (flat SR=128.0,
        dismissal%=0.05, etc., regardless of who's actually batting/bowling)
        for an unseen pair, and — worse — trusted a RAW matchup entry
        completely even if it was based on only 1-2 real balls. A batter who's
        faced a given bowler only a couple of times, by chance including a
        dismissal, would show a wildly elevated bvb_rw_dismissal_pct with
        nothing to pull it back toward reality — exactly the extreme
        wicket-probability spikes seen for lightly-matched-up pairs.
        """
        raw = self._bvb.get((batter, bowler))
        career_bat  = self.batter(batter)
        career_bowl = self.bowler(bowler)

        fallback_sr       = career_bat.get("bat_rw_sr", self.DEFAULT_BATTER["bat_rw_sr"])
        fallback_avg      = career_bat.get("bat_rw_avg", self.DEFAULT_BATTER.get("bat_rw_avg", 26.0))
        fallback_dot      = career_bat.get("bat_rw_dot_pct", self.DEFAULT_BATTER["bat_rw_dot_pct"])
        fallback_boundary = career_bat.get("bat_rw_boundary_pct", self.DEFAULT_BATTER["bat_rw_boundary_pct"])
        fallback_six      = career_bat.get("bat_rw_six_pct", self.DEFAULT_BATTER["bat_rw_six_pct"])
        fallback_dismiss  = career_bowl.get("bowl_rw_wicket_pct", self.DEFAULT_BOWLER["bowl_rw_wicket_pct"])

        out = {
            "bvb_rw_sr": fallback_sr, "bvb_rw_dismissal_pct": fallback_dismiss,
            "bvb_rw_dot_pct": fallback_dot, "bvb_rw_boundary_pct": fallback_boundary,
            "bvb_rw_six_pct": fallback_six, "bvb_balls": 0.0,
        }
        for band_tag in ("1_6", "7_10", "11_15", "16_20"):
            out[f"bvb_{band_tag}_sr"]  = fallback_sr
            out[f"bvb_{band_tag}_avg"] = fallback_avg

        if raw is None:
            return out

        balls = raw.get("bvb_balls", 0.0)
        shrink_other   = balls / (balls + BVB_OTHER_SHRINK_K)
        shrink_dismiss = balls / (balls + BVB_DISMISSAL_SHRINK_K)

        out["bvb_balls"] = balls
        out["bvb_rw_sr"] = (shrink_other * raw.get("bvb_rw_sr", fallback_sr)
                             + (1 - shrink_other) * fallback_sr)
        out["bvb_rw_dot_pct"] = (shrink_other * raw.get("bvb_rw_dot_pct", fallback_dot)
                                  + (1 - shrink_other) * fallback_dot)
        out["bvb_rw_boundary_pct"] = (shrink_other * raw.get("bvb_rw_boundary_pct", fallback_boundary)
                                       + (1 - shrink_other) * fallback_boundary)
        out["bvb_rw_six_pct"] = (shrink_other * raw.get("bvb_rw_six_pct", fallback_six)
                                  + (1 - shrink_other) * fallback_six)
        out["bvb_rw_dismissal_pct"] = (shrink_dismiss * raw.get("bvb_rw_dismissal_pct", fallback_dismiss)
                                        + (1 - shrink_dismiss) * fallback_dismiss)
        # Over-band split — same shrink factor as the top-level SR blend
        # above; a raw sparse pairing doesn't have a separate per-band ball
        # count to shrink against independently, so this is a reasonable
        # single approximation rather than 4 separate (and even noisier) ones.
        for band_tag in ("1_6", "7_10", "11_15", "16_20"):
            out[f"bvb_{band_tag}_sr"] = (shrink_other * raw.get(f"bvb_{band_tag}_sr", fallback_sr)
                                          + (1 - shrink_other) * fallback_sr)
            out[f"bvb_{band_tag}_avg"] = (shrink_other * raw.get(f"bvb_{band_tag}_avg", fallback_avg)
                                           + (1 - shrink_other) * fallback_avg)
        return out

    def venue(self, v: str) -> dict:
        hit = self._venue.get(v)
        if hit is None:
            print(f"⚠ Venue lookup miss for '{v}' — using DEFAULT_VENUE "
                  f"(generic league average, not this ground's real profile)")
        return hit if hit is not None else self.DEFAULT_VENUE

    def batter_venue(self, name: str, venue: str) -> dict:
        """Highest-precedence tier. Falls back to this player's own career
        SR/boundary% (NOT a hardcoded global default) when there's no
        venue-specific history yet — matches feature_engineer.py's Step C
        behavior of blending toward the player's real career number rather
        than a league-wide constant."""
        hit = self._batter_venue.get((name, venue))
        if hit is not None:
            return hit
        career = self.batter(name)
        adj_sr = career.get("bat_rw_sr", self.DEFAULT_BATTER["bat_rw_sr"])
        adj_avg = career.get("bat_rw_avg", self.DEFAULT_BATTER.get("bat_rw_avg", 26.0))
        out = {
            "bat_venue_adj_sr": adj_sr,
            "bat_venue_adj_boundary_pct": career.get(
                "bat_rw_boundary_pct", self.DEFAULT_BATTER["bat_rw_boundary_pct"]),
            "bat_venue_rw_balls": 0.0,
        }
        # Over-band split — with no venue-specific history at all, fall all
        # the way back to the player's own career SR/avg for every band
        # (matches feature_engineer.py's shrink-to-0 behavior when
        # bat_venue_{band}_balls is 0).
        for band_tag in ("1_6", "7_10", "11_15", "16_20"):
            out[f"bat_venue_{band_tag}_sr"] = adj_sr
            out[f"bat_venue_{band_tag}_avg"] = adj_avg
        return out

    def bowler_venue(self, name: str, venue: str) -> dict:
        """Same fallback logic as batter_venue(), for bowlers."""
        hit = self._bowler_venue.get((name, venue))
        if hit is not None:
            return hit
        career = self.bowler(name)
        adj_econ = career.get("bowl_rw_economy", self.DEFAULT_BOWLER["bowl_rw_economy"])
        adj_wkt = career.get("bowl_rw_wicket_pct", self.DEFAULT_BOWLER["bowl_rw_wicket_pct"])
        out = {
            "bowl_venue_adj_economy": adj_econ,
            "bowl_venue_adj_wicket_pct": adj_wkt,
            "bowl_venue_rw_balls": 0.0,
        }
        # Over-band split — same career-only fallback rationale as
        # batter_venue() above.
        for band_tag in ("1_6", "7_10", "11_15", "16_20"):
            out[f"bowl_venue_{band_tag}_economy"] = adj_econ
            out[f"bowl_venue_{band_tag}_wicket_pct"] = adj_wkt
        return out


# ═══════════════════════════════════════════════════════════════════════════════
# Simulator
# ═══════════════════════════════════════════════════════════════════════════════

class MatchSimulator:
    """
    Simulates a full T20 match.

    Usage
    -----
    stats  = StatsStore()
    stats.load_from_csv()          # optional — loads pre-computed stats
    sim    = MatchSimulator(predictor, stats)
    result = sim.simulate(
        team1="Mumbai Indians",
        team2="Chennai Super Kings",
        batting_order_1=[...],     # 11 players
        batting_order_2=[...],
        bowling_rotation_1=[...],  # 20 bowler names (one per over)
        bowling_rotation_2=[...],
        venue="Wankhede Stadium",
    )
    """

    def __init__(self, predictor=None, stats_store: Optional[StatsStore] = None,
                 verbose: bool = False):
        self.predictor   = predictor or load_predictor()
        self.stats_store = stats_store or StatsStore()
        self.verbose      = verbose

    # ── Public API ────────────────────────────────────────────────────────────

    def simulate(
        self,
        team1: str,
        team2: str,
        batting_order_1:   List[str],
        batting_order_2:   List[str],
        bowling_rotation_1: List[str],
        bowling_rotation_2: List[str],
        venue: str = "Unknown",
        toss_winner: str = None,
        toss_choice: str = "bat",
    ) -> MatchResult:

        # No canonicalization/allowlist here anymore — venue_mapping.py was
        # solving a naming-variant problem this dataset doesn't actually
        # have. venue is used exactly as passed in (cleaner.py already
        # normalized whitespace at data-prep time). If it's a venue with no
        # stats on file (typo, or a ground genuinely absent from training
        # data), stats_store.venue() falls back to DEFAULT_VENUE rather than
        # erroring — that's a soft degrade worth knowing about, so we warn
        # instead of silently proceeding.
        venue = str(venue).strip()
        if venue not in self.stats_store._venue:
            print(f"⚠ No venue stats on file for '{venue}' — falling back to "
                  f"DEFAULT_VENUE (league-average profile) for this match.")

        if toss_winner == team2 and toss_choice == "bat":
            batting_first, batting_second   = team2, team1
            order_first, order_second       = batting_order_2, batting_order_1
            bowl_rot_first, bowl_rot_second = bowling_rotation_1, bowling_rotation_2
        else:
            batting_first, batting_second   = team1, team2
            order_first, order_second       = batting_order_1, batting_order_2
            bowl_rot_first, bowl_rot_second = bowling_rotation_2, bowling_rotation_1

        inn1 = self._simulate_innings(
            innings_num=1, batting_team=batting_first, bowling_team=batting_second,
            batting_order=order_first, bowling_rotation=bowl_rot_first, venue=venue,
        )

        target = inn1.score + 1
        inn2 = self._simulate_innings(
            innings_num=2, batting_team=batting_second, bowling_team=batting_first,
            batting_order=order_second, bowling_rotation=bowl_rot_second,
            venue=venue, target=target,
        )

        if inn2.score >= target:
            winner    = batting_second
            win_type  = "wickets"
            win_margin = inn2.wickets_remaining
        else:
            winner    = batting_first
            win_type  = "runs"
            win_margin = inn1.score - inn2.score

        if self.verbose:
            print(f"\n{'=' * 60}\nFINAL: {batting_first} {inn1.score}/{inn1.wickets}  vs  "
                  f"{batting_second} {inn2.score}/{inn2.wickets}\n{'=' * 60}")

        return MatchResult(
            batting_team_1=batting_first, batting_team_2=batting_second,
            score_1=inn1.score, score_2=inn2.score,
            wickets_1=inn1.wickets, wickets_2=inn2.wickets,
            winner=winner, win_margin=win_margin, win_type=win_type,
            innings_1_log=inn1.ball_log, innings_2_log=inn2.ball_log,
            batter_stats_1={n: v.__dict__ for n, v in inn1.batter_states.items()},
            batter_stats_2={n: v.__dict__ for n, v in inn2.batter_states.items()},
            bowler_stats_1={n: v.__dict__ for n, v in inn1.bowler_states.items()},
            bowler_stats_2={n: v.__dict__ for n, v in inn2.bowler_states.items()},
        )

    def _log_ball(self, state, over, ball_in_over, striker, bowler_name,
                   outcome, runs, extras, is_wicket, probs):
        """The ONE place ball-by-ball prediction output gets printed.

        Previously there were three overlapping print mechanisms: an
        unconditional full-context-dict dump that fired on every ball
        regardless of the verbose flag (extremely noisy — this is what
        produced hundred-line-per-ball logs), a separate _print_ball()
        called mid-branch for only two of the four extra_kind branches, and
        a third, redundant end-of-ball summary line. Consolidated into this
        single verbose-gated call site so every ball — including wides/
        byes/leg-byes, which never call the model — gets exactly one clean
        line plus (when the model was actually called) its full predicted
        distribution.
        """
        header = (f"[Inn{state.innings_num}] {over}.{ball_in_over}  "
                  f"{striker} vs {bowler_name}  "
                  f"[{state.batting_team} {state.score}/{state.wickets}]")
        print(header)
        if probs is not None:
            ranked = sorted(probs.items(), key=lambda kv: -kv[1])
            dist = "  ".join(f"{k}:{v:.3f}" for k, v in ranked)
            print(f"    model probs -> {dist}")
        else:
            print("    (extra delivery — sampled from EXTRA_RATES, model not called)")
        print(f"    result -> {outcome}  (runs={runs}, extras={extras}, "
              f"wicket={is_wicket})  score={state.score}/{state.wickets}")

    # ── Innings simulation ────────────────────────────────────────────────────

    def _simulate_innings(
        self,
        innings_num:     int,
        batting_team:    str,
        bowling_team:    str,
        batting_order:   List[str],
        bowling_rotation: List[str],
        venue:           str,
        target:          Optional[int] = None,
    ) -> InningsState:

        state = InningsState(
            batting_team=batting_team, bowling_team=bowling_team,
            batting_order=list(batting_order),
            bowling_rotation=list(bowling_rotation),
            venue=venue, innings_num=innings_num, target=target,
            par_rr=self.stats_store.modern_par_rr,
        )
        for i in range(min(2, len(state.batting_order))):
            name = state.batting_order[i]
            state.batter_states[name] = BatterState(name=name)

        for over in range(20):
            if state.is_complete():
                break
            bowler_name = (bowling_rotation[over]
                           if over < len(bowling_rotation)
                           else bowling_rotation[-1])
            if bowler_name not in state.bowler_states:
                state.bowler_states[bowler_name] = BowlerState(name=bowler_name)
            self._simulate_over(state, over, bowler_name)

        return state

    def _simulate_over(self, state: InningsState, over: int, bowler_name: str):
        legal_in_over = 0
        ball_in_over  = 0

        while legal_in_over < 6 and not state.is_complete():
            ball_in_over += 1
            striker      = state.striker()
            phase        = _phase(over)

            # ── Build full context dict ───────────────────────────────────────
            bs   = self.stats_store.batter(striker)
            bls  = self.stats_store.bowler(bowler_name)
            bvbs = self.stats_store.bvb(striker, bowler_name)
            vs   = self.stats_store.venue(state.venue)
            bsv  = self.stats_store.batter_venue(striker, state.venue)
            blsv = self.stats_store.bowler_venue(bowler_name, state.venue)

            ctx = {
                # categorical
                "striker":      striker,
                "bowler":       bowler_name,
                "batting_team": state.batting_team,
                "bowling_team": state.bowling_team,
                "venue":        state.venue,
                "phase":        phase,
                # match state
                "over_num":             over,
                "ball_num":             legal_in_over + 1,
                "cumulative_runs":      state.score,
                "cumulative_wickets":   state.wickets,
                "balls_remaining":      state.balls_remaining,
                "wickets_remaining":    state.wickets_remaining,
                "crr":                  state.crr,
                # batter career
                **bs,
                # bowler career
                **bls,
                # player-at-venue (highest precedence tier — was missing
                # entirely before; every prediction was silently zero-filling
                # bat_venue_adj_sr etc. for every ball of every match)
                **bsv,
                **blsv,
                # matchup
                **bvbs,
                # momentum
                "batter_balls_faced":   state.batter_innings_balls.get(striker, 0),
                "batter_runs_scored":   state.batter_innings_runs.get(striker, 0),
                "batter_innings_sr":    _innings_sr(state, striker),
                # batter-specific "this matchup, this innings" — keyed on the
                # (striker, bowler) PAIR, matching feature_engineer.py. Was
                # previously the bowler's whole-innings aggregate across every
                # batter, which made this feature identical for every batter
                # facing the same bowler (see InningsState.pair_balls/pair_runs).
                "balls_vs_bowler":      state.pair_balls.get((striker, bowler_name), 0),
                "runs_vs_bowler":       state.pair_runs.get((striker, bowler_name), 0),
                "current_matchup_sr":   _current_matchup_sr(state, striker, bowler_name),
                "runs_last6":           state.runs_last6(),
                "runs_last12":          state.runs_last12(),
                "runs_last18":          state.runs_last18(),
                "runs_last_over":       state.runs_last_over(over),
                "consec_dots":          state.consec_dots,
                "consec_boundaries":    state.consec_boundaries,
                "partnership_runs":     state.partnership_runs,
                "partnership_balls":    state.partnership_balls,
                "partnership_run_rate": state.partnership_run_rate,
                "prev_ball_outcome":    state.prev_outcome(1),
                "prev2_ball_outcome":   state.prev_outcome(2),
                "prev3_ball_outcome":   state.prev_outcome(3),
                # venue
                **vs,
                # context
                "is_batting_first": int(state.innings_num == 1),
                "is_chasing":       int(state.innings_num == 2),
                "target":           state.target or 0,
                "runs_needed":      state.runs_needed,
                "rrr":              state.rrr,
                "pressure_index":   state.pressure_index,
                # These 4 were trained on (feature_engineer.py's "batting
                # context / pressure" group) but never supplied here — every
                # prediction was silently zero-filling them via predictor.py's
                # reindex(fill_value=0), which especially hurts realism of
                # second-innings chases (wickets-in-hand-adjusted pressure).
                # Formulas match feature_engineer.py exactly, computed from
                # the same state properties already used above.
                "required_runs_per_wicket": (
                    round(state.runs_needed / state.wickets_remaining, 2)
                    if state.innings_num == 2 and state.wickets_remaining > 0 else 0.0
                ),
                "balls_per_required_run": (
                    round(state.balls_remaining / state.runs_needed, 2)
                    if state.innings_num == 2 and state.runs_needed > 0
                    else state.balls_remaining
                ),
                "pressure_weighted_rrr": (
                    round(state.rrr * (1 + (10 - state.wickets_remaining) / 10), 2)
                    if state.innings_num == 2 else 0.0
                ),
                "pressure_weighted_aggression": (
                    round(state.pressure_index * (state.wickets_remaining / 10), 2)
                    if state.innings_num == 2 else 0.0
                ),
            }

            # ── Extras layer — sampled BEFORE the model is ever called, from
            #    fixed empirical rates. Only "none" reaches predict_proba(),
            #    which keeps the model exactly in the domain it was trained
            #    on (legal deliveries only). ──────────────────────────────────
            extra_kind = _sample_extra()
            probs = None   # stays None for wide/bye/legbye — no model call happens for those
            if extra_kind == "wide":
                runs, is_wicket, is_wide, is_noball = 0, False, True, False
                extras_this_ball = 1
                outcome = "wide+1"

            elif extra_kind == "noball":
                # No-ball: 1 automatic extra, but the batter can still score
                # off the bat, so the model IS still called for that part.
                probs = self.predictor.predict_proba(ctx)
                probs = apply_collapse_adjustment(probs, state.wickets, over)
                bat_outcome = _sample_outcome(probs)
                bat_runs, _, _, _ = _parse_outcome(bat_outcome)
                runs, is_wicket, is_wide, is_noball = bat_runs, False, False, True
                extras_this_ball = 1
                outcome = f"noball+{bat_runs}"

            elif extra_kind in ("bye", "legbye"):
                bye_runs = random.choices(
                    list(BYE_RUN_DIST.keys()), weights=list(BYE_RUN_DIST.values()), k=1
                )[0]
                runs, is_wicket, is_wide, is_noball = 0, False, False, False  # legal ball
                extras_this_ball = bye_runs
                outcome = f"{extra_kind}+{bye_runs}"

            else:
                probs = self.predictor.predict_proba(ctx)
                probs = apply_collapse_adjustment(probs, state.wickets, over)
                outcome = _sample_outcome(probs)
                print("\n" + "=" * 70)
                print(f"Ball : {over}.{ball_in_over}")
                print(f"{striker} vs {bowler_name}")

                print("\nPredicted Probabilities")
                for k, v in sorted(probs.items()):
                    print(f"{k:>2} : {v:.4%}")

                print(f"\nSelected Outcome : {outcome}")
                print("=" * 70)
                runs, is_wicket, is_wide, is_noball = _parse_outcome(outcome)
                extras_this_ball = 0

            # ── Update batter state ───────────────────────────────────────────
            striker_state = state.batter_states[striker]
            if not is_wide:
                striker_state.balls += 1
                state.batter_innings_balls[striker] = state.batter_innings_balls.get(striker, 0) + 1
            if extra_kind not in ("bye", "legbye"):
                # byes/leg-byes are not credited to the batter's own runs
                striker_state.runs += runs
                state.batter_innings_runs[striker] = state.batter_innings_runs.get(striker, 0) + runs
            if runs == 4: striker_state.fours += 1
            if runs == 6: striker_state.sixes += 1

            # ── Update bowler state ───────────────────────────────────────────
            bstate = state.bowler_states[bowler_name]
            if not is_wide: bstate.balls += 1
            bstate.runs += runs
            if is_wide:   bstate.wides   += 1
            if is_noball: bstate.noballs += 1

            extras = extras_this_ball
            state.score  += runs + extras
            state.extras += extras

            bowl_runs_this = runs + extras
            state.bowler_innings_balls[bowler_name] = state.bowler_innings_balls.get(bowler_name, 0) + (0 if is_wide else 1)
            state.bowler_innings_runs[bowler_name]  = state.bowler_innings_runs.get(bowler_name, 0) + bowl_runs_this

            # ── pair-level (batter, bowler) tracking — this is what actually
            #    feeds balls_vs_bowler / runs_vs_bowler / current_matchup_sr.
            #    Wides don't count as a ball faced by the striker; runs off
            #    the bat still count toward the pair (matching feature_engineer's
            #    per-row loop, which only advances on legal deliveries the
            #    striker faced — byes/leg-byes aren't credited to the batter
            #    there either, so mirror that here). ─────────────────────────
            pair_key = (striker, bowler_name)
            if not is_wide:
                state.pair_balls[pair_key] = state.pair_balls.get(pair_key, 0) + 1
            if extra_kind not in ("bye", "legbye"):
                state.pair_runs[pair_key] = state.pair_runs.get(pair_key, 0) + runs

            # ── Wicket ────────────────────────────────────────────────────────
            if is_wicket:
                state.wickets += 1
                striker_state.dismissed = True
                bstate.wickets += 1
                state.partnership_runs  = 0
                state.partnership_balls = 0
                if (state.wickets < 10 and
                        state.next_batter_idx < len(state.batting_order)):
                    new_name = state.batting_order[state.next_batter_idx]
                    state.batter_states[new_name] = BatterState(name=new_name)
                    state.striker_idx  = state.next_batter_idx
                    state.next_batter_idx += 1
            else:
                if runs % 2 == 1:                # odd runs → swap strike
                    state.striker_idx, state.non_striker_idx = (
                        state.non_striker_idx, state.striker_idx
                    )

            # ── Legal delivery bookkeeping ────────────────────────────────────
            if not is_wide and not is_noball:
                legal_in_over  += 1
                state.legal_balls += 1
                state.legal_ball_runs.append(runs)
                state.partnership_runs  += runs
                state.partnership_balls += 1

            state.over_runs[over] = state.over_runs.get(over, 0) + runs + extras

            # ── Streak counters ───────────────────────────────────────────────
            is_boundary = runs in (4, 6)
            if is_wicket or (runs == 0 and not is_wide and not is_noball):
                state.consec_dots       = state.consec_dots + 1 if not is_boundary else 0
                state.consec_boundaries = 0
                if is_wicket:
                    state.consec_dots = 0
            elif is_boundary:
                state.consec_boundaries += 1
                state.consec_dots = 0
            else:
                state.consec_dots       = 0
                state.consec_boundaries = 0

            state.outcome_history.append(outcome)

            state.ball_log.append({
                "innings": state.innings_num,
                "over":    over, "ball": ball_in_over,
                "striker": striker if not is_wicket else striker_state.name,
                "bowler":  bowler_name,
                "outcome": outcome, "runs": runs, "extras": extras,
                "is_wicket": is_wicket,
                "score": state.score, "wickets": state.wickets,
                "probs": probs,   # None for wide/bye/legbye; real dict for noball/normal balls
            })

            if self.verbose:
                self._log_ball(state, over, ball_in_over, striker, bowler_name,
                                outcome, runs, extras, is_wicket, probs)

        # end-of-over: swap strike
        state.striker_idx, state.non_striker_idx = (
            state.non_striker_idx, state.striker_idx
        )
    

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _phase(over: int) -> str:
    if over < 6:  return "powerplay"
    if over < 15: return "middle"
    return "death"


def _innings_sr(state: InningsState, striker: str) -> float:
    b = state.batter_innings_balls.get(striker, 0)
    r = state.batter_innings_runs.get(striker, 0)
    return round(r / b * 100, 1) if b > 0 else 0.0


def _current_matchup_sr(state: InningsState, striker: str, bowler_name: str) -> float:
    """Strike rate for THIS batter against THIS bowler so far in THIS innings —
    matches feature_engineer.py's `current_matchup_sr` (computed from the same
    (striker, bowler) pair_balls/pair_runs the model was trained on)."""
    key = (striker, bowler_name)
    b = state.pair_balls.get(key, 0)
    r = state.pair_runs.get(key, 0)
    return round(r / b * 100, 2) if b > 0 else 0.0


def _sample_outcome(probs: Dict[str, float]) -> str:
    labels  = list(probs.keys())
    weights = [probs[l] for l in labels]
    total   = sum(weights)
    if total == 0:
        return "0"
    norm = [w / total for w in weights]
    return random.choices(labels, weights=norm, k=1)[0]


def _parse_outcome(outcome: str) -> Tuple[int, bool, bool, bool]:
    """Returns (runs, is_wicket, is_wide, is_noball).
    Outcome '5' exists in the dataset (~74 times) — treated as 5 runs.
    Any unrecognised value is treated as 0 runs, not a wicket.
    """
    if outcome == "W":
        return 0, True, False, False
    try:
        runs = int(outcome)
        return runs, False, False, False
    except (ValueError, TypeError):
        return 0, False, False, False