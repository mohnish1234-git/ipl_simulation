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

import random
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.model.predictor import load_predictor

# ── Outcome encoding (for prev_ball_outcome features) ─────────────────────────
_OUTCOME_ENCODE = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "6": 6, "W": 7}


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

    # per-bowler in-match stats (this innings)
    bowler_innings_balls: Dict[str, int] = field(default_factory=dict)
    bowler_innings_runs:  Dict[str, int] = field(default_factory=dict)

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
        return round(self.rrr - self.crr, 2) if self.innings_num == 2 else 0.0

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

    def runs_last_over(self, over: int) -> int:
        return int(self.over_runs.get(over - 1, 0))

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
        bat_rw_avg=6.5, bat_rw_sr=128.0,
        bat_rw_boundary_pct=0.15, bat_rw_six_pct=0.06, bat_rw_dot_pct=0.33,
        bat_pp_rw_sr=125.0, bat_mid_rw_sr=130.0, bat_death_rw_sr=145.0,
        bat_pp_rw_boundary_pct=0.14, bat_death_rw_boundary_pct=0.22,
    )
    DEFAULT_BOWLER = dict(
        bowl_rw_economy=8.2, bowl_rw_wicket_pct=0.055, bowl_rw_dot_pct=0.33,
        bowl_rw_boundary_pct=0.15,
        bowl_pp_rw_economy=7.5, bowl_mid_rw_economy=8.0, bowl_death_rw_economy=9.2,
        bowl_pp_rw_wicket_pct=0.07, bowl_death_rw_wicket_pct=0.055,
    )
    DEFAULT_BVB = dict(
        bvb_balls=0, bvb_rw_sr=128.0, bvb_rw_dismissal_pct=0.055,
        bvb_rw_dot_pct=0.33, bvb_rw_boundary_pct=0.15, bvb_rw_six_pct=0.06,
    )
    DEFAULT_VENUE = dict(
        venue_rw_avg_1st_innings=165.0, venue_rw_avg_2nd_innings=155.0,
        venue_rw_boundary_pct=0.17, venue_rw_six_pct=0.08,
        venue_rw_dot_pct=0.31, venue_rw_wicket_pct=0.054,
        venue_rw_pp_sr=140.0, venue_rw_death_sr=165.0,
    )

    def __init__(self):
        self._batter: Dict[str, dict] = {}
        self._bowler: Dict[str, dict] = {}
        self._bvb:    Dict[Tuple, dict] = {}
        self._venue:  Dict[str, dict] = {}

    def load_from_csv(self, processed_dir: str = "data/processed"):
        """
        Attempt to load pre-computed stats from CSVs saved by prepare_data.py.
        Silently falls back to defaults if files are not present.
        """
        import json
        from pathlib import Path
        p = Path(processed_dir)

        for fname, attr in [
            ("player_batter_stats.json", "_batter"),
            ("player_bowler_stats.json", "_bowler"),
            ("bvb_stats.json",           "_bvb"),
            ("venue_stats.json",         "_venue"),
        ]:
            fpath = p / fname
            if fpath.exists():
                with open(fpath) as f:
                    data = json.load(f)
                if attr == "_bvb":
                    setattr(self, attr, {tuple(k.split("|||")): v for k, v in data.items()})
                else:
                    setattr(self, attr, data)
                print(f"  Loaded {fname}")

    def batter(self, name: str) -> dict:
        return self._batter.get(name, self.DEFAULT_BATTER)

    def bowler(self, name: str) -> dict:
        return self._bowler.get(name, self.DEFAULT_BOWLER)

    def bvb(self, batter: str, bowler: str) -> dict:
        return self._bvb.get((batter, bowler), self.DEFAULT_BVB)

    def venue(self, v: str) -> dict:
        return self._venue.get(v, self.DEFAULT_VENUE)


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

    def __init__(self, predictor=None, stats_store: Optional[StatsStore] = None):
        self.predictor   = predictor or load_predictor()
        self.stats_store = stats_store or StatsStore()

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
                # matchup
                **bvbs,
                # momentum
                "batter_balls_faced":   state.batter_innings_balls.get(striker, 0),
                "batter_runs_scored":   state.batter_innings_runs.get(striker, 0),
                "batter_innings_sr":    _innings_sr(state, striker),
                "balls_vs_bowler":      state.bowler_innings_balls.get(bowler_name, 0),
                "runs_vs_bowler":       state.bowler_innings_runs.get(bowler_name, 0),
                "runs_last6":           state.runs_last6(),
                "runs_last_over":       state.runs_last_over(over),
                "consec_dots":          state.consec_dots,
                "consec_boundaries":    state.consec_boundaries,
                "partnership_runs":     state.partnership_runs,
                "partnership_balls":    state.partnership_balls,
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
            }

            probs   = self.predictor.predict_proba(ctx)
            outcome = _sample_outcome(probs)
            runs, is_wicket, is_wide, is_noball = _parse_outcome(outcome)

            # ── Update batter state ───────────────────────────────────────────
            striker_state = state.batter_states[striker]
            if not is_wide:
                striker_state.balls += 1
                state.batter_innings_balls[striker] = state.batter_innings_balls.get(striker, 0) + 1
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

            extras = 1 if (is_wide or is_noball) else 0
            state.score  += runs + extras
            state.extras += extras

            bowl_runs_this = runs + extras
            state.bowler_innings_balls[bowler_name] = state.bowler_innings_balls.get(bowler_name, 0) + (0 if is_wide else 1)
            state.bowler_innings_runs[bowler_name]  = state.bowler_innings_runs.get(bowler_name, 0) + bowl_runs_this

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
            })

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


def _sample_outcome(probs: Dict[str, float]) -> str:
    labels  = list(probs.keys())
    weights = [probs[l] for l in labels]
    total   = sum(weights)
    if total == 0:
        return "0"
    norm = [w / total for w in weights]
    return random.choices(labels, weights=norm, k=1)[0]


def _parse_outcome(outcome: str) -> Tuple[int, bool, bool, bool]:
    """Returns (runs, is_wicket, is_wide, is_noball)."""
    if outcome == "W":
        return 0, True, False, False
    try:
        return int(outcome), False, False, False
    except ValueError:
        return 0, False, False, False