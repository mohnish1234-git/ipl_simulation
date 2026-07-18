"""
Monte Carlo match simulation + per-player fantasy-value aggregation.

Runs the existing MatchSimulator N times for a FIXED match configuration
(same two XIs, same bowling rotations, same venue, same toss) and
aggregates:
  - win probability for each team (+ tie rate)
  - score distribution (mean/median/p10/p90) for each innings
  - per-player aggregated batting/bowling stats across all N simulated
    matches, plus a fantasy-points estimate per player — the actual output
    the "most suitable player to pick for fantasy" use case needs, not
    just "who won this one simulated match."

This does NOT reach into match_simulator.py's internals — it only calls
MatchSimulator.simulate() N times and aggregates the MatchResult objects it
already returns (batter_stats_1/2, bowler_stats_1/2, score_1/2, winner),
so it stays correct automatically if the underlying simulation logic
changes, and never duplicates ball-by-ball logic that already lives there.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.simulation.match_simulator import MatchSimulator, MatchResult
from src.model.predictor import load_predictor


# ═══════════════════════════════════════════════════════════════════════════
# Fantasy scoring
# ═══════════════════════════════════════════════════════════════════════════
# Standard Dream11-style point values. Every real platform's exact numbers
# differ slightly (and some are configurable/seasonal) — treat this as a
# reasonable, clearly-isolated default rather than an exact reproduction of
# any one platform's rulebook. Swap these two functions out if you need to
# match a specific platform exactly; nothing else in this file depends on
# the particular point values chosen here.

def _batting_fantasy_points(runs: int, balls: int, fours: int, sixes: int,
                             dismissed: bool) -> float:
    pts = float(runs) + fours * 1 + sixes * 2
    if runs >= 100:
        pts += 8
    elif runs >= 50:
        pts += 4
    if dismissed and runs == 0 and balls > 0:
        pts -= 2  # duck
    # Strike-rate bonus/penalty only applies with enough balls faced to be
    # meaningful — a single boundary off 2 balls isn't a "170+ SR" read.
    if balls >= 10:
        sr = runs / balls * 100
        if sr > 170:
            pts += 6
        elif sr >= 150:
            pts += 4
        elif sr >= 130:
            pts += 2
        elif sr < 50:
            pts -= 6
        elif sr < 60:
            pts -= 4
        elif sr < 70:
            pts -= 2
    return pts


def _bowling_fantasy_points(wickets: int, runs: int, balls: int) -> float:
    pts = wickets * 25.0
    if wickets >= 5:
        pts += 12
    elif wickets >= 4:
        pts += 8
    overs = balls / 6
    if overs >= 2:  # economy bonus/penalty needs a meaningful sample
        econ = runs / overs
        if econ < 5:
            pts += 6
        elif econ < 6:
            pts += 4
        elif econ < 7:
            pts += 2
        elif econ >= 12:
            pts -= 6
        elif econ >= 11:
            pts -= 4
        elif econ >= 10:
            pts -= 2
    return pts


# ═══════════════════════════════════════════════════════════════════════════
# Aggregation
# ═══════════════════════════════════════════════════════════════════════════

def _pctile(xs: List[float], p: float):
    if not xs:
        return 0
    xs_sorted = sorted(xs)
    k = min(len(xs_sorted) - 1, max(0, round(p * (len(xs_sorted) - 1))))
    return xs_sorted[k]


def _mean(xs: List[float]) -> float:
    return round(sum(xs) / len(xs), 2) if xs else 0.0


@dataclass
class PlayerAggregate:
    """Accumulates one player's stats across every simulation they appeared
    in (as batter and/or bowler — an all-rounder's fantasy_points entry for
    a given simulation already includes BOTH sides, see run_monte_carlo)."""
    name: str
    runs: List[int] = field(default_factory=list)
    balls_faced: List[int] = field(default_factory=list)
    fours: List[int] = field(default_factory=list)
    sixes: List[int] = field(default_factory=list)
    dismissals: int = 0
    wickets: List[int] = field(default_factory=list)
    runs_conceded: List[int] = field(default_factory=list)
    balls_bowled: List[int] = field(default_factory=list)
    fantasy_points: List[float] = field(default_factory=list)

    def summary(self) -> dict:
        n = len(self.fantasy_points)
        total_bowl_balls = sum(self.balls_bowled)
        return {
            "player": self.name,
            "simulations_appeared": n,
            "batting": {
                "mean_runs": _mean(self.runs),
                "p10_runs": _pctile(self.runs, 0.10),
                "p90_runs": _pctile(self.runs, 0.90),
                "mean_balls_faced": _mean(self.balls_faced),
                "mean_fours": _mean(self.fours),
                "mean_sixes": _mean(self.sixes),
                "dismissal_rate": round(self.dismissals / len(self.runs), 3) if self.runs else 0.0,
            } if self.runs else None,
            "bowling": {
                "mean_wickets": _mean(self.wickets),
                "p10_wickets": _pctile(self.wickets, 0.10),
                "p90_wickets": _pctile(self.wickets, 0.90),
                "mean_economy": round(sum(self.runs_conceded) / (total_bowl_balls / 6), 2)
                                if total_bowl_balls > 0 else 0.0,
            } if self.wickets else None,
            "fantasy_points": {
                "mean": _mean(self.fantasy_points),
                "p10": _pctile(self.fantasy_points, 0.10),
                "p90": _pctile(self.fantasy_points, 0.90),
                "std_dev": round(statistics.pstdev(self.fantasy_points), 2) if n > 1 else 0.0,
            },
        }


@dataclass
class MonteCarloResult:
    num_simulations: int
    team1: str
    team2: str
    team1_win_pct: float
    team2_win_pct: float
    tie_pct: float
    score_1: dict
    score_2: dict
    player_summaries: List[dict]              # sorted by mean fantasy points, desc
    raw_results: Optional[List[MatchResult]] = None   # only populated if keep_raw=True


def run_monte_carlo(
    team1: str,
    team2: str,
    batting_order_1: List[str],
    batting_order_2: List[str],
    bowling_rotation_1: List[str],
    bowling_rotation_2: List[str],
    venue: str = "Unknown",
    num_simulations: int = 200,
    toss_winner: Optional[str] = None,
    toss_choice: str = "bat",
    predictor=None,
    stats_store=None,
    keep_raw: bool = False,
) -> MonteCarloResult:
    """
    Runs MatchSimulator.simulate() `num_simulations` times with the SAME
    lineups/venue/toss every time — only the ball-by-ball sampling varies
    (that's the genuinely stochastic part: predictor.predict_proba() feeding
    _sample_outcome()). Aggregates into win probabilities, score
    distributions, and per-player fantasy value.

    predictor/stats_store are built ONCE (or passed in already-built) and
    reused across all N runs — reloading the model N times would dominate
    the runtime otherwise. Pass an existing predictor/stats_store in if the
    caller already has one warmed up (e.g. a long-lived API process).

    keep_raw=True keeps every individual MatchResult (for e.g. a "sample
    innings log" drill-down in the UI) — off by default since it's the
    single biggest memory cost of a large N.
    """
    sim = MatchSimulator(predictor=predictor or load_predictor(), stats_store=stats_store)

    team1_wins = team2_wins = ties = 0
    scores_1: List[int] = []
    scores_2: List[int] = []
    players: Dict[str, PlayerAggregate] = {}
    raw: List[MatchResult] = []

    def _agg(name: str) -> PlayerAggregate:
        return players.setdefault(name, PlayerAggregate(name=name))

    for _ in range(num_simulations):
        result = sim.simulate(
            team1=team1, team2=team2,
            batting_order_1=batting_order_1, batting_order_2=batting_order_2,
            bowling_rotation_1=bowling_rotation_1, bowling_rotation_2=bowling_rotation_2,
            venue=venue, toss_winner=toss_winner, toss_choice=toss_choice,
        )
        if keep_raw:
            raw.append(result)

        scores_1.append(result.score_1)
        scores_2.append(result.score_2)
        if result.winner == team1:
            team1_wins += 1
        elif result.winner == team2:
            team2_wins += 1
        else:
            ties += 1

        # Accumulate this simulation's fantasy points per player in a
        # per-round dict first, THEN append once per player at the end of
        # the round — an all-rounder who both bats and bowls in the same
        # simulated match needs their batting + bowling points combined
        # into ONE entry for this round, not two separate entries that
        # would silently misalign every list's length against the others.
        round_points: Dict[str, float] = {}

        for stats_dict in (result.batter_stats_1, result.batter_stats_2):
            for name, s in stats_dict.items():
                p = _agg(name)
                p.runs.append(s["runs"])
                p.balls_faced.append(s["balls"])
                p.fours.append(s["fours"])
                p.sixes.append(s["sixes"])
                if s["dismissed"]:
                    p.dismissals += 1
                round_points[name] = round_points.get(name, 0.0) + _batting_fantasy_points(
                    s["runs"], s["balls"], s["fours"], s["sixes"], s["dismissed"]
                )

        for stats_dict in (result.bowler_stats_1, result.bowler_stats_2):
            for name, s in stats_dict.items():
                p = _agg(name)
                p.wickets.append(s["wickets"])
                p.runs_conceded.append(s["runs"])
                p.balls_bowled.append(s["balls"])
                round_points[name] = round_points.get(name, 0.0) + _bowling_fantasy_points(
                    s["wickets"], s["runs"], s["balls"]
                )

        for name, pts in round_points.items():
            _agg(name).fantasy_points.append(pts)

    def _score_summary(xs: List[int]) -> dict:
        return {
            "mean": round(sum(xs) / len(xs), 1) if xs else 0,
            "median": _pctile(xs, 0.5),
            "p10": _pctile(xs, 0.10),
            "p90": _pctile(xs, 0.90),
        }

    player_summaries = sorted(
        (p.summary() for p in players.values()),
        key=lambda s: s["fantasy_points"]["mean"],
        reverse=True,
    )

    return MonteCarloResult(
        num_simulations=num_simulations,
        team1=team1, team2=team2,
        team1_win_pct=round(team1_wins / num_simulations * 100, 1),
        team2_win_pct=round(team2_wins / num_simulations * 100, 1),
        tie_pct=round(ties / num_simulations * 100, 1),
        score_1=_score_summary(scores_1),
        score_2=_score_summary(scores_2),
        player_summaries=player_summaries,
        raw_results=raw if keep_raw else None,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Fantasy squad optimizer
# ═══════════════════════════════════════════════════════════════════════════
# Deliberately bounded in scope: real fantasy platforms (Dream11 etc.) add
# role quotas (min/max WK/BAT/AR/BOWL), a credit budget per player, and a
# captain(2x)/vice-captain(1.5x) pick — all of which need data this project
# doesn't have on file (per-player credit costs, role tags). What CAN be
# done purely from Monte Carlo output: rank all 22 probable players by
# simulated mean fantasy points + consistency, and suggest a captain/vice
# captain from the top of that ranking. Treat this as a "who's actually
# worth picking for THIS matchup/venue" signal to feed into whatever squad
# constraints the real platform enforces, not a full auto-drafted squad.

def suggest_fantasy_picks(mc_result: MonteCarloResult, top_n: int = 11) -> dict:
    ranked = mc_result.player_summaries[:top_n]
    captain = ranked[0]["player"] if ranked else None
    vice_captain = ranked[1]["player"] if len(ranked) > 1 else None
    return {
        "recommended_xi": ranked,
        "captain": captain,
        "vice_captain": vice_captain,
        "note": (
            "Ranked by simulated mean fantasy points for this specific "
            "opponent/venue matchup. Does not enforce role quotas or a "
            "credit budget — cross-check against your platform's actual "
            "squad rules before finalizing."
        ),
    }
