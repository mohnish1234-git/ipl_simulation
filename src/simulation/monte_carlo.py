"""
Monte Carlo Simulation Engine
Runs N simulations of a match to derive probabilistic outcomes:
  - Win probability
  - Average score / score distribution
  - Batter & bowler projections
  - Confidence intervals
"""

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from tqdm import tqdm

from src.simulation.match_simulator import MatchSimulator, MatchResult


@dataclass
class MonteCarloResult:
    n_simulations: int
    team1: str
    team2: str

    # Win probabilities
    team1_win_prob: float
    team2_win_prob: float

    # Score distributions (team batting first)
    avg_score_1: float
    std_score_1: float
    score_dist_1: List[int]           # raw scores from all sims
    score_p10_1: float
    score_p50_1: float
    score_p90_1: float

    # Score distributions (team batting second)
    avg_score_2: float
    std_score_2: float
    score_dist_2: List[int]
    score_p10_2: float
    score_p50_2: float
    score_p90_2: float

    # Win margins
    avg_win_margin_runs: float
    avg_win_margin_wickets: float

    # Player projections
    batter_projections: Dict[str, dict]
    bowler_projections: Dict[str, dict]

    # Confidence
    confidence_interval_95: Dict[str, tuple]   # 95% CI for each team's score


def run_monte_carlo(
    simulator: MatchSimulator,
    team1: str,
    team2: str,
    batting_order_1: List[str],
    batting_order_2: List[str],
    bowling_rotation_1: List[str],
    bowling_rotation_2: List[str],
    venue: str = "Unknown",
    n_simulations: int = 1000,
    n_workers: int = 4,
    verbose: bool = True,
) -> MonteCarloResult:
    """
    Run `n_simulations` independent match simulations and aggregate results.

    Parameters
    ----------
    n_simulations : int
        Recommended: 500 for quick preview, 1000 for standard, 5000 for deep analysis.
    n_workers : int
        Parallel threads. Set to 1 to disable parallelism.
    """

    results: List[MatchResult] = []

    sim_kwargs = dict(
        team1=team1,
        team2=team2,
        batting_order_1=batting_order_1,
        batting_order_2=batting_order_2,
        bowling_rotation_1=bowling_rotation_1,
        bowling_rotation_2=bowling_rotation_2,
        venue=venue,
    )

    if n_workers == 1:
        it = range(n_simulations)
        if verbose:
            it = tqdm(it, desc="Simulating")
        for _ in it:
            results.append(simulator.simulate(**sim_kwargs))
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(simulator.simulate, **sim_kwargs) for _ in range(n_simulations)]
            it = as_completed(futures)
            if verbose:
                it = tqdm(it, total=n_simulations, desc="Simulating")
            for f in it:
                results.append(f.result())

    return _aggregate(results, team1, team2)


def _aggregate(results: List[MatchResult], team1: str, team2: str) -> MonteCarloResult:
    n = len(results)

    # ── Win probability ───────────────────────────────────────────────────────
    t1_wins = sum(1 for r in results if r.winner == r.batting_team_1)
    t2_wins = n - t1_wins

    # ── Score distributions ───────────────────────────────────────────────────
    scores_1 = np.array([r.score_1 for r in results])
    scores_2 = np.array([r.score_2 for r in results])

    # ── Win margins ───────────────────────────────────────────────────────────
    run_wins = [r.win_margin for r in results if r.win_type == "runs"]
    wkt_wins = [r.win_margin for r in results if r.win_type == "wickets"]

    # ── Player projections ────────────────────────────────────────────────────
    batter_proj = _aggregate_batters(results)
    bowler_proj = _aggregate_bowlers(results)

    # ── 95% CI using percentile method ───────────────────────────────────────
    ci = {
        team1: (float(np.percentile(scores_1, 2.5)), float(np.percentile(scores_1, 97.5))),
        team2: (float(np.percentile(scores_2, 2.5)), float(np.percentile(scores_2, 97.5))),
    }

    return MonteCarloResult(
        n_simulations=n,
        team1=team1,
        team2=team2,
        team1_win_prob=round(t1_wins / n, 4),
        team2_win_prob=round(t2_wins / n, 4),
        avg_score_1=round(float(scores_1.mean()), 1),
        std_score_1=round(float(scores_1.std()), 1),
        score_dist_1=scores_1.tolist(),
        score_p10_1=float(np.percentile(scores_1, 10)),
        score_p50_1=float(np.percentile(scores_1, 50)),
        score_p90_1=float(np.percentile(scores_1, 90)),
        avg_score_2=round(float(scores_2.mean()), 1),
        std_score_2=round(float(scores_2.std()), 1),
        score_dist_2=scores_2.tolist(),
        score_p10_2=float(np.percentile(scores_2, 10)),
        score_p50_2=float(np.percentile(scores_2, 50)),
        score_p90_2=float(np.percentile(scores_2, 90)),
        avg_win_margin_runs=round(np.mean(run_wins), 1) if run_wins else 0.0,
        avg_win_margin_wickets=round(np.mean(wkt_wins), 1) if wkt_wins else 0.0,
        batter_projections=batter_proj,
        bowler_projections=bowler_proj,
        confidence_interval_95=ci,
    )


def _aggregate_batters(results: List[MatchResult]) -> Dict[str, dict]:
    runs_map: Dict[str, List[int]] = {}
    balls_map: Dict[str, List[int]] = {}
    fours_map: Dict[str, List[int]] = {}
    sixes_map: Dict[str, List[int]] = {}

    for r in results:
        for stats in (r.batter_stats_1, r.batter_stats_2):
            for name, s in stats.items():
                runs_map.setdefault(name, []).append(s["runs"])
                balls_map.setdefault(name, []).append(s["balls"])
                fours_map.setdefault(name, []).append(s["fours"])
                sixes_map.setdefault(name, []).append(s["sixes"])

    out = {}
    for name in runs_map:
        runs = np.array(runs_map[name])
        balls = np.array(balls_map[name])
        out[name] = {
            "avg_runs": round(float(runs.mean()), 1),
            "std_runs": round(float(runs.std()), 1),
            "p10_runs": float(np.percentile(runs, 10)),
            "p90_runs": float(np.percentile(runs, 90)),
            "avg_balls": round(float(balls.mean()), 1),
            "avg_sr": round(float(runs.sum() / balls.sum() * 100), 1) if balls.sum() else 0,
            "avg_fours": round(float(np.mean(fours_map[name])), 1),
            "avg_sixes": round(float(np.mean(sixes_map[name])), 1),
        }
    return out


def _aggregate_bowlers(results: List[MatchResult]) -> Dict[str, dict]:
    runs_map: Dict[str, List[int]] = {}
    balls_map: Dict[str, List[int]] = {}
    wkts_map: Dict[str, List[int]] = {}

    for r in results:
        for stats in (r.bowler_stats_1, r.bowler_stats_2):
            for name, s in stats.items():
                runs_map.setdefault(name, []).append(s["runs"])
                balls_map.setdefault(name, []).append(s["balls"])
                wkts_map.setdefault(name, []).append(s["wickets"])

    out = {}
    for name in runs_map:
        runs = np.array(runs_map[name])
        balls = np.array(balls_map[name])
        wkts = np.array(wkts_map[name])
        overs = balls / 6
        out[name] = {
            "avg_runs": round(float(runs.mean()), 1),
            "avg_wickets": round(float(wkts.mean()), 2),
            "avg_economy": round(float(runs.sum() / overs.sum()), 2) if overs.sum() else 0,
            "p90_wickets": float(np.percentile(wkts, 90)),
        }
    return out
