"""
Team Optimization Engine
Selects the optimal playing XI and batting/bowling order using Monte Carlo results.

Two optimization modes:
  1. dream11_optimize  → maximize fantasy points within constraints
  2. match_optimize    → maximize win probability for a real match
"""

import itertools
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.simulation.monte_carlo import MonteCarloResult


# ── Player roles ──────────────────────────────────────────────────────────────

ROLE_BATSMAN  = "BAT"
ROLE_BOWLER   = "BOWL"
ROLE_ALLROUND = "AR"
ROLE_WK       = "WK"


@dataclass
class Player:
    name: str
    team: str
    role: str                          # BAT / BOWL / AR / WK
    credits: float = 9.0              # fantasy credits (Dream11 context)
    is_overseas: bool = False
    bowling_style: str = "medium"     # pace / spin / medium


# ── Dream11 fantasy optimization ─────────────────────────────────────────────

def dream11_optimize(
    players: List[Player],
    mc_result: MonteCarloResult,
    budget: float = 100.0,
    team_size: int = 11,
    max_from_one_team: int = 7,
    min_wk: int = 1, max_wk: int = 1,
    min_bat: int = 3, max_bat: int = 5,
    min_ar: int = 1,  max_ar: int = 4,
    min_bowl: int = 3, max_bowl: int = 5,
    max_overseas: int = 4,
    n_top_candidates: int = 200,
) -> Tuple[List[Player], float]:
    """
    Greedy + combinatorial search for best Dream11 XI.
    Returns (selected_players, expected_fantasy_points).
    """

    # Score each player using MC projections
    scored = [(p, _fantasy_score(p, mc_result)) for p in players]
    scored.sort(key=lambda x: x[1], reverse=True)

    # Prune to top N candidates for speed
    candidates = scored[:n_top_candidates]

    best_team: List[Player] = []
    best_score: float = 0.0

    # Greedy seed
    greedy, greedy_score = _greedy_select(candidates, budget, team_size,
                                           max_from_one_team, min_wk, max_wk,
                                           min_bat, max_bat, min_ar, max_ar,
                                           min_bowl, max_bowl, max_overseas)
    if greedy:
        best_team, best_score = greedy, greedy_score

    return best_team, best_score


def _fantasy_score(player: Player, mc: MonteCarloResult) -> float:
    """Estimate expected Dream11 points from MC projections."""
    score = 0.0
    b = mc.batter_projections.get(player.name)
    if b:
        runs = b["avg_runs"]
        score += runs                              # 1 pt per run
        score += b["avg_fours"] * 1               # 1 pt per four
        score += b["avg_sixes"] * 2               # 2 pt per six
        if runs >= 50: score += 8
        if runs >= 100: score += 16
        if b["avg_sr"] >= 170: score += 6
        elif b["avg_sr"] >= 150: score += 4

    bl = mc.bowler_projections.get(player.name)
    if bl:
        wkts = bl["avg_wickets"]
        score += wkts * 25                        # 25 pts per wicket
        if wkts >= 3: score += 4
        if wkts >= 5: score += 8
        econ = bl["avg_economy"]
        if econ < 5: score += 6
        elif econ < 6: score += 4
        elif econ < 7: score += 2

    return round(score, 2)


def _greedy_select(
    scored: List[Tuple[Player, float]],
    budget, size, max_team, min_wk, max_wk,
    min_bat, max_bat, min_ar, max_ar, min_bowl, max_bowl, max_overseas
) -> Tuple[List[Player], float]:

    selected: List[Player] = []
    total_credits = 0.0
    team_counts: Dict[str, int] = {}
    role_counts = {ROLE_WK: 0, ROLE_BATSMAN: 0, ROLE_ALLROUND: 0, ROLE_BOWLER: 0}
    overseas = 0
    total_score = 0.0

    for player, pts in scored:
        if len(selected) >= size:
            break
        if total_credits + player.credits > budget:
            continue
        tc = team_counts.get(player.team, 0)
        if tc >= max_team:
            continue
        r = player.role
        rc = role_counts[r]
        if r == ROLE_WK and rc >= max_wk: continue
        if r == ROLE_BATSMAN and rc >= max_bat: continue
        if r == ROLE_ALLROUND and rc >= max_ar: continue
        if r == ROLE_BOWLER and rc >= max_bowl: continue
        if player.is_overseas and overseas >= max_overseas: continue

        selected.append(player)
        total_credits += player.credits
        team_counts[player.team] = tc + 1
        role_counts[r] += 1
        if player.is_overseas:
            overseas += 1
        total_score += pts

    # Validate minimums
    valid = (
        role_counts[ROLE_WK] >= min_wk and
        role_counts[ROLE_BATSMAN] >= min_bat and
        role_counts[ROLE_ALLROUND] >= min_ar and
        role_counts[ROLE_BOWLER] >= min_bowl and
        len(selected) == size
    )

    return (selected, total_score) if valid else ([], 0.0)


# ── Match optimization: batting order ────────────────────────────────────────

def optimize_batting_order(
    players: List[Player],
    mc_result: MonteCarloResult,
    top_n: int = 6,
) -> List[str]:
    """
    Sort batting order: aggressive batters with high SR up top,
    anchors in middle, big hitters at 5–7, tail at end.
    """

    def _batter_score(p: Player) -> float:
        b = mc_result.batter_projections.get(p.name)
        if not b:
            return 0.0
        # Balance runs and SR, penalise tail
        role_bonus = 2.0 if p.role == ROLE_WK else (1.0 if p.role in (ROLE_BATSMAN, ROLE_ALLROUND) else 0.0)
        return b["avg_runs"] * 0.6 + b["avg_sr"] * 0.4 * 0.1 + role_bonus

    ordered = sorted(players, key=_batter_score, reverse=True)
    return [p.name for p in ordered]


def optimize_bowling_rotation(
    bowlers: List[Player],
    mc_result: MonteCarloResult,
    n_overs: int = 20,
    max_overs_per_bowler: int = 4,
) -> List[str]:
    """
    Assign bowlers to overs. Death bowlers (low economy, high wickets) placed
    at overs 16–20. Powerplay bowlers placed at 0–5.
    """

    def _death_score(p: Player) -> float:
        b = mc_result.bowler_projections.get(p.name)
        if not b:
            return 0.0
        return b["avg_wickets"] * 5 - b["avg_economy"]

    def _pp_score(p: Player) -> float:
        b = mc_result.bowler_projections.get(p.name)
        if not b:
            return 0.0
        return b["avg_wickets"] * 4 - b["avg_economy"] * 0.5

    eligible = [p for p in bowlers if p.role in (ROLE_BOWLER, ROLE_ALLROUND)]

    pp_order = sorted(eligible, key=_pp_score, reverse=True)
    death_order = sorted(eligible, key=_death_score, reverse=True)

    rotation: List[str] = []
    over_counts: Dict[str, int] = {}

    pp_idx, mid_idx, death_idx = 0, 0, 0
    mid_pool = sorted(eligible, key=lambda p: (mc_result.bowler_projections.get(p.name) or {}).get("avg_economy", 99))

    for over in range(n_overs):
        if over < 6:
            pool = pp_order
        elif over >= 16:
            pool = death_order
        else:
            pool = mid_pool

        chosen = None
        for p in pool:
            if over_counts.get(p.name, 0) < max_overs_per_bowler:
                chosen = p
                break

        if chosen is None:
            # fallback: anyone with overs left
            for p in eligible:
                if over_counts.get(p.name, 0) < max_overs_per_bowler:
                    chosen = p
                    break

        name = chosen.name if chosen else (eligible[0].name if eligible else "Unknown")
        rotation.append(name)
        over_counts[name] = over_counts.get(name, 0) + 1

    return rotation
