"""
src/api/app.py
FastAPI backend — serves simulation, Monte Carlo, and optimization endpoints.

Start:
    uvicorn src.api.app:app --reload --port 8000
"""

import json
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.model.predictor import load_predictor
from src.simulation.match_simulator import MatchSimulator
from src.simulation.monte_carlo import run_monte_carlo
from src.optimization.team_optimizer import (
    Player, optimize_batting_order, optimize_bowling_rotation, dream11_optimize
)

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="IPL Simulation API",
    description="AI-powered IPL match simulator and team optimizer",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load once at startup
predictor   = load_predictor()          # falls back to MockPredictor if no model file
simulator   = MatchSimulator(predictor)

META_PATH = Path("data/processed/meta.json")
_meta: dict = {}
if META_PATH.exists():
    with open(META_PATH) as f:
        _meta = json.load(f)


# ── Request / Response schemas ────────────────────────────────────────────────

class SimulateRequest(BaseModel):
    team1: str
    team2: str
    batting_order_1: List[str] = Field(..., min_items=11, max_items=11)
    batting_order_2: List[str] = Field(..., min_items=11, max_items=11)
    bowling_rotation_1: List[str] = Field(..., min_items=20, max_items=20,
                                           description="Bowler name for each of the 20 overs")
    bowling_rotation_2: List[str] = Field(..., min_items=20, max_items=20)
    venue: str = "Unknown"
    toss_winner: Optional[str] = None
    toss_choice: str = "bat"


class MonteCarloRequest(SimulateRequest):
    n_simulations: int = Field(500, ge=100, le=5000)
    n_workers: int = Field(4, ge=1, le=8)


class PlayerSchema(BaseModel):
    name: str
    team: str
    role: str                   # BAT / BOWL / AR / WK
    credits: float = 9.0
    is_overseas: bool = False
    bowling_style: str = "medium"


class Dream11Request(BaseModel):
    players: List[PlayerSchema]
    mc_result_team1: str
    mc_result_team2: str
    # Minimal MC result passed from frontend after a MC run
    batter_projections: dict
    bowler_projections: dict
    budget: float = 100.0


class OptimizeOrderRequest(BaseModel):
    players: List[PlayerSchema]
    batter_projections: dict
    bowler_projections: dict
    team1: str
    team2: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "message": "IPL Simulation API is running"}


@app.get("/meta")
def get_meta():
    """Return known teams, venues, batters, bowlers from the dataset."""
    return _meta


@app.post("/simulate")
def simulate_match(req: SimulateRequest):
    """Run a single match simulation. Fast (~50ms)."""
    try:
        result = simulator.simulate(
            team1=req.team1,
            team2=req.team2,
            batting_order_1=req.batting_order_1,
            batting_order_2=req.batting_order_2,
            bowling_rotation_1=req.bowling_rotation_1,
            bowling_rotation_2=req.bowling_rotation_2,
            venue=req.venue,
            toss_winner=req.toss_winner,
            toss_choice=req.toss_choice,
        )
        return {
            "batting_team_1": result.batting_team_1,
            "batting_team_2": result.batting_team_2,
            "score_1": result.score_1,
            "wickets_1": result.wickets_1,
            "score_2": result.score_2,
            "wickets_2": result.wickets_2,
            "winner": result.winner,
            "win_margin": result.win_margin,
            "win_type": result.win_type,
            "batter_stats_1": result.batter_stats_1,
            "batter_stats_2": result.batter_stats_2,
            "bowler_stats_1": result.bowler_stats_1,
            "bowler_stats_2": result.bowler_stats_2,
            "innings_1_log": result.innings_1_log,
            "innings_2_log": result.innings_2_log,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/monte-carlo")
def monte_carlo(req: MonteCarloRequest):
    """Run N simulations and return aggregated win probabilities and projections."""
    try:
        mc = run_monte_carlo(
            simulator=simulator,
            team1=req.team1,
            team2=req.team2,
            batting_order_1=req.batting_order_1,
            batting_order_2=req.batting_order_2,
            bowling_rotation_1=req.bowling_rotation_1,
            bowling_rotation_2=req.bowling_rotation_2,
            venue=req.venue,
            n_simulations=req.n_simulations,
            n_workers=req.n_workers,
            verbose=False,
        )
        return {
            "n_simulations": mc.n_simulations,
            "team1": mc.team1,
            "team2": mc.team2,
            "team1_win_prob": mc.team1_win_prob,
            "team2_win_prob": mc.team2_win_prob,
            "score_summary": {
                mc.team1: {
                    "avg": mc.avg_score_1, "std": mc.std_score_1,
                    "p10": mc.score_p10_1, "p50": mc.score_p50_1, "p90": mc.score_p90_1,
                    "distribution": mc.score_dist_1,
                },
                mc.team2: {
                    "avg": mc.avg_score_2, "std": mc.std_score_2,
                    "p10": mc.score_p10_2, "p50": mc.score_p50_2, "p90": mc.score_p90_2,
                    "distribution": mc.score_dist_2,
                },
            },
            "win_margins": {
                "avg_runs": mc.avg_win_margin_runs,
                "avg_wickets": mc.avg_win_margin_wickets,
            },
            "batter_projections": mc.batter_projections,
            "bowler_projections": mc.bowler_projections,
            "confidence_interval_95": mc.confidence_interval_95,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/optimize/batting-order")
def optimize_batting(req: OptimizeOrderRequest):
    """Return an optimized batting order given MC projections."""
    from src.simulation.monte_carlo import MonteCarloResult
    import numpy as np
    mc = _build_mock_mc(req.team1, req.team2, req.batter_projections, req.bowler_projections)
    players = [Player(**p.dict()) for p in req.players]
    order = optimize_batting_order(players, mc)
    return {"batting_order": order}


@app.post("/optimize/bowling-rotation")
def optimize_bowling(req: OptimizeOrderRequest):
    """Return an optimized 20-over bowling rotation."""
    mc = _build_mock_mc(req.team1, req.team2, req.batter_projections, req.bowler_projections)
    players = [Player(**p.dict()) for p in req.players]
    bowlers = [p for p in players if p.role in ("BOWL", "AR")]
    rotation = optimize_bowling_rotation(bowlers, mc)
    return {"bowling_rotation": rotation}


@app.post("/optimize/dream11")
def optimize_dream11(req: Dream11Request):
    """Return optimal Dream11 XI within budget and role constraints."""
    from src.simulation.monte_carlo import MonteCarloResult
    mc = _build_mock_mc(
        req.mc_result_team1, req.mc_result_team2,
        req.batter_projections, req.bowler_projections
    )
    players = [Player(**p.dict()) for p in req.players]
    team, score = dream11_optimize(players, mc, budget=req.budget)
    return {
        "selected_players": [p.name for p in team],
        "total_credits": sum(p.credits for p in team),
        "expected_fantasy_points": score,
        "player_details": [p.__dict__ for p in team],
    }


@app.post("/predict-ball")
def predict_ball(ball_context: dict):
    """Raw ball probability prediction endpoint — useful for debugging."""
    probs = predictor.predict_proba(ball_context)
    return {"probabilities": probs}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_mock_mc(team1, team2, batter_proj, bowler_proj):
    """Construct a minimal MonteCarloResult from raw projection dicts."""
    from src.simulation.monte_carlo import MonteCarloResult
    import numpy as np
    return MonteCarloResult(
        n_simulations=0, team1=team1, team2=team2,
        team1_win_prob=0.5, team2_win_prob=0.5,
        avg_score_1=160, std_score_1=15, score_dist_1=[], score_p10_1=140, score_p50_1=160, score_p90_1=180,
        avg_score_2=155, std_score_2=15, score_dist_2=[], score_p10_2=135, score_p50_2=155, score_p90_2=175,
        avg_win_margin_runs=10, avg_win_margin_wickets=3,
        batter_projections=batter_proj,
        bowler_projections=bowler_proj,
        confidence_interval_95={team1: (140, 180), team2: (135, 175)},
    )
