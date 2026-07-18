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

# Force UTF-8 output on Windows — prevents UnicodeEncodeError from ⚠ / ✓ chars
# printed during model/stats loading on cp1252 consoles.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.model.predictor import load_predictor
from src.simulation.match_simulator import MatchSimulator, StatsStore
from src.simulation.monte_carlo import run_monte_carlo

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

# ── Load everything once at startup ───────────────────────────────────────────

predictor = load_predictor()          # strict by default: raises loudly if the real trained model can't load, no silent mock fallback

# StatsStore: loads recency-weighted player/venue stats from processed JSONs
stats_store = StatsStore()
stats_store.load_from_csv("data/processed")   # silent no-op if files absent

simulator = MatchSimulator(predictor, stats_store)

META_PATH = Path("data/processed/meta.json")
_meta: dict = {}
if META_PATH.exists():
    with open(META_PATH) as f:
        _meta = json.load(f)
    print(f"Meta loaded: {len(_meta.get('batters',[]))} batters, "
          f"{len(_meta.get('venues',[]))} venues")


# ── Request / Response schemas ────────────────────────────────────────────────

class SimulateRequest(BaseModel):
    team1: str
    team2: str
    batting_order_1: List[str] = Field(..., min_length=11, max_length=11)
    batting_order_2: List[str] = Field(..., min_length=11, max_length=11)
    bowling_rotation_1: List[str] = Field(..., min_length=20, max_length=20,
                                          description="Bowler name for each of the 20 overs")
    bowling_rotation_2: List[str] = Field(..., min_length=20, max_length=20)
    venue: str = "Unknown"
    toss_winner: Optional[str] = None
    toss_choice: str = "bat"


class MonteCarloRequest(SimulateRequest):
    n_simulations: int = Field(500, ge=100, le=5000)
    n_workers: int = Field(1, ge=1, le=8)   # default 1 — avoids threading issues on Windows


class PlayerSchema(BaseModel):
    name: str
    team: str
    role: str
    credits: float = 9.0
    is_overseas: bool = False
    bowling_style: str = "medium"


class Dream11Request(BaseModel):
    players: List[PlayerSchema]
    mc_result_team1: str
    mc_result_team2: str
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


@app.get("/stats/batter/{name}")
def get_batter_stats(name: str):
    """Return recency-weighted career stats for a batter."""
    return stats_store.batter(name)


@app.get("/stats/bowler/{name}")
def get_bowler_stats(name: str):
    """Return recency-weighted career stats for a bowler."""
    return stats_store.bowler(name)


@app.get("/stats/venue/{name}")
def get_venue_stats(name: str):
    """Return recency-weighted stats for a venue."""
    return stats_store.venue(name)

@app.post("/simulate")
def simulate_match(req: SimulateRequest):
    """Run a single match simulation. Fast (~100ms)."""
    print("=" * 60)
    print("REQUEST RECEIVED")

    try:
        # Pydantic v2
        print(req.model_dump())
    except AttributeError:
        # Pydantic v1
        print(req.dict())

    print("=" * 60)
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
            "batting_team_1":  result.batting_team_1,
            "batting_team_2":  result.batting_team_2,
            "score_1":         result.score_1,
            "wickets_1":       result.wickets_1,
            "score_2":         result.score_2,
            "wickets_2":       result.wickets_2,
            "winner":          result.winner,
            "win_margin":      result.win_margin,
            "win_type":        result.win_type,
            "batter_stats_1":  result.batter_stats_1,
            "batter_stats_2":  result.batter_stats_2,
            "bowler_stats_1":  result.bowler_stats_1,
            "bowler_stats_2":  result.bowler_stats_2,
            "innings_1_log":   result.innings_1_log,
            "innings_2_log":   result.innings_2_log,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    for ball in result.innings_1_log:
        print("=" * 80)
        print(f"{ball['over']}.{ball['ball']}")
        print(f"{ball['striker']} vs {ball['bowler']}")
        print(f"Outcome : {ball['outcome']}")
        print(f"Score   : {ball['score']}/{ball['wickets']}")

        print("\nMatch Situation")
        for k, v in ball["context"].items():
            print(f"{k:35}: {v}")

        print("\nPredicted Probabilities")
        if ball["probs"]:
            for outcome, prob in sorted(ball["probs"].items()):
                print(f"{outcome:>3} : {prob:.4f}")

@app.post("/monte-carlo")
def monte_carlo(req: MonteCarloRequest):
    """Run N simulations and return aggregated win probabilities, top players, and score stats."""
    try:
        mc = run_monte_carlo(
            team1=req.team1,
            team2=req.team2,
            batting_order_1=req.batting_order_1,
            batting_order_2=req.batting_order_2,
            bowling_rotation_1=req.bowling_rotation_1,
            bowling_rotation_2=req.bowling_rotation_2,
            venue=req.venue,
            num_simulations=req.n_simulations,
            toss_winner=req.toss_winner,
            toss_choice=req.toss_choice,
            predictor=predictor,
            stats_store=stats_store,
        )

        # ── Derive most probable winner ───────────────────────────────────────
        if mc.team1_win_pct >= mc.team2_win_pct:
            most_probable_winner = mc.team1
            winner_pct = mc.team1_win_pct
        else:
            most_probable_winner = mc.team2
            winner_pct = mc.team2_win_pct

        # ── Top 3 batters (highest mean runs across all simulations) ──────────
        batters_ranked = sorted(
            [s for s in mc.player_summaries if s.get("batting") and s["batting"].get("mean_runs", 0) > 0],
            key=lambda s: s["batting"]["mean_runs"],
            reverse=True,
        )
        top3_batters = [
            {
                "player": s["player"],
                "mean_runs": s["batting"]["mean_runs"],
                "p90_runs": s["batting"]["p90_runs"],
                "mean_balls_faced": s["batting"]["mean_balls_faced"],
            }
            for s in batters_ranked[:3]
        ]

        # ── Top 3 bowlers (highest mean wickets across all simulations) ───────
        bowlers_ranked = sorted(
            [s for s in mc.player_summaries if s.get("bowling") and s["bowling"].get("mean_wickets", 0) > 0],
            key=lambda s: s["bowling"]["mean_wickets"],
            reverse=True,
        )
        top3_bowlers = [
            {
                "player": s["player"],
                "mean_wickets": s["bowling"]["mean_wickets"],
                "p90_wickets": s["bowling"]["p90_wickets"],
                "mean_economy": s["bowling"]["mean_economy"],
            }
            for s in bowlers_ranked[:3]
        ]

        return {
            "num_simulations":      mc.num_simulations,
            "team1":                mc.team1,
            "team2":                mc.team2,
            "team1_win_pct":        mc.team1_win_pct,
            "team2_win_pct":        mc.team2_win_pct,
            "tie_pct":              mc.tie_pct,
            "most_probable_winner": most_probable_winner,
            "winner_confidence":    winner_pct,
            "avg_score_team1":      mc.score_1["mean"],
            "avg_score_team2":      mc.score_2["mean"],
            "score_1":              mc.score_1,
            "score_2":              mc.score_2,
            "top3_batters":         top3_batters,
            "top3_bowlers":         top3_bowlers,
            "player_summaries":     mc.player_summaries,
        }
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"{e}\n{traceback.format_exc()}")

@app.post("/predict-ball")
def predict_ball(ball_context: dict):
    """Raw ball probability prediction — useful for debugging."""
    probs = predictor.predict_proba(ball_context)
    return {"probabilities": probs}


# ── Helper ────────────────────────────────────────────────────────────────────
"""
def _build_mock_mc(team1, team2, batter_proj, bowler_proj):
    from src.simulation.monte_carlo import MonteCarloResult
    return MonteCarloResult(
        n_simulations=0, team1=team1, team2=team2,
        team1_win_prob=0.5, team2_win_prob=0.5,
        avg_score_1=162, std_score_1=18, score_dist_1=[],
        score_p10_1=140, score_p50_1=162, score_p90_1=185,
        avg_score_2=158, std_score_2=18, score_dist_2=[],
        score_p10_2=136, score_p50_2=158, score_p90_2=181,
        avg_win_margin_runs=12, avg_win_margin_wickets=3,
        batter_projections=batter_proj,
        bowler_projections=bowler_proj,
        confidence_interval_95={team1: (140, 185), team2: (136, 181)},
    )
"""

# ── Debug endpoint — remove after confirming fix ──────────────────────────────

@app.get("/debug")
def debug_info():
    """Hit this in browser at http://localhost:8000/debug to verify runtime state."""
    import os
    
    # Check MockPredictor weights
    pred_type = type(predictor).__name__
    mid_w = None
    if hasattr(predictor, 'PHASE_WEIGHTS'):
        mid_w = predictor.PHASE_WEIGHTS.get('middle', {}).get('W')
    
    # Run a quick 5-ball test
    from src.simulation.match_simulator import _phase
    ctx = {
        'striker': 'RG Sharma', 'bowler': 'JJ Bumrah',
        'batting_team': 'MI', 'bowling_team': 'CSK',
        'venue': 'Wankhede Stadium, Mumbai', 'phase': 'powerplay',
        'over_num': 0, 'ball_num': 1,
        'cumulative_runs': 0, 'cumulative_wickets': 0,
        'balls_remaining': 120, 'wickets_remaining': 10, 'crr': 0.0,
        **stats_store.batter('RG Sharma'),
        **stats_store.bowler('JJ Bumrah'),
        **stats_store.bvb('RG Sharma', 'JJ Bumrah'),
        **stats_store.venue('Wankhede Stadium, Mumbai'),
        'batter_balls_faced': 0, 'batter_runs_scored': 0, 'batter_innings_sr': 0,
        'balls_vs_bowler': 0, 'runs_vs_bowler': 0,
        'runs_last6': 0, 'runs_last_over': 0,
        'consec_dots': 0, 'consec_boundaries': 0,
        'partnership_runs': 0, 'partnership_balls': 0,
        'prev_ball_outcome': -1, 'prev2_ball_outcome': -1, 'prev3_ball_outcome': -1,
        'is_batting_first': 1, 'is_chasing': 0,
        'target': 0, 'runs_needed': 0, 'rrr': 0.0, 'pressure_index': 0.0,
    }
    probs = predictor.predict_proba(ctx)
    exp_runs = sum(int(k) * v for k, v in probs.items() if k != 'W')
    
    # Quick 5-sim test
    bat = ["RG Sharma","RP Rickelton","TV Samson","DB Brevis","HH Pandya",
           "TH David","KH Pandya","JJ Bumrah","J Yadav","MA Starc","Akash Madhwal"]
    bowl = ["JJ Bumrah","MA Starc","JJ Bumrah","MA Starc","J Yadav",
            "J Yadav","HH Pandya","JJ Bumrah","MA Starc","J Yadav",
            "J Yadav","HH Pandya","JJ Bumrah","MA Starc","J Yadav",
            "JJ Bumrah","MA Starc","JJ Bumrah","MA Starc","JJ Bumrah"]
    sim_scores = []
    for _ in range(5):
        r = simulator.simulate("MI", "CSK", bat, bat, bowl, bowl, venue="Wankhede Stadium, Mumbai")
        sim_scores.append(r.score_1)

    return {
        "working_directory": os.getcwd(),
        "predictor_type": pred_type,
        "mock_middle_W_weight": mid_w,
        "weights_are_calibrated": mid_w == 0.041 if mid_w else False,
        "stats_store_batters": len(stats_store._batter),
        "stats_store_bowlers": len(stats_store._bowler),
        "stats_store_venues": len(stats_store._venue),
        "rohit_sharma_stats": stats_store.batter('RG Sharma'),
        "first_ball_probs_Rohit_vs_Bumrah": probs,
        "expected_runs_per_ball": round(exp_runs, 4),
        "five_sim_scores": sim_scores,
        "five_sim_avg": round(sum(sim_scores) / 5, 1),
    }