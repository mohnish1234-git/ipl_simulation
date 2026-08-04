"""
src/api/app.py
FastAPI backend — serves simulation and fantasy-XI endpoints.

Start:
    uvicorn src.api.app:app --reload --port 8000
"""

import os
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

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Optional convenience: load PG/DB credentials from a .env file if
# python-dotenv is installed. Checked in order: database/.env (this
# project's actual location), then the default search (cwd and parents).
try:
    from dotenv import load_dotenv

    _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    _DB_ENV_PATH = _PROJECT_ROOT / "database" / ".env"

    if _DB_ENV_PATH.exists():
        load_dotenv(dotenv_path=_DB_ENV_PATH)
    else:
        load_dotenv()

except ImportError:
    pass

from src.model.predictor import load_predictor
from src.simulation.match_simulator import MatchSimulator, StatsStore
from src.simulation.monte_carlo import run_monte_carlo

from src.fantasy_engine.feature_builder import FeatureBuilder
from src.fantasy_engine.fantasy_predictor import FantasyPredictor
from src.fantasy_engine.rule_engine import RuleEngine
from src.fantasy_engine.decision_engine import DecisionEngine
from src.fantasy_engine.vicecaptain_selector import ViceCaptainSelector
from src.fantasy_engine.optimizer import Dream11Optimizer

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

# FantasyFeatureBuilder talks to Postgres directly (see feature_builder.py) —
# it doesn't take stats_store. With no dsn given it reads PG_HOST/PG_PORT/
# PG_DATABASE/PG_USER/PG_PASSWORD from the environment.
feature_builder = FeatureBuilder()

# Points to the trained classifier; classifier_encoders.pkl and
# classifier_feature_columns.pkl are auto-discovered next to it.
# There is no regression model in this pipeline anymore — the engine
# ranks players by high_performer_probability, not expected points.
FANTASY_CLASSIFIER_PATH = os.environ.get("FANTASY_CLASSIFIER_PATH", "models/fantasy_classifier.pkl")
fantasy_predictor = FantasyPredictor(FANTASY_CLASSIFIER_PATH)

rule_engine = RuleEngine()
decision_engine = DecisionEngine(rule_engine)
vicecaptain_selector = ViceCaptainSelector()
optimizer = Dream11Optimizer()

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


class FantasyXIRequest(BaseModel):
    team1: str
    team2: str

    playing_xi_team1: List[str] = Field(..., min_length=11, max_length=11)
    playing_xi_team2: List[str] = Field(..., min_length=11, max_length=11)

    venue: str

    toss_winner: Optional[str] = None
    toss_decision: str = "bat"

    # Monte Carlo adds simulation-based variance on top of the ML
    # prediction but costs real time (n simulations of a full match) —
    # off by default, opt in per request. Fantasy XI ranking works fine
    # without it; simulation is a purely optional add-on.
    use_monte_carlo: bool = False
    monte_carlo_simulations: int = Field(5, ge=1, le=100)

    # Required ONLY when use_monte_carlo=True (validated below) — the
    # bowler for each of the 20 overs, per side, exactly like
    # SimulateRequest.bowling_rotation_1/2. No auto-generated rotation:
    # if you want simulation, you supply who bowls each over yourself.
    bowling_rotation_team1: Optional[List[str]] = Field(
        None, min_length=20, max_length=20,
        description="Required if use_monte_carlo=True: bowler for each of team1's 20 overs"
    )
    bowling_rotation_team2: Optional[List[str]] = Field(
        None, min_length=20, max_length=20,
        description="Required if use_monte_carlo=True: bowler for each of team2's 20 overs"
    )


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


import math


def _json_safe_records(records):

    # json.dumps (what Starlette's default JSONResponse uses) raises
    # ValueError on NaN/Infinity — it isn't valid JSON. Doing this on the
    # DataFrame itself doesn't work: pandas has no way to hold None in a
    # float64 column, so df.where(df.notna(), None) silently reverts
    # right back to NaN. Only once to_dict('records') has turned values
    # into plain Python floats does replacing NaN/inf with None actually
    # stick.
    def _clean(value):

        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):

            return None

        return value

    return [

        {key: _clean(value) for key, value in record.items()}

        for record in records

    ]


def _fantasy_points_from_summary(summary):

    # Heuristic Dream11-style scoring from Monte Carlo mean batting/
    # bowling output — 1 pt/run, 25 pts/wicket. Replace with your exact
    # scoring rules (boundary/strike-rate/economy bonuses, catches,
    # etc.) if you want numeric parity with the real format.
    batting = summary.get("batting") or {}
    bowling = summary.get("bowling") or {}

    runs = batting.get("mean_runs", 0) or 0
    wickets = bowling.get("mean_wickets", 0) or 0

    return runs + (wickets * 25)


def _run_fantasy_monte_carlo(req):

    try:

        mc = run_monte_carlo(
            team1=req.team1,
            team2=req.team2,
            batting_order_1=req.playing_xi_team1,
            batting_order_2=req.playing_xi_team2,
            bowling_rotation_1=req.bowling_rotation_team1,
            bowling_rotation_2=req.bowling_rotation_team2,
            venue=req.venue,
            num_simulations=req.monte_carlo_simulations,
            toss_winner=req.toss_winner,
            toss_choice=req.toss_decision,
            predictor=predictor,
            stats_store=stats_store,
        )

        rows = [

            {"player": summary["player"], "simulation_points": _fantasy_points_from_summary(summary)}

            for summary in mc.player_summaries

        ]

        return pd.DataFrame(rows)

    except Exception as exc:

        # Simulation is meant to add variance on top of the ML
        # prediction, not be a hard dependency — if it fails, fall back
        # to an empty frame so rule_engine's left join just zero-fills
        # simulation_points instead of taking the whole endpoint down.
        print(f"Monte Carlo simulation failed, continuing without it: {exc}")
        return pd.DataFrame(columns=["player", "simulation_points"])


@app.post("/predict-fantasy-xi")
def predict_fantasy_xi(req: FantasyXIRequest):

    if req.use_monte_carlo and (not req.bowling_rotation_team1 or not req.bowling_rotation_team2):

        raise HTTPException(

            status_code=400,

            detail=(

                "use_monte_carlo=True requires bowling_rotation_team1 and "
                "bowling_rotation_team2 — the bowler for each of the 20 "
                "overs per side, supplied by you. Simulation doesn't "
                "guess a rotation on its own."

            ),

        )

    try:

        # FantasyFeatureBuilder.build_dataframe(batting_team, bowling_team, venue)
        # already produces one flat, model-ready row per player across BOTH
        # playing XIs (22 rows) — team1/team2 here just mark which side is
        # "batting" for matchup lookups, not who gets a row.
        feature_df = feature_builder.build_dataframe(

            req.playing_xi_team1,

            req.playing_xi_team2,

            req.venue,

            batting_team_name=req.team1,

            bowling_team_name=req.team2,

        )

        ml_predictions = fantasy_predictor.predict_players(feature_df)

        if req.use_monte_carlo:

            simulation_predictions = _run_fantasy_monte_carlo(req)

        else:

            simulation_predictions = pd.DataFrame(columns=["player", "simulation_points"])

        # No dedicated roster source exists yet (see comment further
        # down) — but team1/team2 and both playing XIs are right here in
        # the request, so at minimum every player's actual team can be
        # attached without waiting on that.
        roster = (

            [{"player": p, "team": req.team1} for p in req.playing_xi_team1]

            + [{"player": p, "team": req.team2} for p in req.playing_xi_team2]

        )

        decided = decision_engine.decide(ml_predictions, simulation_predictions, roster=roster)

        # optimizer.optimize() requires a "role" column with real
        # WK/BAT/AR/BOWL values, which isn't part of the
        # historical-stats pipeline or the current request — it has to
        # come from a proper roster source, which the lightweight
        # roster above deliberately doesn't provide (only team is
        # known). Until a real roster with roles is wired in, always
        # fall back to the top-11 ranked pool rather than risk calling
        # the role-constrained optimizer with role values it doesn't
        # recognize.
        if "role" in decided.columns and decided["role"].notna().any():

            recommendation = optimizer.recommend(decided)

            best_xi_df = recommendation["team"]

        else:

            best_xi_df = decided.head(11)

        best_xi_df = best_xi_df.sort_values("final_points", ascending=False).reset_index(drop=True)

        # player_role (from the feature pipeline, e.g. "Batsman"/
        # "Bowler") is a scoring signal, not a validated WK/BAT/AR/BOWL
        # category — safe to show, not safe to feed the optimizer.
        display_role_column = "role" if "role" in best_xi_df.columns else "player_role"

        captain = best_xi_df.iloc[0]["player"] if not best_xi_df.empty else None

        vice_captain = (

            vicecaptain_selector.choose(best_xi_df)["player"]

            if len(best_xi_df) > 1 else None

        )

        best_xi_records = _json_safe_records([

            {

                "player_name": row["player"],
                "team": row.get("team"),
                "role": row.get(display_role_column, "Unknown"),
                "high_performer_probability": row.get("high_performer_probability"),
                "final_points": row.get("final_points"),

            }

            for row in best_xi_df.to_dict("records")

        ])

        return {

            "best_xi": best_xi_records,
            "captain": captain,
            "vice_captain": vice_captain,

        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-ball")
def predict_ball(ball_context: dict):
    """Raw ball probability prediction — useful for debugging."""
    probs = predictor.predict_proba(ball_context)
    return {"probabilities": probs}


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