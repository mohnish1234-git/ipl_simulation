import React, { useState } from "react";
import TeamBuilder from "../components/TeamBuilder";
import { runMonteCarlo } from "../utils/api";
import useMeta from "../hooks/useMeta";

// NOTE: runMonteCarlo needs to exist in utils/api.js and MUST hit the same
// base URL / axios instance that simulateMatch() already uses successfully
// — see the snippet at the bottom of this file. A relative "/monte-carlo"
// path resolves against whatever origin the page is served from (the React
// dev server, typically :3000), NOT the FastAPI backend (typically :8000),
// which is the most likely cause of a clean 404 "Not Found" here.

const blank11 = () => Array(11).fill("");
const blank20 = () => Array(20).fill("");
const SIM_COUNT_OPTIONS = [100, 200, 500, 1000];

function WinnerBanner({ winner, confidence, team1, team2, team1Pct, team2Pct, tiePct }) {
  return (
    <div>
      <div className="result-banner">
        <span className="badge badge-win">{winner}</span>
        <span>most likely winner — {confidence}% of simulations</span>
      </div>
      <div className="prob-bar-wrap" style={{ marginTop: "1rem" }}>
        <div className="prob-bar-label">
          <span>{team1} — {team1Pct}%</span>
          <span>{team2} — {team2Pct}%</span>
        </div>
        <div className="prob-bar-track" style={{ display: "flex" }}>
          <div className="prob-bar-fill" style={{ width: `${team1Pct}%` }} />
          <div className="prob-bar-fill team2" style={{ width: `${team2Pct}%` }} />
        </div>
        {tiePct > 0 && <p className="meta-status" style={{ marginTop: 4 }}>Tie: {tiePct}%</p>}
      </div>
    </div>
  );
}

function ScoreDistribution({ label, dist }) {
  return (
    <div className="stat-box">
      <div className="val">{dist.mean}</div>
      <div className="lbl">{label} — mean score</div>
      <p className="meta-status" style={{ marginTop: 8, marginBottom: 0 }}>
        median {dist.median} · p10 {dist.p10} · p90 {dist.p90}
      </p>
    </div>
  );
}

function TopBatters({ players }) {
  return (
    <table>
      <thead>
        <tr><th>#</th><th>Batter</th><th>Mean Runs</th><th>Ceiling (p90)</th><th>Avg Balls Faced</th></tr>
      </thead>
      <tbody>
        {players.map((p, i) => (
          <tr key={p.player}>
            <td>{i + 1}</td>
            <td>{p.player}</td>
            <td><strong>{p.mean_runs}</strong></td>
            <td>{p.p90_runs}</td>
            <td>{p.mean_balls_faced}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function TopBowlers({ players }) {
  return (
    <table>
      <thead>
        <tr><th>#</th><th>Bowler</th><th>Mean Wickets</th><th>Ceiling (p90)</th><th>Avg Economy</th></tr>
      </thead>
      <tbody>
        {players.map((p, i) => (
          <tr key={p.player}>
            <td>{i + 1}</td>
            <td>{p.player}</td>
            <td><strong>{p.mean_wickets}</strong></td>
            <td>{p.p90_wickets}</td>
            <td>{p.mean_economy}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default function MonteCarloPage() {
  const { meta, loading: metaLoading, error: metaError } = useMeta();

  const [team1, setTeam1] = useState("Mumbai Indians");
  const [team2, setTeam2] = useState("Chennai Super Kings");
  const [venue, setVenue] = useState("");
  const [order1, setOrder1] = useState(blank11());
  const [order2, setOrder2] = useState(blank11());
  const [rot1, setRot1] = useState(blank20());
  const [rot2, setRot2] = useState(blank20());
  const [numSimulations, setNumSimulations] = useState(500);

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleRun = async () => {
    setLoading(true); setError(""); setResult(null);
    try {
      const res = await runMonteCarlo({
        team1, team2, venue: venue || "Unknown",
        batting_order_1: order1.map(p => p || "Unknown"),
        batting_order_2: order2.map(p => p || "Unknown"),
        bowling_rotation_1: rot1.map(b => b || "Unknown"),
        bowling_rotation_2: rot2.map(b => b || "Unknown"),
        // Field name MUST match app.py's MonteCarloRequest exactly — it's
        // n_simulations, not num_simulations (that mismatch was a bug in
        // the previous version of this page).
        n_simulations: numSimulations,
      });
      setResult(res);
    } catch (e) {
      setError(e.response?.data?.detail || e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h1 style={{ marginBottom: "0.5rem", fontSize: "1.4rem" }}>Monte Carlo Simulation</h1>
      <p className="meta-status" style={{ marginBottom: "1.5rem" }}>
        Runs the same matchup many times over to see the SPREAD of outcomes — most
        likely winner, likely score range, and which players are the top fantasy
        performers for this specific opponent and venue.
      </p>

      <div className="card">
        <h2>Match Setup</h2>
        {metaLoading && <p className="meta-status">Loading dropdown options...</p>}
        {metaError && <p className="meta-status warning">Using manual inputs because metadata could not be loaded.</p>}
        <div className="grid-3">
          <div className="field">
            <label>Team 1</label>
            {meta.teams?.length ? (
              <select value={team1} onChange={e => setTeam1(e.target.value)}>
                {meta.teams.map(t => <option key={t}>{t}</option>)}
              </select>
            ) : <input value={team1} onChange={e => setTeam1(e.target.value)} />}
          </div>
          <div className="field">
            <label>Team 2</label>
            {meta.teams?.length ? (
              <select value={team2} onChange={e => setTeam2(e.target.value)}>
                {meta.teams.map(t => <option key={t}>{t}</option>)}
              </select>
            ) : <input value={team2} onChange={e => setTeam2(e.target.value)} />}
          </div>
          <div className="field">
            <label>Venue</label>
            {meta.venues?.length ? (
              <select value={venue} onChange={e => setVenue(e.target.value)}>
                <option value="">Unknown</option>
                {meta.venues.map(v => <option key={v}>{v}</option>)}
              </select>
            ) : <input value={venue} onChange={e => setVenue(e.target.value)} placeholder="Wankhede Stadium" />}
          </div>
        </div>
        <div className="field" style={{ maxWidth: 220 }}>
          <label>Simulations to run</label>
          <select value={numSimulations} onChange={e => setNumSimulations(Number(e.target.value))}>
            {SIM_COUNT_OPTIONS.map(n => <option key={n} value={n}>{n}</option>)}
          </select>
        </div>
      </div>

      <div className="grid-2">
        <TeamBuilder
          label={`${team1} — Squad`}
          players={order1} setPlayers={setOrder1}
          rotation={rot1} setRotation={setRot1}
          knownPlayers={meta.batters_by_team?.[team1] ?? meta.active_batters ?? meta.batters ?? []}
          knownBowlers={meta.bowlers_by_team?.[team1] ?? meta.active_bowlers ?? meta.bowlers ?? []}
        />
        <TeamBuilder
          label={`${team2} — Squad`}
          players={order2} setPlayers={setOrder2}
          rotation={rot2} setRotation={setRot2}
          knownPlayers={meta.batters_by_team?.[team2] ?? meta.active_batters ?? meta.batters ?? []}
          knownBowlers={meta.bowlers_by_team?.[team2] ?? meta.active_bowlers ?? meta.bowlers ?? []}
        />
      </div>

      <button className="btn btn-primary" onClick={handleRun} disabled={loading} style={{ marginBottom: "1.5rem" }}>
        {loading ? `Running ${numSimulations} simulations…` : `▶ Run ${numSimulations} Simulations`}
      </button>

      {error && <div className="card" style={{ color: "#f87171" }}>Error: {error}</div>}
      {loading && (
        <div className="loading">
          <div className="spinner" />
          <p>Simulating {numSimulations} matches — this takes longer than a single match…</p>
        </div>
      )}

      {result && (
        <>
          <div className="card">
            <h2>Result ({result.num_simulations} simulations)</h2>
            <WinnerBanner
              winner={result.most_probable_winner}
              confidence={result.winner_confidence}
              team1={result.team1} team2={result.team2}
              team1Pct={result.team1_win_pct} team2Pct={result.team2_win_pct}
              tiePct={result.tie_pct}
            />
          </div>

          <div className="card">
            <h2>Score Distribution</h2>
            <div className="grid-2">
              <ScoreDistribution label={result.team1} dist={result.score_1} />
              <ScoreDistribution label={result.team2} dist={result.score_2} />
            </div>
          </div>

          <div className="card">
            <h2>Top 3 Batters</h2>
            <p className="section-title">Ranked by mean simulated runs</p>
            <TopBatters players={result.top3_batters} />
          </div>

          <div className="card">
            <h2>Top 3 Bowlers</h2>
            <p className="section-title">Ranked by mean simulated wickets</p>
            <TopBowlers players={result.top3_bowlers} />
          </div>
        </>
      )}
    </div>
  );
}

/* ── Add to utils/api.js (not included in this upload) ─────────────────────

export async function runMonteCarlo(config) {
  // Use the EXACT SAME client/base URL as simulateMatch() — if
  // simulateMatch does e.g. `apiClient.post("/simulate", config)` with
  // apiClient already pointed at http://localhost:8000, this should be
  // apiClient.post("/monte-carlo", config), NOT axios.post("/monte-carlo",
  // config) with no base URL — that's what sends the request to the React
  // dev server instead of FastAPI and produces the clean 404 you're seeing.
  const res = await apiClient.post("/monte-carlo", config);
  return res.data;
}
────────────────────────────────────────────────────────────────────────── */