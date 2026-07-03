import React, { useState } from "react";
import TeamBuilder from "../components/TeamBuilder";
import WinProbChart, { ScoreDistChart } from "../components/WinProbChart";
import { runMonteCarlo } from "../utils/api";
import useMeta from "../hooks/useMeta";

const blank11 = () => Array(11).fill("");
const blank20 = () => Array(20).fill("");

export default function MonteCarloPage() {
  const { meta } = useMeta();

  const [team1, setTeam1]   = useState("Mumbai Indians");
  const [team2, setTeam2]   = useState("Chennai Super Kings");
  const [venue, setVenue]   = useState("");
  const [nSims, setNSims]   = useState(500);
  const [order1, setOrder1] = useState(blank11());
  const [order2, setOrder2] = useState(blank11());
  const [rot1, setRot1]     = useState(blank20());
  const [rot2, setRot2]     = useState(blank20());

  const [result, setResult]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState("");

  const handleRun = async () => {
    setLoading(true); setError(""); setResult(null);
    try {
      const res = await runMonteCarlo({
        team1, team2, venue: venue || "Unknown",
        batting_order_1: order1.map(p => p || "Unknown"),
        batting_order_2: order2.map(p => p || "Unknown"),
        bowling_rotation_1: rot1.map(b => b || "Unknown"),
        bowling_rotation_2: rot2.map(b => b || "Unknown"),
        n_simulations: nSims,
      });
      setResult(res);
    } catch (e) {
      setError(e.response?.data?.detail || e.message);
    } finally {
      setLoading(false);
    }
  };

  const ss = result?.score_summary;

  return (
    <div>
      <h1 style={{ marginBottom: "1.5rem", fontSize: "1.4rem" }}>Monte Carlo Simulation</h1>

      <div className="card">
        <h2>Setup</h2>
        <div className="grid-3">
          <div className="field">
            <label>Team 1</label>
            {meta.teams?.length
              ? <select value={team1} onChange={e => setTeam1(e.target.value)}>{meta.teams.map(t => <option key={t}>{t}</option>)}</select>
              : <input value={team1} onChange={e => setTeam1(e.target.value)} />}
          </div>
          <div className="field">
            <label>Team 2</label>
            {meta.teams?.length
              ? <select value={team2} onChange={e => setTeam2(e.target.value)}>{meta.teams.map(t => <option key={t}>{t}</option>)}</select>
              : <input value={team2} onChange={e => setTeam2(e.target.value)} />}
          </div>
          <div className="field">
            <label>Venue</label>
            {meta.venues?.length
              ? <select value={venue} onChange={e => setVenue(e.target.value)}><option value="">Unknown</option>{meta.venues.map(v => <option key={v}>{v}</option>)}</select>
              : <input value={venue} onChange={e => setVenue(e.target.value)} />}
          </div>
        </div>
        <div className="field" style={{ maxWidth: 200 }}>
          <label>Number of Simulations</label>
          <select value={nSims} onChange={e => setNSims(+e.target.value)}>
            <option value={100}>100 — quick</option>
            <option value={500}>500 — standard</option>
            <option value={1000}>1000 — accurate</option>
            <option value={5000}>5000 — deep</option>
          </select>
        </div>
      </div>

      <div className="grid-2">
        <TeamBuilder label={`${team1}`} players={order1} setPlayers={setOrder1} rotation={rot1} setRotation={setRot1} knownPlayers={meta.batters || []} />
        <TeamBuilder label={`${team2}`} players={order2} setPlayers={setOrder2} rotation={rot2} setRotation={setRot2} knownPlayers={meta.batters || []} />
      </div>

      <button className="btn btn-primary" onClick={handleRun} disabled={loading} style={{ marginBottom: "1.5rem" }}>
        {loading ? `Running ${nSims} simulations…` : `▶ Run ${nSims} Simulations`}
      </button>

      {error  && <div className="card" style={{ color: "#f87171" }}>Error: {error}</div>}
      {loading && <div className="loading"><div className="spinner" /><p>Running Monte Carlo simulations…<br /><small>This may take 10–60 seconds</small></p></div>}

      {result && (
        <>
          <WinProbChart team1={result.team1} team2={result.team2} prob1={result.team1_win_prob} prob2={result.team2_win_prob} />
          {ss && (
            <div className="grid-2">
              <div className="card">
                <h2>{result.team1} Score Summary</h2>
                <div className="grid-3">
                  <div className="stat-box"><div className="val">{ss[result.team1]?.avg}</div><div className="lbl">Avg Score</div></div>
                  <div className="stat-box"><div className="val">{ss[result.team1]?.p50}</div><div className="lbl">Median</div></div>
                  <div className="stat-box"><div className="val">±{ss[result.team1]?.std}</div><div className="lbl">Std Dev</div></div>
                </div>
                <p style={{ color: "#64748b", fontSize: "0.8rem", marginTop: 8 }}>
                  P10–P90 range: {ss[result.team1]?.p10} – {ss[result.team1]?.p90}
                </p>
              </div>
              <div className="card">
                <h2>{result.team2} Score Summary</h2>
                <div className="grid-3">
                  <div className="stat-box"><div className="val">{ss[result.team2]?.avg}</div><div className="lbl">Avg Score</div></div>
                  <div className="stat-box"><div className="val">{ss[result.team2]?.p50}</div><div className="lbl">Median</div></div>
                  <div className="stat-box"><div className="val">±{ss[result.team2]?.std}</div><div className="lbl">Std Dev</div></div>
                </div>
                <p style={{ color: "#64748b", fontSize: "0.8rem", marginTop: 8 }}>
                  P10–P90 range: {ss[result.team2]?.p10} – {ss[result.team2]?.p90}
                </p>
              </div>
            </div>
          )}
          <ScoreDistChart
            team1={result.team1} team2={result.team2}
            dist1={ss?.[result.team1]?.distribution}
            dist2={ss?.[result.team2]?.distribution}
          />
          <PlayerProjections projections={result.batter_projections} bowlerProj={result.bowler_projections} />
        </>
      )}
    </div>
  );
}

function PlayerProjections({ projections, bowlerProj }) {
  const batters = Object.entries(projections || {}).sort((a, b) => b[1].avg_runs - a[1].avg_runs).slice(0, 10);
  const bowlers = Object.entries(bowlerProj || {}).sort((a, b) => b[1].avg_wickets - a[1].avg_wickets).slice(0, 10);
  return (
    <div className="grid-2">
      <div className="card">
        <h2>Batter Projections (avg over {"{N}"} sims)</h2>
        <table>
          <thead><tr><th>Batter</th><th>Avg R</th><th>P10</th><th>P90</th><th>Avg SR</th></tr></thead>
          <tbody>
            {batters.map(([name, s]) => (
              <tr key={name}><td>{name}</td><td><b>{s.avg_runs}</b></td><td>{s.p10_runs}</td><td>{s.p90_runs}</td><td>{s.avg_sr}</td></tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="card">
        <h2>Bowler Projections</h2>
        <table>
          <thead><tr><th>Bowler</th><th>Avg W</th><th>Avg Eco</th><th>P90 W</th></tr></thead>
          <tbody>
            {bowlers.map(([name, s]) => (
              <tr key={name}><td>{name}</td><td><b>{s.avg_wickets}</b></td><td>{s.avg_economy}</td><td>{s.p90_wickets}</td></tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
