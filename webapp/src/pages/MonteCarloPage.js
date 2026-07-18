import React, { useState } from "react";
import TeamBuilder from "../components/TeamBuilder";
import WinProbChart from "../components/WinProbChart";
import { runMonteCarlo } from "../utils/api";
import useMeta from "../hooks/useMeta";

const blank11 = () => Array(11).fill("");
const blank20 = () => Array(20).fill("");

/* ── Win probability bar ─────────────────────────────────────────────────── */
function WinBar({ team1, team2, pct1, pct2, tiePct }) {
  return (
    <div style={{ marginTop: "0.75rem" }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "0.4rem", fontWeight: 600 }}>
        <span>{team1} — <span style={{ color: "#38bdf8" }}>{pct1}%</span></span>
        <span>{team2} — <span style={{ color: "#f472b6" }}>{pct2}%</span></span>
      </div>
      <div style={{ display: "flex", height: 14, borderRadius: 7, overflow: "hidden", background: "#1e293b" }}>
        <div style={{ width: `${pct1}%`, background: "linear-gradient(90deg,#38bdf8,#6366f1)", transition: "width 0.8s ease" }} />
        <div style={{ width: `${pct2}%`, background: "linear-gradient(90deg,#ec4899,#f97316)", transition: "width 0.8s ease" }} />
      </div>
      {tiePct > 0 && <p style={{ color: "#94a3b8", fontSize: "0.78rem", marginTop: 4 }}>Tie probability: {tiePct}%</p>}
    </div>
  );
}

/* ── Winner badge ─────────────────────────────────────────────────────────── */
function WinnerBadge({ winner, confidence }) {
  return (
    <div style={{
      display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
      padding: "1.25rem 2rem", borderRadius: 14,
      background: "linear-gradient(135deg,#0f172a 60%,#1e293b)",
      border: "2px solid #38bdf8", textAlign: "center", gap: "0.4rem",
    }}>
      <div style={{ fontSize: "0.75rem", textTransform: "uppercase", letterSpacing: "0.12em", color: "#94a3b8" }}>
        Most Probable Winner
      </div>
      <div style={{ fontSize: "1.9rem", fontWeight: 800, color: "#38bdf8", letterSpacing: "-0.02em" }}>
        🏆 {winner}
      </div>
      <div style={{ fontSize: "0.88rem", color: "#cbd5e1" }}>
        wins in <strong style={{ color: "#e2e8f0" }}>{confidence}%</strong> of simulations
      </div>
    </div>
  );
}

/* ── Score summary card ───────────────────────────────────────────────────── */
function ScoreCard({ label, dist, color }) {
  return (
    <div style={{
      background: "#0f172a", borderRadius: 12, padding: "1rem 1.25rem",
      border: `1.5px solid ${color}33`,
    }}>
      <div style={{ color: "#94a3b8", fontSize: "0.78rem", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: "0.5rem" }}>
        {label} — Avg Score
      </div>
      <div style={{ fontSize: "2.4rem", fontWeight: 800, color, lineHeight: 1 }}>{dist.mean}</div>
      <div style={{ display: "flex", gap: "1rem", marginTop: "0.5rem", fontSize: "0.8rem", color: "#64748b" }}>
        <span>Median: <b style={{ color: "#94a3b8" }}>{dist.median}</b></span>
        <span>P10: <b style={{ color: "#94a3b8" }}>{dist.p10}</b></span>
        <span>P90: <b style={{ color: "#94a3b8" }}>{dist.p90}</b></span>
      </div>
    </div>
  );
}

/* ── Top 3 batters table ──────────────────────────────────────────────────── */
function TopBatters({ batters }) {
  if (!batters?.length) return <p style={{ color: "#64748b" }}>No batter data available.</p>;
  return (
    <table style={{ width: "100%", borderCollapse: "collapse" }}>
      <thead>
        <tr style={{ borderBottom: "1px solid #334155", color: "#94a3b8", fontSize: "0.78rem", textTransform: "uppercase" }}>
          <th style={{ textAlign: "left", padding: "0.5rem 0.75rem" }}>#</th>
          <th style={{ textAlign: "left", padding: "0.5rem 0.75rem" }}>Batter</th>
          <th style={{ textAlign: "right", padding: "0.5rem 0.75rem" }}>Avg Runs</th>
          <th style={{ textAlign: "right", padding: "0.5rem 0.75rem" }}>P90 Runs</th>
          <th style={{ textAlign: "right", padding: "0.5rem 0.75rem" }}>Avg Balls</th>
        </tr>
      </thead>
      <tbody>
        {batters.map((b, i) => (
          <tr key={b.player} style={{ borderBottom: "1px solid #1e293b", transition: "background 0.2s" }}
            onMouseEnter={e => e.currentTarget.style.background = "#1e293b"}
            onMouseLeave={e => e.currentTarget.style.background = "transparent"}>
            <td style={{ padding: "0.6rem 0.75rem", color: i === 0 ? "#fbbf24" : i === 1 ? "#94a3b8" : "#b45309", fontWeight: 700 }}>
              {i === 0 ? "🥇" : i === 1 ? "🥈" : "🥉"}
            </td>
            <td style={{ padding: "0.6rem 0.75rem", fontWeight: 600, color: "#e2e8f0" }}>{b.player}</td>
            <td style={{ padding: "0.6rem 0.75rem", textAlign: "right", color: "#38bdf8", fontWeight: 700, fontSize: "1.05rem" }}>{b.mean_runs}</td>
            <td style={{ padding: "0.6rem 0.75rem", textAlign: "right", color: "#64748b" }}>{b.p90_runs}</td>
            <td style={{ padding: "0.6rem 0.75rem", textAlign: "right", color: "#64748b" }}>{b.mean_balls_faced}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/* ── Top 3 bowlers table ──────────────────────────────────────────────────── */
function TopBowlers({ bowlers }) {
  if (!bowlers?.length) return <p style={{ color: "#64748b" }}>No bowler data available.</p>;
  return (
    <table style={{ width: "100%", borderCollapse: "collapse" }}>
      <thead>
        <tr style={{ borderBottom: "1px solid #334155", color: "#94a3b8", fontSize: "0.78rem", textTransform: "uppercase" }}>
          <th style={{ textAlign: "left", padding: "0.5rem 0.75rem" }}>#</th>
          <th style={{ textAlign: "left", padding: "0.5rem 0.75rem" }}>Bowler</th>
          <th style={{ textAlign: "right", padding: "0.5rem 0.75rem" }}>Avg Wkts</th>
          <th style={{ textAlign: "right", padding: "0.5rem 0.75rem" }}>P90 Wkts</th>
          <th style={{ textAlign: "right", padding: "0.5rem 0.75rem" }}>Economy</th>
        </tr>
      </thead>
      <tbody>
        {bowlers.map((b, i) => (
          <tr key={b.player} style={{ borderBottom: "1px solid #1e293b", transition: "background 0.2s" }}
            onMouseEnter={e => e.currentTarget.style.background = "#1e293b"}
            onMouseLeave={e => e.currentTarget.style.background = "transparent"}>
            <td style={{ padding: "0.6rem 0.75rem", color: i === 0 ? "#fbbf24" : i === 1 ? "#94a3b8" : "#b45309", fontWeight: 700 }}>
              {i === 0 ? "🥇" : i === 1 ? "🥈" : "🥉"}
            </td>
            <td style={{ padding: "0.6rem 0.75rem", fontWeight: 600, color: "#e2e8f0" }}>{b.player}</td>
            <td style={{ padding: "0.6rem 0.75rem", textAlign: "right", color: "#a78bfa", fontWeight: 700, fontSize: "1.05rem" }}>{b.mean_wickets}</td>
            <td style={{ padding: "0.6rem 0.75rem", textAlign: "right", color: "#64748b" }}>{b.p90_wickets}</td>
            <td style={{ padding: "0.6rem 0.75rem", textAlign: "right", color: "#64748b" }}>{b.mean_economy}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/* ── Main page ───────────────────────────────────────────────────────────── */
export default function MonteCarloPage() {
  const { meta, loading: metaLoading, error: metaError } = useMeta();

  const [team1, setTeam1] = useState("Mumbai Indians");
  const [team2, setTeam2] = useState("Chennai Super Kings");
  const [venue, setVenue] = useState("");
  const [nSims, setNSims] = useState(200);
  const [order1, setOrder1] = useState(blank11());
  const [order2, setOrder2] = useState(blank11());
  const [rot1, setRot1] = useState(blank20());
  const [rot2, setRot2] = useState(blank20());

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
        n_simulations: nSims,
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
      <h1 style={{ marginBottom: "0.4rem", fontSize: "1.5rem" }}>🎲 Monte Carlo Simulation</h1>
      <p style={{ color: "#64748b", marginBottom: "1.75rem", fontSize: "0.9rem" }}>
        Runs the same matchup many times to estimate win probability, likely scores, and standout performers.
      </p>

      {/* Setup card */}
      <div className="card">
        <h2>Match Setup</h2>
        {metaLoading && <p className="meta-status">Loading dropdown options...</p>}
        {metaError && <p className="meta-status warning">Using manual inputs — metadata unavailable.</p>}
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
              : <input value={venue} onChange={e => setVenue(e.target.value)} placeholder="Wankhede Stadium" />}
          </div>
        </div>
        <div className="field" style={{ maxWidth: 200, marginTop: "0.75rem" }}>
          <label>Simulations</label>
          <select value={nSims} onChange={e => setNSims(+e.target.value)}>
            <option value={100}>100 — quick</option>
            <option value={200}>200 — standard</option>
            <option value={500}>500 — accurate</option>
            <option value={1000}>1000 — deep</option>
          </select>
        </div>
      </div>

      {/* Team builders */}
      <div className="grid-2">
        <TeamBuilder label={team1} players={order1} setPlayers={setOrder1} rotation={rot1} setRotation={setRot1}
          knownPlayers={meta.batters_by_team?.[team1] ?? meta.batters ?? []}
          knownBowlers={meta.bowlers_by_team?.[team1] ?? meta.bowlers ?? []} />
        <TeamBuilder label={team2} players={order2} setPlayers={setOrder2} rotation={rot2} setRotation={setRot2}
          knownPlayers={meta.batters_by_team?.[team2] ?? meta.batters ?? []}
          knownBowlers={meta.bowlers_by_team?.[team2] ?? meta.bowlers ?? []} />
      </div>

      <button className="btn btn-primary" onClick={handleRun} disabled={loading} style={{ marginBottom: "1.5rem" }}>
        {loading ? `Running ${nSims} simulations…` : `▶ Run ${nSims} Simulations`}
      </button>

      {error && (
        <div className="card" style={{ color: "#f87171", whiteSpace: "pre-wrap", fontFamily: "monospace", fontSize: "0.8rem" }}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {loading && (
        <div className="loading">
          <div className="spinner" />
          <p>Simulating {nSims} matches — this may take up to a minute…</p>
        </div>
      )}

      {result && (
        <>
          {/* ── Winner + Win probability ──────────────────────────────────── */}
          <div className="card">
            <h2>Result — {result.num_simulations} Simulations</h2>
            <div className="grid-2" style={{ alignItems: "stretch", gap: "1rem", marginTop: "0.5rem" }}>
              <WinnerBadge winner={result.most_probable_winner} confidence={result.winner_confidence} />
              <div style={{ display: "flex", flexDirection: "column", justifyContent: "center" }}>
                <p style={{ color: "#94a3b8", fontSize: "0.82rem", marginBottom: "0.25rem" }}>Win probability breakdown</p>
                <WinBar
                  team1={result.team1} team2={result.team2}
                  pct1={result.team1_win_pct} pct2={result.team2_win_pct}
                  tiePct={result.tie_pct}
                />
              </div>
            </div>
          </div>

          {/* ── Average scores ───────────────────────────────────────────── */}
          <div className="card">
            <h2>Average Scores</h2>
            <div className="grid-2" style={{ gap: "1rem", marginTop: "0.5rem" }}>
              <ScoreCard label={result.team1} dist={result.score_1} color="#38bdf8" />
              <ScoreCard label={result.team2} dist={result.score_2} color="#f472b6" />
            </div>
          </div>

          {/* ── Top performers ───────────────────────────────────────────── */}
          <div className="grid-2">
            <div className="card">
              <h2>🏏 Top 3 Batters</h2>
              <p style={{ color: "#64748b", fontSize: "0.8rem", marginBottom: "0.75rem" }}>Ranked by average runs across all simulations</p>
              <TopBatters batters={result.top3_batters} />
            </div>
            <div className="card">
              <h2>🎳 Top 3 Bowlers</h2>
              <p style={{ color: "#64748b", fontSize: "0.8rem", marginBottom: "0.75rem" }}>Ranked by average wickets across all simulations</p>
              <TopBowlers bowlers={result.top3_bowlers} />
            </div>
          </div>

          {/* ── Full player fantasy table (collapsed by default) ─────────── */}
          {result.player_summaries?.length > 0 && (
            <details className="card" style={{ cursor: "pointer" }}>
              <summary style={{ fontWeight: 700, fontSize: "1rem", color: "#e2e8f0", userSelect: "none" }}>
                📊 All Players — Fantasy Value Ranking
              </summary>
              <div style={{ overflowX: "auto", marginTop: "0.75rem" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.85rem" }}>
                  <thead>
                    <tr style={{ borderBottom: "1px solid #334155", color: "#94a3b8", fontSize: "0.75rem", textTransform: "uppercase" }}>
                      <th style={{ textAlign: "left", padding: "0.4rem 0.6rem" }}>Player</th>
                      <th style={{ textAlign: "right", padding: "0.4rem 0.6rem" }}>Sims</th>
                      <th style={{ textAlign: "right", padding: "0.4rem 0.6rem" }}>Avg R</th>
                      <th style={{ textAlign: "right", padding: "0.4rem 0.6rem" }}>Avg W</th>
                      <th style={{ textAlign: "right", padding: "0.4rem 0.6rem" }}>Avg Pts</th>
                      <th style={{ textAlign: "right", padding: "0.4rem 0.6rem" }}>P10 / P90</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.player_summaries.map((p) => (
                      <tr key={p.player} style={{ borderBottom: "1px solid #1e293b" }}>
                        <td style={{ padding: "0.45rem 0.6rem", color: "#e2e8f0", fontWeight: 500 }}>{p.player}</td>
                        <td style={{ padding: "0.45rem 0.6rem", textAlign: "right", color: "#64748b" }}>{p.simulations_appeared}</td>
                        <td style={{ padding: "0.45rem 0.6rem", textAlign: "right", color: "#38bdf8" }}>{p.batting ? p.batting.mean_runs : "—"}</td>
                        <td style={{ padding: "0.45rem 0.6rem", textAlign: "right", color: "#a78bfa" }}>{p.bowling ? p.bowling.mean_wickets : "—"}</td>
                        <td style={{ padding: "0.45rem 0.6rem", textAlign: "right", color: "#fbbf24", fontWeight: 700 }}>{p.fantasy_points.mean}</td>
                        <td style={{ padding: "0.45rem 0.6rem", textAlign: "right", color: "#64748b" }}>{p.fantasy_points.p10} / {p.fantasy_points.p90}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </details>
          )}
        </>
      )}
    </div>
  );
}
