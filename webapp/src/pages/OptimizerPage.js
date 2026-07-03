import React, { useState } from "react";
import { optimizeBat, optimizeBowl } from "../utils/api";
import useMeta from "../hooks/useMeta";

const ROLES = ["BAT", "BOWL", "AR", "WK"];

const blankPlayer = (i) => ({
  name: "", team: "Team 1", role: "BAT", credits: 9.0, is_overseas: false, bowling_style: "medium"
});

export default function OptimizerPage() {
  const { meta } = useMeta();
  const [team1, setTeam1] = useState("Mumbai Indians");
  const [team2, setTeam2] = useState("Chennai Super Kings");
  const [players, setPlayers] = useState(Array(11).fill(null).map(blankPlayer));
  const [batProj, setBatProj] = useState("{}");
  const [bowlProj, setBowlProj] = useState("{}");
  const [batOrder, setBatOrder] = useState(null);
  const [bowlRot, setBowlRot]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState("");

  const updatePlayer = (i, field, val) => {
    const next = [...players];
    next[i] = { ...next[i], [field]: val };
    setPlayers(next);
  };

  const handleOptimize = async () => {
    setLoading(true); setError(""); setBatOrder(null); setBowlRot(null);
    try {
      const body = {
        players: players.filter(p => p.name),
        team1, team2,
        batter_projections: JSON.parse(batProj || "{}"),
        bowler_projections: JSON.parse(bowlProj || "{}"),
      };
      const [bo, br] = await Promise.all([optimizeBat(body), optimizeBowl(body)]);
      setBatOrder(bo.batting_order);
      setBowlRot(br.bowling_rotation);
    } catch (e) {
      setError(e.response?.data?.detail || e.message || "Parse error — check JSON fields");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h1 style={{ marginBottom: "1.5rem", fontSize: "1.4rem" }}>Team Optimizer</h1>

      <div className="card">
        <h2>Teams</h2>
        <div className="grid-2">
          <div className="field"><label>Team 1</label>
            {meta.teams?.length ? <select value={team1} onChange={e => setTeam1(e.target.value)}>{meta.teams.map(t => <option key={t}>{t}</option>)}</select>
              : <input value={team1} onChange={e => setTeam1(e.target.value)} />}
          </div>
          <div className="field"><label>Team 2</label>
            {meta.teams?.length ? <select value={team2} onChange={e => setTeam2(e.target.value)}>{meta.teams.map(t => <option key={t}>{t}</option>)}</select>
              : <input value={team2} onChange={e => setTeam2(e.target.value)} />}
          </div>
        </div>
      </div>

      <div className="card">
        <h2>Squad (add players to optimize)</h2>
        <p style={{ color: "#64748b", fontSize: "0.8rem", marginBottom: "1rem" }}>
          Run a Monte Carlo simulation first, then paste the batter/bowler projections JSON below for better optimization.
        </p>
        <table>
          <thead>
            <tr><th>#</th><th>Name</th><th>Team</th><th>Role</th><th>Credits</th><th>Overseas</th></tr>
          </thead>
          <tbody>
            {players.map((p, i) => (
              <tr key={i}>
                <td style={{ color: "#64748b" }}>{i + 1}</td>
                <td>
                  {meta.batters?.length
                    ? <select value={p.name} onChange={e => updatePlayer(i, "name", e.target.value)} style={{ fontSize: "0.8rem" }}>
                        <option value="">— pick —</option>
                        {meta.batters.map(n => <option key={n}>{n}</option>)}
                      </select>
                    : <input value={p.name} onChange={e => updatePlayer(i, "name", e.target.value)} placeholder="Player name" style={{ fontSize: "0.8rem" }} />}
                </td>
                <td>
                  <select value={p.team} onChange={e => updatePlayer(i, "team", e.target.value)} style={{ fontSize: "0.8rem" }}>
                    <option value="Team 1">{team1}</option>
                    <option value="Team 2">{team2}</option>
                  </select>
                </td>
                <td>
                  <select value={p.role} onChange={e => updatePlayer(i, "role", e.target.value)} style={{ fontSize: "0.8rem" }}>
                    {ROLES.map(r => <option key={r}>{r}</option>)}
                  </select>
                </td>
                <td>
                  <input type="number" value={p.credits} step="0.5" min="7" max="12"
                    onChange={e => updatePlayer(i, "credits", +e.target.value)}
                    style={{ width: 60, fontSize: "0.8rem" }} />
                </td>
                <td>
                  <input type="checkbox" checked={p.is_overseas}
                    onChange={e => updatePlayer(i, "is_overseas", e.target.checked)} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="grid-2">
        <div className="card">
          <h2>Batter Projections JSON <span style={{ color: "#475569", fontWeight: 400 }}>(optional)</span></h2>
          <textarea rows={5} value={batProj} onChange={e => setBatProj(e.target.value)} placeholder='{"V Kohli": {"avg_runs": 38, "avg_sr": 140, ...}}' />
        </div>
        <div className="card">
          <h2>Bowler Projections JSON <span style={{ color: "#475569", fontWeight: 400 }}>(optional)</span></h2>
          <textarea rows={5} value={bowlProj} onChange={e => setBowlProj(e.target.value)} placeholder='{"JJ Bumrah": {"avg_wickets": 1.8, "avg_economy": 7.2, ...}}' />
        </div>
      </div>

      <button className="btn btn-primary" onClick={handleOptimize} disabled={loading} style={{ marginBottom: "1.5rem" }}>
        {loading ? "Optimizing…" : "⚡ Optimize Order & Rotation"}
      </button>

      {error && <div className="card" style={{ color: "#f87171" }}>Error: {error}</div>}

      {(batOrder || bowlRot) && (
        <div className="grid-2">
          {batOrder && (
            <div className="card">
              <h2>Optimized Batting Order</h2>
              <ol style={{ paddingLeft: "1.2rem" }}>
                {batOrder.map((name, i) => (
                  <li key={i} style={{ padding: "4px 0", color: i < 3 ? "#93c5fd" : "#e2e8f0" }}>{name}</li>
                ))}
              </ol>
            </div>
          )}
          {bowlRot && (
            <div className="card">
              <h2>Optimized Bowling Rotation</h2>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: 6 }}>
                {bowlRot.map((b, i) => (
                  <div key={i} className="stat-box" style={{ padding: "6px" }}>
                    <div style={{ fontSize: "0.7rem", color: "#64748b" }}>Ov {i + 1}</div>
                    <div style={{ fontSize: "0.8rem" }}>{b}</div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
