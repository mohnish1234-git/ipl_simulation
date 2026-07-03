import React, { useState } from "react";
import TeamBuilder from "../components/TeamBuilder";
import ScoreCard from "../components/ScoreCard";
import { simulateMatch } from "../utils/api";
import useMeta from "../hooks/useMeta";

const blank11 = () => Array(11).fill("");
const blank20 = () => Array(20).fill("");

export default function SimulatePage() {
  const { meta } = useMeta();

  const [team1, setTeam1]   = useState("Mumbai Indians");
  const [team2, setTeam2]   = useState("Chennai Super Kings");
  const [venue, setVenue]   = useState("");
  const [order1, setOrder1] = useState(blank11());
  const [order2, setOrder2] = useState(blank11());
  const [rot1, setRot1]     = useState(blank20());
  const [rot2, setRot2]     = useState(blank20());

  const [result, setResult]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState("");

  const handleSimulate = async () => {
    setLoading(true); setError(""); setResult(null);
    try {
      const res = await simulateMatch({
        team1, team2, venue: venue || "Unknown",
        batting_order_1: order1.map(p => p || "Unknown"),
        batting_order_2: order2.map(p => p || "Unknown"),
        bowling_rotation_1: rot1.map(b => b || "Unknown"),
        bowling_rotation_2: rot2.map(b => b || "Unknown"),
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
      <h1 style={{ marginBottom: "1.5rem", fontSize: "1.4rem" }}>Single Match Simulation</h1>

      {/* Match setup */}
      <div className="card">
        <h2>Match Setup</h2>
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
      </div>

      {/* Team builders */}
      <div className="grid-2">
        <TeamBuilder label={`${team1} — Squad`} players={order1} setPlayers={setOrder1} rotation={rot1} setRotation={setRot1} knownPlayers={meta.batters || []} />
        <TeamBuilder label={`${team2} — Squad`} players={order2} setPlayers={setOrder2} rotation={rot2} setRotation={setRot2} knownPlayers={meta.batters || []} />
      </div>

      <button className="btn btn-primary" onClick={handleSimulate} disabled={loading} style={{ marginBottom: "1.5rem" }}>
        {loading ? "Simulating…" : "▶ Simulate Match"}
      </button>

      {error && <div className="card" style={{ color: "#f87171" }}>Error: {error}</div>}
      {loading && <div className="loading"><div className="spinner" /><p>Simulating match…</p></div>}
      {result && <ScoreCard result={result} />}
    </div>
  );
}
