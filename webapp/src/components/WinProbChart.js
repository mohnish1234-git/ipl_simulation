import React from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";

export default function WinProbChart({ team1, team2, prob1, prob2 }) {
  const data = [
    { name: team1, prob: +(prob1 * 100).toFixed(1) },
    { name: team2, prob: +(prob2 * 100).toFixed(1) },
  ];
  return (
    <div className="card">
      <h2>Win Probability</h2>
      <div className="grid-2" style={{ marginBottom: "1rem" }}>
        <div className="stat-box">
          <div className="val" style={{ color: "#3b82f6" }}>{(prob1 * 100).toFixed(1)}%</div>
          <div className="lbl">{team1}</div>
        </div>
        <div className="stat-box">
          <div className="val" style={{ color: "#f59e0b" }}>{(prob2 * 100).toFixed(1)}%</div>
          <div className="lbl">{team2}</div>
        </div>
      </div>

      <div style={{ marginBottom: ".5rem" }}>
        <div className="prob-bar-label"><span>{team1}</span><span>{(prob1 * 100).toFixed(1)}%</span></div>
        <div className="prob-bar-track"><div className="prob-bar-fill" style={{ width: `${prob1 * 100}%` }} /></div>
      </div>
      <div>
        <div className="prob-bar-label"><span>{team2}</span><span>{(prob2 * 100).toFixed(1)}%</span></div>
        <div className="prob-bar-track"><div className="prob-bar-fill team2" style={{ width: `${prob2 * 100}%` }} /></div>
      </div>
    </div>
  );
}

export function ScoreDistChart({ team1, team2, dist1, dist2 }) {
  // Build histogram bins of width 5
  const bin = (scores) => {
    const counts = {};
    scores.forEach(s => { const b = Math.floor(s / 5) * 5; counts[b] = (counts[b] || 0) + 1; });
    return Object.entries(counts).sort((a,b) => +a[0]-+b[0]).map(([k,v]) => ({ score: +k, count: v }));
  };
  const d1 = bin(dist1 || []);
  const d2 = bin(dist2 || []);

  return (
    <div className="card">
      <h2>Score Distribution</h2>
      <p className="section-title" style={{ marginBottom: 4 }}>{team1}</p>
      <ResponsiveContainer width="100%" height={120}>
        <BarChart data={d1} margin={{ top: 4, right: 0, bottom: 0, left: -10 }}>
          <XAxis dataKey="score" tick={{ fontSize: 10, fill: "#64748b" }} />
          <YAxis hide />
          <Tooltip formatter={(v) => [v, "simulations"]} contentStyle={{ background: "#1e293b", border: "1px solid #334155" }} />
          <Bar dataKey="count" fill="#3b82f6" radius={[2,2,0,0]} />
        </BarChart>
      </ResponsiveContainer>
      <p className="section-title" style={{ marginTop: 12, marginBottom: 4 }}>{team2}</p>
      <ResponsiveContainer width="100%" height={120}>
        <BarChart data={d2} margin={{ top: 4, right: 0, bottom: 0, left: -10 }}>
          <XAxis dataKey="score" tick={{ fontSize: 10, fill: "#64748b" }} />
          <YAxis hide />
          <Tooltip formatter={(v) => [v, "simulations"]} contentStyle={{ background: "#1e293b", border: "1px solid #334155" }} />
          <Bar dataKey="count" fill="#f59e0b" radius={[2,2,0,0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
