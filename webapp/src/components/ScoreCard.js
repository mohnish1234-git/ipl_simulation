import React from "react";

export default function ScoreCard({ result }) {
  if (!result) return null;
  const { batting_team_1: t1, batting_team_2: t2,
          score_1, wickets_1, score_2, wickets_2, winner, win_margin, win_type } = result;

  return (
    <div className="card">
      <h2>Match Result</h2>
      <div className="grid-2" style={{ marginBottom: "1rem" }}>
        <div className="stat-box">
          <div className="lbl">{t1}</div>
          <div className="val">{score_1}/{wickets_1}</div>
          <div className="lbl">20 overs</div>
        </div>
        <div className="stat-box">
          <div className="lbl">{t2}</div>
          <div className="val">{score_2}/{wickets_2}</div>
          <div className="lbl">20 overs</div>
        </div>
      </div>
      <div style={{ textAlign: "center", marginBottom: "1rem" }}>
        <span className="badge badge-win">{winner} won</span>
        <span style={{ marginLeft: 8, color: "#94a3b8", fontSize: "0.85rem" }}>
          by {win_margin} {win_type}
        </span>
      </div>
      <BatterTable title={`${t1} Batting`} stats={result.batter_stats_1} />
      <BatterTable title={`${t2} Batting`} stats={result.batter_stats_2} />
      <BowlerTable title={`${t1} Bowling`} stats={result.bowler_stats_1} />
      <BowlerTable title={`${t2} Bowling`} stats={result.bowler_stats_2} />
    </div>
  );
}

function BatterTable({ title, stats }) {
  const rows = Object.values(stats).filter(s => s.balls > 0);
  if (!rows.length) return null;
  return (
    <div style={{ marginBottom: "1rem" }}>
      <p className="section-title">{title}</p>
      <table>
        <thead><tr><th>Batter</th><th>R</th><th>B</th><th>4s</th><th>6s</th><th>SR</th><th></th></tr></thead>
        <tbody>
          {rows.map(r => (
            <tr key={r.name}>
              <td>{r.name}</td>
              <td><b>{r.runs}</b></td>
              <td>{r.balls}</td>
              <td>{r.fours}</td>
              <td>{r.sixes}</td>
              <td>{r.strike_rate}</td>
              <td>{r.dismissed ? <span className="badge badge-loss">out</span> : <span className="badge badge-win">not out</span>}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function BowlerTable({ title, stats }) {
  const rows = Object.values(stats).filter(s => s.balls > 0);
  if (!rows.length) return null;
  return (
    <div style={{ marginBottom: "1rem" }}>
      <p className="section-title">{title}</p>
      <table>
        <thead><tr><th>Bowler</th><th>O</th><th>R</th><th>W</th><th>Eco</th></tr></thead>
        <tbody>
          {rows.map(r => (
            <tr key={r.name}>
              <td>{r.name}</td>
              <td>{(r.balls / 6).toFixed(1)}</td>
              <td>{r.runs}</td>
              <td><b>{r.wickets}</b></td>
              <td>{r.economy}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
