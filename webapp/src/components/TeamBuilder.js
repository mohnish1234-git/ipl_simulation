import React from "react";

export default function TeamBuilder({ label, players, setPlayers, rotation, setRotation, knownPlayers }) {
  const handlePlayerChange = (index, value) => {
    const newPlayers = [...players];
    newPlayers[index] = value;
    setPlayers(newPlayers);
  };

  const handleRotationChange = (index, value) => {
    const newRotation = [...rotation];
    newRotation[index] = value;
    setRotation(newRotation);
  };

  const autofillLineup = () => {
    if (!knownPlayers || knownPlayers.length < 11) return;
    // Pick 11 random known players
    const shuffled = [...knownPlayers].sort(() => 0.5 - Math.random());
    const selected = shuffled.slice(0, 11);
    setPlayers(selected);
  };

  const autofillRotation = () => {
    // Collect non-empty players in lineup or fallback to known batters/bowlers
    const activeLineup = players.filter(Boolean);
    const pool = activeLineup.length >= 3 ? activeLineup : (knownPlayers.length ? knownPlayers : ["Bowler A", "Bowler B", "Bowler C"]);
    const newRot = Array(20).fill("").map(() => pool[Math.floor(Math.random() * pool.length)]);
    setRotation(newRot);
  };

  return (
    <div className="card team-builder" style={{ padding: "1.5rem", borderRadius: "12px", border: "1px solid #334155" }}>
      <div style={{ display: "flex", justifyContent: "between", alignItems: "center", marginBottom: "1rem" }}>
        <h2 style={{ fontSize: "1.2rem", margin: 0, color: "#f8fafc" }}>{label}</h2>
        <div style={{ display: "flex", gap: "8px" }}>
          <button className="btn" onClick={autofillLineup} style={{ fontSize: "0.75rem", padding: "4px 8px" }} type="button">
            🪄 Auto Lineup
          </button>
          <button className="btn" onClick={autofillRotation} style={{ fontSize: "0.75rem", padding: "4px 8px" }} type="button">
            🪄 Auto Bowlers
          </button>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1.5rem" }}>
        {/* Batting order (1-11) */}
        <div>
          <h3 style={{ fontSize: "0.95rem", color: "#94a3b8", marginBottom: "0.75rem", borderBottom: "1px solid #1e293b", paddingBottom: "4px" }}>
            Batting Lineup (1-11)
          </h3>
          <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
            {players.map((p, idx) => (
              <div key={idx} style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <span style={{ minWidth: "22px", fontSize: "0.85rem", color: "#64748b", fontWeight: 600 }}>
                  {idx + 1}
                </span>
                {knownPlayers.length ? (
                  <select
                    value={p}
                    onChange={(e) => handlePlayerChange(idx, e.target.value)}
                    style={{ flexGrow: 1, padding: "5px", borderRadius: "6px", backgroundColor: "#0f172a", border: "1px solid #334155", color: "#e2e8f0", fontSize: "0.85rem" }}
                  >
                    <option value="">— select player —</option>
                    {knownPlayers.map((name) => (
                      <option key={name} value={name}>{name}</option>
                    ))}
                  </select>
                ) : (
                  <input
                    type="text"
                    value={p}
                    onChange={(e) => handlePlayerChange(idx, e.target.value)}
                    placeholder={`Player name`}
                    style={{ flexGrow: 1, padding: "5px", borderRadius: "6px", backgroundColor: "#0f172a", border: "1px solid #334155", color: "#e2e8f0", fontSize: "0.85rem" }}
                  />
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Bowling Rotation (20 Overs) */}
        <div>
          <h3 style={{ fontSize: "0.95rem", color: "#94a3b8", marginBottom: "0.75rem", borderBottom: "1px solid #1e293b", paddingBottom: "4px" }}>
            Bowling Rotation (20 Overs)
          </h3>
          <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: "6px", maxHeight: "400px", overflowY: "auto", paddingRight: "4px" }}>
            {rotation.map((b, idx) => (
              <div key={idx} style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                <span style={{ minWidth: "45px", fontSize: "0.85rem", color: "#64748b", fontWeight: 600 }}>
                  Over {idx + 1}
                </span>
                {knownPlayers.length ? (
                  <select
                    value={b}
                    onChange={(e) => handleRotationChange(idx, e.target.value)}
                    style={{ flexGrow: 1, padding: "5px", borderRadius: "6px", backgroundColor: "#0f172a", border: "1px solid #334155", color: "#e2e8f0", fontSize: "0.85rem" }}
                  >
                    <option value="">— select bowler —</option>
                    {knownPlayers.map((name) => (
                      <option key={name} value={name}>{name}</option>
                    ))}
                  </select>
                ) : (
                  <input
                    type="text"
                    value={b}
                    onChange={(e) => handleRotationChange(idx, e.target.value)}
                    placeholder={`Bowler name`}
                    style={{ flexGrow: 1, padding: "5px", borderRadius: "6px", backgroundColor: "#0f172a", border: "1px solid #334155", color: "#e2e8f0", fontSize: "0.85rem" }}
                  />
                )}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
