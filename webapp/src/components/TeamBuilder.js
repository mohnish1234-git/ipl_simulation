import React from "react";

export default function TeamBuilder({
  label,
  players,
  setPlayers,
  knownPlayers = [],
}) {
  const handlePlayerChange = (index, value) => {
    const updated = [...players];
    updated[index] = value;
    setPlayers(updated);
  };

  const autofillLineup = () => {
    if (!knownPlayers || knownPlayers.length < 11) return;

    const shuffled = [...knownPlayers].sort(() => Math.random() - 0.5);
    setPlayers(shuffled.slice(0, 11));
  };

  const clearLineup = () => {
    setPlayers(Array(11).fill(""));
  };

  return (
    <div
      className="card team-builder"
      style={{
        padding: "1.5rem",
        borderRadius: "12px",
        border: "1px solid #334155",
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: "1rem",
        }}
      >
        <h2
          style={{
            fontSize: "1.2rem",
            margin: 0,
            color: "#f8fafc",
          }}
        >
          {label}
        </h2>

        <div
          style={{
            display: "flex",
            gap: "8px",
          }}
        >
          <button
            type="button"
            className="btn"
            onClick={autofillLineup}
            style={{
              fontSize: "0.75rem",
              padding: "4px 8px",
            }}
          >
            🪄 Auto Fill
          </button>

          <button
            type="button"
            className="btn"
            onClick={clearLineup}
            style={{
              fontSize: "0.75rem",
              padding: "4px 8px",
            }}
          >
            Clear
          </button>
        </div>
      </div>

      <h3
        style={{
          fontSize: "0.95rem",
          color: "#94a3b8",
          marginBottom: "0.75rem",
          borderBottom: "1px solid #1e293b",
          paddingBottom: "4px",
        }}
      >
        Playing XI
      </h3>

      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: "8px",
        }}
      >
        {players.map((player, index) => (
          <div
            key={index}
            style={{
              display: "flex",
              alignItems: "center",
              gap: "10px",
            }}
          >
            <span
              style={{
                minWidth: "24px",
                fontWeight: 600,
                color: "#64748b",
              }}
            >
              {index + 1}
            </span>

            {knownPlayers.length ? (
              <select
                value={player}
                onChange={(e) =>
                  handlePlayerChange(index, e.target.value)
                }
                style={{
                  flex: 1,
                  padding: "6px",
                  borderRadius: "6px",
                  background: "#0f172a",
                  border: "1px solid #334155",
                  color: "#e2e8f0",
                }}
              >
                <option value="">Select Player</option>

                {knownPlayers.map((name) => (
                  <option
                    key={name}
                    value={name}
                  >
                    {name}
                  </option>
                ))}
              </select>
            ) : (
              <input
                type="text"
                value={player}
                placeholder="Player Name"
                onChange={(e) =>
                  handlePlayerChange(index, e.target.value)
                }
                style={{
                  flex: 1,
                  padding: "6px",
                  borderRadius: "6px",
                  background: "#0f172a",
                  border: "1px solid #334155",
                  color: "#e2e8f0",
                }}
              />
            )}
          </div>
        ))}
      </div>
    </div>
  );
}