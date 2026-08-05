import React, { useState } from "react";
import TeamBuilder from "../components/TeamBuilder";
import useMeta from "../hooks/useMeta";
import { predictFantasyXI } from "../utils/api";

const blank11 = () => Array(11).fill("");

export default function FantasyPredictorPage() {
    const { meta, loading: metaLoading, error: metaError } = useMeta();

    const [team1, setTeam1] = useState("Mumbai Indians");
    const [team2, setTeam2] = useState("Chennai Super Kings");
    const [venue, setVenue] = useState("");

    const [playingXI1, setPlayingXI1] = useState(blank11());
    const [playingXI2, setPlayingXI2] = useState(blank11());

    const [tossWinner, setTossWinner] = useState("");
    const [tossDecision, setTossDecision] = useState("bat");

    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");
    const [result, setResult] = useState(null);

    const handlePredict = async () => {
        setLoading(true);
        setError("");
        setResult(null);

        try {
            const response = await predictFantasyXI({
                team1,
                team2,
                venue: venue || "Unknown",

                playing_xi_team1: playingXI1.map((p) => p || "Unknown"),
                playing_xi_team2: playingXI2.map((p) => p || "Unknown"),

                toss_winner: tossWinner || team1,
                toss_decision: tossDecision,
            });

            setResult(response);
        } catch (e) {
            setError(e.response?.data?.detail || e.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div>

            <h1 style={{ marginBottom: "1.5rem", fontSize: "1.4rem" }}>
                AI Fantasy XI Predictor
            </h1>

            <div className="card">

                <h2>Match Details</h2>

                {metaLoading && (
                    <p className="meta-status">
                        Loading match information...
                    </p>
                )}

                {metaError && (
                    <p className="meta-status warning">
                        Metadata unavailable. Manual input enabled.
                    </p>
                )}

                <div className="grid-3">

                    <div className="field">
                        <label>Team 1</label>

                        {meta.teams?.length ? (
                            <select
                                value={team1}
                                onChange={(e) => setTeam1(e.target.value)}
                            >
                                {meta.teams.map((t) => (
                                    <option key={t}>{t}</option>
                                ))}
                            </select>
                        ) : (
                            <input
                                value={team1}
                                onChange={(e) => setTeam1(e.target.value)}
                            />
                        )}
                    </div>

                    <div className="field">
                        <label>Team 2</label>

                        {meta.teams?.length ? (
                            <select
                                value={team2}
                                onChange={(e) => setTeam2(e.target.value)}
                            >
                                {meta.teams.map((t) => (
                                    <option key={t}>{t}</option>
                                ))}
                            </select>
                        ) : (
                            <input
                                value={team2}
                                onChange={(e) => setTeam2(e.target.value)}
                            />
                        )}
                    </div>

                    <div className="field">
                        <label>Venue</label>

                        {meta.venues?.length ? (
                            <select
                                value={venue}
                                onChange={(e) => setVenue(e.target.value)}
                            >
                                <option value="">Select Venue</option>

                                {meta.venues.map((v) => (
                                    <option key={v}>{v}</option>
                                ))}
                            </select>
                        ) : (
                            <input
                                value={venue}
                                placeholder="Venue"
                                onChange={(e) => setVenue(e.target.value)}
                            />
                        )}
                    </div>

                </div>

                <div
                    className="grid-2"
                    style={{ marginTop: "1rem" }}
                >

                    <div className="field">
                        <label>Toss Winner</label>

                        <select
                            value={tossWinner}
                            onChange={(e) => setTossWinner(e.target.value)}
                        >
                            <option value="">Select</option>
                            <option value={team1}>{team1}</option>
                            <option value={team2}>{team2}</option>
                        </select>
                    </div>

                    <div className="field">
                        <label>Toss Decision</label>

                        <select
                            value={tossDecision}
                            onChange={(e) => setTossDecision(e.target.value)}
                        >
                            <option value="bat">Bat First</option>
                            <option value="bowl">Bowl First</option>
                        </select>
                    </div>

                </div>

            </div>

            <div className="grid-2">

                <TeamBuilder
                    label={`${team1} Playing XI`}
                    players={playingXI1}
                    setPlayers={setPlayingXI1}
                    knownPlayers={
                        meta.batters_by_team?.[team1] ??
                        meta.players_by_team?.[team1] ??
                        meta.players ??
                        []
                    }
                />

                <TeamBuilder
                    label={`${team2} Playing XI`}
                    players={playingXI2}
                    setPlayers={setPlayingXI2}
                    knownPlayers={
                        meta.batters_by_team?.[team2] ??
                        meta.players_by_team?.[team2] ??
                        meta.players ??
                        []
                    }
                />

            </div>

            <button
                className="btn btn-primary"
                onClick={handlePredict}
                disabled={loading}
                style={{ marginTop: "1.5rem" }}
            >
                {loading
                    ? "Generating Fantasy XI..."
                    : "Generate Fantasy XI"}
            </button>

            {loading && (
                <div className="loading">
                    <div className="spinner" />
                    <p>Analysing players...</p>
                </div>
            )}

            {error && (
                <div
                    className="card"
                    style={{ color: "#f87171", marginTop: "1rem" }}
                >
                    {error}
                </div>
            )}

            {result && (

                <div className="card" style={{ marginTop: "2rem" }}>

                    <h2>Recommended Fantasy XI</h2>

                    <table className="table">

                        <thead>

                            <tr>
                                <th>Rank</th>
                                <th>Player</th>
                                <th>Team</th>
                                <th>High Performer %</th>
                                <th>Role</th>
                            </tr>

                        </thead>

                        <tbody>

                            {result.best_xi?.map((player, index) => (

                                <tr key={player.player_name}>

                                    <td>{index + 1}</td>

                                    <td>
                                        {player.player_name}

                                        {player.player_name === result.captain &&
                                            " (C)"}

                                        {player.player_name === result.vice_captain &&
                                            " (VC)"}
                                    </td>

                                    <td>{player.team}</td>

                                    <td>
                                        {(player.high_performer_probability * 100).toFixed(1)}%
                                    </td>

                                    <td>{player.role}</td>

                                </tr>

                            ))}

                        </tbody>

                    </table>

                </div>

            )}

        </div>
    );
}