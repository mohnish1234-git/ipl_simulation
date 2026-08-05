import React from "react";
import { BrowserRouter, Routes, Route, Link } from "react-router-dom";
import SimulatePage from "./pages/SimulatePage";
import FantasyPredictorPage from "./pages/FantasyPredictorPage";
import "./App.css";

export default function App() {
    return (
        <BrowserRouter>
            <div className="app">
                <nav className="navbar">
                    <div className="nav-brand">🏏 IPL Fantasy XI Predictor</div>

                    <div className="nav-links">
                        <Link to="/">Simulation</Link>
                        <Link to="/fantasy">Fantasy XI</Link>
                    </div>
                </nav>

                <main className="main-content">
                    <Routes>
                        <Route path="/" element={<SimulatePage />} />
                        <Route path="/fantasy" element={<FantasyPredictorPage />} />
                    </Routes>
                </main>
            </div>
        </BrowserRouter>
    );
}