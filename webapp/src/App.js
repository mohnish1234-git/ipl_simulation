import React from "react";
import { BrowserRouter, Routes, Route, NavLink } from "react-router-dom";
import SimulatePage from "./pages/SimulatePage";
import MonteCarloPage from "./pages/MonteCarloPage";
import OptimizerPage from "./pages/OptimizerPage";
import "./App.css";

export default function App() {
  return (
    <BrowserRouter>
      <div className="app">
        <nav className="navbar">
          <div className="nav-brand">🏏 IPL Simulator</div>
          <div className="nav-links">
            <NavLink to="/"         end>Simulate</NavLink>
            <NavLink to="/monte-carlo">Monte Carlo</NavLink>
            <NavLink to="/optimize">Optimize</NavLink>
          </div>
        </nav>
        <main className="main-content">
          <Routes>
            <Route path="/"            element={<SimulatePage />} />
            <Route path="/monte-carlo" element={<MonteCarloPage />} />
            <Route path="/optimize"    element={<OptimizerPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
