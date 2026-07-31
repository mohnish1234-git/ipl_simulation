# IPL Match Simulation & Fantasy Recommendation System

An AI-powered cricket analytics platform that combines machine learning, ball-by-ball match simulation, Monte Carlo methods, and a PostgreSQL-backed feature store to simulate IPL matches and recommend the optimal Fantasy XI for upcoming games.

---

## Overview

This project consists of two integrated systems:

### 1. IPL Match Simulation Engine

Predicts every ball of an IPL match using a machine learning model trained on historical IPL deliveries.

The simulator models

- Batter vs Bowler interactions
- Venue effects
- Match situation
- Wickets
- Strike rotation
- Boundary probability
- Monte Carlo simulations

to estimate realistic match outcomes.

---

### 2. Fantasy Recommendation Engine

Uses historical player statistics together with Monte Carlo simulation outputs to recommend

- Best Fantasy XI
- Captain
- Vice Captain

using a weighted decision engine.

Instead of relying only on simulations or historical averages, the recommendation combines multiple prediction sources.

---

# Overall Architecture

```
                    IPL Ball-by-Ball Dataset
                               │
                               ▼
                Data Cleaning & Feature Engineering
                               │
                               ▼
                Historical Feature Generation
                               │
                ┌──────────────┴──────────────┐
                ▼                             ▼
        PostgreSQL Feature Store        Ball Outcome ML Model
                │                             │
                │                             ▼
                │                   Ball-by-Ball Simulation
                │                             │
                │                             ▼
                │                  Monte Carlo Simulations
                │                             │
                └──────────────┬──────────────┘
                               ▼
                     Fantasy Decision Engine
                               │
                               ▼
                    Dream11 Team Optimizer
                               │
                               ▼
                 Captain & Vice Captain Selection
                               │
                               ▼
                      FastAPI + React Web App
```

---

# Features

## Match Simulation

- Ball-by-ball IPL simulation
- Machine learning event prediction
- Dynamic strike rotation
- Partnership tracking
- Wicket progression
- Over-by-over simulation
- Monte Carlo simulations
- Venue-aware scoring
- Batter vs Bowler modelling

---

## Fantasy Recommendation

- PostgreSQL feature store
- Historical player analytics
- Venue-adjusted statistics
- Batter vs Bowler history
- Machine learning fantasy prediction
- Simulation-assisted player scoring
- Decision engine
- Dream11 team optimization
- Captain recommendation
- Vice Captain recommendation

---

# Project Structure

```
ipl_simulation/

├── data/
│   ├── raw/
│   └── processed/
│
├── database/
│   ├── create_database.py
│   ├── db.py
│   ├── import_jsons.py
│   ├── queries.py
│   ├── feature_store.py
│   └── test_database.py
│
├── fantasy_engine/
│   ├── feature_builder.py
│   ├── fantasy_predictor.py
│   ├── simulator_adapter.py
│   ├── decision_engine.py
│   ├── optimizer.py
│   ├── captain_selector.py
│   ├── vicecaptain_selector.py
│   └── output_formatter.py
│
├── models/
│
├── notebooks/
│
├── scripts/
│
├── src/
│   ├── api/
│   ├── data/
│   ├── model/
│   ├── optimization/
│   └── simulation/
│
├── webapp/
│
├── requirements.txt
│
└── README.md
```

---

# Technology Stack

### Programming

- Python

### Machine Learning

- XGBoost
- Scikit-Learn

### Database

- PostgreSQL

### Backend

- FastAPI

### Frontend

- React

### Optimization

- PuLP

### Data Processing

- Pandas
- NumPy

---

# Match Simulation Pipeline

```
Historical IPL Data
        │
        ▼
Feature Engineering
        │
        ▼
Ball Outcome Prediction Model
        │
        ▼
Ball-by-Ball Match Simulator
        │
        ▼
Monte Carlo Simulations
        │
        ▼
Predicted Match Statistics
```

---

# Fantasy Recommendation Pipeline

```
Historical Features
        │
        ▼
PostgreSQL Feature Store
        │
        ▼
Fantasy ML Model
        │
        ├───────────────┐
        ▼               ▼
Historical Score   Monte Carlo Score
        │               │
        └──────┬────────┘
               ▼
       Decision Engine
               ▼
      Dream11 Optimizer
               ▼
Recommended Fantasy XI
```

---

# Setup

1. Clone the repository.

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Place `ipl_final.csv` inside `data/raw/`.

4. Run the preprocessing pipeline.

```bash
python scripts/prepare_data.py
```

5. Create the PostgreSQL database.

```bash
python -m database.create_database
```

6. Import the processed statistics.

```bash
python -m database.import_jsons
```

7. Train the machine learning model.

8. Start the FastAPI backend.

```bash
uvicorn src.api.app:app --reload
```

9. Start the React frontend.

```bash
cd webapp
npm install
npm start
```

---

# Future Improvements

- Toss-aware decision engine
- Weather-aware simulations
- Playing XI prediction
- Live score integration
- Fantasy ownership prediction
- Bayesian player uncertainty modelling
- Reinforcement learning for captain selection
- Multi-match tournament simulation