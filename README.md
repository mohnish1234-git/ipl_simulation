# IPL Match Simulation & Team Optimization System

AI-powered ball-by-ball IPL match simulator using historical data, ML predictions, and Monte Carlo simulations.

## Workflow
```
IPL Ball-by-Ball Dataset
        ↓
Data Cleaning & Feature Engineering  (src/data/)
        ↓
Ball Outcome Prediction Model        (Google Colab → models/)
        ↓
Match Simulation Engine              (src/simulation/)
        ↓
Monte Carlo Simulations              (src/simulation/)
        ↓
Team Optimization Engine             (src/optimization/)
        ↓
Web Application                      (webapp/)
```

## Project Structure
```
ipl_simulation/
├── data/
│   ├── raw/                    # Original ipl_final.csv goes here
│   └── processed/              # Cleaned & engineered features
├── notebooks/
│   └── colab_training.ipynb    # Upload to Google Colab for model training
├── src/
│   ├── data/
│   │   ├── cleaner.py          # Data cleaning pipeline
│   │   └── feature_engineer.py # Feature engineering
│   ├── model/
│   │   └── predictor.py        # Model inference wrapper (loads exported model)
│   ├── simulation/
│   │   ├── match_simulator.py  # Single match simulation engine
│   │   └── monte_carlo.py      # Monte Carlo runner
│   ├── optimization/
│   │   └── team_optimizer.py   # Team selection optimizer
│   └── api/
│       └── app.py              # FastAPI backend
├── models/                     # Trained model files dropped here after Colab
├── scripts/
│   └── prepare_data.py         # Run once: cleans data & saves processed files
├── webapp/                     # React frontend
├── requirements.txt
└── README.md
```

## Setup

### 1. Place your dataset
```bash
cp ipl_final.csv data/raw/
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare data (run once)
```bash
python scripts/prepare_data.py
```

### 4. Train model (Google Colab)
- Upload `notebooks/colab_training.ipynb` to Google Colab
- Upload `data/processed/features.csv` to Colab
- Run all cells
- Download the exported `ipl_ball_model.pkl` and `label_encoders.pkl`
- Place both files in the `models/` folder

### 5. Start the API
```bash
uvicorn src.api.app:app --reload --port 8000
```

### 6. Start the webapp
```bash
cd webapp && npm install && npm start
```
