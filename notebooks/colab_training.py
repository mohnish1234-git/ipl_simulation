# %% [markdown]
# # IPL Ball Outcome Prediction Model — Google Colab Training
# **Context-Aware & Recency-Weighted XGBoost**

# %%
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

# %% [markdown]
# ## Cell 3 — Load features

# %%
df = pd.read_csv("features.csv")
print(f"Loaded: {df.shape}")
print(f"\nOutcome distribution:\n{df['outcome'].value_counts()}")
print(f"\nSample weight range: {df['sample_weight'].min():.4f} – {df['sample_weight'].max():.4f}")

# %% [markdown]
# ## Cell 4 — Define feature groups
#
# All 7 feature groups are listed explicitly.
# XGBoost learns which ones matter most — nothing is hardcoded.

# %%
CATEGORICAL = [
    "striker", "bowler", "batting_team", "bowling_team", "venue", "phase",
]

NUMERICAL = [
    # ── basic match state ────────────────────────────────────────────────────
    "over_num", "ball_num",
    "cumulative_runs", "cumulative_wickets",
    "balls_remaining", "wickets_remaining",
    "crr",

    # ── recency-weighted career stats — batter ───────────────────────────────
    "bat_rw_avg", "bat_rw_sr", "bat_rw_boundary_pct", "bat_rw_six_pct",
    "bat_rw_dot_pct",
    "bat_pp_rw_sr", "bat_mid_rw_sr", "bat_death_rw_sr",
    "bat_pp_rw_boundary_pct", "bat_death_rw_boundary_pct",

    # ── recency-weighted career stats — bowler ───────────────────────────────
    "bowl_rw_economy", "bowl_rw_wicket_pct", "bowl_rw_dot_pct",
    "bowl_rw_boundary_pct",
    "bowl_pp_rw_economy", "bowl_mid_rw_economy", "bowl_death_rw_economy",
    "bowl_pp_rw_wicket_pct", "bowl_death_rw_wicket_pct",

    # ── player-at-venue interaction — HIGHEST precedence tier ────────────────
    # These were being computed by feature_engineer.py (shrunk toward career
    # stats, K=15) but were previously missing from this list entirely, so
    # the model never actually trained on them. Adding them now is what makes
    # the venue-precedence idea real rather than cosmetic.
    "bat_venue_adj_sr", "bat_venue_adj_boundary_pct", "bat_venue_rw_balls",
    "bowl_venue_adj_economy", "bowl_venue_adj_wicket_pct", "bowl_venue_rw_balls",

    # ── batter-vs-bowler matchup ─────────────────────────────────────────────
    "bvb_balls", "bvb_rw_sr", "bvb_rw_dismissal_pct",
    "bvb_rw_dot_pct", "bvb_rw_boundary_pct", "bvb_rw_six_pct",

    # ── in-match momentum ────────────────────────────────────────────────────
    "batter_balls_faced", "batter_runs_scored", "batter_innings_sr",
    "balls_vs_bowler", "runs_vs_bowler",
    "runs_last6", "runs_last_over",
    "consec_dots", "consec_boundaries",
    "partnership_runs", "partnership_balls",
    "prev_ball_outcome", "prev2_ball_outcome", "prev3_ball_outcome",

    # ── venue intelligence ───────────────────────────────────────────────────
    "venue_rw_avg_1st_innings", "venue_rw_avg_2nd_innings",
    "venue_rw_boundary_pct", "venue_rw_six_pct",
    "venue_rw_dot_pct", "venue_rw_wicket_pct",
    "venue_rw_pp_sr", "venue_rw_death_sr",

    # ── batting context / pressure ───────────────────────────────────────────
    "is_batting_first", "is_chasing",
    "target", "runs_needed", "rrr", "pressure_index",
]

TARGET        = "outcome"
WEIGHT_COL    = "sample_weight"
ALL_FEATURES  = CATEGORICAL + NUMERICAL

# Keep only columns present in the file
ALL_FEATURES = [c for c in ALL_FEATURES if c in df.columns]
NUMERICAL    = [c for c in NUMERICAL    if c in df.columns]
CATEGORICAL  = [c for c in CATEGORICAL  if c in df.columns]

print(f"\nTotal features: {len(ALL_FEATURES)}  "
      f"(categorical: {len(CATEGORICAL)}, numerical: {len(NUMERICAL)})")

# drop rows with missing target
df = df.dropna(subset=[TARGET])
print(f"Rows after dropping nulls: {len(df):,}")

# %% [markdown]
# ## Cell 5 — Encode categorical columns

# %%
label_encoders = {}

for col in CATEGORICAL:
    le = LabelEncoder()
    df[col] = df[col].fillna("Unknown").astype(str)
    all_vals = list(df[col].unique()) + ["Unknown"]
    le.fit(all_vals)
    df[col] = le.transform(df[col])
    label_encoders[col] = le
    print(f"  {col}: {len(le.classes_)} classes")

# encode target
le_target = LabelEncoder()
df[TARGET] = le_target.fit_transform(df[TARGET].astype(str))
label_encoders["outcome"] = le_target
print(f"\nTarget classes: {le_target.classes_}")

# %% [markdown]
# ## Cell 6 — Fill numeric nulls

# %%
for col in NUMERICAL:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())

# %% [markdown]
# ## Cell 7 — Train / validation split
#
# We split CHRONOLOGICALLY (by season) rather than randomly.
# The most recent 15% of seasons are held out for validation.
# This prevents leakage — we never validate on matches that preceded training.

# %%
seasons   = sorted(df["season"].unique()) if "season" in df.columns else []
split_idx = int(len(seasons) * 0.85)
val_seasons = set(seasons[split_idx:]) if seasons else set()

if val_seasons:
    train_mask = ~df["season"].isin(val_seasons)
    val_mask   =  df["season"].isin(val_seasons)
    X_train = df.loc[train_mask, ALL_FEATURES]
    X_val   = df.loc[val_mask,   ALL_FEATURES]
    y_train = df.loc[train_mask, TARGET]
    y_val   = df.loc[val_mask,   TARGET]
    w_train = df.loc[train_mask, WEIGHT_COL].values if WEIGHT_COL in df.columns else None
    print(f"Chronological split — Val seasons: {sorted(val_seasons)}")
else:
    # Fallback: random split if season column missing
    X_train, X_val, y_train, y_val = train_test_split(
        df[ALL_FEATURES], df[TARGET], test_size=0.15, random_state=42, stratify=df[TARGET]
    )
    w_train = df.loc[X_train.index, WEIGHT_COL].values if WEIGHT_COL in df.columns else None
    print("Random split (season column not found)")

print(f"Train: {X_train.shape}  |  Val: {X_val.shape}")
if w_train is not None:
    print(f"Sample weights — min: {w_train.min():.4f}  max: {w_train.max():.4f}  mean: {w_train.mean():.4f}")

# %% [markdown]
# ## Cell 8 — Train XGBoost
#
# Key changes vs original:
# - `sample_weight` passed to fit() so recent deliveries influence the model more
# - `scale_pos_weight` not needed (multi-class); class imbalance handled via weights
# - `max_depth=8` to capture more complex context interactions
# - `subsample=0.7` + `colsample_bytree=0.7` for better generalisation with many features
# - `early_stopping_rounds=40` — THIS WAS MISSING BEFORE. With n_estimators=600
#   and no early stopping, the model kept training all 600 rounds even after
#   validation logloss bottomed out (~round 100) and started climbing again —
#   classic overfitting. The final saved model was the round-600 (most
#   overfit) checkpoint, not the best one. early_stopping_rounds=40 means
#   training stops once validation logloss hasn't improved for 40 rounds in
#   a row, and model.best_iteration tells us exactly which round was best —
#   used in Cell 12 to export ONLY the good trees, not the overfit tail.

# %%
model = XGBClassifier(
    n_estimators=600,
    max_depth=8,
    learning_rate=0.04,
    subsample=0.7,
    colsample_bytree=0.7,
    min_child_weight=5,
    gamma=0.1,
    reg_alpha=0.05,
    reg_lambda=1.0,
    eval_metric="mlogloss",
    early_stopping_rounds=40,     # ← stops training once val loss stalls/worsens
    random_state=42,
    n_jobs=-1,
    tree_method="hist",           # switch to "gpu_hist" if GPU runtime is enabled
)

model.fit(
    X_train, y_train,
    sample_weight=w_train,        # ← recency weighting applied here
    eval_set=[(X_val, y_val)],
    verbose=50,
)

print(f"\nBest iteration: {model.best_iteration}  "
      f"(out of {model.get_booster().num_boosted_rounds()} trained)")
print(f"Best validation mlogloss: {model.best_score:.5f}")
if model.best_iteration < model.get_booster().num_boosted_rounds() - 1:
    print("Training stopped early — the tail rounds were overfitting and "
          "have been discarded automatically.")

# %% [markdown]
# ## Cell 9 — Evaluate

# %%
y_pred = model.predict(X_val)
acc    = accuracy_score(y_val, y_pred)
print(f"\nValidation Accuracy: {acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_pred, target_names=le_target.classes_))

# %% [markdown]
# ## Cell 10 — Feature importances (top 20)

# %%
fi = (
    pd.Series(model.feature_importances_, index=ALL_FEATURES)
    .sort_values(ascending=False)
)
print("Top 20 features by importance:")
print(fi.head(20).to_string())

# Quick sanity check: momentum and venue features should appear in the top 20
top20 = set(fi.head(20).index)
print("\nMomentum features in top 20:", top20 & {"runs_last6","consec_dots","batter_balls_faced","pressure_index"})
print("Venue features in top 20:   ", top20 & {"venue_rw_pp_sr","venue_rw_death_sr","venue_rw_boundary_pct"})
print("Matchup features in top 20: ", top20 & {"bvb_rw_sr","bvb_balls","bvb_rw_boundary_pct"})

# %% [markdown]
# ## Cell 11 — Probability calibration check

# %%
proba = model.predict_proba(X_val.iloc[:5])
for i, row in enumerate(proba):
    line = "  ".join(f"{cls}={p:.2f}" for cls, p in zip(le_target.classes_, row))
    print(f"Ball {i+1}: {line}")

# %% [markdown]
# ## Cell 12 — Export model files
#
# Exports the booster SLICED to best_iteration — this is the fix that
# actually matters. Without it, even with early_stopping_rounds set above,
# joblib.dump(model, ...) would still pickle every tree trained (XGBoost
# keeps them all in memory for the eval_set curve), including the overfit
# tail past best_iteration. Slicing the booster is what makes the exported
# model match the checkpoint that actually had the best validation logloss.
#
# Also saves as JSON (not just pickle) — predictor.py already expects
# ipl_ball_model.json for cross-version portability; this is the model
# format it actually loads.

# %%
best_booster = model.get_booster()[: model.best_iteration + 1]
best_booster.save_model("ipl_ball_model.json")

joblib.dump(model,          "ipl_ball_model.pkl",    compress=3)   # legacy/backup only
joblib.dump(label_encoders, "label_encoders.pkl",    compress=3)
joblib.dump(ALL_FEATURES,   "feature_columns.pkl")

print("Saved:")
print(f"  ipl_ball_model.json   (booster sliced to best_iteration={model.best_iteration})")
print("  ipl_ball_model.pkl    (legacy — predictor.py prefers the .json above)")
print("  label_encoders.pkl")
print("  feature_columns.pkl")

# %% [markdown]
# ## Cell 13 — Download files

# %%
try:
    from google.colab import files
    files.download("ipl_ball_model.json")
    files.download("ipl_ball_model.pkl")
    files.download("label_encoders.pkl")
    files.download("feature_columns.pkl")
    print("Downloads triggered.")
except ImportError:
    print("Not in Colab — copy files manually from the Files panel.")

# %% [markdown]
# ## Cell 14 — Sanity test (Kohli vs Bumrah, death over, chasing)

# %%
import joblib, numpy as np, pandas as pd
import xgboost as xgb

booster_ck  = xgb.Booster()
booster_ck.load_model("ipl_ball_model.json")   # the best_iteration-sliced model
encoders_ck = joblib.load("label_encoders.pkl")
feat_cols   = joblib.load("feature_columns.pkl")

def encode_row(ctx: dict) -> dict:
    row = dict(ctx)
    for col in ["striker", "bowler", "batting_team", "bowling_team", "venue", "phase"]:
        if col not in feat_cols:
            continue
        le  = encoders_ck[col]
        val = row.get(col, "Unknown")
        if val not in le.classes_:
            val = "Unknown"
        row[col] = int(le.transform([val])[0])
    return row

test_ctx = {
    # categorical
    "striker":       "V Kohli",
    "bowler":        "JJ Bumrah",
    "batting_team":  "Royal Challengers Bengaluru",
    "bowling_team":  "Mumbai Indians",
    "venue":         "Wankhede Stadium",
    "phase":         "death",
    # match state
    "over_num": 18, "ball_num": 3,
    "cumulative_runs": 148, "cumulative_wickets": 3,
    "balls_remaining": 15, "wickets_remaining": 7,
    "crr": 9.87,
    # batter career stats (recency-weighted)
    "bat_rw_avg": 8.5, "bat_rw_sr": 138.0,
    "bat_rw_boundary_pct": 0.20, "bat_rw_six_pct": 0.08,
    "bat_rw_dot_pct": 0.31,
    "bat_pp_rw_sr": 130.0, "bat_mid_rw_sr": 140.0, "bat_death_rw_sr": 150.0,
    "bat_pp_rw_boundary_pct": 0.16, "bat_death_rw_boundary_pct": 0.24,
    # bowler career stats
    "bowl_rw_economy": 7.2, "bowl_rw_wicket_pct": 0.09,
    "bowl_rw_dot_pct": 0.38, "bowl_rw_boundary_pct": 0.12,
    "bowl_pp_rw_economy": 6.5, "bowl_mid_rw_economy": 7.0, "bowl_death_rw_economy": 8.1,
    "bowl_pp_rw_wicket_pct": 0.10, "bowl_death_rw_wicket_pct": 0.08,
    # player-at-venue (highest precedence tier)
    "bat_venue_adj_sr": 152.0, "bat_venue_adj_boundary_pct": 0.23, "bat_venue_rw_balls": 64.0,
    "bowl_venue_adj_economy": 8.6, "bowl_venue_adj_wicket_pct": 0.05, "bowl_venue_rw_balls": 48.0,
    # matchup
    "bvb_balls": 42, "bvb_rw_sr": 128.0, "bvb_rw_dismissal_pct": 0.07,
    "bvb_rw_dot_pct": 0.35, "bvb_rw_boundary_pct": 0.17, "bvb_rw_six_pct": 0.07,
    # momentum
    "batter_balls_faced": 34, "batter_runs_scored": 48, "batter_innings_sr": 141.2,
    "balls_vs_bowler": 6, "runs_vs_bowler": 8,
    "runs_last6": 11, "runs_last_over": 9,
    "consec_dots": 1, "consec_boundaries": 0,
    "partnership_runs": 42, "partnership_balls": 28,
    "prev_ball_outcome": 1, "prev2_ball_outcome": 0, "prev3_ball_outcome": 4,
    # venue
    "venue_rw_avg_1st_innings": 178.0, "venue_rw_avg_2nd_innings": 162.0,
    "venue_rw_boundary_pct": 0.19, "venue_rw_six_pct": 0.09,
    "venue_rw_dot_pct": 0.30, "venue_rw_wicket_pct": 0.055,
    "venue_rw_pp_sr": 145.0, "venue_rw_death_sr": 168.0,
    # context
    "is_batting_first": 0, "is_chasing": 1,
    "target": 182, "runs_needed": 34, "rrr": 13.6, "pressure_index": 3.73,
}

row = encode_row(test_ctx)
X_t = pd.DataFrame([row])
# keep only the columns the model knows
X_t = X_t.reindex(columns=feat_cols, fill_value=0)

probs  = booster_ck.predict(xgb.DMatrix(X_t, feature_names=feat_cols))[0]
labels = encoders_ck["outcome"].classes_

print("Ball outcome probabilities (Kohli vs Bumrah, death, chasing 34 off 15):")
for lbl, p in sorted(zip(labels, probs), key=lambda x: -x[1]):
    bar = "█" * int(p * 40)
    print(f"  {lbl:>3}  {p:.3f}  {bar}")