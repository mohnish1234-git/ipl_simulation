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
# ## Cell 3b — Drop deliveries for retired/inactive players
#
# Mirrors src/model/retirement_filter.py EXACTLY (same ACTIVE_WINDOW_SEASONS,
# same "active" definition, same mode="either" default) — inlined here rather
# than imported since this script is meant to run standalone in Colab with
# no repo checkout. If you ever change retirement_filter.py's constants or
# logic, update this cell to match, or training and the serving-time active-
# player list (used to validate a simulated squad and build the UI roster)
# will silently drift apart.
#
# "either" keeps a row if AT LEAST ONE side (striker or bowler) is still
# active — a retired batter's dismissal pattern against a currently-active
# bowler is still informative for that bowler, even though we'll never
# simulate that specific batter again. Only rows where BOTH sides are
# retired get dropped. Rationale for doing this at all: every split the
# tree spends learning fine-grained patterns for a player who can no longer
# be selected in a simulation is training capacity that isn't going toward
# players who WILL actually appear — this doesn't fix BvB sparsity (most
# head-to-head pairs are thin regardless of retirement status; that's what
# BVB_OTHER_SHRINK_K / BVB_DISMISSAL_SHRINK_K in feature_engineer.py are
# for), it's a separate, complementary improvement.

ACTIVE_WINDOW_SEASONS = 3   # keep in sync with retirement_filter.py

def _compute_active_players(df, window_seasons=ACTIVE_WINDOW_SEASONS):
    latest_season = df["season"].max()
    cutoff_season = latest_season - window_seasons + 1
    recent = df[df["season"] >= cutoff_season]
    return set(recent["striker"].dropna().unique()) | set(recent["bowler"].dropna().unique())

def _filter_to_active_players(df, active_players, mode="either"):
    striker_active = df["striker"].isin(active_players)
    bowler_active  = df["bowler"].isin(active_players)
    keep = (striker_active | bowler_active) if mode == "either" else (striker_active & bowler_active)
    before = len(df)
    out = df[keep].copy()
    print(f"Retirement filter ({mode}, last {ACTIVE_WINDOW_SEASONS} seasons): "
          f"{before - len(out):,} rows removed, {len(out):,} remaining ({len(active_players)} active players)")
    return out

active_players = _compute_active_players(df, window_seasons=ACTIVE_WINDOW_SEASONS)
df = _filter_to_active_players(df, active_players, mode="either")

# Save alongside the other model artifacts so prepare_data.py / the live
# simulator can load the SAME active-player set this model was trained
# against, instead of recomputing it separately from a possibly different
# snapshot of the data (that mismatch is exactly the kind of train/serve
# skew this whole pipeline has been bitten by before).
joblib.dump(sorted(active_players), "active_players.pkl")
print(f"Saved active_players.pkl ({len(active_players)} players)")

# %% [markdown]
# ## Cell 4 — Define feature groups
#
# All 7 feature groups are listed explicitly.
# XGBoost learns which ones matter most — nothing is hardcoded.

CATEGORICAL = [
    "striker",
    "bowler",
    "batting_team",
    "bowling_team",
    "venue",
    "phase",
]

NUMERICAL = [
    # ───────────────────────────────────────────────────────────────
    # Match State
    # ───────────────────────────────────────────────────────────────
    "over_num",
    "ball_num",
    "cumulative_runs",
    "cumulative_wickets",
    "balls_remaining",
    "wickets_remaining",
    "crr",

    # ───────────────────────────────────────────────────────────────
    # Batter Career
    # ───────────────────────────────────────────────────────────────
    "bat_rw_avg",
    "bat_rw_sr",
    "bat_rw_boundary_pct",
    "bat_rw_six_pct",
    "bat_rw_dot_pct",

    "bat_pp_rw_sr",
    "bat_mid_rw_sr",
    "bat_death_rw_sr",

    "bat_pp_rw_boundary_pct",
    "bat_mid_rw_boundary_pct",
    "bat_death_rw_boundary_pct",

    "bat_pp_rw_dot_pct",
    "bat_mid_rw_dot_pct",
    "bat_death_rw_dot_pct",

    # ───────────────────────────────────────────────────────────────
    # Bowler Career
    # ───────────────────────────────────────────────────────────────
    "bowl_rw_economy",
    "bowl_rw_wicket_pct",
    "bowl_rw_dot_pct",
    "bowl_rw_boundary_pct",

    "bowl_pp_rw_economy",
    "bowl_mid_rw_economy",
    "bowl_death_rw_economy",

    "bowl_pp_rw_wicket_pct",
    "bowl_mid_rw_wicket_pct",
    "bowl_death_rw_wicket_pct",

    "bowl_pp_rw_dot_pct",
    "bowl_mid_rw_dot_pct",
    "bowl_death_rw_dot_pct",

    "bowl_pp_rw_boundary_pct",
    "bowl_mid_rw_boundary_pct",
    "bowl_death_rw_boundary_pct",

    # ───────────────────────────────────────────────────────────────
    # Batter @ Venue
    # ───────────────────────────────────────────────────────────────
    "bat_venue_adj_sr",
    "bat_venue_adj_boundary_pct",
    "bat_venue_rw_balls",

    # ───────────────────────────────────────────────────────────────
    # Bowler @ Venue
    # ───────────────────────────────────────────────────────────────
    "bowl_venue_adj_economy",
    "bowl_venue_adj_wicket_pct",
    "bowl_venue_rw_balls",

    # ───────────────────────────────────────────────────────────────
    # Batter vs Bowler
    # ───────────────────────────────────────────────────────────────
    "bvb_balls",
    "bvb_rw_sr",
    "bvb_rw_dismissal_pct",
    "bvb_rw_dot_pct",
    "bvb_rw_boundary_pct",
    "bvb_rw_six_pct",

    # ───────────────────────────────────────────────────────────────
    # In-match Momentum
    # ───────────────────────────────────────────────────────────────
    "batter_balls_faced",
    "batter_runs_scored",
    "batter_innings_sr",

    "balls_vs_bowler",
    "runs_vs_bowler",

    "runs_last6",
    "runs_last12",
    "runs_last18",
    "runs_last_over",

    "consec_dots",
    "consec_boundaries",

    "partnership_runs",
    "partnership_balls",
    "partnership_run_rate",

    "current_matchup_sr",

    "prev_ball_outcome",
    "prev2_ball_outcome",
    "prev3_ball_outcome",

    # ───────────────────────────────────────────────────────────────
    # Venue Intelligence
    # ───────────────────────────────────────────────────────────────
    "venue_rw_avg_1st_innings",
    "venue_rw_avg_2nd_innings",

    "venue_rw_boundary_pct",
    "venue_rw_six_pct",
    "venue_rw_dot_pct",
    "venue_rw_wicket_pct",

    "venue_rw_pp_sr",
    "venue_rw_mid_sr",
    "venue_rw_death_sr",

    "venue_rw_pp_boundary_pct",
    "venue_rw_mid_boundary_pct",
    "venue_rw_death_boundary_pct",

    "venue_rw_pp_wicket_pct",
    "venue_rw_mid_wicket_pct",
    "venue_rw_death_wicket_pct",

    # ───────────────────────────────────────────────────────────────
    # Venue Over-Band Features
    # ───────────────────────────────────────────────────────────────
    "venue_rw_1_6_rr",
    "venue_rw_7_10_rr",
    "venue_rw_11_15_rr",
    "venue_rw_16_20_rr",

    "venue_rw_1_6_wicket_pct",
    "venue_rw_7_10_wicket_pct",
    "venue_rw_11_15_wicket_pct",
    "venue_rw_16_20_wicket_pct",

    # ───────────────────────────────────────────────────────────────
    # Batter @ Venue Over-Bands
    # ───────────────────────────────────────────────────────────────
    "bat_venue_1_6_sr",
    "bat_venue_7_10_sr",
    "bat_venue_11_15_sr",
    "bat_venue_16_20_sr",

    "bat_venue_1_6_avg",
    "bat_venue_7_10_avg",
    "bat_venue_11_15_avg",
    "bat_venue_16_20_avg",

    # ───────────────────────────────────────────────────────────────
    # Bowler @ Venue Over-Bands
    # ───────────────────────────────────────────────────────────────
    "bowl_venue_1_6_economy",
    "bowl_venue_7_10_economy",
    "bowl_venue_11_15_economy",
    "bowl_venue_16_20_economy",

    "bowl_venue_1_6_wicket_pct",
    "bowl_venue_7_10_wicket_pct",
    "bowl_venue_11_15_wicket_pct",
    "bowl_venue_16_20_wicket_pct",

    # ───────────────────────────────────────────────────────────────
    # Batter vs Bowler Over-Bands
    # ───────────────────────────────────────────────────────────────
    "bvb_1_6_sr",
    "bvb_7_10_sr",
    "bvb_11_15_sr",
    "bvb_16_20_sr",

    "bvb_1_6_avg",
    "bvb_7_10_avg",
    "bvb_11_15_avg",
    "bvb_16_20_avg",

    # ───────────────────────────────────────────────────────────────
    # Batting Context / Pressure
    # ───────────────────────────────────────────────────────────────
    "is_batting_first",
    "is_chasing",

    "target",
    "runs_needed",
    "rrr",
    "pressure_index",

    "required_runs_per_wicket",
    "balls_per_required_run",
    "pressure_weighted_rrr",
    "pressure_weighted_aggression",
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
    le.fit(list(df[col].unique()) + ["Unknown"])
    df[col] = le.transform(df[col])
    df[col] = df[col].astype("category")      # ← add this line
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
    enable_categorical=True, 
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
X_t = X_t.reindex(columns=feat_cols, fill_value=0)

# cast the same categorical columns back to category dtype so this
# matches what the booster was trained on
for col in ["striker", "bowler", "batting_team", "bowling_team", "venue", "phase"]:
    if col in X_t.columns:
        X_t[col] = X_t[col].astype("category")

probs  = booster_ck.predict(
    xgb.DMatrix(X_t, feature_names=feat_cols, enable_categorical=True)   # ← add enable_categorical=True here too
)[0]
labels = encoders_ck["outcome"].classes_

print("Ball outcome probabilities (Kohli vs Bumrah, death, chasing 34 off 15):")
for lbl, p in sorted(zip(labels, probs), key=lambda x: -x[1]):
    bar = "█" * int(p * 40)
    print(f"  {lbl:>3}  {p:.3f}  {bar}")