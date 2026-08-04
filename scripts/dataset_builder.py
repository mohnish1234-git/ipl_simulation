"""
Fantasy Dataset Builder
========================
Builds the training dataset for the Dream11 fantasy-points regression model.

Pipeline:
  1. Load raw ball-by-ball data and clean it with the EXISTING cleaning
     pipeline (cleaner.clean) — same cleaning the match simulator uses.
  2. Run the EXISTING feature_engineer.build_features() over the full
     cleaned history. That function is already strictly anti-leakage (every
     recency-weighted stat only ever looks at seasons STRICTLY BEFORE the
     row's own season — see feature_engineer.py's module docstring), so its
     bat_/bowl_/bvb_/venue_ columns are safe to reuse here verbatim.
  3. For every (match_id, player) pair, take exactly ONE pre-match feature
     snapshot — these historical columns are already constant across an
     entire match (they're keyed by season, not by ball), so any one row
     for that player in that match carries the correct value. LIVE
     match-state columns (cumulative_runs, crr, momentum, partnership,
     pressure_index, etc.) are explicitly dropped here — they don't exist
     before a ball has been bowled and must never leak into a pre-match
     prediction target.
  4. Independently compute each player's ACTUAL Dream11 fantasy points for
     that match from the cleaned ball-by-ball data (batting + bowling +
     fielding, standard Dream11 T20 scoring). This is the label, and is
     computed from the match's own outcome — it is never used as a feature.
  5. Join pre-match features (X) with actual points (y) on (match_id,
     player), validate, dedupe, fill any residual missing values, and
     export data/processed/fantasy_dataset.csv.

Usage:
    python -m scripts.dataset_builder
"""

from pathlib import Path
import numpy as np
import pandas as pd

from src.data.cleaner import load_raw, clean
import src.data.feature_engineer as fe

PROCESSED_DIR = Path("data/processed")
OUTPUT_PATH = PROCESSED_DIR / "fantasy_dataset.csv"

# ── Columns that describe LIVE, in-match state. These must NEVER be used as
# pre-match prediction features (a fantasy team is picked before a ball is
# bowled), so they're explicitly excluded when building the per-player
# feature snapshot, even though build_features() computes them.
LIVE_MATCH_STATE_COLS = {
    "over_num", "ball_num",
    "cumulative_runs", "cumulative_wickets",
    "balls_remaining", "wickets_remaining", "crr",
    "batter_balls_faced", "batter_runs_scored", "batter_innings_sr",
    "balls_vs_bowler", "runs_vs_bowler",
    "runs_last6", "runs_last12", "runs_last18", "runs_last_over",
    "consec_dots", "consec_boundaries",
    "partnership_runs", "partnership_balls", "partnership_run_rate",
    "current_matchup_sr",
    "prev_ball_outcome", "prev2_ball_outcome", "prev3_ball_outcome",
    "target", "runs_needed", "rrr", "pressure_index",
    "required_runs_per_wicket", "balls_per_required_run",
    "pressure_weighted_rrr", "pressure_weighted_aggression",
    "sample_weight", "outcome",
}

# Columns that are the same for every player in a match (context, not a
# per-player skill signal) — kept, since venue/team/phase-agnostic-season
# info is legitimate pre-match context, just not "live state".
MATCH_CONTEXT_COLS = ["match_id", "season", "date", "venue", "is_batting_first"]


# ═══════════════════════════════════════════════════════════════════════════
# Dream11-style fantasy points (standard T20 scoring)
# ═══════════════════════════════════════════════════════════════════════════

def _batting_points(runs: int, balls: int, fours: int, sixes: int, dismissed: bool) -> float:
    pts = runs * 1.0
    pts += fours * 1.0
    pts += sixes * 2.0
    if runs >= 100:
        pts += 16
    elif runs >= 50:
        pts += 8
    elif runs >= 30:
        pts += 4
    if dismissed and runs == 0:
        pts -= 2
    if balls >= 10:
        sr = runs / balls * 100
        if sr > 170:
            pts += 6
        elif sr >= 150.01:
            pts += 4
        elif sr >= 130:
            pts += 2
        elif sr <= 50 - 1e-9:
            pts -= 6
        elif sr <= 59.99:
            pts -= 4
        elif sr < 70:
            pts -= 2
    return pts


def _bowling_points(wickets: int, balls: int, runs_conceded: int, maidens: int) -> float:
    pts = wickets * 25.0
    if wickets >= 5:
        pts += 16
    elif wickets == 4:
        pts += 8
    pts += maidens * 12.0
    overs = balls / 6.0
    if overs >= 2:
        econ = runs_conceded / overs
        if econ < 5:
            pts += 6
        elif econ <= 5.99:
            pts += 4
        elif econ <= 7:
            pts += 2
        elif econ <= 10:
            pass
        elif econ <= 11:
            pts -= 2
        elif econ <= 12:
            pts -= 4
        else:
            pts -= 6
    return pts


def _compute_actual_fantasy_points(cleaned: pd.DataFrame) -> pd.DataFrame:
    """One row per (match_id, player) with the ACTUAL fantasy points earned
    in that match (batting + bowling [+ fielding if raw data supports it]).
    """
    rows = []

    # ── Batting ────────────────────────────────────────────────────────────
    bat = (
        cleaned[cleaned["is_legal"] == 1]
        .groupby(["match_id", "striker"])
        .agg(
            runs=("runs_of_bat", "sum"),
            balls=("is_legal", "sum"),
            fours=("is_four", "sum"),
            sixes=("is_six", "sum"),
        )
        .reset_index()
    )
    # dismissed = the striker was the one out on a ball charged against them.
    # wicket_type-based is_wicket doesn't tell us WHO was dismissed for
    # non-bowled-style dismissals (e.g. run-outs of the non-striker), but the
    # vast majority of wicket_type dismissals are of the striker on strike;
    # this is a reasonable approximation given the columns available upstream.
    dismissed = (
        cleaned[cleaned["is_wicket"] == 1]
        .groupby(["match_id", "striker"])
        .size()
        .rename("dismissed_count")
        .reset_index()
    )
    bat = bat.merge(dismissed, on=["match_id", "striker"], how="left")
    bat["dismissed_count"] = bat["dismissed_count"].fillna(0)

    for r in bat.itertuples(index=False):
        pts = _batting_points(r.runs, r.balls, r.fours, r.sixes, r.dismissed_count > 0)
        rows.append({"match_id": r.match_id, "player": r.striker, "batting_points": pts})
    bat_pts = pd.DataFrame(rows)

    # ── Bowling ────────────────────────────────────────────────────────────
    bowl_rows = []
    legal = cleaned[cleaned["is_legal"] == 1]
    bowl = legal.groupby(["match_id", "bowler"]).agg(
        balls=("is_legal", "sum"),
        runs_conceded=("total_runs", "sum"),
        wickets=("is_wicket", "sum"),
    ).reset_index()
    # Maidens: overs (6 legal balls bowled by the same bowler, same
    # match/innings/over_num) with zero runs conceded off the bat + extras.
    over_group = legal.groupby(["match_id", "innings", "bowler", "over_num"]).agg(
        legal_balls=("is_legal", "sum"),
        over_runs=("total_runs", "sum"),
    ).reset_index()
    maidens = (
        over_group[(over_group["legal_balls"] == 6) & (over_group["over_runs"] == 0)]
        .groupby(["match_id", "bowler"])
        .size()
        .rename("maidens")
        .reset_index()
    )
    bowl = bowl.merge(maidens, on=["match_id", "bowler"], how="left")
    bowl["maidens"] = bowl["maidens"].fillna(0)

    for r in bowl.itertuples(index=False):
        pts = _bowling_points(r.wickets, r.balls, r.runs_conceded, r.maidens)
        bowl_rows.append({"match_id": r.match_id, "player": r.bowler, "bowling_points": pts})
    bowl_pts = pd.DataFrame(bowl_rows)

    # ── Fielding ───────────────────────────────────────────────────────────
    # The upstream raw schema (see cleaner.py) exposes only `wicket_type`,
    # with no `fielder` / `player_dismissed` columns to attribute a catch,
    # stumping, or run-out to a specific fielder. Fielding points are
    # therefore NOT computable from this data and are set to 0 for every
    # player rather than guessed. If a `fielder` column is added upstream in
    # future, wire it in here (catch=+8, 3-catch bonus=+4, stumping=+12,
    # run-out direct=+12 / thrower+catcher=+6 each).
    print("  ⚠ Fielding points set to 0 for all players — raw data has no "
          "fielder/player_dismissed column to attribute catches/stumpings/"
          "run-outs to a specific player.")

    all_players = pd.concat([
        bat[["match_id", "striker"]].rename(columns={"striker": "player"}),
        bowl[["match_id", "bowler"]].rename(columns={"bowler": "player"}),
    ]).drop_duplicates()

    out = all_players.merge(bat_pts, on=["match_id", "player"], how="left")
    out = out.merge(bowl_pts, on=["match_id", "player"], how="left")
    out["batting_points"] = out["batting_points"].fillna(0)
    out["bowling_points"] = out["bowling_points"].fillna(0)
    out["fielding_points"] = 0.0
    out["fantasy_points"] = out["batting_points"] + out["bowling_points"] + out["fielding_points"]
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Opportunity / role features
# ═══════════════════════════════════════════════════════════════════════════
# Skill stats (SR, average, economy, etc.) only tell you how good a player
# is WHEN they get the ball. They say nothing about whether they'll actually
# get 40 balls to face or 4 overs to bowl — and that opportunity is the
# single biggest driver of raw fantasy points (a set batter who faces 10
# balls scores less than a mediocre one who faces 50). Everything below is
# built the same causal way as feature_engineer.py: sorted chronologically
# per player, with `.shift(1)` before every rolling/expanding aggregate, so
# a player's opportunity/form features for match N only ever see matches
# STRICTLY BEFORE match N — never the match itself.
#
# Two roles CANNOT be derived from this data and are called out rather than
# guessed: pace vs. spin (the raw schema has no bowler-style column) and
# wicketkeeper (no fielder/keeper column). `player_role` below is therefore
# limited to {batter, bowler, allrounder, unknown} based on batting/bowling
# involvement volume — see its docstring.

RECENT_FORM_WINDOWS = (5, 10)
POWERPLAY_OVERS = (0, 5)   # over_num 0-5 inclusive
DEATH_OVERS = (15, 19)     # over_num 15-19 inclusive
WORKLOAD_WINDOW_DAYS = 14  # rolling window (days) for the recent-workload/fatigue count


def _compute_match_involvement(cleaned: pd.DataFrame) -> pd.DataFrame:
    """One row per (match_id, player) describing what actually happened in
    that match — balls faced, overs bowled, batting position, phase usage,
    team run/wicket share. This is the RAW per-match involvement data that
    the causal rolling/expanding features below are built from; it is never
    used directly as a model feature itself (it's post-match, same as
    fantasy_points).
    """
    legal = cleaned[cleaned["is_legal"] == 1].copy()
    legal["is_powerplay"] = legal["over_num"].between(*POWERPLAY_OVERS).astype(int)
    legal["is_death"] = legal["over_num"].between(*DEATH_OVERS).astype(int)

    # ── Batting involvement ──────────────────────────────────────────────
    bat = legal.groupby(["match_id", "striker", "batting_team"]).agg(
        balls_faced=("is_legal", "sum"),
        runs_scored=("runs_of_bat", "sum"),
        pp_balls_faced=("is_powerplay", "sum"),
        death_balls_faced=("is_death", "sum"),
    ).reset_index().rename(columns={"striker": "player", "batting_team": "team"})

    # Batting position: order of first appearance at the crease within each
    # (match, innings), sorted chronologically — a genuine pre-ball-known
    # quantity once the match is underway, and highly stable match-to-match
    # for a given player, which is exactly why history of it is predictive.
    first_ball = (
        legal.sort_values(["match_id", "innings", "over_num", "ball_num"])
        .drop_duplicates(subset=["match_id", "innings", "striker"], keep="first")
        .copy()
    )
    first_ball["batting_position"] = (
        first_ball.groupby(["match_id", "innings"]).cumcount() + 1
    )
    bat = bat.merge(
        first_ball[["match_id", "striker", "batting_position"]].rename(columns={"striker": "player"}),
        on=["match_id", "player"], how="left",
    )

    # ── Bowling involvement ──────────────────────────────────────────────
    bowl = legal.groupby(["match_id", "bowler", "bowling_team"]).agg(
        balls_bowled=("is_legal", "sum"),
        wickets_taken=("is_wicket", "sum"),
        pp_balls_bowled=("is_powerplay", "sum"),
        death_balls_bowled=("is_death", "sum"),
    ).reset_index().rename(columns={"bowler": "player", "bowling_team": "team"})
    bowl["overs_bowled"] = bowl["balls_bowled"] / 6.0
    bowl["bowled_4_overs"] = (bowl["balls_bowled"] >= 24).astype(int)

    # ── Team totals (for run share / wicket share) ───────────────────────
    team_runs = legal.groupby(["match_id", "batting_team"])["total_runs"].sum().rename("team_runs").reset_index()
    team_wkts = legal.groupby(["match_id", "bowling_team"])["is_wicket"].sum().rename("team_wickets").reset_index()

    bat = bat.merge(team_runs, left_on=["match_id", "team"], right_on=["match_id", "batting_team"], how="left")
    bat["run_share"] = np.where(bat["team_runs"] > 0, bat["runs_scored"] / bat["team_runs"], 0.0)

    bowl = bowl.merge(team_wkts, left_on=["match_id", "team"], right_on=["match_id", "bowling_team"], how="left")
    bowl["wicket_share"] = np.where(bowl["team_wickets"] > 0, bowl["wickets_taken"] / bowl["team_wickets"], 0.0)

    involvement = bat[[
        "match_id", "player", "team", "balls_faced", "runs_scored", "batting_position",
        "pp_balls_faced", "death_balls_faced", "run_share",
    ]].merge(
        bowl[[
            "match_id", "player", "team", "balls_bowled", "overs_bowled", "wickets_taken",
            "bowled_4_overs", "pp_balls_bowled", "death_balls_bowled", "wicket_share",
        ]],
        on=["match_id", "player"], how="outer",
        suffixes=("_bat", "_bowl"),
    )
    # A player's team is the same whichever side (bat/bowl) it came from —
    # coalesce rather than pick one, since a pure bowler has no team_bat and
    # a pure batter has no team_bowl.
    involvement["team"] = involvement["team_bat"].combine_first(involvement["team_bowl"])
    involvement = involvement.drop(columns=["team_bat", "team_bowl"])

    num_cols = [c for c in involvement.columns if c not in ("match_id", "player", "team")]
    involvement[num_cols] = involvement[num_cols].fillna(0)
    return involvement


def _compute_opportunity_role_features(involvement: pd.DataFrame,
                                        actual_pts: pd.DataFrame,
                                        match_meta: pd.DataFrame) -> pd.DataFrame:
    """Turns raw per-match involvement + fantasy points into CAUSAL pre-match
    opportunity/role/form features, one row per (match_id, player). Every
    aggregate is `.shift(1)`-ed before the rolling/expanding window so it
    only reflects matches strictly before the one being featurized.
    """
    df = involvement.merge(actual_pts[["match_id", "player", "fantasy_points"]],
                            on=["match_id", "player"], how="left")
    df = df.merge(match_meta, on="match_id", how="left")
    df = df.sort_values(["player", "date", "match_id"]).reset_index(drop=True)

    g = df.groupby("player", sort=False)

    def causal_mean(col):
        return g[col].apply(lambda s: s.shift(1).expanding().mean()).reset_index(level=0, drop=True)

    def causal_rolling_mean(col, window):
        return g[col].apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean()).reset_index(level=0, drop=True)

    def causal_std(col):
        return g[col].apply(lambda s: s.shift(1).expanding().std()).reset_index(level=0, drop=True)

    def causal_max(col):
        return g[col].apply(lambda s: s.shift(1).expanding().max()).reset_index(level=0, drop=True)

    def causal_min(col):
        return g[col].apply(lambda s: s.shift(1).expanding().min()).reset_index(level=0, drop=True)

    # ── Opportunity: batting position / opening probability ──────────────
    df["avg_batting_position"] = causal_mean("batting_position")
    # Role/positional CONSISTENCY — a player nailed into one slot is a far
    # more predictable fantasy source than one whose role has been
    # shuffled recently; std is a genuinely different signal from the mean
    # above, not a duplicate of it.
    df["batting_position_std"] = causal_std("batting_position").fillna(0)
    df["_is_opener"] = (df["batting_position"] <= 2).astype(int)
    df["opening_probability"] = causal_mean("_is_opener")
    df.drop(columns=["_is_opener"], inplace=True)

    # ── Opportunity: workload ─────────────────────────────────────────────
    df["avg_balls_faced"] = causal_mean("balls_faced")
    df["avg_overs_bowled"] = causal_mean("overs_bowled")
    df["prob_bowling_4_overs"] = causal_mean("bowled_4_overs")

    # ── Opportunity: phase usage share (of the balls a player faced/bowled,
    # what fraction came in the powerplay / death overs — captures role
    # within the innings, e.g. death-overs finisher vs. top-order anchor) ──
    df["_pp_share_bat"] = np.where(df["balls_faced"] > 0, df["pp_balls_faced"] / df["balls_faced"], 0.0)
    df["_death_share_bat"] = np.where(df["balls_faced"] > 0, df["death_balls_faced"] / df["balls_faced"], 0.0)
    df["_pp_share_bowl"] = np.where(df["balls_bowled"] > 0, df["pp_balls_bowled"] / df["balls_bowled"], 0.0)
    df["_death_share_bowl"] = np.where(df["balls_bowled"] > 0, df["death_balls_bowled"] / df["balls_bowled"], 0.0)
    df["powerplay_usage_pct"] = causal_mean("_pp_share_bat") + causal_mean("_pp_share_bowl")
    df["death_overs_usage_pct"] = causal_mean("_death_share_bat") + causal_mean("_death_share_bowl")
    df.drop(columns=["_pp_share_bat", "_death_share_bat", "_pp_share_bowl", "_death_share_bowl"], inplace=True)

    # ── Team contribution ─────────────────────────────────────────────────
    df["avg_run_share"] = causal_mean("run_share")
    df["avg_wicket_share"] = causal_mean("wicket_share")

    # ── Recent fantasy form (last 5 / last 10 matches) ───────────────────
    for w in RECENT_FORM_WINDOWS:
        df[f"fantasy_form_last{w}"] = causal_rolling_mean("fantasy_points", w)

    # ── Fantasy consistency + overall history ─────────────────────────────
    df["career_avg_fantasy_points"] = causal_mean("fantasy_points")
    df["career_std_fantasy_points"] = causal_std("fantasy_points").fillna(0)
    df["career_ceiling_fantasy_points"] = causal_max("fantasy_points")
    df["career_floor_fantasy_points"] = causal_min("fantasy_points")
    df["career_matches_played"] = g.cumcount()

    # Most-recent-match points, kept SEPARATE from the smoothed rolling
    # averages above — a single recent big score/failure is real signal a
    # multi-match average dilutes away, and "hot/cold right now" is a
    # different question from "good on average."
    df["last_match_fantasy_points"] = g["fantasy_points"].shift(1)

    # Form trend: is the player currently running hotter or colder than
    # their own career average? (fantasy_form_last5 is assigned above in
    # the RECENT_FORM_WINDOWS loop, so it's already available here.)
    df["form_trend"] = df["fantasy_form_last5"] - df["career_avg_fantasy_points"]

    # ── Expected match involvement (composite workload share, 0-2) ────────
    df["expected_match_involvement"] = (
        df["avg_balls_faced"].fillna(0) / 20.0 + df["avg_overs_bowled"].fillna(0) / 4.0
    )

    # ── Workload / availability ─────────────────────────────────────────
    # Days since last match needs no shift() of its own: it's the gap
    # between THIS row's own match date and the player's PREVIOUS match
    # date, which by construction can only ever look backward. A player's
    # first-ever match has no previous date -> NaN -> filled with a
    # neutral "well rested" default (30 days) rather than 0, which would
    # misleadingly read as "played yesterday" for a debutant.
    df["days_since_last_match"] = g["date"].diff().dt.days
    df["days_since_last_match"] = df["days_since_last_match"].fillna(30.0)

    def _matches_in_prior_window(dates: pd.Series, window_days: int) -> pd.Series:
        vals = dates.values
        out = np.empty(len(vals), dtype=int)
        for i, d in enumerate(vals):
            window_start = d - np.timedelta64(window_days, "D")
            prior = vals[:i]
            out[i] = int(((prior >= window_start) & (prior < d)).sum())
        return pd.Series(out, index=dates.index)

    # Recent-workload / fatigue proxy: how many matches the player has
    # already played in the WORKLOAD_WINDOW_DAYS days strictly before this
    # match (current match itself and anything after it excluded by
    # construction — the window only ever looks at `prior`, i.e. dates
    # before index i).
    df[f"matches_last_{WORKLOAD_WINDOW_DAYS}_days"] = g["date"].apply(
        lambda s: _matches_in_prior_window(s, WORKLOAD_WINDOW_DAYS)
    ).reset_index(level=0, drop=True)

    # Expected workload in balls (batting + bowling combined) — a simpler,
    # more directly interpretable companion to expected_match_involvement's
    # normalized 0-2 composite score.
    df["expected_workload_balls"] = df["avg_balls_faced"].fillna(0) + df["avg_overs_bowled"].fillna(0) * 6.0

    # ── Rate stat + explicit projection formula ───────────────────────────
    # career_avg_fantasy_points blends "great player, rarely used" and
    # "average player, huge opportunity" into the same number — it can't
    # tell skill-per-chance apart from how many chances the player gets.
    # fantasy_points_per_involvement_ball is a genuine RATE (points per
    # ball of batting+bowling involvement, sum-of-points over sum-of-balls
    # across a player's causal history — not an average of per-match
    # ratios, which would let low-involvement matches dominate the number).
    # Multiplying that rate by expected_workload_balls turns "how good per
    # opportunity" and "how much opportunity" back into a single explicit
    # projected-points prior, instead of leaving the tree to rediscover
    # that multiplication from two separate columns on its own.
    df["_involvement_balls"] = df["balls_faced"] + df["balls_bowled"]

    def causal_rate(num_col, den_col):
        num_cum = g[num_col].apply(lambda s: s.shift(1).expanding().sum()).reset_index(level=0, drop=True)
        den_cum = g[den_col].apply(lambda s: s.shift(1).expanding().sum()).reset_index(level=0, drop=True)
        return np.where(den_cum > 0, num_cum / den_cum, np.nan)

    df["fantasy_points_per_involvement_ball"] = causal_rate("fantasy_points", "_involvement_balls")
    df.drop(columns=["_involvement_balls"], inplace=True)

    df["expected_fantasy_points_prior"] = (
        pd.Series(df["fantasy_points_per_involvement_ball"], index=df.index).fillna(0)
        * df["expected_workload_balls"]
    )

    # ── Role classification (batter / bowler / allrounder / unknown) ──────
    # Heuristic thresholds on CAUSAL prior totals (30 balls faced / 30 balls
    # bowled ≈ 5 overs) — deliberately cannot distinguish pace/spin bowlers
    # or flag wicketkeepers; the raw schema has no bowling-style or
    # fielder/keeper column to derive either from (see module note above).
    df["_prior_balls_faced_total"] = df.groupby("player")["balls_faced"].apply(
        lambda s: s.shift(1).expanding().sum()).reset_index(level=0, drop=True)
    df["_prior_balls_bowled_total"] = df.groupby("player")["balls_bowled"].apply(
        lambda s: s.shift(1).expanding().sum()).reset_index(level=0, drop=True)

    def _role(row):
        bat_hist = row["_prior_balls_faced_total"]
        bowl_hist = row["_prior_balls_bowled_total"]
        if pd.isna(bat_hist) or pd.isna(bowl_hist) or (bat_hist == 0 and bowl_hist == 0):
            return "unknown"
        if bat_hist >= 30 and bowl_hist >= 30:
            return "allrounder"
        if bowl_hist >= 30:
            return "bowler"
        if bat_hist >= 30:
            return "batter"
        return "unknown"

    df["player_role"] = df.apply(_role, axis=1)

    # ── Richer sub-role: opener / top-order / middle-order / finisher for
    # specialist batters, plus a split of "allrounder" into
    # batting-leaning vs. bowling-leaning. NOTE: pace-vs-spin and
    # wicketkeeper flags are NOT derivable from this data — same schema
    # limitation called out in dataset_builder.py's fielding-points
    # comment (no bowler-style or fielder/keeper column upstream) — so
    # player_sub_role stays limited to batting-order + allrounder-lean
    # roles. That's a real, if partial, upgrade over player_role above.
    def _sub_role(row):
        bat_hist = row["_prior_balls_faced_total"]
        bowl_hist = row["_prior_balls_bowled_total"]
        if pd.isna(bat_hist) or pd.isna(bowl_hist) or (bat_hist == 0 and bowl_hist == 0):
            return "unknown"
        if bat_hist >= 30 and bowl_hist >= 30:
            return "batting_allrounder" if bat_hist >= bowl_hist else "bowling_allrounder"
        if bowl_hist >= 30:
            return "bowler"
        if bat_hist >= 30:
            pos = row["avg_batting_position"]
            if pd.isna(pos):
                return "batter"
            if pos <= 2:
                return "opener"
            if pos <= 4:
                return "top_order"
            if pos <= 6:
                return "middle_order"
            return "finisher"
        return "unknown"

    df["player_sub_role"] = df.apply(_sub_role, axis=1)
    df.drop(columns=["_prior_balls_faced_total", "_prior_balls_bowled_total"], inplace=True)

    opportunity_cols = [
        "match_id", "player",
        "avg_batting_position", "batting_position_std", "opening_probability",
        "avg_balls_faced", "avg_overs_bowled", "prob_bowling_4_overs",
        "powerplay_usage_pct", "death_overs_usage_pct",
        "avg_run_share", "avg_wicket_share",
        *[f"fantasy_form_last{w}" for w in RECENT_FORM_WINDOWS],
        "career_avg_fantasy_points", "career_std_fantasy_points",
        "career_ceiling_fantasy_points", "career_floor_fantasy_points",
        "career_matches_played", "last_match_fantasy_points", "form_trend",
        "expected_match_involvement", "player_role",
        "player_sub_role", "days_since_last_match",
        f"matches_last_{WORKLOAD_WINDOW_DAYS}_days", "expected_workload_balls",
        "fantasy_points_per_involvement_ball", "expected_fantasy_points_prior",
    ]
    return df[opportunity_cols]


# ═══════════════════════════════════════════════════════════════════════════
# Team context / opposition-strength / venue-familiarity features
# ═══════════════════════════════════════════════════════════════════════════
# Everything above describes a player in isolation. It says nothing about
# (a) how strong the player's OWN team is around them — a set batter in a
# deep, powerful lineup gets fewer opportunities than the same batter as
# their team's one reliable scorer, and a bowler on a side short of good
# bowling options bowls more overs by default — or (b) how strong the
# OPPOSITION is — the same skill nets more runs against a weak attack and
# fewer wickets against a deep batting lineup, which the existing bvb_*
# batter-vs-bowler columns don't capture at the team level. Built the same
# causal way as everything else here: per (match, team) summaries, sorted
# chronologically, `.shift(1)` before any rolling/expanding aggregate, so a
# team's strength/venue-familiarity entering match N only reflects matches
# STRICTLY BEFORE match N.

TEAM_FORM_WINDOW = 10              # rolling window (matches) for team strength
BATTING_DEPTH_BALL_THRESHOLD = 6   # min legal balls faced to count as a
                                    # "contributing" batter for depth purposes


def _compute_team_match_stats(cleaned: pd.DataFrame, match_meta: pd.DataFrame) -> pd.DataFrame:
    """One row per (match_id, team): that team's own output IN that match —
    runs scored, wickets lost batting, runs conceded, wickets taken
    bowling, batting depth (# batters who reached
    BATTING_DEPTH_BALL_THRESHOLD legal balls faced), bowling options (#
    distinct bowlers used). This is POST-match, same as fantasy_points —
    never used directly as a model feature, only as the input to the
    causal rolling team-strength features below.
    """
    legal = cleaned[cleaned["is_legal"] == 1]

    bat_side = legal.groupby(["match_id", "batting_team"]).agg(
        runs_scored=("total_runs", "sum"),
        wickets_lost=("is_wicket", "sum"),
    ).reset_index().rename(columns={"batting_team": "team"})

    per_batter_balls = (
        legal.groupby(["match_id", "batting_team", "striker"])["is_legal"]
        .sum().reset_index(name="balls_faced")
    )
    per_batter_balls["contributed"] = (
        per_batter_balls["balls_faced"] >= BATTING_DEPTH_BALL_THRESHOLD
    ).astype(int)
    depth = (
        per_batter_balls.groupby(["match_id", "batting_team"])["contributed"]
        .sum().reset_index(name="batting_depth")
        .rename(columns={"batting_team": "team"})
    )

    bowl_side = legal.groupby(["match_id", "bowling_team"]).agg(
        runs_conceded=("total_runs", "sum"),
        wickets_taken=("is_wicket", "sum"),
    ).reset_index().rename(columns={"bowling_team": "team"})

    options = (
        legal.groupby(["match_id", "bowling_team"])["bowler"]
        .nunique().reset_index(name="bowling_options")
        .rename(columns={"bowling_team": "team"})
    )

    team_stats = bat_side.merge(depth, on=["match_id", "team"], how="left")
    team_stats = team_stats.merge(bowl_side, on=["match_id", "team"], how="outer")
    team_stats = team_stats.merge(options, on=["match_id", "team"], how="outer")
    team_stats = team_stats.merge(match_meta, on="match_id", how="left")

    num_cols = ["runs_scored", "wickets_lost", "batting_depth",
                "runs_conceded", "wickets_taken", "bowling_options"]
    team_stats[num_cols] = team_stats[num_cols].fillna(0)
    return team_stats


def _compute_team_strength_and_venue_features(team_stats: pd.DataFrame) -> pd.DataFrame:
    """Causal, per (match_id, team) team-strength + venue-familiarity
    features.

    Team "home venue" isn't available as raw data (no team->city mapping
    upstream, and the allowed-venue list mixes traditional home grounds
    with newer neutral venues), so it's estimated causally from playing
    history: at any point in time, a team's estimated home venue is
    whichever venue it has played at MOST OFTEN SO FAR.
    `is_estimated_home_venue` flags whether the CURRENT match's venue
    matches that running estimate. This is a heuristic derived from the
    data, not verified ground truth — called out explicitly rather than
    silently presented as a real home/away field.
    """
    df = team_stats.sort_values(["team", "date", "match_id"]).reset_index(drop=True)
    g = df.groupby("team", sort=False)

    def causal_rolling_mean(col, window=TEAM_FORM_WINDOW):
        return g[col].apply(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        ).reset_index(level=0, drop=True)

    df["team_batting_strength"] = causal_rolling_mean("runs_scored")
    # Lower = a STRONGER bowling unit (fewer runs conceded on average) —
    # called out since, unlike every other "_strength" column here, higher
    # is not better for this one.
    df["team_bowling_strength"] = causal_rolling_mean("runs_conceded")
    df["team_batting_depth"] = causal_rolling_mean("batting_depth")
    df["team_bowling_options"] = causal_rolling_mean("bowling_options")

    # Venue familiarity: how many matches this team has ALREADY played at
    # this exact venue. cumcount() is 0 on a team's first-ever match at a
    # venue — inherently prior-only, no extra shift needed.
    df["team_venue_matches_played"] = df.groupby(["team", "venue"]).cumcount()

    # Estimated home venue: an explicit running per-team venue counter, so
    # "current" match is never included in its own estimate.
    home_flags = pd.Series(0, index=df.index)
    for _, idx in df.groupby("team", sort=False).groups.items():
        venue_counts = {}
        for i in idx:
            v = df.at[i, "venue"]
            home_venue = max(venue_counts, key=venue_counts.get) if venue_counts else None
            home_flags.at[i] = int(v == home_venue)
            venue_counts[v] = venue_counts.get(v, 0) + 1
    df["is_estimated_home_venue"] = home_flags

    keep = ["match_id", "team", "team_batting_strength", "team_bowling_strength",
            "team_batting_depth", "team_bowling_options",
            "team_venue_matches_played", "is_estimated_home_venue"]
    return df[keep]


def _build_player_opponent_map(involvement: pd.DataFrame,
                                team_features: pd.DataFrame) -> pd.DataFrame:
    """One row per (match_id, player): the player's own team plus the
    OPPOSING team for that match. Shared by _attach_team_and_opposition_
    features and _compute_matchup_history_features so the "which team did
    this player face" logic exists in exactly one place.
    """
    player_team = involvement[["match_id", "player", "team"]].drop_duplicates(
        subset=["match_id", "player"]
    )

    teams_per_match = (
        team_features[["match_id", "team"]]
        .drop_duplicates()
        .groupby("match_id")["team"]
        .apply(list)
        .to_dict()
    )

    def opponent_of(row):
        teams = teams_per_match.get(row["match_id"], [])
        others = [t for t in teams if t != row["team"]]
        return others[0] if others else None

    player_team = player_team.copy()
    player_team["opponent_team"] = player_team.apply(opponent_of, axis=1)
    return player_team


def _attach_team_and_opposition_features(involvement: pd.DataFrame,
                                          team_features: pd.DataFrame) -> pd.DataFrame:
    """One row per (match_id, player): the player's OWN team's strength /
    venue-familiarity features, plus the OPPOSING team's — a batter's
    opposition context is the team's bowling strength/options they face; a
    bowler's opposition context is the team's batting strength/depth they
    bowl at. Captures matchup context beyond individual batter-vs-bowler
    history (bvb_* columns), which only ever sees one bowler/batter at a
    time, not the whole XI's overall strength.
    """
    player_team = _build_player_opponent_map(involvement, team_features)

    own_cols = {c: f"own_{c}" for c in team_features.columns if c not in ("match_id", "team")}
    opp_cols = {c: f"opp_{c}" for c in team_features.columns if c not in ("match_id", "team")}

    out = player_team.merge(
        team_features.rename(columns=own_cols),
        on=["match_id", "team"], how="left",
    )
    out = out.merge(
        team_features.rename(columns={**opp_cols, "team": "opponent_team"}),
        on=["match_id", "opponent_team"], how="left",
    )
    out = out.drop(columns=["team", "opponent_team"])

    num_cols = [c for c in out.columns if c not in ("match_id", "player")]
    out[num_cols] = out[num_cols].fillna(0)
    return out


# Shrinkage constant (in PRIOR matches vs this specific opponent) for the
# head-to-head fantasy-points feature below. With this many prior meetings,
# the raw head-to-head average sits halfway between itself and the
# player's overall career average; below that it leans toward the career
# number, since a handful of matches against one specific opponent is a
# noisy sample on its own (same K/(K+n) shrinkage style used throughout
# feature_engineer.py for the same reason).
OPPONENT_HISTORY_SHRINK_K = 5.0


def _compute_matchup_history_features(involvement: pd.DataFrame,
                                       actual_pts: pd.DataFrame,
                                       match_meta: pd.DataFrame,
                                       team_features: pd.DataFrame,
                                       opportunity: pd.DataFrame) -> pd.DataFrame:
    """One row per (match_id, player): this player's own causal history of
    fantasy points SPECIFICALLY against the upcoming opponent — different
    from opp_team_batting_strength/opp_team_bowling_strength (the
    opponent's general quality), this is "how has THIS player personally
    done against THIS opponent before." Shrunk toward the player's overall
    career average at low sample size (see OPPONENT_HISTORY_SHRINK_K)
    since most player/opponent pairings only have a handful of meetings.
    """
    df = _build_player_opponent_map(involvement, team_features)
    df = df.merge(actual_pts[["match_id", "player", "fantasy_points"]],
                   on=["match_id", "player"], how="left")
    df = df.merge(match_meta, on="match_id", how="left")
    df = df.merge(opportunity[["match_id", "player", "career_avg_fantasy_points"]],
                   on=["match_id", "player"], how="left")
    df = df.sort_values(["player", "opponent_team", "date", "match_id"]).reset_index(drop=True)

    g2 = df.groupby(["player", "opponent_team"], sort=False)
    # cumcount() is inherently prior-only (0 on the first-ever meeting) —
    # no extra shift needed, same reasoning as team_venue_matches_played.
    df["matches_vs_opponent_played"] = g2.cumcount()
    raw_avg_vs_opp = g2["fantasy_points"].apply(
        lambda s: s.shift(1).expanding().mean()
    ).reset_index(level=[0, 1], drop=True)

    n = df["matches_vs_opponent_played"].astype(float)
    weight = n / (n + OPPONENT_HISTORY_SHRINK_K)
    prior = df["career_avg_fantasy_points"].fillna(0)
    df["fantasy_points_vs_opponent"] = np.where(
        n > 0,
        raw_avg_vs_opp.fillna(prior) * weight + prior * (1 - weight),
        prior,
    )

    return df[["match_id", "player", "fantasy_points_vs_opponent", "matches_vs_opponent_played"]]


# ═══════════════════════════════════════════════════════════════════════════
# Cricket-specific interaction features
# ═══════════════════════════════════════════════════════════════════════════

def _add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """A handful of explicit interaction terms. XGBoost can in principle
    learn feature interactions on its own via successive tree splits, but
    handing it a well-chosen product directly makes a known domain
    interaction available in ONE split instead of requiring the tree to
    rediscover a multi-level split pattern from a finite, noisy sample.
    Cheap to add; guarded with fillna(0) since this runs before
    _validate_and_clean's own NaN pass.
    """
    def safe_mul(a, b):
        return df[a].fillna(0) * df[b].fillna(0)

    # Batting opportunity (balls likely faced) × underlying strike rate —
    # fantasy upside needs BOTH a chance to bat and the skill to score
    # fast once in, not either alone.
    if {"avg_balls_faced", "bat_rw_sr"}.issubset(df.columns):
        df["batting_opportunity_x_sr"] = safe_mul("avg_balls_faced", "bat_rw_sr")

    # Bowling workload (overs likely to bowl) × wicket-taking ability — a
    # strike bowler who only gets 1 over has a capped wicket ceiling; a
    # containment bowler who bowls all 4 overs still won't rack up wickets.
    if {"avg_overs_bowled", "bowl_rw_wicket_pct"}.issubset(df.columns):
        df["bowling_workload_x_wicket_pct"] = safe_mul("avg_overs_bowled", "bowl_rw_wicket_pct")

    # Opening probability × how much this venue historically favors 1st
    # innings scoring — an opener's fantasy ceiling is much higher at a
    # high-scoring venue than a slow, low-scoring one.
    if {"opening_probability", "venue_rw_avg_1st_innings"}.issubset(df.columns):
        df["opening_prob_x_venue_scoring"] = safe_mul("opening_probability", "venue_rw_avg_1st_innings")

    return df


# ═══════════════════════════════════════════════════════════════════════════
# Pre-match feature snapshot (one row per player per match)
# ═══════════════════════════════════════════════════════════════════════════

def _build_pre_match_features(features: pd.DataFrame) -> pd.DataFrame:
    """features = full output of feature_engineer.build_features(). Collapses
    it to one row per (match_id, player) WITHOUT splitting into separate
    batter/bowler tables and merging them back together — that split
    approach threw away every specialist's other-side columns (a pure
    batter has no bowler row -> all bowl_* NaN, and vice versa), which both
    inflated NaN counts and quietly dropped ~half the engineered columns
    from the final dataset.

    Instead: a player's involvement in a match — as striker OR as bowler —
    is unioned into one long table, sorted chronologically, and the FIRST
    row for that (match_id, player) is kept as-is, with only the live
    match-state columns stripped out. Since every non-live engineered
    column here (bat_/bowl_/bvb_/venue_) is causal and keyed by season (see
    feature_engineer.py), it's already constant across the whole match, so
    taking the first appearance loses nothing and needs no merge.
    """
    # Sort chronologically so "first row" == the player's earliest
    # involvement in the match, before any live-state column is dropped.
    sort_cols = [c for c in ["match_id", "innings", "over_num", "ball_num"] if c in features.columns]
    feat_sorted = features.sort_values(sort_cols).reset_index(drop=True)

    as_striker = feat_sorted.assign(player=feat_sorted["striker"])
    as_bowler = feat_sorted.assign(player=feat_sorted["bowler"])
    long = pd.concat([as_striker, as_bowler], ignore_index=True)

    pre_match = long.drop_duplicates(subset=["match_id", "player"], keep="first")

    keep_cols = ["player"] + [c for c in pre_match.columns
                               if c not in LIVE_MATCH_STATE_COLS and c not in ("striker", "bowler", "player")]
    pre_match = pre_match[keep_cols].reset_index(drop=True)
    return pre_match


# ═══════════════════════════════════════════════════════════════════════════
# Validation / cleanup
# ═══════════════════════════════════════════════════════════════════════════

def _validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["match_id", "player"]).reset_index(drop=True)
    print(f"  Deduped: {before:,} -> {len(df):,} rows")

    # A player row with no batting AND no bowling history at all (both sides
    # of the outer-merge were empty, e.g. a fielder-only substitute with no
    # recorded ball-by-ball involvement) can't be featurized -> drop.
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in
                    ("match_id", "season", "batting_points", "bowling_points",
                     "fielding_points", "fantasy_points", "career_matches_played")]

    all_na_mask = df[numeric_cols].isna().all(axis=1) if numeric_cols else pd.Series(False, index=df.index)
    dropped = int(all_na_mask.sum())
    if dropped:
        print(f"  Dropping {dropped:,} rows with zero engineered feature history")
    df = df[~all_na_mask].reset_index(drop=True)

    # Median-fill any remaining sparse NaNs in NUMERIC columns (a player's
    # first-ever match, a brand-new venue, etc.) — same philosophy as
    # feature_engineer's own final NaN safety net.
    for col in numeric_cols:
        n_missing = int(df[col].isna().sum())
        if n_missing:
            median = df[col].median()
            if pd.isna(median):
                median = 0.0
            df[col] = df[col].fillna(median)

    # The numeric fill above skips object/string columns entirely (venue,
    # batting_team, bowling_team, phase, date, etc.) — select_dtypes(number)
    # never selects them, so any NaN in those columns previously survived
    # untouched all the way to the exported CSV. Fill those here too.
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols and c != "fantasy_points"]
    for col in non_numeric_cols:
        n_missing = int(df[col].isna().sum())
        if not n_missing:
            continue
        if col == "date":
            df[col] = df[col].fillna(method="ffill")
        else:
            df[col] = df[col].fillna("Unknown")

    df["fantasy_points"] = df["fantasy_points"].fillna(0)

    total_na = int(df.isna().sum().sum())
    print(f"  Total remaining NaN after cleanup: {total_na:,}")
    if total_na:
        print(f"  ⚠ NaN still present in: \n{df.isna().sum()[df.isna().sum() > 0]}")

    return df


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def build_dataset() -> pd.DataFrame:
    print("Step 1: loading + cleaning raw data …")
    raw = load_raw()
    cleaned = clean(raw)

    print("\nStep 2: engineering historical features (reusing feature_engineer.py) …")
    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
    features = fe.build_features(cleaned)

    print("\nStep 3: building pre-match player feature snapshots …")
    pre_match = _build_pre_match_features(features)
    print(f"  {len(pre_match):,} (match, player) pre-match feature rows")

    print("\nStep 4: computing actual Dream11 fantasy points per player per match …")
    actual_pts = _compute_actual_fantasy_points(cleaned)
    print(f"  {len(actual_pts):,} (match, player) actual-points rows")

    print("\nStep 5: computing opportunity / role / recent-form features …")
    involvement = _compute_match_involvement(cleaned)
    match_meta = cleaned[["match_id", "date", "season", "venue"]].drop_duplicates(subset=["match_id"])
    opportunity = _compute_opportunity_role_features(involvement, actual_pts, match_meta[["match_id", "date"]])
    print(f"  {len(opportunity):,} (match, player) opportunity-feature rows")

    print("\nStep 6: computing team-strength / opposition-strength / venue-familiarity features …")
    team_stats = _compute_team_match_stats(cleaned, match_meta)
    team_features = _compute_team_strength_and_venue_features(team_stats)
    team_opposition = _attach_team_and_opposition_features(involvement, team_features)
    print(f"  {len(team_opposition):,} (match, player) team/opposition-feature rows")

    print("\nStep 6b: computing player-vs-opponent head-to-head fantasy history …")
    matchup_history = _compute_matchup_history_features(
        involvement, actual_pts, match_meta[["match_id", "date"]], team_features, opportunity
    )
    print(f"  {len(matchup_history):,} (match, player) matchup-history rows")

    print("\nStep 7: joining skill features (X) + opportunity features (X) + team/opposition "
          "features (X) + matchup-history features (X) + actual points (y) …")
    dataset = pre_match.merge(actual_pts, on=["match_id", "player"], how="inner")
    dataset = dataset.merge(opportunity, on=["match_id", "player"], how="left")
    dataset = dataset.merge(team_opposition, on=["match_id", "player"], how="left")
    dataset = dataset.merge(matchup_history, on=["match_id", "player"], how="left")
    print(f"  {len(dataset):,} joined rows")

    print("\nStep 8: adding cricket-specific interaction features …")
    dataset = _add_interaction_features(dataset)

    print("\nStep 9: validating + cleaning …")
    dataset = _validate_and_clean(dataset)

    print(f"\nFinal fantasy dataset shape: {dataset.shape}")
    print(f"Fantasy points distribution:\n{dataset['fantasy_points'].describe()}")
    return dataset


if __name__ == "__main__":
    dataset = build_dataset()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved → {OUTPUT_PATH}")