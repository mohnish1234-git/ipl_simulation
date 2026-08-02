# feature_builder.py

import os
import ast
import json
import logging
import operator

import pandas as pd
import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)

##############################################################
# DERIVED / ENGINEERED FEATURES
#
# Columns like expected_match_involvement, avg_run_share,
# prob_bowling_4_overs, etc. don't exist as raw columns in any skill
# table — they were computed at training time from raw stats. The
# real formulas should live in meta.derived_feature_formulas (JSON),
# keyed by output column name, as arithmetic expressions over any
# flattened raw feature name (e.g. "bat_rw_sr / 150"). These DEFAULTS
# are placeholders used only if meta doesn't define a formula for a
# given column — they exist so the pipeline never silently ships a
# zero-filled derived column, but they should be replaced with the
# exact training-time formulas for numerical parity with the model.
#
# career_avg_fantasy_points (and the rest of the fantasy-history /
# role / team-strength / workload columns) used to live here as a
# crude placeholder formula. They now have a real source — the
# PLAYER_SNAPSHOT_TABLE below — so they're no longer approximated.
##############################################################
DEFAULT_DERIVED_FORMULAS = {

    "expected_match_involvement": "(bat_rw_sr / 200) + (bowl_rw_wicket_pct)",
    "expected_workload_balls": "bat_venue_rw_balls + bowl_venue_rw_balls",
    "avg_run_share": "bat_rw_sr / 150",
    "avg_wicket_share": "bowl_rw_wicket_pct",
    "prob_bowling_4_overs": "1 - bowl_rw_dot_pct",

    # Mirror dataset_builder.py's _add_interaction_features exactly.
    # avg_balls_faced/avg_overs_bowled/opening_probability come from
    # the player's snapshot (SNAPSHOT_HISTORY_COLUMNS); bat_rw_sr/
    # bowl_rw_wicket_pct/venue_rw_avg_1st_innings come from the live
    # skill/venue tables — combining live signal with historical
    # opportunity, rather than reusing a stale snapshot value that
    # was computed for a different venue/opponent.
    "batting_opportunity_x_sr": "avg_balls_faced * bat_rw_sr",
    "bowling_workload_x_wicket_pct": "avg_overs_bowled * bowl_rw_wicket_pct",
    "opening_prob_x_venue_scoring": "opening_probability * venue_rw_avg_1st_innings",

}

##############################################################
# PLAYER SNAPSHOT TABLE
#
# fantasy_classifier_dataset holds one row per (match_id, player) as
# exported by scripts/fantasy_classifier_dataset.py — every column
# fantasy_classifier.pkl was trained on, including the categories
# that have no other source in this database: player_role,
# player_sub_role, fantasy-history/form, career stats, workload/
# recency, and team-strength. Per dataset_builder.py's own docstring,
# these are "constant across a match, keyed by season" — i.e. a real
# pre-match snapshot — so a player's MOST RECENT row is a valid
# stand-in for their current pre-match state ahead of a new match.
##############################################################
SNAPSHOT_TABLE = "fantasy_classifier_dataset"

# Columns pulled as-is from the player's latest snapshot row. These
# describe the player/their team's standing history, not the specific
# match being predicted — batting_team/bowling_team/venue/phase/
# innings/is_batting_first/is_chasing are NOT here on purpose: those
# describe the upcoming match and are always supplied fresh by the
# caller instead (see _build_match_context), never read from a past
# snapshot. fantasy_points_vs_opponent/matches_vs_opponent_played
# aren't here either — those are opponent-specific and are
# recomputed live against the ACTUAL upcoming opponent, not whichever
# opponent happened to appear in the player's last stored match.
SNAPSHOT_HISTORY_COLUMNS = [

    "player_role", "player_sub_role",
    "fantasy_form_last5", "fantasy_form_last10",
    "career_avg_fantasy_points", "career_std_fantasy_points",
    "career_ceiling_fantasy_points", "career_floor_fantasy_points",
    "career_matches_played", "last_match_fantasy_points", "form_trend",
    "matches_last_14_days",
    "expected_fantasy_points_prior", "fantasy_points_per_involvement_ball",
    "own_team_batting_strength", "own_team_bowling_strength",
    "own_team_batting_depth", "own_team_bowling_options",
    "own_team_venue_matches_played", "own_is_estimated_home_venue",

    # Opportunity / batting-order-usage columns — same situation as
    # the history columns above: no live table computes these, so
    # they come from the latest snapshot too. avg_run_share,
    # avg_wicket_share, prob_bowling_4_overs, expected_match_involvement,
    # and expected_workload_balls are NOT here on purpose — those stay
    # as DEFAULT_DERIVED_FORMULAS computed off the live skill/venue
    # tables above, which is more accurate for a new venue/opponent
    # than reusing a past snapshot's value.
    "modern_par_rr", "avg_batting_position", "batting_position_std",
    "opening_probability", "avg_balls_faced", "avg_overs_bowled",
    "powerplay_usage_pct", "death_overs_usage_pct",

]

# Leakage / identifier columns from the training CSV that must never
# be treated as a feature or copied into a live prediction row.
SNAPSHOT_EXCLUDED_COLUMNS = {

    "match_id", "date", "player",
    "is_high_performer", "fantasy_points",
    "batting_points", "bowling_points", "fielding_points",

}

_SAFE_BIN_OPS = {

    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,

}

_SAFE_UNARY_OPS = {ast.USub: operator.neg}


##############################################################
# FEATURE BUILDER
##############################################################

class FeatureBuilder:

    def __init__(self, dsn=None, classifier_feature_columns_path=None):

        self.dsn = dsn or self._build_dsn_from_env()

        self.conn = self._connect()

        self._batter_cache = {}
        self._bowler_cache = {}
        self._matchup_cache = {}
        self._venue_cache = {}
        self._batter_venue_cache = {}
        self._bowler_venue_cache = {}
        self._snapshot_cache = {}
        self._fantasy_vs_opponent_cache = {}

        self._meta = None
        self._tailender_defaults = None

        self.get_meta()
        self.get_tailender_defaults()

        # classifier_feature_columns.pkl (the exact schema
        # fantasy_classifier.pkl was trained on) is now the
        # authoritative column list/order — it's what the model
        # actually needs, and it's stricter than trusting meta to
        # have stayed in sync with it. meta-based lookups remain a
        # fallback for environments that haven't wired the path in
        # yet.
        self._feature_columns = (
            self._load_classifier_feature_columns(classifier_feature_columns_path)
            or self._resolve_feature_columns()
        )
        self._derived_formulas = self._resolve_derived_formulas()


    def _load_classifier_feature_columns(self, path):

        if not path or not os.path.exists(path):

            return None

        try:

            import joblib

            columns = joblib.load(path)
            return list(columns)

        except Exception as exc:

            logger.warning("Could not load classifier_feature_columns.pkl at %s: %s", path, exc)
            return None


    ##########################################################
    # CONNECTION HELPERS
    ##########################################################

    def _build_dsn_from_env(self):

        def env(db_key, pg_key, default):

            return os.environ.get(db_key, os.environ.get(pg_key, default))

        return {

            "host": env("DB_HOST", "PG_HOST", "localhost"),
            "port": env("DB_PORT", "PG_PORT", "5432"),
            "dbname": env("DB_NAME", "PG_DATABASE", "cricket"),
            "user": env("DB_USER", "PG_USER", "postgres"),
            "password": env("DB_PASSWORD", "PG_PASSWORD", ""),

        }


    def _connect(self):

        try:

            conn = psycopg2.connect(**self.dsn) if isinstance(self.dsn, dict) else psycopg2.connect(self.dsn)
            conn.autocommit = True

            return conn

        except Exception as exc:

            logger.error("Failed to connect to PostgreSQL: %s", exc)
            raise


    def _ensure_connection(self):

        try:

            if self.conn is None or self.conn.closed:

                self.conn = self._connect()

        except Exception as exc:

            logger.error("Failed to re-establish PostgreSQL connection: %s", exc)
            raise


    def _fetch_one(self, query, params):

        self._ensure_connection()

        try:

            with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:

                cur.execute(query, params)

                return cur.fetchone()

        except Exception as exc:

            logger.error("Query failed [%s] params=%s: %s", query, params, exc)
            return None


    def _fetch_all(self, query, params):

        self._ensure_connection()

        try:

            with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:

                cur.execute(query, params)

                return cur.fetchall()

        except Exception as exc:

            logger.error("Query failed [%s] params=%s: %s", query, params, exc)
            return []


    ##########################################################
    # META / DEFAULTS
    ##########################################################

    def get_meta(self):

        if self._meta is not None:

            return self._meta

        row = self._fetch_one("SELECT * FROM meta LIMIT 1", ())

        if not row:

            logger.warning("meta table returned no rows; using empty metadata")
            self._meta = {}

            return self._meta

        raw = row.get("meta") or row.get("data") or row.get("json") or row

        if isinstance(raw, str):

            try:

                raw = json.loads(raw)

            except Exception as exc:

                logger.error("Failed to parse meta JSON: %s", exc)
                raw = {}

        self._meta = raw or {}

        return self._meta


    def _resolve_feature_columns(self):

        for key in ("feature_columns", "regression_feature_columns", "columns"):

            columns = self._meta.get(key)

            if columns:

                return list(columns)

        logger.warning("meta has no feature_columns; falling back to natural flatten order")

        return None


    def _resolve_derived_formulas(self):

        formulas = dict(DEFAULT_DERIVED_FORMULAS)

        meta_formulas = self._meta.get("derived_feature_formulas") if isinstance(self._meta, dict) else None

        if meta_formulas:

            formulas.update(meta_formulas)

        else:

            logger.warning(

                "meta has no derived_feature_formulas; using placeholder defaults for %s",
                list(DEFAULT_DERIVED_FORMULAS.keys())

            )

        return formulas


    def _safe_eval(self, expr, variables):

        try:

            node = ast.parse(expr, mode="eval").body

            return self._eval_node(node, variables)

        except Exception as exc:

            logger.warning("Failed to evaluate derived feature expression '%s': %s", expr, exc)
            return 0.0


    def _eval_node(self, node, variables):

        if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_BIN_OPS:

            left = self._eval_node(node.left, variables)
            right = self._eval_node(node.right, variables)

            try:

                return _SAFE_BIN_OPS[type(node.op)](left, right)

            except ZeroDivisionError:

                return 0.0

        if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_UNARY_OPS:

            return _SAFE_UNARY_OPS[type(node.op)](self._eval_node(node.operand, variables))

        if isinstance(node, ast.Name):

            return self._to_number(variables.get(node.id, 0.0))

        if isinstance(node, ast.Constant):

            return node.value

        raise ValueError(f"unsupported expression node: {type(node).__name__}")


    def _compute_derived_features(self, flat):

        derived = {}

        for column, formula in self._derived_formulas.items():

            if column in flat:

                # Already supplied directly by a DB table — don't overwrite.
                continue

            derived[column] = self._safe_eval(formula, flat)

        return derived


    def get_tailender_defaults(self):

        if self._tailender_defaults is not None:

            return self._tailender_defaults

        row = self._fetch_one("SELECT * FROM tailender_default LIMIT 1", ())

        self._tailender_defaults = self._clean_row(row) if row else {}

        return self._tailender_defaults


    ##########################################################
    # ROW CLEANUP
    ##########################################################

    def _clean_row(self, row):

        if row is None:

            return {}

        return {key: value for key, value in dict(row).items()}


    def _neutral_matchup(self):

        return {

            "bvb_balls": 0,
            "bvb_rw_dismissal_pct": 0.0,
            "bvb_rw_sr": 0.0,
            "bvb_rw_dot_pct": 0.0,
            "bvb_rw_boundary_pct": 0.0,
            "bvb_rw_six_pct": 0.0,
            "bvb_1_6_sr": 0.0,
            "bvb_1_6_avg": 0.0,
            "bvb_7_10_sr": 0.0,
            "bvb_7_10_avg": 0.0,
            "bvb_11_15_sr": 0.0,
            "bvb_11_15_avg": 0.0,
            "bvb_16_20_sr": 0.0,
            "bvb_16_20_avg": 0.0,

        }


    ##########################################################
    # BATTER STATS
    ##########################################################

    def get_batter_stats(self, batter):

        if batter in self._batter_cache:

            return self._batter_cache[batter]

        row = self._fetch_one(

            "SELECT * FROM player_batter_stats WHERE player_name = %s",
            (batter,)

        )

        if not row:

            logger.info("No batting stats for %s; using tailender defaults", batter)
            stats = dict(self.get_tailender_defaults())
            stats["player_name"] = batter

        else:

            stats = self._clean_row(row)

        self._batter_cache[batter] = stats

        return stats


    ##########################################################
    # BOWLER STATS
    ##########################################################

    def get_bowler_stats(self, bowler):

        if bowler in self._bowler_cache:

            return self._bowler_cache[bowler]

        row = self._fetch_one(

            "SELECT * FROM player_bowler_stats WHERE player_name = %s",
            (bowler,)

        )

        stats = self._clean_row(row) if row else {"player_name": bowler}

        self._bowler_cache[bowler] = stats

        return stats


    ##########################################################
    # BATTER VS BOWLER (single pair)
    ##########################################################

    def _get_single_matchup(self, batter, bowler):

        key = (batter, bowler)

        if key in self._matchup_cache:

            return self._matchup_cache[key]

        row = self._fetch_one(

            "SELECT * FROM batter_vs_bowler_stats WHERE batter_name = %s AND bowler_name = %s",
            (batter, bowler)

        )

        stats = self._clean_row(row) if row else None

        self._matchup_cache[key] = stats

        return stats


    ##########################################################
    # MATCHUP AGGREGATION (batter vs a group of bowlers)
    ##########################################################

    def get_matchup_stats(self, batter, bowlers):

        if isinstance(bowlers, str):

            bowlers = [bowlers]

        rows = []

        for bowler in bowlers:

            matchup = self._get_single_matchup(batter, bowler)

            if matchup:

                rows.append(matchup)

        if not rows:

            return self._neutral_matchup()

        return self._weighted_average_matchups(rows)


    def _weighted_average_matchups(self, rows):

        numeric_keys = [

            key for key in rows[0].keys()

            if key not in ("batter_name", "bowler_name") and self._is_numeric(rows[0][key])

        ]

        total_balls = sum(self._to_number(row.get("bvb_balls", 0)) for row in rows)

        aggregated = {}

        for key in numeric_keys:

            if key == "bvb_balls":

                aggregated[key] = total_balls
                continue

            if total_balls > 0:

                weighted_sum = sum(

                    self._to_number(row.get(key, 0)) * self._to_number(row.get("bvb_balls", 0))
                    for row in rows

                )

                aggregated[key] = weighted_sum / total_balls

            else:

                values = [self._to_number(row.get(key, 0)) for row in rows]
                aggregated[key] = sum(values) / len(values) if values else 0.0

        return aggregated


    def _is_numeric(self, value):

        return isinstance(value, (int, float)) and not isinstance(value, bool)


    def _to_number(self, value):

        try:

            return float(value)

        except (TypeError, ValueError):

            return 0.0


    ##########################################################
    # VENUE STATS
    ##########################################################

    def get_venue_stats(self, venue):

        if venue in self._venue_cache:

            return self._venue_cache[venue]

        row = self._fetch_one(

            "SELECT * FROM venue_stats WHERE venue_name = %s",
            (venue,)

        )

        stats = self._clean_row(row) if row else {"venue_name": venue}

        self._venue_cache[venue] = stats

        return stats


    ##########################################################
    # BATTER-VENUE STATS
    ##########################################################

    def get_batter_venue_stats(self, batter, venue):

        key = (batter, venue)

        if key in self._batter_venue_cache:

            return self._batter_venue_cache[key]

        row = self._fetch_one(

            "SELECT * FROM batter_venue_stats WHERE player_name = %s AND venue_name = %s",
            (batter, venue)

        )

        if row:

            stats = self._clean_row(row)

        else:

            logger.info(

                "No batter-venue stats for %s at %s; falling back to player-level stats",
                batter, venue

            )

            stats = self._fallback_batter_venue(batter)

        self._batter_venue_cache[key] = stats

        return stats


    def _fallback_batter_venue(self, batter):

        player_stats = self.get_batter_stats(batter)

        return {

            "bat_venue_rw_balls": 0,
            "bat_venue_adj_sr": player_stats.get("bat_rw_sr", 0.0),
            "bat_venue_adj_boundary_pct": player_stats.get("bat_rw_boundary_pct", 0.0),
            "bat_venue_1_6_sr": player_stats.get("bat_1_6_rw_sr", 0.0),
            "bat_venue_1_6_avg": player_stats.get("bat_rw_avg", 0.0),
            "bat_venue_7_10_sr": player_stats.get("bat_7_10_rw_sr", 0.0),
            "bat_venue_7_10_avg": player_stats.get("bat_rw_avg", 0.0),
            "bat_venue_11_15_sr": player_stats.get("bat_11_15_rw_sr", 0.0),
            "bat_venue_11_15_avg": player_stats.get("bat_rw_avg", 0.0),
            "bat_venue_16_20_sr": player_stats.get("bat_16_20_rw_sr", 0.0),
            "bat_venue_16_20_avg": player_stats.get("bat_rw_avg", 0.0),

        }


    ##########################################################
    # BOWLER-VENUE STATS
    ##########################################################

    def get_bowler_venue_stats(self, bowler, venue):

        key = (bowler, venue)

        if key in self._bowler_venue_cache:

            return self._bowler_venue_cache[key]

        row = self._fetch_one(

            "SELECT * FROM bowler_venue_stats WHERE player_name = %s AND venue_name = %s",
            (bowler, venue)

        )

        if row:

            stats = self._clean_row(row)

        else:

            logger.info(

                "No bowler-venue stats for %s at %s; falling back to player-level stats",
                bowler, venue

            )

            stats = self._fallback_bowler_venue(bowler)

        self._bowler_venue_cache[key] = stats

        return stats


    def _fallback_bowler_venue(self, bowler):

        player_stats = self.get_bowler_stats(bowler)

        return {

            "bowl_venue_rw_balls": 0,
            "bowl_venue_adj_economy": player_stats.get("bowl_rw_economy", 0.0),
            "bowl_venue_adj_wicket_pct": player_stats.get("bowl_rw_wicket_pct", 0.0),
            "bowl_venue_1_6_economy": player_stats.get("bowl_1_6_rw_economy", 0.0),
            "bowl_venue_1_6_wicket_pct": player_stats.get("bowl_1_6_rw_wicket_pct", 0.0),
            "bowl_venue_7_10_economy": player_stats.get("bowl_7_10_rw_economy", 0.0),
            "bowl_venue_7_10_wicket_pct": player_stats.get("bowl_7_10_rw_wicket_pct", 0.0),
            "bowl_venue_11_15_economy": player_stats.get("bowl_11_15_rw_economy", 0.0),
            "bowl_venue_11_15_wicket_pct": player_stats.get("bowl_11_15_rw_wicket_pct", 0.0),
            "bowl_venue_16_20_economy": player_stats.get("bowl_16_20_rw_economy", 0.0),
            "bowl_venue_16_20_wicket_pct": player_stats.get("bowl_16_20_rw_wicket_pct", 0.0),

        }


    ##########################################################
    # BOWLER-GROUP AGGREGATION (opposition bowling attack)
    ##########################################################

    def _mean_of_dicts(self, dicts):

        if not dicts:

            return {}

        if len(dicts) == 1:

            return dict(dicts[0])

        numeric_keys = [

            key for key in dicts[0].keys()

            if self._is_numeric(dicts[0][key])

        ]

        aggregated = {}

        for key in numeric_keys:

            values = [self._to_number(entry.get(key, 0)) for entry in dicts]
            aggregated[key] = sum(values) / len(values) if values else 0.0

        return aggregated


    def _get_bowler_group_stats(self, bowlers):

        return self._mean_of_dicts([self.get_bowler_stats(bowler) for bowler in bowlers])


    def _get_bowler_group_venue_stats(self, bowlers, venue):

        return self._mean_of_dicts([self.get_bowler_venue_stats(bowler, venue) for bowler in bowlers])


    def _get_batter_group_stats(self, batters):

        return self._mean_of_dicts([self.get_batter_stats(batter) for batter in batters])


    ##########################################################
    # MATCHUP FROM THE BOWLER'S SIDE (same table, reversed role)
    ##########################################################

    def get_matchup_stats_as_bowler(self, bowler, batters):

        if isinstance(batters, str):

            batters = [batters]

        rows = [

            matchup for matchup in (

                self._get_single_matchup(batter, bowler) for batter in batters

            ) if matchup

        ]

        if not rows:

            return self._neutral_matchup()

        return self._weighted_average_matchups(rows)


    ##########################################################
    # PLAYER SNAPSHOT (fantasy_classifier_dataset)
    ##########################################################

    def _fetch_latest_snapshot_row(self, player):

        if player in self._snapshot_cache:

            return self._snapshot_cache[player]

        row = self._fetch_one(

            f"SELECT * FROM {SNAPSHOT_TABLE} WHERE player = %s ORDER BY date DESC LIMIT 1",
            (player,)

        )

        cleaned = self._clean_row(row) if row else None

        self._snapshot_cache[player] = cleaned

        return cleaned


    def get_player_history_features(self, player, match_date=None):

        """Player-level history/role/team-strength columns pulled from
        this player's most recent fantasy_classifier_dataset row.
        Match-context columns (venue/teams/phase/innings) are
        deliberately excluded — see SNAPSHOT_HISTORY_COLUMNS.
        """

        row = self._fetch_latest_snapshot_row(player)

        if not row:

            logger.info(

                "No fantasy_classifier_dataset history for %s; "
                "history/role/team-strength columns will default to 0.0",
                player

            )

            return {}

        history = {

            column: row.get(column)

            for column in SNAPSHOT_HISTORY_COLUMNS

            if column in row

        }

        # days_since_last_match in the stored snapshot is relative to
        # THAT past match, not the match we're predicting for — if we
        # know the upcoming match's date, recompute it relative to
        # that so it isn't stale.
        snapshot_date = row.get("date")

        if match_date is not None and snapshot_date is not None:

            try:

                delta_days = (pd.to_datetime(match_date) - pd.to_datetime(snapshot_date)).days
                history["days_since_last_match"] = max(float(delta_days), 0.0)

            except Exception as exc:

                logger.warning("Could not recompute days_since_last_match for %s: %s", player, exc)
                history["days_since_last_match"] = row.get("days_since_last_match", 30.0)

        else:

            history["days_since_last_match"] = row.get("days_since_last_match", 30.0)

        return history


    ##########################################################
    # OPPONENT TEAM STRENGTH (aggregated from the opposing XI's
    # own team-strength snapshot values)
    ##########################################################

    def get_opponent_team_strength(self, opponent_players):

        own_side_columns = [

            "own_team_batting_strength", "own_team_bowling_strength",
            "own_team_batting_depth", "own_team_bowling_options",
            "own_team_venue_matches_played", "own_is_estimated_home_venue",

        ]

        rows = []

        for player in opponent_players:

            row = self._fetch_latest_snapshot_row(player)

            if row:

                rows.append(row)

        if not rows:

            return {}

        aggregated = self._mean_of_dicts([

            {column: row.get(column, 0.0) for column in own_side_columns}
            for row in rows

        ])

        return {

            column.replace("own_team_", "opp_team_").replace("own_is_estimated_home_venue", "opp_is_estimated_home_venue"): value

            for column, value in aggregated.items()

        }


    ##########################################################
    # FANTASY HISTORY VS A SPECIFIC OPPONENT (recomputed live
    # against the ACTUAL upcoming opponent, not a stale snapshot)
    ##########################################################

    def get_fantasy_points_vs_opponent(self, player, opponent_team_name, before_date=None):

        if not opponent_team_name:

            return {"fantasy_points_vs_opponent": 0.0, "matches_vs_opponent_played": 0}

        cache_key = (player, opponent_team_name, before_date)

        if cache_key in self._fantasy_vs_opponent_cache:

            return self._fantasy_vs_opponent_cache[cache_key]

        query = (

            f"SELECT fantasy_points FROM {SNAPSHOT_TABLE} "
            "WHERE player = %s AND (batting_team = %s OR bowling_team = %s)"

        )

        params = [player, opponent_team_name, opponent_team_name]

        if before_date is not None:

            query += " AND date < %s"
            params.append(before_date)

        rows = self._fetch_all(query, tuple(params))

        points = [self._to_number(row.get("fantasy_points", 0.0)) for row in rows]

        result = {

            "fantasy_points_vs_opponent": (sum(points) / len(points)) if points else 0.0,
            "matches_vs_opponent_played": len(points),

        }

        self._fantasy_vs_opponent_cache[cache_key] = result

        return result


    ##########################################################
    # MATCH CONTEXT (never read from a stale snapshot — always
    # supplied fresh for the actual match being predicted)
    ##########################################################

    def _build_match_context(

        self,

        player,

        is_batting_first,

        venue,

        own_team_name,

        opponent_team_name,

        season,

        match_date,

        innings,

        phase

    ):

        resolved_season = season

        if resolved_season is None and match_date is not None:

            try:

                resolved_season = pd.to_datetime(match_date).year

            except Exception:

                resolved_season = None

        try:

            resolved_season = int(resolved_season) if resolved_season is not None else 0

        except (TypeError, ValueError):

            logger.warning("Non-numeric season %r supplied; defaulting to 0", resolved_season)
            resolved_season = 0

        return {

            "venue": venue,
            "batting_team": own_team_name if is_batting_first else opponent_team_name,
            "bowling_team": opponent_team_name if is_batting_first else own_team_name,
            "season": resolved_season,
            "innings": innings if innings is not None else (1 if is_batting_first else 2),
            "phase": phase or "pre_match",
            "is_batting_first": 1 if is_batting_first else 0,
            "is_chasing": 0 if is_batting_first else 1,

        }


    ##########################################################
    # BUILD ONE PLAYER (preserved public interface)
    ##########################################################

    def build_player_features(

        self,

        player,

        opponents,

        venue,

        is_batting_first=True,

        own_team_name=None,

        opponent_team_name=None,

        season=None,

        match_date=None,

        innings=None,

        phase=None

    ):

        opponents = list(opponents) if isinstance(opponents, (list, tuple, set)) else [opponents]

        return {

            "player": player,

            # Player's own ability, regardless of role — an allrounder/
            # bowler still needs their own batting numbers and vice versa,
            # since fantasy points come from both disciplines.
            "batter_stats": self.get_batter_stats(player),
            "bowler_stats": self.get_bowler_stats(player),

            # This player's own venue history.
            "batter_venue": self.get_batter_venue_stats(player, venue),
            "bowler_venue": self.get_bowler_venue_stats(player, venue),

            # Head-to-head, both directions: player batting vs the
            # opposing bowlers, and player bowling vs the opposing batters.
            "bvb": self.get_matchup_stats(player, opponents),
            "bvb_bowling": self.get_matchup_stats_as_bowler(player, opponents),

            # Aggregate strength of the opposition this player faces.
            "opposition_bowling": self._get_bowler_group_stats(opponents),
            "opposition_batting": self._get_batter_group_stats(opponents),

            "venue": self.get_venue_stats(venue),

            # Role / fantasy-form / career / team-strength history —
            # sourced from this player's latest fantasy_classifier_dataset
            # snapshot (see get_player_history_features).
            "history": self.get_player_history_features(player, match_date=match_date),

            # Opposing team's strength, aggregated live from the actual
            # opponents in this match — never from a stale snapshot.
            "opponent_strength": self.get_opponent_team_strength(opponents),

            # This player's fantasy history specifically against the
            # ACTUAL upcoming opponent team.
            "vs_opponent": self.get_fantasy_points_vs_opponent(

                player,
                opponent_team_name,
                before_date=match_date

            ),

            # The real context of the match being predicted — never
            # read from a past snapshot.
            "match_context": self._build_match_context(

                player,
                is_batting_first,
                venue,
                own_team_name,
                opponent_team_name,
                season,
                match_date,
                innings,
                phase

            ),

        }


    ##########################################################
    # BUILD MATCH FEATURES (preserved public interface)
    ##########################################################

    def build_match_features(

        self,

        batting_team,

        bowling_team,

        venue,

        batting_team_name=None,

        bowling_team_name=None,

        season=None,

        match_date=None,

        phase=None

    ):

        dataset = []

        # batting_team_name/bowling_team_name are the real team names
        # (e.g. "Mumbai Indians") used for the batting_team/bowling_team
        # categorical columns and for opponent-specific history lookups.
        # If not supplied, those columns fall back to "Unknown" — the
        # same fallback the training pipeline uses for unseen
        # categories — rather than silently mislabeling a team.
        batting_team_name = batting_team_name or "Unknown"
        bowling_team_name = bowling_team_name or "Unknown"

        # batting_team/bowling_team (the player-list args) are kept as
        # the two playing XIs for backward-compatible call signatures,
        # but every player on both sides gets a row — a recommendation
        # engine needs all 22 players ranked, not just the
        # "batting_team" argument.
        for player in batting_team:

            dataset.append(self.build_player_features(

                player, bowling_team, venue,

                is_batting_first=True,
                own_team_name=batting_team_name,
                opponent_team_name=bowling_team_name,
                season=season,
                match_date=match_date,
                phase=phase,

            ))

        for player in bowling_team:

            dataset.append(self.build_player_features(

                player, batting_team, venue,

                is_batting_first=False,
                own_team_name=bowling_team_name,
                opponent_team_name=batting_team_name,
                season=season,
                match_date=match_date,
                phase=phase,

            ))

        return dataset


    ##########################################################
    # FLATTEN ONE PLAYER'S NESTED FEATURES INTO A MODEL ROW
    ##########################################################

    # Sections use their raw DB column names as-is (already namespaced:
    # bat_, bowl_, bvb_, bat_venue_, bowl_venue_, venue_) except the two
    # reversed/aggregated views below, which reuse those same column
    # names for a different context and so need an explicit prefix to
    # avoid clobbering the player's own values.
    _PREFIXED_SECTIONS = {

        "bvb_bowling": "asbowler_",
        "opposition_bowling": "opp_bowl_",
        "opposition_batting": "opp_bat_",

    }

    _PLAIN_SECTIONS = (

        "batter_stats", "bowler_stats", "bvb", "batter_venue", "bowler_venue", "venue",
        "history", "opponent_strength", "vs_opponent", "match_context",

    )

    def _flatten_player_features(self, feature):

        flat = {}

        skip_keys = ("player_name", "venue_name", "batter_name", "bowler_name")

        for section in self._PLAIN_SECTIONS:

            for key, value in (feature.get(section) or {}).items():

                if key in skip_keys:

                    continue

                flat[key] = self._to_number(value) if self._is_numeric(value) else value

        for section, prefix in self._PREFIXED_SECTIONS.items():

            for key, value in (feature.get(section) or {}).items():

                if key in skip_keys:

                    continue

                flat[f"{prefix}{key}"] = self._to_number(value) if self._is_numeric(value) else value

        flat.update(self._compute_derived_features(flat))

        if self._feature_columns:

            ordered = {"player": feature["player"]}

            defaults = self._meta.get("defaults", {}) if isinstance(self._meta, dict) else {}

            for column in self._feature_columns:

                ordered[column] = flat.get(column, defaults.get(column, 0.0))

            return ordered

        flat_with_player = {"player": feature["player"]}
        flat_with_player.update(flat)

        return flat_with_player


    ##########################################################
    # BUILD DATAFRAME (preserved public interface)
    ##########################################################

    def build_dataframe(

        self,

        batting_team,

        bowling_team,

        venue,

        batting_team_name=None,

        bowling_team_name=None,

        season=None,

        match_date=None,

        phase=None

    ):

        features = self.build_match_features(

            batting_team,

            bowling_team,

            venue,

            batting_team_name=batting_team_name,

            bowling_team_name=bowling_team_name,

            season=season,

            match_date=match_date,

            phase=phase,

        )

        rows = [self._flatten_player_features(feature) for feature in features]

        return pd.DataFrame(rows)


    ##########################################################
    # CLEANUP
    ##########################################################

    def close(self):

        try:

            if self.conn and not self.conn.closed:

                self.conn.close()

        except Exception as exc:

            logger.warning("Error closing PostgreSQL connection: %s", exc)


    def __del__(self):

        try:

            self.close()

        except Exception:

            pass