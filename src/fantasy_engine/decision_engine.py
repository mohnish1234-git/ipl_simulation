# decision_engine.py

import numpy as np

import pandas as pd


##############################################################
# DECISION ENGINE
##############################################################
#
# Combines several INDEPENDENT, individually-normalized scoring
# components into a single "cricket intelligence score" per player,
# via configurable weights (no hardcoded blending logic). Each
# component is its own method (see the "COMPONENT SCORERS" section
# below) so any one of them can be tuned, replaced, or reweighted
# without touching the others or the surrounding pipeline.
#
# There is no regression model in this pipeline anymore — the old
# "regression" component (which scored expected_points) has been
# removed. classification (high_performer_probability) is now the
# only ML-model-derived component.
#
# Components:
#   classification  - classifier's high_performer_probability (already 0-1)
#   opportunity     - expected match involvement / batting & bowling workload
#   venue           - venue-suitability signal
#   matchup         - batter-vs-bowler matchup signal
#   role            - player role / sub-role signal
#   consistency     - consistency / reliability signal
#
# optimizer.py, captain_selector.py continue to consume ONLY the
# resulting "final_points" column, exactly as before — this class is
# the only place that changed.
##############################################################

class DecisionEngine:

    ##########################################################
    # Default weight for each component. Renormalized to sum to
    # 1.0 in __init__ so final_points always stays on a
    # comparable 0-1 scale no matter what weights are supplied.
    ##########################################################
    DEFAULT_COMPONENT_WEIGHTS = {

        "classification": 0.35,

        "opportunity":    0.20,

        "venue":          0.15,

        "matchup":        0.10,

        "role":           0.10,

        "consistency":    0.10,

    }

    ##########################################################
    # Column-name patterns each component looks for on the
    # ml_predictions dataframe (substring match against every
    # column name). Lets each component pick up whichever
    # engineered columns the feature pipeline currently exports
    # without this file hardcoding every exact column name.
    ##########################################################
    DEFAULT_COMPONENT_COLUMN_PATTERNS = {

        "opportunity": [

            "avg_balls_faced", "avg_overs_bowled", "opening_probability",

            "batting_opportunity_x_sr", "bowling_workload_x_wicket_pct",

            "involvement", "opportunity",

        ],

        "venue": [

            "venue_rw_", "venue_avg_", "opening_prob_x_venue_scoring",

        ],

        "matchup": [

            "bvb_",

        ],

        "role": [

            "player_role", "player_sub_role", "role_",

        ],

        "consistency": [

            "consistency", "reliability", "season_weight",

            "career_matches_played", "_std", "_variance",

        ],

    }

    # Ordered so ranking()/decide() always evaluate components in the
    # same, predictable sequence.
    COMPONENT_METHODS = {

        "classification": "_score_classification",

        "opportunity":    "_score_opportunity",

        "venue":          "_score_venue",

        "matchup":        "_score_matchup",

        "role":           "_score_role",

        "consistency":    "_score_consistency",

    }

    def __init__(

        self,

        rule_engine,

        component_weights=None,

        component_column_patterns=None

    ):

        self.rule_engine = rule_engine

        self.component_weights = self._normalize_weights(

            component_weights or self.DEFAULT_COMPONENT_WEIGHTS

        )

        self.component_column_patterns = (

            component_column_patterns

            if component_column_patterns is not None

            else self.DEFAULT_COMPONENT_COLUMN_PATTERNS

        )


    ##########################################################
    # WEIGHT NORMALIZATION
    ##########################################################

    def _normalize_weights(

        self,

        weights

    ):

        all_components = set(self.DEFAULT_COMPONENT_WEIGHTS) | set(weights)

        resolved = {

            component: float(weights.get(component, 0.0))

            for component in all_components

        }

        total = sum(resolved.values())

        if total <= 0:

            raise ValueError(

                "component_weights must sum to > 0"

            )

        return {

            component: value / total

            for component, value in resolved.items()

        }


    ##########################################################
    # GENERIC HELPERS
    ##########################################################

    def _normalize(

        self,

        series

    ):

        series = series.astype(float)

        low = series.min()

        high = series.max()

        if not np.isfinite(low) or not np.isfinite(high) or high == low:

            # No spread to rank on (all-equal, all-NaN, or a pool of one)
            # — treat every player as equally scored on this component
            # rather than dividing by zero.
            return pd.Series(0.5, index=series.index)

        return ((series - low) / (high - low)).fillna(0.5)


    def _columns_matching(

        self,

        dataframe,

        patterns

    ):

        matched = []

        for column in dataframe.columns:

            for pattern in patterns:

                if pattern in column:

                    matched.append(column)

                    break

        return matched


    def _blended_component_from_patterns(

        self,

        dataframe,

        component_name

    ):

        patterns = self.component_column_patterns.get(component_name, [])

        matched_columns = self._columns_matching(

            dataframe,

            patterns

        )

        numeric_columns = [

            column for column in matched_columns

            if pd.api.types.is_numeric_dtype(dataframe[column])

        ]

        if not numeric_columns:

            # None of this component's expected columns are present on
            # this dataframe (feature pipeline doesn't export them yet,
            # or this run genuinely has no signal for it) — fall back to
            # a neutral 0.5 for every player so decide() still produces a
            # complete score instead of raising or silently dropping the
            # component's configured weight.
            return pd.Series(0.5, index=dataframe.index)

        normalized_columns = pd.concat(

            [self._normalize(dataframe[column]) for column in numeric_columns],

            axis=1

        )

        return normalized_columns.mean(axis=1)


    ##########################################################
    # COMPONENT SCORERS — each returns a Series in [0, 1],
    # aligned to dataframe.index. Add/replace a component by
    # adding a method here and an entry in COMPONENT_METHODS —
    # nothing else in the pipeline needs to change.
    ##########################################################

    def _score_classification(

        self,

        dataframe

    ):

        if "high_performer_probability" not in dataframe.columns:

            return pd.Series(0.5, index=dataframe.index)

        # Already a 0-1 probability from FantasyPredictor's classifier —
        # no min-max normalization needed, unlike the other components.
        return dataframe["high_performer_probability"].astype(float).clip(0, 1).fillna(0.5)


    def _score_opportunity(

        self,

        dataframe

    ):

        return self._blended_component_from_patterns(

            dataframe,

            "opportunity"

        )


    def _score_venue(

        self,

        dataframe

    ):

        return self._blended_component_from_patterns(

            dataframe,

            "venue"

        )


    def _score_matchup(

        self,

        dataframe

    ):

        return self._blended_component_from_patterns(

            dataframe,

            "matchup"

        )


    def _score_role(

        self,

        dataframe

    ):

        return self._blended_component_from_patterns(

            dataframe,

            "role"

        )


    def _score_consistency(

        self,

        dataframe

    ):

        return self._blended_component_from_patterns(

            dataframe,

            "consistency"

        )


    ##########################################################
    # COMBINE COMPONENTS INTO final_points
    ##########################################################

    def _compute_component_scores(

        self,

        dataframe

    ):

        scores = {}

        for component, method_name in self.COMPONENT_METHODS.items():

            method = getattr(self, method_name)

            scores[component] = method(dataframe)

        return pd.DataFrame(scores, index=dataframe.index)


    def _combine_components(

        self,

        dataframe

    ):

        dataframe = dataframe.copy()

        component_scores = self._compute_component_scores(dataframe)

        # Keep every individual component score on the output — lets
        # anything downstream (debugging, tuning, display) see exactly
        # what drove a player's final score, not just the blended result.
        for component in component_scores.columns:

            dataframe[f"score_{component}"] = component_scores[component]

        weighted_score = sum(

            component_scores[component] * self.component_weights.get(component, 0.0)

            for component in component_scores.columns

        )

        dataframe["cricket_intelligence_score"] = weighted_score

        # 0-1 scale, same as every individual component — optimizer.py /
        # captain_selector.py only ever sort/rank on this column, so an
        # absolute "fantasy points" unit isn't required here.
        dataframe["final_points"] = weighted_score

        return dataframe


    ##########################################################
    # DECISION
    ##########################################################

    def decide(

        self,

        ml_predictions,

        simulation_predictions,

        roster=None

    ):

        dataframe = self.rule_engine.apply(

            ml_predictions,

            simulation_predictions,

            roster=roster

        )

        # rule_engine.apply() only knows about the columns it has always
        # produced (high_performer_probability, simulation_points,
        # final_points, ...). Every engineered signal the component scorers above read
        # — opportunity/venue/matchup/role/consistency columns, plus the
        # classifier's high_performer_probability — lives on
        # ml_predictions and needs to be re-attached here before scoring.
        extra_columns = [

            column for column in ml_predictions.columns

            if column not in dataframe.columns and column != "player"

        ]

        if extra_columns:

            dataframe = dataframe.merge(

                ml_predictions[["player"] + extra_columns],

                on="player",

                how="left"

            )

        dataframe = self._combine_components(dataframe)

        dataframe = dataframe.sort_values(

            "final_points",

            ascending=False

        ).reset_index(

            drop=True

        )

        return dataframe


    ##########################################################
    # TOP PLAYERS
    ##########################################################

    def top_players(

        self,

        ml_predictions,

        simulation_predictions,

        n=11

    ):

        dataframe = self.decide(

            ml_predictions,

            simulation_predictions

        )

        return dataframe.head(

            n

        )


    ##########################################################
    # PLAYER RANKING
    ##########################################################

    def ranking(

        self,

        ml_predictions,

        simulation_predictions

    ):

        dataframe = self.decide(

            ml_predictions,

            simulation_predictions

        )

        dataframe = dataframe.reset_index(

            drop=True

        )

        dataframe["rank"] = (

            dataframe.index + 1

        )

        cols = [

            "rank",

            "player",

            "simulation_points",

            "score_classification",

            "score_opportunity",

            "score_venue",

            "score_matchup",

            "score_role",

            "score_consistency",

            "cricket_intelligence_score",

            "final_points"

        ]

        cols = [column for column in cols if column in dataframe.columns]

        return dataframe[cols]


    ##########################################################
    # SINGLE PLAYER
    ##########################################################

    def player_score(

        self,

        player,

        ml_predictions,

        simulation_predictions

    ):

        dataframe = self.decide(

            ml_predictions,

            simulation_predictions

        )

        dataframe = dataframe.loc[

            dataframe["player"] == player

        ]

        if dataframe.empty:

            return None

        return dataframe.iloc[0]


    ##########################################################
    # EXPORT
    ##########################################################

    def export(

        self,

        ml_predictions,

        simulation_predictions,

        filename

    ):

        dataframe = self.decide(

            ml_predictions,

            simulation_predictions

        )

        dataframe.to_csv(

            filename,

            index=False

        )

        return filename