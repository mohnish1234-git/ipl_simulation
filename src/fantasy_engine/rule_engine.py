# rule_engine.py

import pandas as pd


##############################################################
# RULE ENGINE
##############################################################

class RuleEngine:

    def __init__(self):

        pass


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

        if not pd.notna(low) or not pd.notna(high) or high == low:

            # No spread to rank on (all-equal, all-NaN, or a pool of one)
            # — treat every player as equally scored rather than divide
            # by zero.
            return pd.Series(0.5, index=series.index)

        return ((series - low) / (high - low)).fillna(0.5)


    ##########################################################
    # MINIMUM MATCHES
    ##########################################################

    def filter_min_matches(

        self,

        dataframe,

        minimum_matches=5

    ):

        if "matches" not in dataframe.columns:

            return dataframe

        return dataframe.loc[

            dataframe["matches"] >= minimum_matches

        ].reset_index(drop=True)


    ##########################################################
    # REMOVE NEGATIVE
    ##########################################################

    def remove_negative_predictions(

        self,

        dataframe

    ):

        # high_performer_probability (the classifier's only output) is
        # always in [0, 1], so there's nothing negative to filter here
        # anymore — expected_points doesn't exist since the regression
        # model was removed. Kept as a no-op passthrough so callers that
        # still invoke this method don't break.

        return dataframe


    ##########################################################
    # APPLY WEIGHTS
    ##########################################################

    def combine_predictions(

        self,

        ml_dataframe,

        simulation_dataframe,

        classifier_weight=0.70,

        simulation_weight=0.30

    ):

        # Left join off the ML predictions (which cover every player the
        # feature pipeline built a row for) — an inner join would silently
        # drop any player the Monte Carlo simulator happened not to
        # produce points for in a given run, shrinking the pool optimizer.py
        # draws from before it ever sees it.
        dataframe = ml_dataframe.merge(

            simulation_dataframe,

            on="player",

            how="left"

        )

        if "simulation_points" in dataframe.columns:

            dataframe["simulation_points"] = dataframe["simulation_points"].fillna(0.0)

        else:

            dataframe["simulation_points"] = 0.0

        if "high_performer_probability" not in dataframe.columns:

            raise ValueError(

                "ml_dataframe must contain 'high_performer_probability' "
                "(produced by FantasyPredictor.predict_players()). "
                "'expected_points' no longer exists — there is no "
                "regression model in this pipeline anymore."

            )

        # high_performer_probability is already 0-1; simulation_points is
        # on a raw fantasy-points scale (runs + wickets*25, see
        # app.py:_fantasy_points_from_summary). Min-max normalize
        # simulation_points before blending so one component doesn't
        # swamp the other just because of unit scale.
        probability_score = dataframe["high_performer_probability"].astype(float).clip(0, 1).fillna(0.5)

        simulation_score = self._normalize(dataframe["simulation_points"])

        dataframe["final_points"] = (

            probability_score * classifier_weight

            +

            simulation_score * simulation_weight

        )

        return dataframe


    ##########################################################
    # SORT
    ##########################################################

    def sort_players(

        self,

        dataframe

    ):

        return dataframe.sort_values(

            "final_points",

            ascending=False

        ).reset_index(

            drop=True

        )


    ##########################################################
    # ATTACH ROSTER METADATA (role / team)
    #
    # role and team-name aren't in any historical-stats table — they
    # come from wherever the playing XI itself is sourced. Pass a
    # roster dataframe/list of dicts with at least "player", "role",
    # "team" columns and this merges them in; downstream (optimizer.py,
    # output_formatter.py) needs "role" to build a valid XI.
    ##########################################################

    def attach_roster(

        self,

        dataframe,

        roster

    ):

        if roster is None:

            return dataframe

        roster_df = roster if isinstance(roster, pd.DataFrame) else pd.DataFrame(roster)

        roster_columns = [column for column in ("player", "role", "team") if column in roster_df.columns]

        return dataframe.merge(

            roster_df[roster_columns],

            on="player",

            how="left"

        )


    ##########################################################
    # COMPLETE PIPELINE
    ##########################################################

    def apply(

        self,

        ml_dataframe,

        simulation_dataframe,

        roster=None

    ):

        dataframe = self.combine_predictions(

            ml_dataframe,

            simulation_dataframe

        )

        dataframe = self.attach_roster(

            dataframe,

            roster

        )

        dataframe = self.remove_negative_predictions(

            dataframe

        )

        dataframe = self.filter_min_matches(

            dataframe

        )

        dataframe = self.sort_players(

            dataframe

        )

        return dataframe