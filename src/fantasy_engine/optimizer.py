# optimizer.py

import pandas as pd


##############################################################
# DREAM11 OPTIMIZER
##############################################################

class Dream11Optimizer:

    def __init__(self):

        pass


    ##########################################################
    # ROLE LIMITS
    ##########################################################

    def role_constraints(

        self,

        dataframe

    ):

        if "role" not in dataframe.columns:

            raise ValueError(

                "optimize() requires a 'role' column (WK/BAT/AR/BOWL) on the dataframe. "

                "Role isn't part of the historical-stats pipeline — attach it via "

                "RuleEngine.attach_roster() (or merge it in yourself) before calling optimize()."

            )

        wk = dataframe.loc[
            dataframe["role"] == "WK"
        ]

        bat = dataframe.loc[
            dataframe["role"] == "BAT"
        ]

        ar = dataframe.loc[
            dataframe["role"] == "AR"
        ]

        bowl = dataframe.loc[
            dataframe["role"] == "BOWL"
        ]

        return wk, bat, ar, bowl


    ##########################################################
    # SELECT TEAM
    ##########################################################

    MAX_PER_REAL_TEAM = 7

    def _pick(

        self,

        candidates,

        count,

        team_counts,

        selected_names

    ):

        picked = []

        has_team_column = "team" in candidates.columns

        for _, row in candidates.iterrows():

            if len(picked) >= count:

                break

            if row["player"] in selected_names:

                continue

            team_name = row["team"] if has_team_column else None

            if team_name is not None and team_counts.get(team_name, 0) >= self.MAX_PER_REAL_TEAM:

                continue

            picked.append(row.to_dict())
            selected_names.add(row["player"])

            if team_name is not None:

                team_counts[team_name] = team_counts.get(team_name, 0) + 1

        return picked


    def optimize(

        self,

        dataframe

    ):

        dataframe = dataframe.sort_values(

            "final_points",

            ascending=False

        )

        wk, bat, ar, bowl = self.role_constraints(

            dataframe

        )

        team = []
        selected_names = set()
        team_counts = {}

        ######################################################
        # 1 WK / 3 BAT / 2 AR / 4 BOWL, respecting the 7-per-
        # real-team cap (skipped only when no "team" column exists,
        # matching the earlier behavior for callers that don't have
        # roster metadata attached yet).
        ######################################################

        team.extend(self._pick(wk, 1, team_counts, selected_names))
        team.extend(self._pick(bat, 3, team_counts, selected_names))
        team.extend(self._pick(ar, 2, team_counts, selected_names))
        team.extend(self._pick(bowl, 4, team_counts, selected_names))

        ######################################################
        # REMAINING SLOT
        ######################################################

        if len(team) < 11:

            remaining = dataframe.loc[

                ~dataframe["player"].isin(selected_names)

            ]

            team.extend(

                self._pick(remaining, 11 - len(team), team_counts, selected_names)

            )

        return pd.DataFrame(team)


    ##########################################################
    # BENCH
    ##########################################################

    def bench(

        self,

        dataframe,

        selected_team

    ):

        return dataframe.loc[

            ~dataframe["player"].isin(

                selected_team["player"]

            )

        ].reset_index(

            drop=True

        )


    ##########################################################
    # COMPLETE
    ##########################################################

    def recommend(

        self,

        dataframe

    ):

        team = self.optimize(

            dataframe

        )

        bench = self.bench(

            dataframe,

            team

        )

        return {

            "team": team,

            "bench": bench

        }