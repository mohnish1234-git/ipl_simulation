# output_formatter.py

import pandas as pd


##############################################################
# OUTPUT FORMATTER
##############################################################

class OutputFormatter:

    def __init__(self):

        pass


    def _select_existing(

        self,

        dataframe,

        columns

    ):

        return dataframe[[column for column in columns if column in dataframe.columns]]


    ##########################################################
    # TEAM
    ##########################################################

    def team(

        self,

        dataframe

    ):

        columns = [

            "player",

            "role",

            "team",

            "high_performer_probability",

            "simulation_points",

            "final_points"

        ]

        return self._select_existing(dataframe, columns)


    ##########################################################
    # WITH TAGS
    ##########################################################

    def with_tags(

        self,

        dataframe

    ):

        columns = [

            "player",

            "role",

            "team",

            "tag",

            "high_performer_probability",

            "simulation_points",

            "final_points"

        ]

        return self._select_existing(dataframe, columns)


    ##########################################################
    # TOP PLAYERS
    ##########################################################

    def top_players(

        self,

        dataframe,

        n=20

    ):

        dataframe = dataframe.sort_values(

            "final_points",

            ascending=False

        )

        return dataframe.head(

            n

        )


    ##########################################################
    # EXPORT CSV
    ##########################################################

    def export_csv(

        self,

        dataframe,

        filename

    ):

        dataframe.to_csv(

            filename,

            index=False

        )

        return filename


    ##########################################################
    # EXPORT EXCEL
    ##########################################################

    def export_excel(

        self,

        dataframe,

        filename

    ):

        dataframe.to_excel(

            filename,

            index=False

        )

        return filename


    ##########################################################
    # SUMMARY
    ##########################################################

    def summary(

        self,

        dataframe

    ):

        print("=" * 70)

        print("Recommended Dream11 Team")

        print("=" * 70)

        print(

            dataframe.to_string(

                index=False

            )

        )

        print("=" * 70)

        print("Total Players :", len(dataframe))

        print("Average Points :", round(

            dataframe["final_points"].mean(),

            2

        ))

        print("Highest Points :", round(

            dataframe["final_points"].max(),

            2

        ))

        print("=" * 70)

        return dataframe