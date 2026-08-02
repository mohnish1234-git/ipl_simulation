# captain_selector.py

import pandas as pd


##############################################################
# CAPTAIN SELECTOR
##############################################################

class CaptainSelector:

    def __init__(self):

        pass


    ##########################################################
    # CAPTAIN
    ##########################################################

    def choose_captain(

        self,

        dataframe

    ):

        dataframe = dataframe.sort_values(

            "final_points",

            ascending=False

        )

        return dataframe.iloc[0]


    ##########################################################
    # VICE CAPTAIN
    ##########################################################

    def choose_vice_captain(

        self,

        dataframe

    ):

        dataframe = dataframe.sort_values(

            "final_points",

            ascending=False

        )

        return dataframe.iloc[1]


    ##########################################################
    # BOTH
    ##########################################################

    def select(

        self,

        dataframe

    ):

        captain = self.choose_captain(

            dataframe

        )

        vice_captain = self.choose_vice_captain(

            dataframe

        )

        return {

            "captain": captain,

            "vice_captain": vice_captain

        }


    ##########################################################
    # TAG TEAM
    ##########################################################

    def apply_tags(

        self,

        dataframe

    ):

        dataframe = dataframe.copy()

        dataframe["tag"] = ""

        captain = self.choose_captain(

            dataframe

        )["player"]

        vice = self.choose_vice_captain(

            dataframe

        )["player"]

        dataframe.loc[

            dataframe["player"] == captain,

            "tag"

        ] = "C"

        dataframe.loc[

            dataframe["player"] == vice,

            "tag"

        ] = "VC"

        return dataframe