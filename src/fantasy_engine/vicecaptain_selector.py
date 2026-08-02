# vicecaptain_selector.py

import pandas as pd


##############################################################
# VICE CAPTAIN SELECTOR
##############################################################

class ViceCaptainSelector:

    def __init__(self):

        pass


    ##########################################################
    # VICE CAPTAIN
    ##########################################################

    def choose(

        self,

        dataframe

    ):

        dataframe = dataframe.sort_values(

            "final_points",

            ascending=False

        )

        return dataframe.iloc[1]


    ##########################################################
    # TOP K OPTIONS
    ##########################################################

    def top_candidates(

        self,

        dataframe,

        k=5

    ):

        dataframe = dataframe.sort_values(

            "final_points",

            ascending=False

        )

        return dataframe.iloc[1:k + 1]


    ##########################################################
    # PLAYER ONLY
    ##########################################################

    def player(

        self,

        dataframe

    ):

        return self.choose(

            dataframe

        )["player"]


    ##########################################################
    # POINTS
    ##########################################################

    def points(

        self,

        dataframe

    ):

        return self.choose(

            dataframe

        )["final_points"]


    ##########################################################
    # SUMMARY
    ##########################################################

    def summary(

        self,

        dataframe

    ):

        vice = self.choose(

            dataframe

        )

        return {

            "player": vice["player"],

            "high_performer_probability": vice["high_performer_probability"],

            "simulation_points": vice["simulation_points"],

            "final_points": vice["final_points"]

        }