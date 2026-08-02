# simulator_adapter.py

import pandas as pd


##############################################################
# SIMULATOR ADAPTER
##############################################################

class SimulatorAdapter:

    def __init__(

        self,

        simulator

    ):

        self.simulator = simulator


    ##########################################################
    # RUN ONE SIMULATION
    ##########################################################

    def simulate_match(

        self,

        batting_team,

        bowling_team,

        venue

    ):

        return self.simulator.simulate_match(

            batting_team,

            bowling_team,

            venue

        )


    ##########################################################
    # RUN MULTIPLE SIMULATIONS
    ##########################################################

    def simulate(

        self,

        batting_team,

        bowling_team,

        venue,

        simulations=100

    ):

        return self.simulator.simulate(

            batting_team,

            bowling_team,

            venue,

            simulations

        )


    ##########################################################
    # PLAYER FANTASY POINTS
    ##########################################################

    def get_player_points(

        self,

        batting_team,

        bowling_team,

        venue,

        simulations=100

    ):

        result = self.simulate(

            batting_team,

            bowling_team,

            venue,

            simulations

        )

        return result["player_points"]


    ##########################################################
    # TEAM SCORES
    ##########################################################

    def get_team_scores(

        self,

        batting_team,

        bowling_team,

        venue,

        simulations=100

    ):

        result = self.simulate(

            batting_team,

            bowling_team,

            venue,

            simulations

        )

        return result["team_scores"]


    ##########################################################
    # DATAFRAME
    ##########################################################

    def get_dataframe(

        self,

        batting_team,

        bowling_team,

        venue,

        simulations=100

    ):

        player_points = self.get_player_points(

            batting_team,

            bowling_team,

            venue,

            simulations

        )

        rows = []

        for player, points in player_points.items():

            rows.append(

                {

                    "player": player,

                    "simulation_points": points

                }

            )

        return pd.DataFrame(rows)