# feature_store.py

from .queries import (
    get_player_batting,
    get_player_bowling,
    get_batter_venue,
    get_bowler_venue,
    get_batter_vs_bowler,
    get_venue_features,
    get_tailender_default
)


##############################################################
# SAFE LOOKUPS
##############################################################

def safe_player_batting(player):

    data = get_player_batting(player)

    if data is None:

        return None

    return data


def safe_player_bowling(player):

    data = get_player_bowling(player)

    if data is None:

        return None

    return data


def safe_batter_venue(player, venue):

    data = get_batter_venue(player, venue)

    if data is None:

        return None

    return data


def safe_bowler_venue(player, venue):

    data = get_bowler_venue(player, venue)

    if data is None:

        return None

    return data


def safe_bvb(batter, bowler):

    data = get_batter_vs_bowler(batter, bowler)

    if data is None:

        return None

    return data


##############################################################
# FALLBACKS
##############################################################

def get_tailender_features():

    return get_tailender_default()


def get_default_batter(player, venue):

    batter = safe_player_batting(player)

    if batter is not None:

        return batter

    return get_tailender_features()


def get_default_bowler(player):

    bowler = safe_player_bowling(player)

    if bowler is not None:

        return bowler

    return None


##############################################################
# COMPLETE FEATURE STORE
##############################################################

def build_feature_store(

    batter,
    bowler,
    venue

):

    return {

        "batter_stats": get_default_batter(
            batter,
            venue
        ),

        "bowler_stats": get_default_bowler(
            bowler
        ),

        "batter_venue": safe_batter_venue(
            batter,
            venue
        ),

        "bowler_venue": safe_bowler_venue(
            bowler,
            venue
        ),

        "bvb": safe_bvb(
            batter,
            bowler
        ),

        "venue": get_venue_features(
            venue
        )

    }
    ##############################################################
# FEATURE VECTOR BUILDER
##############################################################

def build_model_input(

    batter,
    bowler,
    venue

):

    features = build_feature_store(

        batter,
        bowler,
        venue

    )

    model_input = {}

    ##########################################################
    # BATTER
    ##########################################################

    model_input["batter_stats"] = features["batter_stats"]

    ##########################################################
    # BOWLER
    ##########################################################

    model_input["bowler_stats"] = features["bowler_stats"]

    ##########################################################
    # BATTER VENUE
    ##########################################################

    model_input["batter_venue"] = features["batter_venue"]

    ##########################################################
    # BOWLER VENUE
    ##########################################################

    model_input["bowler_venue"] = features["bowler_venue"]

    ##########################################################
    # BATTER VS BOWLER
    ##########################################################

    model_input["bvb"] = features["bvb"]

    ##########################################################
    # VENUE
    ##########################################################

    model_input["venue"] = features["venue"]

    return model_input


##############################################################
# TEAM FEATURES
##############################################################

def build_team_features(

    players,
    venue

):

    team_features = {}

    for player in players:

        team_features[player] = {

            "batting": safe_player_batting(player),

            "bowling": safe_player_bowling(player),

            "batting_venue": safe_batter_venue(

                player,
                venue

            ),

            "bowling_venue": safe_bowler_venue(

                player,
                venue

            )

        }

    return team_features


##############################################################
# MATCH FEATURE STORE
##############################################################

def build_match_feature_store(

    batting_team,

    bowling_team,

    venue

):

    return {

        "batting_team": build_team_features(

            batting_team,

            venue

        ),

        "bowling_team": build_team_features(

            bowling_team,

            venue

        ),

        "venue": get_venue_features(

            venue

        )

    }
    ##############################################################
# MERGED FEATURE VECTOR
##############################################################

def merge_features(

    batter,
    bowler,
    venue

):

    feature_store = build_feature_store(

        batter,
        bowler,
        venue

    )

    merged = {

        "batter": batter,

        "bowler": bowler,

        "venue": venue

    }

    merged["batting_stats"] = feature_store["batter_stats"]

    merged["bowling_stats"] = feature_store["bowler_stats"]

    merged["batter_venue"] = feature_store["batter_venue"]

    merged["bowler_venue"] = feature_store["bowler_venue"]

    merged["bvb"] = feature_store["bvb"]

    merged["venue_stats"] = feature_store["venue"]

    return merged


##############################################################
# FEATURE VALIDATION
##############################################################

def validate_player(player):

    return (

        safe_player_batting(player) is not None

        or

        safe_player_bowling(player) is not None

    )


def validate_venue(venue):

    return get_venue_features(venue) is not None


##############################################################
# MATCH VALIDATION
##############################################################

def validate_match(

    batting_team,

    bowling_team,

    venue

):

    if not validate_venue(venue):

        return False

    for player in batting_team:

        if not validate_player(player):

            return False

    for player in bowling_team:

        if not validate_player(player):

            return False

    return True


##############################################################
# DEBUG
##############################################################

if __name__ == "__main__":

    batter = "Virat Kohli"

    bowler = "Jasprit Bumrah"

    venue = "Wankhede Stadium"

    print(

        build_feature_store(

            batter,

            bowler,

            venue

        )

    )
    