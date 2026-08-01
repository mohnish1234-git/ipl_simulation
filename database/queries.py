# queries.py

from .db import get_connection, return_connection


##############################################################
# PLAYER FEATURES
##############################################################

def get_player_batting(player_name):

    conn = get_connection()

    cur = conn.cursor()

    cur.execute(

        """

        SELECT *

        FROM player_batter_stats

        WHERE player_name=%s

        """,

        (player_name,)

    )

    row = cur.fetchone()

    cur.close()

    return_connection(conn)

    return row


def get_player_bowling(player_name):

    conn = get_connection()

    cur = conn.cursor()

    cur.execute(

        """

        SELECT *

        FROM player_bowler_stats

        WHERE player_name=%s

        """,

        (player_name,)

    )

    row = cur.fetchone()

    cur.close()

    return_connection(conn)

    return row


##############################################################
# VENUE FEATURES
##############################################################

def get_venue_features(venue):

    conn = get_connection()

    cur = conn.cursor()

    cur.execute(

        """

        SELECT *

        FROM venue_stats

        WHERE venue_name=%s

        """,

        (venue,)

    )

    row = cur.fetchone()

    cur.close()

    return_connection(conn)

    return row


##############################################################
# BATTER VENUE
##############################################################

def get_batter_venue(player, venue):

    conn = get_connection()

    cur = conn.cursor()

    cur.execute(

        """

        SELECT *

        FROM batter_venue_stats

        WHERE player_name=%s

        AND venue_name=%s

        """,

        (player, venue)

    )

    row = cur.fetchone()

    cur.close()

    return_connection(conn)

    return row


##############################################################
# BOWLER VENUE
##############################################################

def get_bowler_venue(player, venue):

    conn = get_connection()

    cur = conn.cursor()

    cur.execute(

        """

        SELECT *

        FROM bowler_venue_stats

        WHERE player_name=%s

        AND venue_name=%s

        """,

        (player, venue)

    )

    row = cur.fetchone()

    cur.close()

    return_connection(conn)

    return row


##############################################################
# BATTER VS BOWLER
##############################################################

def get_batter_vs_bowler(batter, bowler):

    conn = get_connection()

    cur = conn.cursor()

    cur.execute(

        """

        SELECT *

        FROM batter_vs_bowler_stats

        WHERE batter=%s

        AND bowler=%s

        """,

        (batter, bowler)

    )

    row = cur.fetchone()

    cur.close()

    return_connection(conn)

    return row
    ##############################################################
# META
##############################################################

def get_meta(feature_name):

    conn = get_connection()

    cur = conn.cursor()

    cur.execute(

        """

        SELECT feature_value

        FROM meta

        WHERE feature_name=%s

        """,

        (feature_name,)

    )

    row = cur.fetchone()

    cur.close()

    return_connection(conn)

    if row is None:

        return None

    return row[0]


##############################################################
# TAILENDER DEFAULT
##############################################################

def get_tailender_default():

    conn = get_connection()

    cur = conn.cursor()

    cur.execute(

        """

        SELECT *

        FROM tailender_default

        LIMIT 1

        """

    )

    row = cur.fetchone()

    cur.close()

    return_connection(conn)

    return row


##############################################################
# PLAYER EXISTENCE
##############################################################

def player_exists(player_name):

    conn = get_connection()

    cur = conn.cursor()

    cur.execute(

        """

        SELECT EXISTS(

            SELECT 1

            FROM player_batter_stats

            WHERE player_name=%s

        )

        """,

        (player_name,)

    )

    exists = cur.fetchone()[0]

    cur.close()

    return_connection(conn)

    return exists


##############################################################
# LIST ALL PLAYERS
##############################################################

def get_all_players():

    conn = get_connection()

    cur = conn.cursor()

    cur.execute(

        """

        SELECT player_name

        FROM player_batter_stats

        ORDER BY player_name

        """

    )

    players = [row[0] for row in cur.fetchall()]

    cur.close()

    return_connection(conn)

    return players


##############################################################
# LIST ALL VENUES
##############################################################

def get_all_venues():

    conn = get_connection()

    cur = conn.cursor()

    cur.execute(

        """

        SELECT venue_name

        FROM venue_stats

        ORDER BY venue_name

        """

    )

    venues = [row[0] for row in cur.fetchall()]

    cur.close()

    return_connection(conn)

    return venues
    ##############################################################
# SEARCH HELPERS
##############################################################

def get_players_by_prefix(prefix):

    conn = get_connection()

    cur = conn.cursor()

    cur.execute(

        """

        SELECT player_name

        FROM player_batter_stats

        WHERE player_name ILIKE %s

        ORDER BY player_name

        """,

        (prefix + "%",)

    )

    players = [row[0] for row in cur.fetchall()]

    cur.close()

    return_connection(conn)

    return players


def get_venues_by_prefix(prefix):

    conn = get_connection()

    cur = conn.cursor()

    cur.execute(

        """

        SELECT venue_name

        FROM venue_stats

        WHERE venue_name ILIKE %s

        ORDER BY venue_name

        """,

        (prefix + "%",)

    )

    venues = [row[0] for row in cur.fetchall()]

    cur.close()

    return_connection(conn)

    return venues


##############################################################
# FEATURE STORE
##############################################################

def get_complete_player_features(player, venue):

    features = {}

    batting = get_player_batting(player)

    bowling = get_player_bowling(player)

    batter_venue = get_batter_venue(player, venue)

    bowler_venue = get_bowler_venue(player, venue)

    venue_features = get_venue_features(venue)

    features["batting"] = batting
    features["bowling"] = bowling
    features["batter_venue"] = batter_venue
    features["bowler_venue"] = bowler_venue
    features["venue"] = venue_features

    return features


##############################################################
# MATCHUP FEATURE STORE
##############################################################

def get_matchup_features(batter, bowler, venue):

    return {

        "batter": get_player_batting(batter),

        "bowler": get_player_bowling(bowler),

        "bvb": get_batter_vs_bowler(batter, bowler),

        "batter_venue": get_batter_venue(batter, venue),

        "bowler_venue": get_bowler_venue(bowler, venue),

        "venue": get_venue_features(venue)

    }
