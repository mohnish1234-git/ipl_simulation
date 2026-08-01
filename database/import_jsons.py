# import_jsons.py

import os

import json

from tqdm import tqdm

from psycopg2.extras import execute_values

from .db import get_connection, return_connection


##############################################################
# DATA PATH
##############################################################

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_FOLDER = BASE_DIR.parent / "data" / "processed"


##############################################################
# LOAD JSON
##############################################################

def load_json(filename):

    with open(DATA_FOLDER / filename, "r", encoding="utf-8") as f:

        return json.load(f)


##############################################################
# PLAYER BATTER STATS
##############################################################

def import_player_batter_stats():

    print("\nImporting Player Batter Stats...")

    data = load_json(

        "player_batter_stats.json"

    )

    rows = []

    for player, stats in tqdm(data.items()):

        rows.append(

            (

                player,

                stats.get("bat_rw_avg"),

                stats.get("bat_rw_sr"),

                stats.get("bat_rw_boundary_pct"),

                stats.get("bat_rw_six_pct"),

                stats.get("bat_rw_dot_pct"),

                stats.get("bat_role_bowler_ratio"),

                stats.get("bat_1_6_rw_sr"),

                stats.get("bat_1_6_rw_boundary_pct"),

                stats.get("bat_1_6_rw_dot_pct"),

                stats.get("bat_7_10_rw_sr"),

                stats.get("bat_7_10_rw_boundary_pct"),

                stats.get("bat_7_10_rw_dot_pct"),

                stats.get("bat_11_15_rw_sr"),

                stats.get("bat_11_15_rw_boundary_pct"),

                stats.get("bat_11_15_rw_dot_pct"),

                stats.get("bat_16_20_rw_sr"),

                stats.get("bat_16_20_rw_boundary_pct"),

                stats.get("bat_16_20_rw_dot_pct")

            )

        )

    conn = get_connection()

    cur = conn.cursor()

    execute_values(

        cur,

        """

        INSERT INTO player_batter_stats

        VALUES %s

        ON CONFLICT(player_name)

        DO UPDATE SET

        bat_rw_avg=EXCLUDED.bat_rw_avg,

        bat_rw_sr=EXCLUDED.bat_rw_sr,

        bat_rw_boundary_pct=EXCLUDED.bat_rw_boundary_pct,

        bat_rw_six_pct=EXCLUDED.bat_rw_six_pct,

        bat_rw_dot_pct=EXCLUDED.bat_rw_dot_pct,

        bat_role_bowler_ratio=EXCLUDED.bat_role_bowler_ratio,

        bat_1_6_rw_sr=EXCLUDED.bat_1_6_rw_sr,

        bat_1_6_rw_boundary_pct=EXCLUDED.bat_1_6_rw_boundary_pct,

        bat_1_6_rw_dot_pct=EXCLUDED.bat_1_6_rw_dot_pct,

        bat_7_10_rw_sr=EXCLUDED.bat_7_10_rw_sr,

        bat_7_10_rw_boundary_pct=EXCLUDED.bat_7_10_rw_boundary_pct,

        bat_7_10_rw_dot_pct=EXCLUDED.bat_7_10_rw_dot_pct,

        bat_11_15_rw_sr=EXCLUDED.bat_11_15_rw_sr,

        bat_11_15_rw_boundary_pct=EXCLUDED.bat_11_15_rw_boundary_pct,

        bat_11_15_rw_dot_pct=EXCLUDED.bat_11_15_rw_dot_pct,

        bat_16_20_rw_sr=EXCLUDED.bat_16_20_rw_sr,

        bat_16_20_rw_boundary_pct=EXCLUDED.bat_16_20_rw_boundary_pct,

        bat_16_20_rw_dot_pct=EXCLUDED.bat_16_20_rw_dot_pct

        """,

        rows

    )

    conn.commit()

    cur.close()

    return_connection(conn)

    print("Imported", len(rows), "batters")
    ##############################################################
# PLAYER BOWLER STATS
##############################################################

def import_player_bowler_stats():

    print("\nImporting Player Bowler Stats...")

    data = load_json(

        "player_bowler_stats.json"

    )

    rows = []

    for player, stats in tqdm(data.items()):

        rows.append(

            (

                player,

                stats.get("bowl_rw_economy"),

                stats.get("bowl_rw_wicket_pct"),

                stats.get("bowl_rw_dot_pct"),

                stats.get("bowl_rw_boundary_pct"),

                stats.get("bowl_1_6_rw_economy"),

                stats.get("bowl_1_6_rw_wicket_pct"),

                stats.get("bowl_1_6_rw_dot_pct"),

                stats.get("bowl_1_6_rw_boundary_pct"),

                stats.get("bowl_7_10_rw_economy"),

                stats.get("bowl_7_10_rw_wicket_pct"),

                stats.get("bowl_7_10_rw_dot_pct"),

                stats.get("bowl_7_10_rw_boundary_pct"),

                stats.get("bowl_11_15_rw_economy"),

                stats.get("bowl_11_15_rw_wicket_pct"),

                stats.get("bowl_11_15_rw_dot_pct"),

                stats.get("bowl_11_15_rw_boundary_pct"),

                stats.get("bowl_16_20_rw_economy"),

                stats.get("bowl_16_20_rw_wicket_pct"),

                stats.get("bowl_16_20_rw_dot_pct"),

                stats.get("bowl_16_20_rw_boundary_pct")

            )

        )

    conn = get_connection()

    cur = conn.cursor()

    execute_values(

        cur,

        """

        INSERT INTO player_bowler_stats

        VALUES %s

        ON CONFLICT(player_name)

        DO UPDATE SET

        bowl_rw_economy=EXCLUDED.bowl_rw_economy,

        bowl_rw_wicket_pct=EXCLUDED.bowl_rw_wicket_pct,

        bowl_rw_dot_pct=EXCLUDED.bowl_rw_dot_pct,

        bowl_rw_boundary_pct=EXCLUDED.bowl_rw_boundary_pct,

        bowl_1_6_rw_economy=EXCLUDED.bowl_1_6_rw_economy,

        bowl_1_6_rw_wicket_pct=EXCLUDED.bowl_1_6_rw_wicket_pct,

        bowl_1_6_rw_dot_pct=EXCLUDED.bowl_1_6_rw_dot_pct,

        bowl_1_6_rw_boundary_pct=EXCLUDED.bowl_1_6_rw_boundary_pct,

        bowl_7_10_rw_economy=EXCLUDED.bowl_7_10_rw_economy,

        bowl_7_10_rw_wicket_pct=EXCLUDED.bowl_7_10_rw_wicket_pct,

        bowl_7_10_rw_dot_pct=EXCLUDED.bowl_7_10_rw_dot_pct,

        bowl_7_10_rw_boundary_pct=EXCLUDED.bowl_7_10_rw_boundary_pct,

        bowl_11_15_rw_economy=EXCLUDED.bowl_11_15_rw_economy,

        bowl_11_15_rw_wicket_pct=EXCLUDED.bowl_11_15_rw_wicket_pct,

        bowl_11_15_rw_dot_pct=EXCLUDED.bowl_11_15_rw_dot_pct,

        bowl_11_15_rw_boundary_pct=EXCLUDED.bowl_11_15_rw_boundary_pct,

        bowl_16_20_rw_economy=EXCLUDED.bowl_16_20_rw_economy,

        bowl_16_20_rw_wicket_pct=EXCLUDED.bowl_16_20_rw_wicket_pct,

        bowl_16_20_rw_dot_pct=EXCLUDED.bowl_16_20_rw_dot_pct,

        bowl_16_20_rw_boundary_pct=EXCLUDED.bowl_16_20_rw_boundary_pct

        """,

        rows

    )

    conn.commit()

    cur.close()

    return_connection(conn)

    print("Imported", len(rows), "bowlers")
    ##############################################################
# VENUE STATS
##############################################################

def import_venue_stats():

    print("\nImporting Venue Stats...")

    data = load_json(

        "venue_stats.json"

    )

    rows = []

    for venue, stats in tqdm(data.items()):

        rows.append(

            (

                venue,

                stats.get("venue_rw_avg_1st_innings"),

                stats.get("venue_rw_avg_2nd_innings"),

                stats.get("venue_rw_boundary_pct"),

                stats.get("venue_rw_six_pct"),

                stats.get("venue_rw_dot_pct"),

                stats.get("venue_rw_wicket_pct"),

                stats.get("venue_rw_1_6_rr"),

                stats.get("venue_rw_1_6_wicket_pct"),

                stats.get("venue_rw_7_10_rr"),

                stats.get("venue_rw_7_10_wicket_pct"),

                stats.get("venue_rw_11_15_rr"),

                stats.get("venue_rw_11_15_wicket_pct"),

                stats.get("venue_rw_16_20_rr"),

                stats.get("venue_rw_16_20_wicket_pct")

            )

        )

    conn = get_connection()

    cur = conn.cursor()

    execute_values(

        cur,

        """

        INSERT INTO venue_stats

        VALUES %s

        ON CONFLICT(venue_name)

        DO UPDATE SET

        venue_rw_avg_1st_innings = EXCLUDED.venue_rw_avg_1st_innings,

        venue_rw_avg_2nd_innings = EXCLUDED.venue_rw_avg_2nd_innings,

        venue_rw_boundary_pct = EXCLUDED.venue_rw_boundary_pct,

        venue_rw_six_pct = EXCLUDED.venue_rw_six_pct,

        venue_rw_dot_pct = EXCLUDED.venue_rw_dot_pct,

        venue_rw_wicket_pct = EXCLUDED.venue_rw_wicket_pct,

        venue_rw_1_6_rr = EXCLUDED.venue_rw_1_6_rr,

        venue_rw_1_6_wicket_pct = EXCLUDED.venue_rw_1_6_wicket_pct,

        venue_rw_7_10_rr = EXCLUDED.venue_rw_7_10_rr,

        venue_rw_7_10_wicket_pct = EXCLUDED.venue_rw_7_10_wicket_pct,

        venue_rw_11_15_rr = EXCLUDED.venue_rw_11_15_rr,

        venue_rw_11_15_wicket_pct = EXCLUDED.venue_rw_11_15_wicket_pct,

        venue_rw_16_20_rr = EXCLUDED.venue_rw_16_20_rr,

        venue_rw_16_20_wicket_pct = EXCLUDED.venue_rw_16_20_wicket_pct

        """,

        rows

    )

    conn.commit()

    cur.close()

    return_connection(conn)

    print("Imported", len(rows), "venues")
    ##############################################################
# VENUE STATS
##############################################################

def import_venue_stats():

    print("\nImporting Venue Stats...")

    data = load_json(

        "venue_stats.json"

    )

    rows = []

    for venue, stats in tqdm(data.items()):

        rows.append(

            (

                venue,

                stats.get("venue_rw_avg_1st_innings"),

                stats.get("venue_rw_avg_2nd_innings"),

                stats.get("venue_rw_boundary_pct"),

                stats.get("venue_rw_six_pct"),

                stats.get("venue_rw_dot_pct"),

                stats.get("venue_rw_wicket_pct"),

                stats.get("venue_rw_1_6_rr"),

                stats.get("venue_rw_1_6_wicket_pct"),

                stats.get("venue_rw_7_10_rr"),

                stats.get("venue_rw_7_10_wicket_pct"),

                stats.get("venue_rw_11_15_rr"),

                stats.get("venue_rw_11_15_wicket_pct"),

                stats.get("venue_rw_16_20_rr"),

                stats.get("venue_rw_16_20_wicket_pct")

            )

        )

    conn = get_connection()

    cur = conn.cursor()

    execute_values(

        cur,

        """

        INSERT INTO venue_stats

        VALUES %s

        ON CONFLICT(venue_name)

        DO UPDATE SET

        venue_rw_avg_1st_innings = EXCLUDED.venue_rw_avg_1st_innings,

        venue_rw_avg_2nd_innings = EXCLUDED.venue_rw_avg_2nd_innings,

        venue_rw_boundary_pct = EXCLUDED.venue_rw_boundary_pct,

        venue_rw_six_pct = EXCLUDED.venue_rw_six_pct,

        venue_rw_dot_pct = EXCLUDED.venue_rw_dot_pct,

        venue_rw_wicket_pct = EXCLUDED.venue_rw_wicket_pct,

        venue_rw_1_6_rr = EXCLUDED.venue_rw_1_6_rr,

        venue_rw_1_6_wicket_pct = EXCLUDED.venue_rw_1_6_wicket_pct,

        venue_rw_7_10_rr = EXCLUDED.venue_rw_7_10_rr,

        venue_rw_7_10_wicket_pct = EXCLUDED.venue_rw_7_10_wicket_pct,

        venue_rw_11_15_rr = EXCLUDED.venue_rw_11_15_rr,

        venue_rw_11_15_wicket_pct = EXCLUDED.venue_rw_11_15_wicket_pct,

        venue_rw_16_20_rr = EXCLUDED.venue_rw_16_20_rr,

        venue_rw_16_20_wicket_pct = EXCLUDED.venue_rw_16_20_wicket_pct

        """,

        rows

    )

    conn.commit()

    cur.close()

    return_connection(conn)

    print("Imported", len(rows), "venues")
    ##############################################################
# BATTER VENUE STATS
##############################################################

def import_batter_venue_stats():

    print("\nImporting Batter Venue Stats...")

    data = load_json(

        "batter_venue_stats.json"

    )

    rows = []

    for key, stats in tqdm(data.items()):

        player, venue = key.split("|||")

        rows.append(

            (

                player,

                venue,

                stats.get("bat_venue_rw_balls"),

                stats.get("bat_venue_adj_sr"),

                stats.get("bat_venue_adj_boundary_pct"),

                stats.get("bat_venue_1_6_sr"),

                stats.get("bat_venue_1_6_avg"),

                stats.get("bat_venue_7_10_sr"),

                stats.get("bat_venue_7_10_avg"),

                stats.get("bat_venue_11_15_sr"),

                stats.get("bat_venue_11_15_avg"),

                stats.get("bat_venue_16_20_sr"),

                stats.get("bat_venue_16_20_avg")

            )

        )

    conn = get_connection()

    cur = conn.cursor()

    execute_values(

        cur,

        """

        INSERT INTO batter_venue_stats

        VALUES %s

        ON CONFLICT(player_name, venue_name)

        DO UPDATE SET

        bat_venue_rw_balls = EXCLUDED.bat_venue_rw_balls,

        bat_venue_adj_sr = EXCLUDED.bat_venue_adj_sr,

        bat_venue_adj_boundary_pct = EXCLUDED.bat_venue_adj_boundary_pct,

        bat_venue_1_6_sr = EXCLUDED.bat_venue_1_6_sr,

        bat_venue_1_6_avg = EXCLUDED.bat_venue_1_6_avg,

        bat_venue_7_10_sr = EXCLUDED.bat_venue_7_10_sr,

        bat_venue_7_10_avg = EXCLUDED.bat_venue_7_10_avg,

        bat_venue_11_15_sr = EXCLUDED.bat_venue_11_15_sr,

        bat_venue_11_15_avg = EXCLUDED.bat_venue_11_15_avg,

        bat_venue_16_20_sr = EXCLUDED.bat_venue_16_20_sr,

        bat_venue_16_20_avg = EXCLUDED.bat_venue_16_20_avg

        """,

        rows

    )

    conn.commit()

    cur.close()

    return_connection(conn)

    print("Imported", len(rows), "batter venue records")
    ##############################################################
# BOWLER VENUE STATS
##############################################################

def import_bowler_venue_stats():

    print("\nImporting Bowler Venue Stats...")

    data = load_json(

        "bowler_venue_stats.json"

    )

    rows = []

    for key, stats in tqdm(data.items()):

        player, venue = key.split("|||")

        rows.append(

            (

                player,

                venue,

                stats.get("bowl_venue_rw_balls"),

                stats.get("bowl_venue_adj_economy"),

                stats.get("bowl_venue_adj_wicket_pct"),

                stats.get("bowl_venue_1_6_economy"),

                stats.get("bowl_venue_1_6_wicket_pct"),

                stats.get("bowl_venue_7_10_economy"),

                stats.get("bowl_venue_7_10_wicket_pct"),

                stats.get("bowl_venue_11_15_economy"),

                stats.get("bowl_venue_11_15_wicket_pct"),

                stats.get("bowl_venue_16_20_economy"),

                stats.get("bowl_venue_16_20_wicket_pct")

            )

        )

    conn = get_connection()

    cur = conn.cursor()

    execute_values(

        cur,

        """

        INSERT INTO bowler_venue_stats

        VALUES %s

        ON CONFLICT(player_name, venue_name)

        DO UPDATE SET

        bowl_venue_rw_balls = EXCLUDED.bowl_venue_rw_balls,

        bowl_venue_adj_economy = EXCLUDED.bowl_venue_adj_economy,

        bowl_venue_adj_wicket_pct = EXCLUDED.bowl_venue_adj_wicket_pct,

        bowl_venue_1_6_economy = EXCLUDED.bowl_venue_1_6_economy,

        bowl_venue_1_6_wicket_pct = EXCLUDED.bowl_venue_1_6_wicket_pct,

        bowl_venue_7_10_economy = EXCLUDED.bowl_venue_7_10_economy,

        bowl_venue_7_10_wicket_pct = EXCLUDED.bowl_venue_7_10_wicket_pct,

        bowl_venue_11_15_economy = EXCLUDED.bowl_venue_11_15_economy,

        bowl_venue_11_15_wicket_pct = EXCLUDED.bowl_venue_11_15_wicket_pct,

        bowl_venue_16_20_economy = EXCLUDED.bowl_venue_16_20_economy,

        bowl_venue_16_20_wicket_pct = EXCLUDED.bowl_venue_16_20_wicket_pct

        """,

        rows

    )

    conn.commit()

    cur.close()

    return_connection(conn)

    print("Imported", len(rows), "bowler venue records")
    ##############################################################
# BATTER VS BOWLER STATS
##############################################################

def import_batter_vs_bowler_stats():

    print("\nImporting Batter vs Bowler Stats...")

    data = load_json(

        "bvb_stats.json"

    )

    rows = []

    for key, stats in tqdm(data.items()):

        batter, bowler = key.split("|||")

        rows.append(

            (

                batter,

                bowler,

                stats.get("bvb_balls"),

                stats.get("bvb_rw_dismissal_pct"),

                stats.get("bvb_rw_sr"),

                stats.get("bvb_rw_dot_pct"),

                stats.get("bvb_rw_boundary_pct"),

                stats.get("bvb_rw_six_pct"),

                stats.get("bvb_1_6_sr"),

                stats.get("bvb_1_6_avg"),

                stats.get("bvb_7_10_sr"),

                stats.get("bvb_7_10_avg"),

                stats.get("bvb_11_15_sr"),

                stats.get("bvb_11_15_avg"),

                stats.get("bvb_16_20_sr"),

                stats.get("bvb_16_20_avg")

            )

        )

    conn = get_connection()

    cur = conn.cursor()

    execute_values(

        cur,

        """

        INSERT INTO batter_vs_bowler_stats

        VALUES %s

        ON CONFLICT(batter_name, bowler_name)

        DO UPDATE SET

        bvb_balls = EXCLUDED.bvb_balls,

        bvb_rw_dismissal_pct = EXCLUDED.bvb_rw_dismissal_pct,

        bvb_rw_sr = EXCLUDED.bvb_rw_sr,

        bvb_rw_dot_pct = EXCLUDED.bvb_rw_dot_pct,

        bvb_rw_boundary_pct = EXCLUDED.bvb_rw_boundary_pct,

        bvb_rw_six_pct = EXCLUDED.bvb_rw_six_pct,

        bvb_1_6_sr = EXCLUDED.bvb_1_6_sr,

        bvb_1_6_avg = EXCLUDED.bvb_1_6_avg,

        bvb_7_10_sr = EXCLUDED.bvb_7_10_sr,

        bvb_7_10_avg = EXCLUDED.bvb_7_10_avg,

        bvb_11_15_sr = EXCLUDED.bvb_11_15_sr,

        bvb_11_15_avg = EXCLUDED.bvb_11_15_avg,

        bvb_16_20_sr = EXCLUDED.bvb_16_20_sr,

        bvb_16_20_avg = EXCLUDED.bvb_16_20_avg

        """,

        rows

    )

    conn.commit()

    cur.close()

    return_connection(conn)

    print("Imported", len(rows), "batter vs bowler records")
    ##############################################################
# META
##############################################################

def import_meta():

    print("\nImporting Meta Information...")

    data = load_json(

        "meta.json"

    )

    rows = []

    for key, value in data.items():

        rows.append(

            (

                key,

                json.dumps(value)

            )

        )

    conn = get_connection()

    cur = conn.cursor()

    execute_values(

        cur,

        """

        INSERT INTO meta

        VALUES %s

        ON CONFLICT(feature_name)

        DO UPDATE SET

        feature_value = EXCLUDED.feature_value

        """,

        rows

    )

    conn.commit()

    cur.close()

    return_connection(conn)

    print("Imported", len(rows), "meta records")


##############################################################
# TAILENDER DEFAULT
##############################################################

def import_tailender_default():

    print("\nImporting Tailender Default...")

    data = load_json(

        "tailender_default.json"

    )

    conn = get_connection()

    cur = conn.cursor()

    cur.execute(

        "TRUNCATE TABLE tailender_default"

    )

    cur.execute(

        """

        INSERT INTO tailender_default(

            id,

            bat_rw_avg,

            bat_rw_sr,

            bat_rw_boundary_pct,

            bat_rw_six_pct,

            bat_rw_dot_pct,

            bat_role_bowler_ratio,

            bat_1_6_rw_sr,

            bat_1_6_rw_boundary_pct,

            bat_1_6_rw_dot_pct,

            bat_7_10_rw_sr,

            bat_7_10_rw_boundary_pct,

            bat_7_10_rw_dot_pct,

            bat_11_15_rw_sr,

            bat_11_15_rw_boundary_pct,

            bat_11_15_rw_dot_pct,

            bat_16_20_rw_sr,

            bat_16_20_rw_boundary_pct,

            bat_16_20_rw_dot_pct

        )

        VALUES(

            1,

            %s,

            %s,

            %s,

            %s,

            %s,

            %s,

            %s,

            %s,

            %s,

            %s,

            %s,

            %s,

            %s,

            %s,

            %s,

            %s,

            %s,

            %s

        )

        """,

        (

            data.get("bat_rw_avg"),

            data.get("bat_rw_sr"),

            data.get("bat_rw_boundary_pct"),

            data.get("bat_rw_six_pct"),

            data.get("bat_rw_dot_pct"),

            data.get("bat_role_bowler_ratio"),

            data.get("bat_1_6_rw_sr"),

            data.get("bat_1_6_rw_boundary_pct"),

            data.get("bat_1_6_rw_dot_pct"),

            data.get("bat_7_10_rw_sr"),

            data.get("bat_7_10_rw_boundary_pct"),

            data.get("bat_7_10_rw_dot_pct"),

            data.get("bat_11_15_rw_sr"),

            data.get("bat_11_15_rw_boundary_pct"),

            data.get("bat_11_15_rw_dot_pct"),

            data.get("bat_16_20_rw_sr"),

            data.get("bat_16_20_rw_boundary_pct"),

            data.get("bat_16_20_rw_dot_pct")

        )

    )

    conn.commit()

    cur.close()

    return_connection(conn)

    print("Imported tailender default")
    ##############################################################
# IMPORT EVERYTHING
##############################################################

def import_all():

    import_player_batter_stats()

    import_player_bowler_stats()

    import_venue_stats()

    import_batter_venue_stats()

    import_bowler_venue_stats()

    import_batter_vs_bowler_stats()

    import_meta()

    import_tailender_default()

    print()

    print("=" * 60)

    print("ALL JSON FILES IMPORTED SUCCESSFULLY")

    print("=" * 60)


##############################################################
# MAIN
##############################################################

if __name__ == "__main__":

    import_all()