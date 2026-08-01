# create_database.py

import psycopg2

from dotenv import load_dotenv

import os


##############################################################
# LOAD ENVIRONMENT
##############################################################

load_dotenv("database/.env")


DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")


##############################################################
# CONNECT
##############################################################

conn = psycopg2.connect(

    dbname=DB_NAME,

    user=DB_USER,

    password=DB_PASSWORD,

    host=DB_HOST,

    port=DB_PORT

)

cur = conn.cursor()


##############################################################
# PLAYER BATTER STATS
##############################################################

cur.execute("""

CREATE TABLE IF NOT EXISTS player_batter_stats(

    player_name TEXT PRIMARY KEY,

    bat_rw_avg DOUBLE PRECISION,

    bat_rw_sr DOUBLE PRECISION,

    bat_rw_boundary_pct DOUBLE PRECISION,

    bat_rw_six_pct DOUBLE PRECISION,

    bat_rw_dot_pct DOUBLE PRECISION,

    bat_role_bowler_ratio DOUBLE PRECISION,

    bat_1_6_rw_sr DOUBLE PRECISION,

    bat_1_6_rw_boundary_pct DOUBLE PRECISION,

    bat_1_6_rw_dot_pct DOUBLE PRECISION,

    bat_7_10_rw_sr DOUBLE PRECISION,

    bat_7_10_rw_boundary_pct DOUBLE PRECISION,

    bat_7_10_rw_dot_pct DOUBLE PRECISION,

    bat_11_15_rw_sr DOUBLE PRECISION,

    bat_11_15_rw_boundary_pct DOUBLE PRECISION,

    bat_11_15_rw_dot_pct DOUBLE PRECISION,

    bat_16_20_rw_sr DOUBLE PRECISION,

    bat_16_20_rw_boundary_pct DOUBLE PRECISION,

    bat_16_20_rw_dot_pct DOUBLE PRECISION

);

""")


##############################################################
# PLAYER BOWLER STATS
##############################################################

cur.execute("""

CREATE TABLE IF NOT EXISTS player_bowler_stats(

    player_name TEXT PRIMARY KEY,

    bowl_rw_economy DOUBLE PRECISION,

    bowl_rw_wicket_pct DOUBLE PRECISION,

    bowl_rw_dot_pct DOUBLE PRECISION,

    bowl_rw_boundary_pct DOUBLE PRECISION,

    bowl_1_6_rw_economy DOUBLE PRECISION,

    bowl_1_6_rw_wicket_pct DOUBLE PRECISION,

    bowl_1_6_rw_dot_pct DOUBLE PRECISION,

    bowl_1_6_rw_boundary_pct DOUBLE PRECISION,

    bowl_7_10_rw_economy DOUBLE PRECISION,

    bowl_7_10_rw_wicket_pct DOUBLE PRECISION,

    bowl_7_10_rw_dot_pct DOUBLE PRECISION,

    bowl_7_10_rw_boundary_pct DOUBLE PRECISION,

    bowl_11_15_rw_economy DOUBLE PRECISION,

    bowl_11_15_rw_wicket_pct DOUBLE PRECISION,

    bowl_11_15_rw_dot_pct DOUBLE PRECISION,

    bowl_11_15_rw_boundary_pct DOUBLE PRECISION,

    bowl_16_20_rw_economy DOUBLE PRECISION,

    bowl_16_20_rw_wicket_pct DOUBLE PRECISION,

    bowl_16_20_rw_dot_pct DOUBLE PRECISION,

    bowl_16_20_rw_boundary_pct DOUBLE PRECISION

);

""")
##############################################################
# VENUE STATS
##############################################################

cur.execute("""

CREATE TABLE IF NOT EXISTS venue_stats(

    venue_name TEXT PRIMARY KEY,

    venue_rw_avg_1st_innings DOUBLE PRECISION,

    venue_rw_avg_2nd_innings DOUBLE PRECISION,

    venue_rw_boundary_pct DOUBLE PRECISION,

    venue_rw_six_pct DOUBLE PRECISION,

    venue_rw_dot_pct DOUBLE PRECISION,

    venue_rw_wicket_pct DOUBLE PRECISION,

    venue_rw_1_6_rr DOUBLE PRECISION,

    venue_rw_1_6_wicket_pct DOUBLE PRECISION,

    venue_rw_7_10_rr DOUBLE PRECISION,

    venue_rw_7_10_wicket_pct DOUBLE PRECISION,

    venue_rw_11_15_rr DOUBLE PRECISION,

    venue_rw_11_15_wicket_pct DOUBLE PRECISION,

    venue_rw_16_20_rr DOUBLE PRECISION,

    venue_rw_16_20_wicket_pct DOUBLE PRECISION

);

""")


##############################################################
# BATTER VENUE STATS
##############################################################

cur.execute("""

CREATE TABLE IF NOT EXISTS batter_venue_stats(

    player_name TEXT,

    venue_name TEXT,

    bat_venue_rw_balls DOUBLE PRECISION,

    bat_venue_adj_sr DOUBLE PRECISION,

    bat_venue_adj_boundary_pct DOUBLE PRECISION,

    bat_venue_1_6_sr DOUBLE PRECISION,

    bat_venue_1_6_avg DOUBLE PRECISION,

    bat_venue_7_10_sr DOUBLE PRECISION,

    bat_venue_7_10_avg DOUBLE PRECISION,

    bat_venue_11_15_sr DOUBLE PRECISION,

    bat_venue_11_15_avg DOUBLE PRECISION,

    bat_venue_16_20_sr DOUBLE PRECISION,

    bat_venue_16_20_avg DOUBLE PRECISION,

    PRIMARY KEY(player_name, venue_name)

);

""")


##############################################################
# BOWLER VENUE STATS
##############################################################

cur.execute("""

CREATE TABLE IF NOT EXISTS bowler_venue_stats(

    player_name TEXT,

    venue_name TEXT,

    bowl_venue_rw_balls DOUBLE PRECISION,

    bowl_venue_adj_economy DOUBLE PRECISION,

    bowl_venue_adj_wicket_pct DOUBLE PRECISION,

    bowl_venue_1_6_economy DOUBLE PRECISION,

    bowl_venue_1_6_wicket_pct DOUBLE PRECISION,

    bowl_venue_7_10_economy DOUBLE PRECISION,

    bowl_venue_7_10_wicket_pct DOUBLE PRECISION,

    bowl_venue_11_15_economy DOUBLE PRECISION,

    bowl_venue_11_15_wicket_pct DOUBLE PRECISION,

    bowl_venue_16_20_economy DOUBLE PRECISION,

    bowl_venue_16_20_wicket_pct DOUBLE PRECISION,

    PRIMARY KEY(player_name, venue_name)

);

""")
##############################################################
# BATTER VS BOWLER STATS
##############################################################

cur.execute("""

CREATE TABLE IF NOT EXISTS batter_vs_bowler_stats(

    batter_name TEXT,

    bowler_name TEXT,

    bvb_balls DOUBLE PRECISION,

    bvb_rw_dismissal_pct DOUBLE PRECISION,

    bvb_rw_sr DOUBLE PRECISION,

    bvb_rw_dot_pct DOUBLE PRECISION,

    bvb_rw_boundary_pct DOUBLE PRECISION,

    bvb_rw_six_pct DOUBLE PRECISION,

    bvb_1_6_sr DOUBLE PRECISION,

    bvb_1_6_avg DOUBLE PRECISION,

    bvb_7_10_sr DOUBLE PRECISION,

    bvb_7_10_avg DOUBLE PRECISION,

    bvb_11_15_sr DOUBLE PRECISION,

    bvb_11_15_avg DOUBLE PRECISION,

    bvb_16_20_sr DOUBLE PRECISION,

    bvb_16_20_avg DOUBLE PRECISION,

    PRIMARY KEY(

        batter_name,

        bowler_name

    )

);

""")


##############################################################
# META
##############################################################

cur.execute("""

CREATE TABLE IF NOT EXISTS meta(

    feature_name TEXT PRIMARY KEY,

    feature_value JSONB

);

""")


##############################################################
# TAILENDER DEFAULT
##############################################################

cur.execute("""

CREATE TABLE IF NOT EXISTS tailender_default(

    id INTEGER PRIMARY KEY,

    bat_rw_avg DOUBLE PRECISION,

    bat_rw_sr DOUBLE PRECISION,

    bat_rw_boundary_pct DOUBLE PRECISION,

    bat_rw_six_pct DOUBLE PRECISION,

    bat_rw_dot_pct DOUBLE PRECISION,

    bat_role_bowler_ratio DOUBLE PRECISION,

    bat_1_6_rw_sr DOUBLE PRECISION,

    bat_1_6_rw_boundary_pct DOUBLE PRECISION,

    bat_1_6_rw_dot_pct DOUBLE PRECISION,

    bat_7_10_rw_sr DOUBLE PRECISION,

    bat_7_10_rw_boundary_pct DOUBLE PRECISION,

    bat_7_10_rw_dot_pct DOUBLE PRECISION,

    bat_11_15_rw_sr DOUBLE PRECISION,

    bat_11_15_rw_boundary_pct DOUBLE PRECISION,

    bat_11_15_rw_dot_pct DOUBLE PRECISION,

    bat_16_20_rw_sr DOUBLE PRECISION,

    bat_16_20_rw_boundary_pct DOUBLE PRECISION,

    bat_16_20_rw_dot_pct DOUBLE PRECISION

);

""")
##############################################################
# INDEXES
##############################################################

cur.execute("""

CREATE INDEX IF NOT EXISTS idx_batter_name

ON player_batter_stats(player_name);

""")

cur.execute("""

CREATE INDEX IF NOT EXISTS idx_bowler_name

ON player_bowler_stats(player_name);

""")

cur.execute("""

CREATE INDEX IF NOT EXISTS idx_venue_name

ON venue_stats(venue_name);

""")

cur.execute("""

CREATE INDEX IF NOT EXISTS idx_batter_venue_player

ON batter_venue_stats(player_name);

""")

cur.execute("""

CREATE INDEX IF NOT EXISTS idx_batter_venue_ground

ON batter_venue_stats(venue_name);

""")

cur.execute("""

CREATE INDEX IF NOT EXISTS idx_bowler_venue_player

ON bowler_venue_stats(player_name);

""")

cur.execute("""

CREATE INDEX IF NOT EXISTS idx_bowler_venue_ground

ON bowler_venue_stats(venue_name);

""")

cur.execute("""

CREATE INDEX IF NOT EXISTS idx_bvb_batter

ON batter_vs_bowler_stats(batter_name);

""")

cur.execute("""

CREATE INDEX IF NOT EXISTS idx_bvb_bowler

ON batter_vs_bowler_stats(bowler_name);

""")


##############################################################
# COMMIT
##############################################################

conn.commit()


##############################################################
# SUCCESS MESSAGE
##############################################################

print("=" * 60)

print("PostgreSQL Database Created Successfully")

print("=" * 60)

print()

print("Tables Created")

print("------------------------------")

print("player_batter_stats")

print("player_bowler_stats")

print("venue_stats")

print("batter_venue_stats")

print("bowler_venue_stats")

print("batter_vs_bowler_stats")

print("meta")

print("tailender_default")

print()

print("Indexes Created Successfully")

print("=" * 60)


##############################################################
# CLOSE
##############################################################

cur.close()

conn.close()
