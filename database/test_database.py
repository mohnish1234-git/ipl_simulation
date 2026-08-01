# test_database.py

from .queries import *

from .feature_store import *

print("=" * 60)
print("DATABASE TEST")
print("=" * 60)

print("\nTesting player lookup...")
print(get_player_batting("Virat Kohli"))

print("\nTesting bowler lookup...")
print(get_player_bowling("Jasprit Bumrah"))

print("\nTesting venue...")
print(get_venue_features("Wankhede Stadium"))

print("\nTesting batter venue...")
print(get_batter_venue("Virat Kohli", "Wankhede Stadium"))

print("\nTesting bowler venue...")
print(get_bowler_venue("Jasprit Bumrah", "Wankhede Stadium"))

print("\nTesting BvB...")
print(get_batter_vs_bowler("Virat Kohli", "Jasprit Bumrah"))

print("\nTesting tailender...")
print(get_tailender_default())

print("\nTesting all players...")
print(len(get_all_players()))

print("\nTesting all venues...")
print(get_all_venues())

print("\nTesting prefix search...")
print(get_players_by_prefix("Vir"))

print("\nTesting feature store...")
print(

    build_feature_store(

        "Virat Kohli",

        "Jasprit Bumrah",

        "Wankhede Stadium"

    )

)

print("\nTesting merged features...")

print(

    merge_features(

        "Virat Kohli",

        "Jasprit Bumrah",

        "Wankhede Stadium"

    )

)

print("\nTesting validation...")

print(

    validate_match(

        [

            "Virat Kohli",

            "Rajat Patidar",

            "Phil Salt"

        ],

        [

            "Jasprit Bumrah",

            "Trent Boult",

            "Hardik Pandya"

        ],

        "Wankhede Stadium"

    )

)

print("\nEverything completed successfully.")

print("=" * 60)
