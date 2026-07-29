"""
src/data/team_mapping.py

Canonicalizes IPL team-name variants (short codes, sponsorship/franchise
renames) into ONE canonical name per franchise, mirroring venue_mapping.py.

Why this matters: "CSK", "Chennai Super Kings" and (if it ever appears)
any other spelling of the same franchise must collapse to one string, or
every team-level stat, roster grouping (batters_by_team/bowlers_by_team),
and the model's batting_team/bowling_team label encoder silently splits
one team's history into multiple smaller, noisier buckets.

Rebrand vs. distinct-franchise judgment calls (kept consistent with how
most public IPL datasets, and the model's own historical labels, treat
these):
  - Delhi Daredevils -> Delhi Capitals: SAME franchise, renamed. Merged.
  - Kings XI Punjab -> Punjab Kings: SAME franchise, renamed. Merged.
  - Royal Challengers Bangalore -> Royal Challengers Bengaluru: SAME
    franchise, city-spelling rename only. Merged.
  - Rising Pune Supergiant (2016) / Rising Pune Supergiants (2017): SAME
    franchise, one-letter spelling difference between seasons. Merged.
  - Deccan Chargers vs. Sunrisers Hyderabad: DIFFERENT franchises (the
    Hyderabad IPL slot was auctioned to a new owner after Deccan Chargers
    was terminated in 2012) — kept SEPARATE, not merged.
  - Gujarat Lions (2016-2017) vs. Gujarat Titans (2022-): DIFFERENT
    franchises (different ownership groups, non-contiguous seasons) —
    kept SEPARATE, not merged.
  - Kochi Tuskers Kerala, Pune Warriors: one-off defunct franchises with
    no successor — kept standalone.

Usage:
    from src.data.team_mapping import canonicalize_team, ALLOWED_TEAMS

    df["batting_team"] = df["batting_team"].map(canonicalize_team)
    df["bowling_team"] = df["bowling_team"].map(canonicalize_team)

As with venue_mapping.py: if new raw variants show up in your data, run
print_unmapped_teams(df) and add them to the matching list in TEAM_ALIASES.
"""

import pandas as pd
from typing import Optional

TEAM_ALIASES = {
    "Chennai Super Kings": [
        "Chennai Super Kings",
        "CSK",
    ],
    "Delhi Capitals": [
        "Delhi Capitals",
        "Delhi Daredevils",
        "DC",
    ],
    "Gujarat Titans": [
        "Gujarat Titans",
        "GT",
    ],
    "Gujarat Titams": [
        "Gujarat Titans",
    ],
    "Kolkata Knight Riders": [
        "Kolkata Knight Riders",
        "KKR",
    ],
    "Punjab Kings": [
        "Punjab Kings",
        "Kings XI Punjab",
        "PBKS",
    ],
    "Lucknow Super Giants": [
        "Lucknow Super Giants",
        "LSG",
    ],
    "Mumbai Indians": [
        "Mumbai Indians",
        "MI",
    ],
    "Rajasthan Royals": [
        "Rajasthan Royals",
        "RR",
    ],
    "Royal Challengers Bengaluru": [
        "Royal Challengers Bengaluru",
        "Royal Challengers Bangalore",
        "RCB",
    ],
    "Sunrisers Hyderabad": [
        "Sunrisers Hyderabad",
        "SRH",
    ],
}

ALLOWED_TEAMS = sorted(TEAM_ALIASES.keys())


def _norm(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


_REVERSE_LOOKUP = {}
for canonical, variants in TEAM_ALIASES.items():
    for v in variants:
        _REVERSE_LOOKUP[_norm(v)] = canonical
    _REVERSE_LOOKUP[_norm(canonical)] = canonical


def canonicalize_team(raw_name) -> Optional[str]:
    """Returns the canonical franchise name, or None if raw_name isn't a
    known team or alias."""
    if pd.isna(raw_name):
        return None
    return _REVERSE_LOOKUP.get(_norm(raw_name))


def print_unmapped_teams(df: pd.DataFrame,
                          team_cols=("batting_team", "bowling_team")) -> None:
    """Diagnostic — run against real raw data to see which team strings
    aren't recognized yet, so you can add them to TEAM_ALIASES above
    instead of silently dropping/misgrouping that data."""
    raw_teams = set()
    for col in team_cols:
        if col in df.columns:
            raw_teams.update(df[col].dropna().unique())
    unmapped = sorted(v for v in raw_teams if canonicalize_team(v) is None)
    if not unmapped:
        print("All team strings in the data matched a known alias. ✓")
        return
    print(f"{len(unmapped)} team strings did NOT match any known alias:")
    for v in unmapped:
        print(f"    '{v}'")
    print("\nIf any of these are a known franchise under a new name/spelling, "
          "add it to the matching list in TEAM_ALIASES.")