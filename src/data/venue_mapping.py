"""
src/data/venue_mapping.py

Restricts the entire pipeline to a fixed set of 13 venues, and canonicalizes
the many raw name variants IPL data uses for the same physical ground across
different seasons (sponsorship renames, "Stadium" vs "Cricket Stadium" vs
city-suffixed strings, etc.) into ONE canonical name per ground.

Why this matters: if "Wankhede Stadium" and "Wankhede Stadium, Mumbai" are
treated as two different venues, every venue-level and player-at-venue stat
for that ground gets silently split into two smaller, noisier buckets. This
file exists so that never happens for the 13 grounds you actually care about.

Usage:
    from src.data.venue_mapping import canonicalize_venue, ALLOWED_VENUES

    df["venue"] = df["venue"].map(canonicalize_venue)
    df = df[df["venue"].isin(ALLOWED_VENUES)].reset_index(drop=True)

IMPORTANT — this list is now built directly from the confirmed unique venue
strings in your actual data/raw/ipl_final.csv (as of the values you shared),
not guessed variants. If you re-pull or extend the raw dataset later and get
new spelling variants, run `print_unmapped_venues(df)` again and add them.
"""

import pandas as pd
from typing import Optional

# ── Canonical venue → exact raw strings confirmed in your dataset ──────────
# Everything NOT listed here (Dr DY Patil, Brabourne, Barabati, Vidarbha CA,
# Nehru Stadium, Holkar, Subrata Roy Sahara, Maharashtra CA, JSCA, Saurashtra
# CA, Green Park, and every neutral UAE/South Africa venue used during
# 2009/2014/2020/2021) is INTENTIONALLY excluded — it will be filtered out
# by filter_to_allowed_venues(), which is exactly what you asked for.
VENUE_ALIASES = {
    "Chennai": [
        "MA Chidambaram Stadium, Chennai",
    ],
    "Wankhede": [
        "Wankhede Stadium, Mumbai",
    ],
    "Delhi": [
        "Arun Jaitley Stadium, Delhi",
    ],
    "Kolkata": [
        "Eden Gardens, Kolkata",
    ],
    "Ahmedabad": [
        "Sardar Patel Stadium, Motera",
        "Narendra Modi Stadium, Ahmedabad",
    ],
    "Raipur": [
        "Shaheed Veer Narayan Singh International Stadium, Raipur",
    ],
    "Guwahati": [
        "Barsapara Cricket Stadium, Guwahati",
    ],
    "Vizag": [
        "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam",
    ],
    "Hyderabad": [
        "Rajiv Gandhi International Stadium, Hyderabad",
        "Rajiv Gandhi International Stadium, Hyderabad, Hyderabad",  # dupe-suffix variant seen in raw data
    ],
    "Bengaluru": [
        "M Chinnaswamy Stadium, Bengaluru",
    ],
    "Lucknow": [
        "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow",
    ],
    "Jaipur": [
        "Sawai Mansingh Stadium, Jaipur",
    ],
    "Dharamsala": [
        "Himachal Pradesh Cricket Association Stadium, Dharamsala",
    ],
    "Chandigarh": [
        "Punjab Cricket Association Stadium, Mohali",
        "Punjab Cricket Association IS Bindra Stadium, Mohali",
        "Punjab Cricket Association IS Bindra Stadium",
        "Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh",
        "Maharaja Yadavindra Singh International Cricket Stadium, New Chandigarh",
    ],
}

ALLOWED_VENUES = sorted(VENUE_ALIASES.keys())

# ── Build the reverse lookup once, normalized ───────────────────────────────
def _norm(s: str) -> str:
    return " ".join(str(s).strip().lower().split())

_REVERSE_LOOKUP = {}
for canonical, variants in VENUE_ALIASES.items():
    for v in variants:
        _REVERSE_LOOKUP[_norm(v)] = canonical
    _REVERSE_LOOKUP[_norm(canonical)] = canonical  # canonical name maps to itself


def canonicalize_venue(raw_name) -> Optional[str]:
    """Returns the canonical venue name, or None if raw_name isn't one of the
    13 allowed grounds (or any known alias of them)."""
    if pd.isna(raw_name):
        return None
    return _REVERSE_LOOKUP.get(_norm(raw_name))


def filter_to_allowed_venues(df: pd.DataFrame, venue_col: str = "venue") -> pd.DataFrame:
    """Canonicalizes df[venue_col] and drops every row whose venue isn't one
    of the 13 allowed grounds. Prints how much data was kept/dropped."""
    before = len(df)
    df = df.copy()
    df[venue_col] = df[venue_col].map(canonicalize_venue)
    dropped_unmatched = df[venue_col].isna().sum()
    df = df[df[venue_col].notna()].reset_index(drop=True)

    print(f"Venue filter: {before:,} rows → {len(df):,} rows "
          f"({dropped_unmatched:,} dropped — venue not in the allowed 13)")
    for v in ALLOWED_VENUES:
        n = (df[venue_col] == v).sum()
        print(f"    {v:<12} {n:,} deliveries")
    return df


def print_unmapped_venues(df: pd.DataFrame, venue_col: str = "venue") -> None:
    """Diagnostic — run this ONCE against your real raw data (before
    filtering) to see which venue strings in your CSV aren't recognized yet,
    so you can add them to VENUE_ALIASES above instead of silently losing
    that data."""
    raw_venues = df[venue_col].dropna().unique()
    unmapped = sorted(v for v in raw_venues if canonicalize_venue(v) is None)
    if not unmapped:
        print("All venue strings in the data matched a known alias. ✓")
        return
    print(f"{len(unmapped)} venue strings did NOT match any known alias:")
    for v in unmapped:
        print(f"    '{v}'")
    print("\nIf any of these are one of your 13 target grounds under a new "
          "sponsorship/spelling, add it to the matching list in VENUE_ALIASES.")