"""
Player archetypes + aggression/dismissal risk coupling.

Why this exists (and why it's NOT duplicating what XGBoost already learns):

  * Archetype identity (Anchor / Accumulator / Aggressive / Finisher, and
    Powerplay / Death / Spinner / All-phase for bowlers) is never a feature
    the model was trained on at all — it's derived here on top of the same
    recency-weighted stats the model *does* see, so it can't be learned by
    the model no matter how good the training data is.

  * The model treats {0,1,2,3,4,6,W} as one softmax draw per ball. In reality,
    a batter's decision to "go for it" is a single underlying choice that
    pushes boundary probability AND dismissal probability up TOGETHER
    (mistimed big shots, aerial risk). A per-ball independent multiclass
    sample doesn't reproduce that coupling on its own — this module adds a
    small, bounded, reversible nudge that moves probability mass between
    dot/single and boundary+wicket buckets in a correlated way, gated by
    signals the model genuinely doesn't have (archetype identity, a
    settling period distinct from raw balls-faced, fine-grained over-by-over
    escalation within "death", and venue/bowler-archetype synergy).

  * Everything here operates ON TOP of the model's own output distribution.
    It never invents runs/wickets from scratch and is capped so it can only
    move a few percentage points of probability at most.
"""

from enum import Enum
from typing import Dict


# ═══════════════════════════════════════════════════════════════════════════
# Archetypes
# ═══════════════════════════════════════════════════════════════════════════

class BatterArchetype(Enum):
    ANCHOR       = "anchor"
    ACCUMULATOR  = "accumulator"
    AGGRESSIVE   = "aggressive"
    FINISHER     = "finisher"


class BowlerArchetype(Enum):
    POWERPLAY  = "powerplay_specialist"
    DEATH      = "death_specialist"
    SPINNER    = "spinner"
    ALL_PHASE  = "all_phase"


def classify_batter(stats: dict) -> BatterArchetype:
    """Classifies from recency-weighted career stats already in StatsStore —
    same numbers the model sees, just used to derive a trait the model
    never had access to."""
    sr           = stats.get("bat_rw_sr", 128.0)
    boundary_pct = stats.get("bat_rw_boundary_pct", 0.15)
    six_pct      = stats.get("bat_rw_six_pct", 0.06)
    death_sr     = stats.get("bat_death_rw_sr", sr)
    dot_pct      = stats.get("bat_rw_dot_pct", 0.33)

    # Finisher: strike rate jumps sharply specifically in death overs
    if (death_sr - sr) > 25 and six_pct >= 0.06:
        return BatterArchetype.FINISHER
    # Aggressive: high SR everywhere, high combined boundary rate
    if sr >= 145 and (boundary_pct + six_pct) >= 0.22:
        return BatterArchetype.AGGRESSIVE
    # Anchor: happy to absorb dots, low six-hitting, moderate-low SR
    if sr <= 118 and dot_pct >= 0.36 and six_pct < 0.05:
        return BatterArchetype.ANCHOR
    return BatterArchetype.ACCUMULATOR


def classify_bowler(stats: dict) -> BowlerArchetype:
    pp_eco     = stats.get("bowl_pp_rw_economy", 7.5)
    mid_eco    = stats.get("bowl_mid_rw_economy", 8.0)
    death_eco  = stats.get("bowl_death_rw_economy", 9.2)
    death_wkt  = stats.get("bowl_death_rw_wicket_pct", 0.055)
    pp_wkt     = stats.get("bowl_pp_rw_wicket_pct", 0.07)
    dot_pct    = stats.get("bowl_rw_dot_pct", 0.33)

    spread = max(pp_eco, mid_eco, death_eco) - min(pp_eco, mid_eco, death_eco)
    if spread < 1.0:
        return BowlerArchetype.ALL_PHASE

    if death_eco > 9.0 and death_wkt >= 0.06 and death_wkt >= pp_wkt:
        return BowlerArchetype.DEATH
    if pp_eco < mid_eco - 1.0 and pp_eco < death_eco - 1.0:
        return BowlerArchetype.POWERPLAY
    # No bowling_style feature in this dataset — a dot-heavy, containing
    # profile through the middle overs is the closest available proxy for spin.
    if dot_pct >= 0.36 and mid_eco <= pp_eco:
        return BowlerArchetype.SPINNER
    return BowlerArchetype.ALL_PHASE


# ═══════════════════════════════════════════════════════════════════════════
# Settling period (new batter caution beyond raw balls-faced)
# ═══════════════════════════════════════════════════════════════════════════

SETTLE_THRESHOLDS = {
    BatterArchetype.AGGRESSIVE:   4,
    BatterArchetype.FINISHER:     5,
    BatterArchetype.ACCUMULATOR:  8,
    BatterArchetype.ANCHOR:      12,
}


def is_settled(archetype: BatterArchetype, balls_faced_this_innings: int) -> bool:
    return balls_faced_this_innings >= SETTLE_THRESHOLDS.get(archetype, 8)


# ═══════════════════════════════════════════════════════════════════════════
# Venue x bowler-archetype synergy (richer venue intelligence)
# ═══════════════════════════════════════════════════════════════════════════

def venue_bowler_synergy(venue_stats: dict, bowler_archetype: BowlerArchetype) -> float:
    """Small bounded risk delta (negative = favours the bowler) when a venue's
    known character matches a bowler's specialism — a batter/bowler-archetype x
    venue interaction the model can't see because archetype isn't a feature."""
    wicket_pct = venue_stats.get("venue_rw_wicket_pct", 0.054)
    death_sr   = venue_stats.get("venue_rw_death_sr", 165.0)

    delta = 0.0
    if bowler_archetype == BowlerArchetype.SPINNER and wicket_pct >= 0.06:
        delta -= 0.05   # turning/wicket-taking venue suits a specialist spinner
    if bowler_archetype == BowlerArchetype.DEATH and death_sr <= 155:
        delta -= 0.05   # venue where death overs are historically contained
    return delta


# ═══════════════════════════════════════════════════════════════════════════
# Risk index: archetype + situation -> a single bounded [-1, 1] aggression signal
# ═══════════════════════════════════════════════════════════════════════════

def compute_risk_index(
    archetype: BatterArchetype,
    over: int,
    balls_faced_this_innings: int,
    is_chasing: bool,
    pressure_index: float,
    partnership_balls: int,
    bowler_archetype: BowlerArchetype,
    venue_stats: dict,
) -> float:
    """Returns a bounded aggression signal in [-1, 1]. Positive = batter is
    taking on more risk than their raw stats alone imply; negative = playing
    it safer. Feeds into apply_risk_adjustment() below."""

    settled = is_settled(archetype, balls_faced_this_innings)
    partner_recently_dismissed = partnership_balls < 3   # fresh, unsettled pair

    risk = {
        BatterArchetype.AGGRESSIVE:   0.35,
        BatterArchetype.FINISHER:     0.15,
        BatterArchetype.ACCUMULATOR:  0.0,
        BatterArchetype.ANCHOR:      -0.25,
    }[archetype]

    # Finishers specifically escalate late — finer-grained than the model's
    # single coarse "death" phase bucket (overs 15-20 individually matter).
    if archetype == BatterArchetype.FINISHER and over >= 15:
        risk += 0.10 * min(over - 14, 6) / 6

    # New-batter settling period, worse if the partnership itself is fresh
    if not settled:
        risk -= 0.20
        if partner_recently_dismissed:
            risk -= 0.10

    # Chasing under real pressure nudges any archetype toward more risk
    if is_chasing and pressure_index > 2:
        risk += min(pressure_index * 0.02, 0.25)

    risk += venue_bowler_synergy(venue_stats, bowler_archetype)

    return max(-1.0, min(1.0, risk))


# ═══════════════════════════════════════════════════════════════════════════
# Applying the risk signal to a model output distribution
# ═══════════════════════════════════════════════════════════════════════════

_MAX_TRANSFER = 0.06   # hard cap: at most 6 percentage points of probability moved


def apply_risk_adjustment(probs: Dict[str, float], risk: float) -> Dict[str, float]:
    """
    Couples boundary-hitting and dismissal probability together, bounded and
    reversible, on top of the model's own predicted distribution.

    risk > 0: move mass from dot/single into fours/sixes/wickets together
              (an attacking shot is more likely to go for a boundary AND more
              likely to get the batter out than a defensive one).
    risk < 0: move mass the other way (fewer big shots, fewer risk-driven
              dismissals) — pure phase/venue-forced caution, not modelled
              as "safer against getting out to bad luck".
    """
    if risk == 0:
        return probs

    p = dict(probs)
    move = min(abs(risk), 1.0) * _MAX_TRANSFER

    if risk > 0:
        take_from = ["0", "1"]
        give_to   = [("4", 0.45), ("6", 0.35), ("W", 0.20)]
    else:
        take_from = ["4", "6", "W"]
        give_to   = [("0", 0.6), ("1", 0.4)]

    available = sum(p.get(k, 0.0) for k in take_from)
    if available <= 0:
        return probs
    actual_move = min(move, available * 0.5)   # never drain a bucket below half itself

    for k in take_from:
        share = p.get(k, 0.0) / available
        p[k] = max(0.0, p.get(k, 0.0) - actual_move * share)
    for k, w in give_to:
        p[k] = p.get(k, 0.0) + actual_move * w

    total = sum(p.values())
    if total <= 0:
        return probs
    return {k: v / total for k, v in p.items()}
