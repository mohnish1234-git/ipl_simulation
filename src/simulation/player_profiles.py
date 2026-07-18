"""
Small, match-state-only adjustment layer on top of the model's own output.

Why this file got much smaller:

  The model is trained on features.csv, which already includes recency-
  weighted batter/bowler stats, batter-vs-bowler matchup numbers, venue and
  player-at-venue stats, and live momentum features (partnership, recent
  runs, consecutive dots/boundaries, etc.) — see feature_engineer.py. All of
  that signal is genuinely learned by the model already.

  The previous version of this file added a second, parallel system on top
  of that: batter/bowler "archetypes" derived from the same stats, a
  percentile-fitted risk index, a settling-period concept, and a venue x
  bowler-archetype synergy term — all just to nudge probabilities that the
  model was, in principle, already free to learn from its own features.
  That's more moving parts than this pipeline needs, and it duplicates
  signal the model already has direct access to.

  What's genuinely NOT in the model's feature set is pure in-match team
  SITUATION — specifically, "the top order has just collapsed" as a live
  event within THIS match. So that's the one thing kept here: a single,
  transparent rule based only on wickets down and the current over, with no
  player classification, no fitted thresholds, and no venue interaction.
"""

from typing import Dict

_MAX_TRANSFER = 0.06   # hard cap: at most 6 percentage points of probability moved


def apply_collapse_adjustment(probs: Dict[str, float], wickets_down: int,
                               over: int) -> Dict[str, float]:
    """
    If 3+ wickets are already down and it's still early in the innings
    (before over 15), new/incoming batters are more likely to play it safe —
    take the single, rotate strike, avoid the big shot — rather than assume
    the model's own per-ball features fully capture "the top order just
    fell apart" as a live situational signal.

    Bounded and reversible, same shape as any other post-hoc nudge: moves at
    most a few percentage points from boundaries into singles/dots, scaled
    by how bad the collapse is (3 down = mild, 5+ down = more caution).
    No-op with fewer than 3 wickets down, or from over 15 onward (death
    overs call for risk regardless of how the innings started).
    """
    if wickets_down < 3 or over >= 15:
        return probs

    severity = min((wickets_down - 2) * 0.02, _MAX_TRANSFER)
    if severity <= 0:
        return probs

    p = dict(probs)
    take_from = ["4", "6"]
    give_to   = [("1", 0.7), ("0", 0.3)]

    available = sum(p.get(k, 0.0) for k in take_from)
    if available <= 0:
        return probs
    move = min(severity, available * 0.5)   # never drain a bucket below half itself

    for k in take_from:
        share = p.get(k, 0.0) / available
        p[k] = max(0.0, p.get(k, 0.0) - move * share)
    for k, w in give_to:
        p[k] = p.get(k, 0.0) + move * w

    total = sum(p.values())
    if total <= 0:
        return probs
    return {k: v / total for k, v in p.items()}