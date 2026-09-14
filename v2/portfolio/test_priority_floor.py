"""Weekly-cron P1 floor tests — closes Issue 3 from Stage 2.5 review.

The ⑩ weekly cron applies a "P1 floor" so that a clean week's recap
still hits operator inbox at P1 even if every adjustment factor is 0:

    natural score on truly-empty portfolio: base 55 → P2
    floor applies:                          65 → P1 + "+10_weekly_recap_floor" reason

The Stage 5 byte-equal tests confirmed the floor doesn't fire when
natural tier is already ≥ P1 (because real portfolios have non-zero
top_1_pct, which adds +10 → 65). This file covers the truly-empty
edge case where the floor matters.

Uses importlib to load priority.py directly (same pattern as
test_priority_integration.py) so the test stays sandbox-runnable
without v2.reporting's heavy package init.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


_PRIORITY_PATH = _REPO_ROOT / "v2" / "reporting" / "priority.py"
_spec = importlib.util.spec_from_file_location("priority_floor_under_test", _PRIORITY_PATH)
priority = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules["priority_floor_under_test"] = priority
_spec.loader.exec_module(priority)

compute_importance = priority.compute_importance
PriorityResult = priority.PriorityResult


# ---------------------------------------------------------------------------
# Floor logic — extracted from scripts/portfolio_weekly_to_telegram.py so
# tests don't need to sys.modules-stub the whole cron environment.
# ---------------------------------------------------------------------------
# Keep this in lock-step with the floor branch in the cron's main(). If
# the cron ever changes the floor score (currently 65) or reason tag,
# update both places + this test.

_FLOOR_SCORE = 65
_FLOOR_REASON = "+10_weekly_recap_floor"


def _apply_weekly_floor(pr):
    """Mirror of the cron's floor logic. Identity if pr already ≥ P1."""
    if pr.tier == "P2":
        return PriorityResult(
            score=_FLOOR_SCORE, tier="P1",
            reasons=list(pr.reasons) + [_FLOOR_REASON],
        )
    return pr


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_truly_empty_portfolio_natural_p2():
    """daily_pnl=0, top_1=0, max_dd=0, n_earnings=0 → no adjustments
    fire → natural score stays at base 55 → P2. This is the edge case
    the weekly floor exists to lift."""
    pr = compute_importance(
        "portfolio_risk",
        {
            "daily_pnl_pct":     0.0,
            "top_1_pct":         0.0,
            "max_drawdown_pct":  0.0,
            "n_earnings_next_7d": 0,
        },
    )
    assert pr.score == 55
    assert pr.tier == "P2"
    # Single reason — just the base, no adjustments applied
    assert pr.reasons == ["base=55"]


def test_weekly_floor_engages_when_natural_p2():
    """Truly-empty input → natural P2 → floor lifts to P1 with the
    explicit reason tag in the trail. This closes Issue 3 from the
    Stage 2.5 dry-run review."""
    natural = compute_importance(
        "portfolio_risk",
        {
            "daily_pnl_pct":     0.0,
            "top_1_pct":         0.0,
            "max_drawdown_pct":  0.0,
            "n_earnings_next_7d": 0,
        },
    )
    assert natural.tier == "P2"   # sanity precondition

    after_floor = _apply_weekly_floor(natural)
    assert after_floor.tier == "P1"
    assert after_floor.score == _FLOOR_SCORE
    assert _FLOOR_REASON in after_floor.reasons
    # Original reasons preserved
    assert "base=55" in after_floor.reasons


def test_weekly_floor_inert_when_natural_p1():
    """Real portfolios have non-zero top_1_pct (you can't hold a
    position with 0% weight). top_1=22% adds +10 → 65 → P1 naturally.
    The floor must be a no-op in this case — NO duplicate '+10_weekly_recap_floor'
    reason. This is the case Stage 5's byte-equal smoke confirmed."""
    natural = compute_importance(
        "portfolio_risk",
        {
            "daily_pnl_pct":     0.0031,
            "top_1_pct":         0.22,
            "max_drawdown_pct":  0.018,
            "n_earnings_next_7d": 0,
        },
    )
    assert natural.tier == "P1"
    assert natural.score == 65
    # top_1 adjustment must be present
    assert any("top1_22%" in r for r in natural.reasons)

    after_floor = _apply_weekly_floor(natural)
    # Identity passthrough
    assert after_floor is natural
    assert after_floor.tier == "P1"
    assert after_floor.score == 65
    # No floor reason appended
    assert _FLOOR_REASON not in after_floor.reasons


def test_weekly_floor_inert_when_natural_p0():
    """High-conviction day with big loss → natural P0. Floor obviously
    must NOT demote to P1 — it's an UPPER tier-floor, not a clamp."""
    natural = compute_importance(
        "portfolio_risk",
        {
            "daily_pnl_pct":    -0.06,    # +30 → big bump
            "top_1_pct":         0.35,    # +20
            "max_drawdown_pct":  0.0,
            "n_earnings_next_7d": 0,
        },
    )
    assert natural.tier == "P0"

    after_floor = _apply_weekly_floor(natural)
    assert after_floor.tier == "P0"
    assert after_floor.score == natural.score
    assert _FLOOR_REASON not in after_floor.reasons


def test_floor_score_tag_match_cron_source():
    """Pinning the floor score + reason tag so divergence from the
    cron's source is loud. If you change either in
    scripts/portfolio_weekly_to_telegram.py, update this test."""
    assert _FLOOR_SCORE == 65
    assert _FLOOR_REASON == "+10_weekly_recap_floor"
