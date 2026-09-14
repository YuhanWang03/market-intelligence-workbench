"""Integration tests for the earnings-flavoured priority kinds.

These pin the Stage-0/Stage-2 ladder so the cron output behaviour can't
silently regress:

    D-3 / pending  → P2 (45)
    D-1 / D-0      → P1 (60)
    summary base   → P1 (70)
    summary |Δ| ≥ 10% → P0 (70 + 30 = 100, capped)
    held position  → +15  (universal)
    watchlist only → +10  (universal)
    guidance_lowered → +10  (summary only)
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Load priority.py via importlib so we don't pull v2.reporting's package
# __init__ (which transitively needs matplotlib / v2.data through the
# formatters module) while keeping this integration test self-contained.
_PRIORITY_PATH = _REPO_ROOT / "v2" / "reporting" / "priority.py"
_spec = importlib.util.spec_from_file_location("priority_under_test", _PRIORITY_PATH)
priority = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules["priority_under_test"] = priority
_spec.loader.exec_module(priority)

BASE_SCORES = priority.BASE_SCORES
compute_importance = priority.compute_importance


# ---------------------------------------------------------------------------
# Reminder ladder (D-3 / D-1 / D-0)
# ---------------------------------------------------------------------------

class TestReminderPriorityLadder:

    def test_d3_uses_p2_score_45(self):
        r = compute_importance("earnings_reminder_d3", {})
        assert r.score == 45
        assert r.tier == "P2"
        assert "base=45" in r.reasons[0]

    def test_d1_promoted_to_p1(self):
        r = compute_importance("earnings_reminder_d1", {})
        assert r.score == 60
        assert r.tier == "P1"

    def test_d0_uses_p1(self):
        r = compute_importance("earnings_reminder_d0", {})
        assert r.score == 60
        assert r.tier == "P1"

    def test_d3_held_position_promoted_to_p1(self):
        """D-3 base 45 + held 15 = 60 → just barely P1."""
        r = compute_importance(
            "earnings_reminder_d3",
            {"is_held_position": True},
        )
        assert r.score == 60
        assert r.tier == "P1"
        assert any("held_position" in s for s in r.reasons)

    def test_d3_watchlist_only_stays_p2(self):
        """D-3 base 45 + watchlist 10 = 55 → still P2 (need 60 for P1)."""
        r = compute_importance(
            "earnings_reminder_d3",
            {"is_watchlist": True},
        )
        assert r.score == 55
        assert r.tier == "P2"


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

class TestSummaryPriority:

    def test_summary_base_is_p1(self):
        r = compute_importance("earnings_summary", {})
        assert r.score == 70
        assert r.tier == "P1"

    def test_summary_big_surprise_promoted_to_p0(self):
        """|surprise| ≥ 10% adds +30. 70 + 30 = 100 (capped). P0."""
        r = compute_importance("earnings_summary", {"surprise_pct": 0.12})
        assert r.score == 100   # clamped at 100
        assert r.tier == "P0"
        assert any("big_surprise" in s for s in r.reasons)

    def test_summary_with_held_position_promoted(self):
        """Held = +15. 70 + 15 = 85 → P0 (≥80)."""
        r = compute_importance(
            "earnings_summary",
            {"is_held_position": True},
        )
        assert r.score == 85
        assert r.tier == "P0"

    def test_summary_small_surprise_stays_p1(self):
        """|surprise| < 10% → no adjustment. Stays at 70 = P1."""
        r = compute_importance("earnings_summary", {"surprise_pct": 0.05})
        assert r.score == 70
        assert r.tier == "P1"

    def test_summary_negative_big_surprise_also_promoted(self):
        """abs() takes the magnitude — MISS by 12% is just as P0 as BEAT."""
        r = compute_importance("earnings_summary", {"surprise_pct": -0.12})
        assert r.score == 100
        assert r.tier == "P0"

    def test_guidance_lowered_adds_10(self):
        """Stage 2 metadata hook: guidance_lowered → +10."""
        r = compute_importance(
            "earnings_summary",
            {"guidance_lowered": True},
        )
        assert r.score == 80
        assert r.tier == "P0"   # 70 + 10 = 80, exactly the P0 cutoff
        assert any("guidance_lowered" in s for s in r.reasons)

    def test_summary_compound_held_plus_big_surprise_caps_at_100(self):
        r = compute_importance(
            "earnings_summary",
            {
                "is_held_position": True,
                "surprise_pct": 0.15,
                "guidance_lowered": True,
            },
        )
        # base 70 + 30 (big surprise) + 10 (guidance) + 15 (held) = 125 → 100
        assert r.score == 100
        assert r.tier == "P0"


# ---------------------------------------------------------------------------
# Pending placeholder
# ---------------------------------------------------------------------------

class TestPendingPriority:

    def test_pending_uses_p2(self):
        r = compute_importance("earnings_pending", {})
        assert r.score == 45
        assert r.tier == "P2"

    def test_pending_held_promotes_to_p1(self):
        r = compute_importance(
            "earnings_pending",
            {"is_held_position": True},
        )
        assert r.score == 60
        assert r.tier == "P1"


# ---------------------------------------------------------------------------
# held trumps watchlist
# ---------------------------------------------------------------------------

class TestHeldTrumpsWatchlist:

    def test_held_priority_overrides_watchlist(self):
        """When both flags are True, compute_importance only adds the
        held bonus (+15), NOT both (+15 and +10 = +25)."""
        r = compute_importance(
            "earnings_reminder_d1",
            {"is_held_position": True, "is_watchlist": True},
        )
        # base 60 + held 15 = 75 (not 60 + 25 = 85)
        assert r.score == 75
        assert any("held_position" in s for s in r.reasons)
        assert not any("watchlist" in s and "held" not in s for s in r.reasons)


# ---------------------------------------------------------------------------
# Stage 2 BASE_SCORES table sanity
# ---------------------------------------------------------------------------

class TestBaseScoresShape:

    def test_all_earnings_kinds_registered(self):
        """Every earnings kind the crons/responders emit must have a base
        score so unknown kinds don't silently fall back to default(=55)."""
        for kind in (
            "earnings_reminder_d3",
            "earnings_reminder_d1",
            "earnings_reminder_d0",
            "earnings_summary",
            "earnings_pending",
        ):
            assert kind in BASE_SCORES, f"missing base score for {kind}"

    def test_base_scores_match_spec(self):
        """Spec from Stage 2:
            D-3 = 45 (P2), D-1 = 60 (P1), D-0 = 60 (P1),
            pending = 45 (P2), summary = 70 (P1)."""
        assert BASE_SCORES["earnings_reminder_d3"] == 45
        assert BASE_SCORES["earnings_reminder_d1"] == 60
        assert BASE_SCORES["earnings_reminder_d0"] == 60
        assert BASE_SCORES["earnings_pending"] == 45
        assert BASE_SCORES["earnings_summary"] == 70
