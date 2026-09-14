"""Priority-ladder integration tests for the portfolio_risk event_kind.

Pins the exact thresholds in v2/reporting/priority.py for the Phase-2
Stage-2 spec. Mirrors the test_priority_integration.py pattern used by
the earnings agent — loads priority.py via importlib so the test stays
sandbox-runnable without v2.reporting's package init.

Ladder reminder:

    base = 55 (P2)

    daily_pnl_pct ≤ -5%       → +30  (→ P0)
    daily_pnl_pct ≤ -2%       → +10
    top_1_pct ≥ 30%           → +20
    top_1_pct ≥ 20%           → +10
    max_drawdown_pct ≥ 10%    → +15
    n_earnings_next_7d ≥ 3    → +10

Design philosophy:
- single elevated factor → P1 (alert, not urgent)
- single-day drop ≥ 5%   → P0 (urgent)
- multi-factor stack     → P0 (systemic risk)
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_PRIORITY_PATH = _REPO_ROOT / "v2" / "reporting" / "priority.py"
_spec = importlib.util.spec_from_file_location("priority_under_test_p2", _PRIORITY_PATH)
priority = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules["priority_under_test_p2"] = priority
_spec.loader.exec_module(priority)

BASE_SCORES = priority.BASE_SCORES
compute_importance = priority.compute_importance


# ---------------------------------------------------------------------------
# Single-factor ladder
# ---------------------------------------------------------------------------

class TestPortfolioRiskSingleFactor:

    def test_normal_day_p2(self):
        """Empty metadata → base 55 → P2."""
        r = compute_importance("portfolio_risk", {})
        assert r.score == 55
        assert r.tier == "P2"
        assert r.reasons == ["base=55"]

    def test_high_top1_alone_p1(self):
        """top_1 = 35% alone: 55 + 20 = 75 → P1 (not P0).

        Concentration is a "concerning, not urgent" signal — it gets
        P1 visibility on its own and only escalates when stacked with
        other factors. Day-of-disaster (-5% loss) is reserved for the
        single-factor P0 bump."""
        r = compute_importance("portfolio_risk", {"top_1_pct": 0.35})
        assert r.score == 75
        assert r.tier == "P1"
        assert any("top1_35%" in s for s in r.reasons)

    def test_top1_20pct_alone_p1(self):
        """top_1 = 20%: 55 + 10 = 65 → P1 (just over the threshold)."""
        r = compute_importance("portfolio_risk", {"top_1_pct": 0.20})
        assert r.score == 65
        assert r.tier == "P1"

    def test_top1_just_below_20pct_stays_p2(self):
        """top_1 = 19.9% — no bump. Stays at base 55 → P2."""
        r = compute_importance("portfolio_risk", {"top_1_pct": 0.199})
        assert r.score == 55
        assert r.tier == "P2"

    def test_5pct_loss_alone_p0(self):
        """-6% loss alone: 55 + 30 = 85 → P0. The only single-factor
        bump that reaches P0."""
        r = compute_importance("portfolio_risk", {"daily_pnl_pct": -0.06})
        assert r.score == 85
        assert r.tier == "P0"
        assert any("daily_loss" in s for s in r.reasons)

    def test_5pct_loss_exact_boundary_p0(self):
        """-5.0% exact boundary still triggers the +30 bump."""
        r = compute_importance("portfolio_risk", {"daily_pnl_pct": -0.05})
        assert r.score == 85
        assert r.tier == "P0"

    def test_2pct_loss_alone_p1(self):
        """-3% loss: 55 + 10 = 65 → P1."""
        r = compute_importance("portfolio_risk", {"daily_pnl_pct": -0.03})
        assert r.score == 65
        assert r.tier == "P1"

    def test_small_loss_stays_p2(self):
        """-1% loss → no bump. Stays at base."""
        r = compute_importance("portfolio_risk", {"daily_pnl_pct": -0.01})
        assert r.score == 55
        assert r.tier == "P2"

    def test_drawdown_10pct_alone_p1(self):
        """1M max_drawdown = -12%: 55 + 15 = 70 → P1.

        Drawdown is also "concerning, not urgent" — the single-factor
        bump is only +15 (lands P1)."""
        r = compute_importance("portfolio_risk", {"max_drawdown_pct": -0.12})
        assert r.score == 70
        assert r.tier == "P1"
        assert any("drawdown_12%" in s for s in r.reasons)

    def test_drawdown_handles_unsigned(self):
        """The scorer uses abs() so callers can pass either signed or
        unsigned drawdown — both must produce the same bump."""
        r1 = compute_importance("portfolio_risk", {"max_drawdown_pct": -0.15})
        r2 = compute_importance("portfolio_risk", {"max_drawdown_pct": 0.15})
        assert r1.score == r2.score == 70

    def test_3_earnings_alone_p1(self):
        """3 held-position earnings in 7 days: 55 + 10 = 65 → P1.

        Event-cluster risk — a single bad print can whipsaw the whole
        book if multiple holdings report in the same week."""
        r = compute_importance(
            "portfolio_risk", {"n_earnings_next_7d": 3},
        )
        assert r.score == 65
        assert r.tier == "P1"
        assert any("earnings_density_3" in s for s in r.reasons)

    def test_2_earnings_no_bump(self):
        """2 earnings — below threshold, no bump."""
        r = compute_importance(
            "portfolio_risk", {"n_earnings_next_7d": 2},
        )
        assert r.score == 55
        assert r.tier == "P2"


# ---------------------------------------------------------------------------
# Multi-factor stacking
# ---------------------------------------------------------------------------

class TestPortfolioRiskMultiFactor:

    def test_multi_factor_stack_p0(self):
        """top_1 = 35%, daily_pnl = -3%, max_dd = -12%:
            55 + 20 (top1) + 10 (moderate_loss) + 15 (drawdown) = 100 → P0.

        Multiple "concerning, not urgent" factors stack into a
        systemic-risk P0."""
        r = compute_importance(
            "portfolio_risk",
            {
                "top_1_pct":        0.35,
                "daily_pnl_pct":   -0.03,
                "max_drawdown_pct": -0.12,
            },
        )
        assert r.score == 100
        assert r.tier == "P0"
        # All three factors must surface as reasons.
        assert any("top1_35%" in s for s in r.reasons)
        assert any("moderate_loss" in s for s in r.reasons)
        assert any("drawdown" in s for s in r.reasons)

    def test_all_factors_stack_clamps_at_100(self):
        """Worst-case: all 4 factor types fire. The score clamps at 100."""
        r = compute_importance(
            "portfolio_risk",
            {
                "top_1_pct":         0.40,
                "daily_pnl_pct":    -0.07,
                "max_drawdown_pct": -0.15,
                "n_earnings_next_7d": 5,
            },
        )
        # 55 + 30 + 20 + 15 + 10 = 130 → clamped at 100
        assert r.score == 100
        assert r.tier == "P0"

    def test_two_p1_factors_stay_p1(self):
        """top_1 = 25% (+10) + drawdown = -12% (+15) = 80 → P1.
        Below the 5%-loss-equivalent P0 boundary (≥80 with current
        thresholds) — just barely lands P0 actually.

        With existing thresholds (P0 ≥ 80), 55+10+15=80 is exactly P0.
        Documenting the boundary so a future thresholds change is loud."""
        r = compute_importance(
            "portfolio_risk",
            {
                "top_1_pct":        0.25,
                "max_drawdown_pct": -0.12,
            },
        )
        assert r.score == 80
        # 80 lands P0 under the existing 80/60/40 ladder. Two
        # "P1-individual" factors stacking pushes to P0 — that's the
        # systemic-risk philosophy.
        assert r.tier == "P0"

    def test_pnl_picks_higher_bracket_only(self):
        """daily_pnl_pct triggers only the highest applicable threshold,
        not both. -7% loss should add +30, not +30 AND +10."""
        r = compute_importance(
            "portfolio_risk", {"daily_pnl_pct": -0.07},
        )
        # If both brackets applied: 55 + 30 + 10 = 95. With only the
        # higher: 55 + 30 = 85.
        assert r.score == 85
        # Exactly one "daily" reason in the trail.
        daily_reasons = [s for s in r.reasons if "daily_loss" in s or "moderate_loss" in s]
        assert len(daily_reasons) == 1

    def test_top1_picks_higher_bracket_only(self):
        """Same elif-discipline for the top_1 ladder."""
        r = compute_importance(
            "portfolio_risk", {"top_1_pct": 0.40},
        )
        # 55 + 20 = 75, not 55 + 20 + 10 = 85.
        assert r.score == 75
        top_reasons = [s for s in r.reasons if "top1_" in s]
        assert len(top_reasons) == 1


# ---------------------------------------------------------------------------
# BASE_SCORES sanity
# ---------------------------------------------------------------------------

class TestPortfolioBaseScores:

    def test_portfolio_risk_base_is_55(self):
        assert BASE_SCORES["portfolio_risk"] == 55

    def test_portfolio_alert_base_is_85(self):
        """The 'portfolio_alert' kind exists for severity-1 overrides
        the cron may dispatch separately (e.g. a hand-tuned operator
        ping). Pinning to 85 so it lands P0 with no adjustments."""
        assert BASE_SCORES["portfolio_alert"] == 85
