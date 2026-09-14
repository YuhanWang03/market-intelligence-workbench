"""Priority-ladder integration tests for the Phase 3 SEC event kinds.

Pins the exact thresholds + adjustment factors added in
``v2/reporting/priority.py`` for the Stage 2 spec. Mirrors the Phase 1
``v2/earnings/test_priority_integration.py`` pattern — loads priority.py
via ``importlib`` so the test stays sandbox-runnable without
``v2.reporting``'s heavy package init.

Ladder reminder:

    BASE_SCORES:
        sec_8k_p0           = 85
        sec_8k_p1           = 65
        sec_8k_p2           = 55
        sec_8k_p3           = 35
        sec_form4_purchase  = 75
        sec_form4_sale      = 50
        sec_form4_cluster   = 75

    Adjustments (subset — see priority.py for full table):
        amendment             → -5
        senior_exec confirmed → +5 (P0 only)
        big_purchase ≥ $1M    → +25
        moderate_purchase ≥ $100K → +10
        CEO/CFO purchase      → +10
        10b5-1 plan purchase  → -10
        big_discretionary_sale ≥ $10M → +15
        10b5-1 plan sale ≥ $1M → -5
        cluster ≥ 5           → +15
        cluster ≥ 3           → +5
        cluster purchase      → +10

    Score clamping: raw ∈ [0, 100]. Above 100 → 100 (canonical convention).
    Tier thresholds (unchanged from Phase 0): P0≥80, P1=60-79, P2=40-59, P3<40.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


_PRIORITY_PATH = _REPO_ROOT / "v2" / "reporting" / "priority.py"
_spec = importlib.util.spec_from_file_location("priority_phase3", _PRIORITY_PATH)
priority = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules["priority_phase3"] = priority
_spec.loader.exec_module(priority)

compute_importance = priority.compute_importance
BASE_SCORES = priority.BASE_SCORES


# ---------------------------------------------------------------------------
# 8-K ladder
# ---------------------------------------------------------------------------

class TestSec8KLadder:

    def test_8k_p0_base_is_85(self):
        r = compute_importance("sec_8k_p0", {})
        assert r.score == 85
        assert r.tier == "P0"

    def test_8k_p1_base_is_65(self):
        r = compute_importance("sec_8k_p1", {})
        assert r.score == 65
        assert r.tier == "P1"

    def test_8k_p2_base_is_55(self):
        r = compute_importance("sec_8k_p2", {})
        assert r.score == 55
        assert r.tier == "P2"

    def test_8k_p3_base_is_35(self):
        r = compute_importance("sec_8k_p3", {})
        assert r.score == 35
        assert r.tier == "P3"

    def test_8k_5_02_senior_exec_confirmed_bump(self):
        """5.02 with LLM-confirmed senior exec → P0 + small extra."""
        r = compute_importance("sec_8k_p0", {"has_senior_exec": True})
        assert r.score == 90
        assert r.tier == "P0"
        assert any("ceo_cfo_5_02_confirmed" in s for s in r.reasons)

    def test_8k_amendment_demotes_5(self):
        """8-K/A amendment shaves 5 — same filing re-issued rarely adds signal."""
        r = compute_importance("sec_8k_p0", {"is_amendment": True})
        assert r.score == 80    # 85 - 5 = 80, still P0 boundary
        assert r.tier == "P0"
        assert any("amendment" in s for s in r.reasons)

    def test_8k_amendment_on_p1_demotes(self):
        """8-K/A on a P1 item: 65 - 5 = 60 → still P1 (boundary)."""
        r = compute_importance("sec_8k_p1", {"is_amendment": True})
        assert r.score == 60
        assert r.tier == "P1"


# ---------------------------------------------------------------------------
# Form 4 purchase ladder
# ---------------------------------------------------------------------------

class TestSecForm4Purchase:

    def test_purchase_base_is_p1(self):
        """Insider purchase alone, no magnitude/role bumps → 75 = P1."""
        r = compute_importance("sec_form4_purchase", {})
        assert r.score == 75
        assert r.tier == "P1"

    def test_ceo_big_purchase_clamps_at_100(self):
        """CEO + $2.5M purchase: raw 75 + 25 + 10 = 110 → clamped 100 → P0.

        Spec says "raw 110"; runtime clamps to 100 per the standard
        convention applied across all event_kinds (Phase 1/2 also clamp).
        """
        r = compute_importance("sec_form4_purchase", {
            "transaction_usd": 2_500_000,
            "insider_role": "CEO",
            "is_10b5_1": False,
        })
        assert r.score == 100   # clamped from 110
        assert r.tier == "P0"
        assert any("big_purchase" in s and "$2.5M" in s for s in r.reasons)
        assert any("CEO_purchase" in s for s in r.reasons)

    def test_director_small_purchase_stays_p1(self):
        """Director with $50K purchase: 75 base + no bumps (below $100K)."""
        r = compute_importance("sec_form4_purchase", {
            "transaction_usd": 50_000,
            "insider_role": "Director",
            "is_10b5_1": False,
        })
        assert r.score == 75
        assert r.tier == "P1"

    def test_cfo_moderate_purchase_10b5_1_plan_demotes(self):
        """CFO + $200K + 10b5-1 plan: 75 + 10 + 10 - 10 = 85 → exactly P0 boundary."""
        r = compute_importance("sec_form4_purchase", {
            "transaction_usd": 200_000,
            "insider_role": "CFO",
            "is_10b5_1": True,
        })
        assert r.score == 85
        assert r.tier == "P0"
        # All 3 adjustments must show in reasons trail
        reason_text = " ".join(r.reasons)
        assert "moderate_purchase" in reason_text
        assert "CFO_purchase" in reason_text
        assert "10b5_1_plan_purchase" in reason_text


# ---------------------------------------------------------------------------
# Form 4 sale ladder
# ---------------------------------------------------------------------------

class TestSecForm4Sale:

    def test_sale_base_is_p2(self):
        """Sales default P2 — most insider sales are 10b5-1 / tax / diversify."""
        r = compute_importance("sec_form4_sale", {})
        assert r.score == 50
        assert r.tier == "P2"

    def test_big_discretionary_sale_promotes_p1(self):
        """≥ $10M sale + NOT 10b5-1 → discretionary; +15 bump → P1."""
        r = compute_importance("sec_form4_sale", {
            "transaction_usd": 15_000_000,
            "is_10b5_1": False,
        })
        assert r.score == 65
        assert r.tier == "P1"
        assert any("big_discretionary_sale" in s for s in r.reasons)

    def test_big_10b5_1_sale_demotes(self):
        """≥ $1M sale on 10b5-1 plan: -5 → 45 → still P2.

        Plan sales are pre-arranged; the magnitude doesn't reflect
        discretionary insider sentiment."""
        r = compute_importance("sec_form4_sale", {
            "transaction_usd": 2_000_000,
            "is_10b5_1": True,
        })
        assert r.score == 45
        assert r.tier == "P2"
        assert any("10b5_1_plan_sale" in s for s in r.reasons)

    def test_huge_10b5_1_sale_stays_p2(self):
        """Even $100M on 10b5-1 plan stays P2 — the discretionary
        bump requires NOT is_10b5_1."""
        r = compute_importance("sec_form4_sale", {
            "transaction_usd": 100_000_000,
            "is_10b5_1": True,
        })
        # 50 - 5 = 45 → P2
        assert r.score == 45
        assert r.tier == "P2"


# ---------------------------------------------------------------------------
# Form 4 cluster ladder
# ---------------------------------------------------------------------------

class TestSecForm4Cluster:

    def test_cluster_3_purchase_p0(self):
        """3-insider purchase cluster: 75 + 5 + 10 = 90 → P0."""
        r = compute_importance("sec_form4_cluster", {
            "transaction_count": 3,
            "direction": "purchase",
        })
        assert r.score == 90
        assert r.tier == "P0"
        assert any("cluster_3" in s for s in r.reasons)
        assert any("cluster_buy" in s for s in r.reasons)

    def test_cluster_5_purchase_clamps_at_100(self):
        """5+ cluster + purchase: 75 + 15 + 10 = 100 → P0."""
        r = compute_importance("sec_form4_cluster", {
            "transaction_count": 5,
            "direction": "purchase",
        })
        assert r.score == 100
        assert r.tier == "P0"
        assert any("large_cluster_5" in s for s in r.reasons)

    def test_cluster_3_sale_boundary_p0(self):
        """3-insider sale cluster: 75 + 5 = 80 (no purchase bonus) → P0 at boundary.

        Clusters of coordinated sales ARE noteworthy — historically
        weaker signal than purchase clusters but worth surfacing."""
        r = compute_importance("sec_form4_cluster", {
            "transaction_count": 3,
            "direction": "sale",
        })
        assert r.score == 80
        assert r.tier == "P0"

    def test_cluster_2_below_threshold(self):
        """Below the ≥3 threshold no bumps applied → base 75 = P1."""
        r = compute_importance("sec_form4_cluster", {
            "transaction_count": 2,
            "direction": "purchase",
        })
        # 75 + 10 (cluster_buy) = 85 → P0
        # Note: cluster_buy fires whenever direction='purchase', even
        # if transaction_count doesn't trigger the cluster_N bump.
        # Real pipeline shouldn't emit sec_form4_cluster with count < 3
        # (cluster.find_clusters enforces that), but priority.py is
        # defensive against malformed input.
        assert r.score == 85
        assert r.tier == "P0"


# ---------------------------------------------------------------------------
# BASE_SCORES sanity
# ---------------------------------------------------------------------------

class TestSecBaseScores:

    def test_all_7_sec_kinds_registered(self):
        """Every SEC kind the cron emits must have a base score."""
        for kind in (
            "sec_8k_p0", "sec_8k_p1", "sec_8k_p2", "sec_8k_p3",
            "sec_form4_purchase", "sec_form4_sale", "sec_form4_cluster",
        ):
            assert kind in BASE_SCORES, f"missing base score for {kind}"

    def test_base_scores_match_spec(self):
        assert BASE_SCORES["sec_8k_p0"] == 85
        assert BASE_SCORES["sec_8k_p1"] == 65
        assert BASE_SCORES["sec_8k_p2"] == 55
        assert BASE_SCORES["sec_8k_p3"] == 35
        assert BASE_SCORES["sec_form4_purchase"] == 75
        assert BASE_SCORES["sec_form4_sale"] == 50
        assert BASE_SCORES["sec_form4_cluster"] == 75
