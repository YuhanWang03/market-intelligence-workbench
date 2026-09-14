"""Priority ladder integration tests for Phase 4 macro kinds.

Pins the 7 new BASE_SCORES + the adjustment-factor math added in
``v2/reporting/priority.py``. Mirrors the structure of the Phase 3
``v2/sec/test_priority_integration.py`` (importlib-loads ``priority.py``
directly to avoid pulling the heavy v2.reporting init chain into the
sandbox).
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_PRIORITY_SRC = _REPO_ROOT / "v2" / "reporting" / "priority.py"


def _load_priority():
    """Load v2/reporting/priority.py without triggering v2.reporting
    __init__ (which pulls matplotlib + v2.data via the formatter shim
    chain)."""
    spec = importlib.util.spec_from_file_location(
        "priority_phase4", _PRIORITY_SRC,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["priority_phase4"] = mod
    spec.loader.exec_module(mod)
    return mod


priority = _load_priority()
compute_importance = priority.compute_importance
BASE_SCORES = priority.BASE_SCORES


# ---------------------------------------------------------------------------
# BASE_SCORES wiring
# ---------------------------------------------------------------------------

def test_seven_macro_base_scores_registered():
    """Phase 4 ack'd kinds + scores pinned."""
    assert BASE_SCORES["macro_release_p2"] == 55
    assert BASE_SCORES["macro_release_p1"] == 65
    assert BASE_SCORES["macro_release_p0"] == 85
    assert BASE_SCORES["macro_snapshot_p3"] == 35
    assert BASE_SCORES["macro_vix_spike"] == 85
    assert BASE_SCORES["macro_curve_flip"] == 65
    assert BASE_SCORES["macro_weekly"] == 65


# ---------------------------------------------------------------------------
# macro_release_* — surprise sigma ladder
# ---------------------------------------------------------------------------

def test_macro_release_in_line_p2():
    """< 1σ → no adjustment → 55 → P2."""
    r = compute_importance("macro_release_p2", {"surprise_sigma": 0.3})
    assert r.score == 55
    assert r.tier == "P2"


def test_macro_release_moderate_1sigma_p1():
    """1σ ≤ sigma < 2σ → +5 → 70 → P1."""
    r = compute_importance("macro_release_p1", {"surprise_sigma": 1.2})
    assert r.score == 70, f"expected 70, got {r.score} reasons={r.reasons}"
    assert r.tier == "P1"


def test_macro_release_2sigma_above_p1():
    """2σ ≤ sigma < 3σ → +10 → 75 → P1 strong."""
    r = compute_importance("macro_release_p1", {"surprise_sigma": 2.1})
    assert r.score == 75
    assert r.tier == "P1"
    assert any("big_surprise" in reason for reason in r.reasons)


def test_macro_release_extreme_3sigma_p0_clamps():
    """≥ 3σ → +20 → 85 + 20 = 105 → clamps to 100 → P0."""
    r = compute_importance("macro_release_p0", {"surprise_sigma": 3.5})
    assert r.score == 100
    assert r.tier == "P0"
    assert any("extreme_surprise" in reason for reason in r.reasons)


def test_macro_release_negative_sigma_uses_abs():
    """Surprise direction doesn't matter — magnitude does."""
    r = compute_importance("macro_release_p0", {"surprise_sigma": -3.2})
    assert r.score == 100
    assert r.tier == "P0"


# ---------------------------------------------------------------------------
# FOMC SEP shift
# ---------------------------------------------------------------------------

def test_fomc_sep_hawkish_shift_p0():
    """FOMC + hawkish SEP shift → +15 → 100 → P0."""
    r = compute_importance("macro_release_p0", {
        "is_fomc": True, "sep_shift": "hawkish_shift",
    })
    assert r.score == 100
    assert r.tier == "P0"
    assert any("sep_hawkish_shift" in reason for reason in r.reasons)


def test_fomc_sep_dovish_shift_p0():
    """FOMC + dovish SEP shift → +15."""
    r = compute_importance("macro_release_p0", {
        "is_fomc": True, "sep_shift": "dovish_shift",
    })
    assert r.score == 100
    assert r.tier == "P0"


def test_fomc_no_sep_change_no_nudge():
    """No SEP shift → no nudge from the SEP factor."""
    r = compute_importance("macro_release_p1", {
        "is_fomc": True, "sep_shift": "no_change",
    })
    assert r.score == 65         # base only, no surprise sigma
    assert r.tier == "P1"


def test_fomc_sell_side_hawkish_unexpected():
    """Sell-side majority hawkish → +10 nudge."""
    r = compute_importance("macro_release_p1", {
        "is_fomc": True,
        "sell_side_consensus": "hawkish_unexpected",
    })
    assert r.score == 75         # 65 + 10
    assert r.tier == "P1"
    assert any("sell_side_hawkish" in reason for reason in r.reasons)


# ---------------------------------------------------------------------------
# macro_vix_spike
# ---------------------------------------------------------------------------

def test_vix_spike_strong_p0():
    """VIX +25% → +10 → 95 → P0."""
    r = compute_importance("macro_vix_spike", {"vix_pct_change_1d": 0.25})
    assert r.score == 95
    assert r.tier == "P0"
    assert any("vix_strong" in reason for reason in r.reasons)


def test_vix_spike_extreme_30pct_p0_clamps():
    """VIX +32% → +20 → 105 → clamps 100 → P0."""
    r = compute_importance("macro_vix_spike", {"vix_pct_change_1d": 0.32})
    assert r.score == 100
    assert r.tier == "P0"
    assert any("vix_extreme" in reason for reason in r.reasons)


def test_vix_spike_base_only_when_pct_missing():
    """Empty metadata → no nudge → base 85 → P0."""
    r = compute_importance("macro_vix_spike", {})
    assert r.score == 85
    assert r.tier == "P0"


# ---------------------------------------------------------------------------
# macro_curve_flip
# ---------------------------------------------------------------------------

def test_curve_flip_p1():
    r = compute_importance("macro_curve_flip", {})
    assert r.score == 75         # 65 + 10
    assert r.tier == "P1"
    assert any("yield_curve_inverted" in reason for reason in r.reasons)


# ---------------------------------------------------------------------------
# macro_weekly + macro_snapshot_p3
# ---------------------------------------------------------------------------

def test_macro_weekly_default_p1():
    r = compute_importance("macro_weekly", {})
    assert r.score == 65
    assert r.tier == "P1"


def test_macro_snapshot_p3_is_ambient():
    """Daily ambient snapshot — bare base, no adjustments fire."""
    r = compute_importance("macro_snapshot_p3", {})
    assert r.score == 35
    assert r.tier == "P3"
