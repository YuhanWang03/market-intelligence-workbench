"""Unit tests for the money-flow divergence pipeline.

Indicators are checked against hand-reasoned series; the detector is checked
against every branch of its truth table. No network, no LLM.
"""

from __future__ import annotations

import types

import pytest

from v2.moneyflow.cards import format_view_card
from v2.moneyflow.detector import detect_divergence, read_axes
from v2.moneyflow.indicators import (
    chaikin_money_flow,
    cmf_series,
    rsi,
    window_return,
)
from v2.moneyflow.models import DivergenceConfig

CFG = DivergenceConfig()


# --------------------------------------------------------------------------
# bar builders — CMF sign is controlled by where close sits in [low, high],
# independent of the close-to-close trend that drives RSI / price return.
# --------------------------------------------------------------------------

def _bar(close: float, flow: str):
    if flow == "inflow":          # close near the high → positive multiplier
        high, low = close + 0.1, close - 1.0
    elif flow == "outflow":       # close near the low → negative multiplier
        high, low = close + 1.0, close - 0.1
    else:                         # centered → ~0 multiplier
        high, low = close + 0.5, close - 0.5
    return types.SimpleNamespace(high=high, low=low, close=close, volume=1000)


def _bars(closes, flow):
    return [_bar(c, flow) for c in closes]


# --------------------------------------------------------------------------
# indicators
# --------------------------------------------------------------------------

def test_cmf_all_inflow_is_positive():
    closes = [100.0] * 20
    assert chaikin_money_flow(
        [c + 0.1 for c in closes], [c - 1.0 for c in closes],
        closes, [1000] * 20, window=20,
    ) > 0.5


def test_cmf_all_outflow_is_negative():
    closes = [100.0] * 20
    assert chaikin_money_flow(
        [c + 1.0 for c in closes], [c - 0.1 for c in closes],
        closes, [1000] * 20, window=20,
    ) < -0.5


def test_cmf_flat_bar_contributes_zero():
    # high == low on every bar → multiplier 0 → CMF 0
    assert chaikin_money_flow([100] * 20, [100] * 20, [100] * 20, [1000] * 20) == 0.0


def test_cmf_none_on_short_history():
    assert chaikin_money_flow([1, 2], [1, 2], [1, 2], [1, 1], window=20) is None


def test_rsi_monotonic_up_near_100():
    closes = [100 + i for i in range(30)]
    assert rsi(closes, 14) > 99


def test_rsi_flat_is_fifty():
    assert rsi([100.0] * 30, 14) == 50.0


def test_window_return():
    closes = [100.0] * 20 + [110.0]
    assert window_return(closes, 20) == pytest.approx(0.10)


# --------------------------------------------------------------------------
# detector truth table
# --------------------------------------------------------------------------

def test_accumulation_strong_flat_price_inflow_low_rsi():
    # 40 flat bars, then a mild 20-bar decline → 20d return ≈ -2% (flat band)
    # and a recent downtrend that drags RSI into the oversold/low zone.
    closes = [100.0] * 40 + [100.0 - 0.1 * i for i in range(1, 21)]
    sig = detect_divergence("AAA", _bars(closes, "inflow"), CFG)
    assert sig is not None
    assert sig.kind == "accumulation"
    assert sig.strength == "strong"
    assert sig.flow_state == "inflow"


def test_distribution_strong_flat_price_outflow_high_rsi():
    closes = [100.0] * 40 + [100.0 + 0.1 * i for i in range(1, 21)]
    sig = detect_divergence("BBB", _bars(closes, "outflow"), CFG)
    assert sig is not None
    assert sig.kind == "distribution"
    assert sig.strength == "strong"


def test_contradictory_accumulation_dropped():
    # Sustained uptrend (RSI pegged overbought) then a flat plateau: recent
    # price is flat but RSI is still hot AND hasn't risen over the window
    # (no bullish divergence) → accumulation branch hits the contradictory
    # cell → dropped. (A flat/down price with RSI *rising* would instead be
    # a legit bullish-divergence strong signal — that's covered elsewhere.)
    closes = [100.0 + i for i in range(40)] + [140.0] * 20
    sig = detect_divergence("CCC", _bars(closes, "inflow"), CFG)
    assert sig is None


def test_no_divergence_when_price_and_flow_agree():
    # strong uptrend + inflow → price rising with money → not a divergence.
    closes = [100.0 + 0.5 * i for i in range(60)]
    sig = detect_divergence("DDD", _bars(closes, "inflow"), CFG)
    assert sig is None


def test_none_on_insufficient_history():
    closes = [100.0] * 10
    assert detect_divergence("EEE", _bars(closes, "inflow"), CFG) is None


# --------------------------------------------------------------------------
# read_axes (bot pull path) — always returns a reading when data suffices,
# even when no verdict fires
# --------------------------------------------------------------------------

def test_read_axes_returns_reading_even_without_verdict():
    # steady uptrend + inflow → no divergence verdict, but the raw read
    # must still be available for the on-demand bot card.
    closes = [100.0 + 0.5 * i for i in range(60)]
    reading = read_axes("FFF", _bars(closes, "inflow"), CFG)
    assert reading is not None
    assert reading.price_state == "up"
    assert reading.flow_state == "inflow"
    assert 0 <= reading.rsi <= 100
    # and detect_divergence agrees there's no clean verdict here
    assert detect_divergence("FFF", _bars(closes, "inflow"), CFG) is None


def test_read_axes_none_on_short_history():
    assert read_axes("GGG", _bars([100.0] * 10, "inflow"), CFG) is None


# --------------------------------------------------------------------------
# view card (bot)
# --------------------------------------------------------------------------

def test_view_card_no_verdict_states_plainly():
    closes = [100.0 + 0.5 * i for i in range(60)]
    reading = read_axes("FFF", _bars(closes, "inflow"), CFG)
    card = format_view_card(reading, None)
    assert "资金流分析 · FFF" in card
    assert "无明显量价背离" in card
    assert "CMF" in card and "RSI" in card


def test_view_card_with_verdict_shows_kind_and_narration():
    closes = [100.0] * 40 + [100.0 + 0.1 * i for i in range(1, 21)]
    sig = detect_divergence("BBB", _bars(closes, "outflow"), CFG)
    assert sig is not None
    sig.bull, sig.bear = "或为获利了结", "高位量能背离，承接乏力"
    card = format_view_card(sig, sig)
    assert "疑似派发/出货" in card
    assert "空头视角" in card


# --------------------------------------------------------------------------
# cmf_series (charting) + chart render
# --------------------------------------------------------------------------

def test_cmf_series_alignment_and_sign():
    closes = [100.0] * 30
    highs = [c + 0.1 for c in closes]   # close near high → positive
    lows = [c - 1.0 for c in closes]
    vols = [1000] * 30
    series = cmf_series(highs, lows, closes, vols, window=20)
    assert len(series) == len(closes) - 20 + 1   # aligned to closes[window-1:]
    assert all(v is not None and v > 0.5 for v in series)


def _matplotlib_can_render() -> bool:
    """Probe: some sandbox envs ship an incompatible numpy×matplotlib pair
    (e.g. numpy 2.4 + matplotlib 3.7) where fill_between/savefig raise. Skip
    the render test there — it's an env issue, not a code issue. Production
    (VPS) renders fine."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from io import BytesIO
        fig, ax = plt.subplots()
        ax.fill_between([0, 1, 2], [1, 2, 1], 0)
        fig.savefig(BytesIO(), format="png")
        plt.close(fig)
        return True
    except Exception:
        return False


def test_render_chart_returns_png_bytes():
    pytest.importorskip("matplotlib")
    if not _matplotlib_can_render():
        pytest.skip("matplotlib/numpy rendering broken in this env (not a code issue)")
    from v2.moneyflow.chart import render_moneyflow_chart

    closes = [100.0] * 40 + [100.0 + 0.1 * i for i in range(1, 21)]
    sig = detect_divergence("BBB", _bars(closes, "outflow"), CFG)
    png = render_moneyflow_chart("BBB", _bars(closes, "outflow"), CFG, signal=sig)
    assert isinstance(png, (bytes, bytearray))
    assert png[:8] == b"\x89PNG\r\n\x1a\n"   # PNG magic number
