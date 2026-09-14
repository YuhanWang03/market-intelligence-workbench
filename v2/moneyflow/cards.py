"""Telegram card formatting for money-flow divergence signals.

Numbers come from the detector (Python-computed); the 多头/空头 lines come
from the narrator (LLM, no numbers). A fixed disclaimer footer makes the
proxy nature explicit on every card.
"""

from __future__ import annotations

import html

from v2.moneyflow.models import MoneyFlowReading, MoneyFlowSignal

_KIND_ZH = {"accumulation": "疑似吸筹", "distribution": "疑似派发/出货"}
_STRENGTH_ZH = {"strong": "强信号", "moderate": "中等强度"}
_PRICE_ZH = {"up": "上涨", "flat": "横盘", "down": "下跌"}
_FLOW_ZH = {"inflow": "净流入", "outflow": "净流出", "neutral": "中性"}
_ZONE_ZH = {
    "oversold": "超卖区", "low": "低位区", "neutral": "中性区",
    "high": "高位区", "overbought": "超买区",
}
_DIVERGENCE_ZH = {"bullish": "底背离", "bearish": "顶背离"}


def format_signal_card(signal: MoneyFlowSignal, *, price_window: int = 20) -> str:
    """Render one divergence signal as an HTML Telegram card."""
    kind_zh = _KIND_ZH.get(signal.kind, signal.kind)
    strength_zh = _STRENGTH_ZH.get(signal.strength, signal.strength)

    # Only surface the RSI-divergence chip when it *confirms* the verdict —
    # a bullish divergence on a distribution card (or vice-versa) would read
    # as self-contradictory. Such opposing divergences never drive the
    # strength grade either (see detector), so hiding them here is honest.
    _confirming = {"accumulation": "bullish", "distribution": "bearish"}
    div_suffix = ""
    if signal.rsi_divergence == _confirming.get(signal.kind):
        div_suffix = f" · {_DIVERGENCE_ZH[signal.rsi_divergence]}"

    lines = [
        f"📊 <b>资金流背离 · {html.escape(signal.ticker)}</b>",
        f"{kind_zh} · {strength_zh}",
        "",
        f"价格：近 {price_window} 日{_PRICE_ZH.get(signal.price_state, signal.price_state)}"
        f" ({signal.price_return:+.1%})",
        f"资金流：CMF {signal.cmf:+.2f}（{_FLOW_ZH.get(signal.flow_state, signal.flow_state)}）",
        f"动量：RSI {signal.rsi:.0f}（{_ZONE_ZH.get(signal.rsi_zone, signal.rsi_zone)}{div_suffix}）",
    ]

    if signal.bull:
        lines += ["", f"💡 多头视角：{html.escape(signal.bull)}"]
    if signal.bear:
        lines.append(f"🔻 空头视角：{html.escape(signal.bear)}")

    lines += [
        "",
        "⚠️ 量价代理指标（CMF/RSI）推断，非逐笔主力资金；背离信号存在假信号，"
        "请结合基本面与机构持仓交叉验证。",
    ]
    return "\n".join(lines)


def _axes_lines(reading: MoneyFlowReading, price_window: int, show_div: bool) -> list[str]:
    div_suffix = ""
    if show_div and reading.rsi_divergence in _DIVERGENCE_ZH:
        div_suffix = f" · {_DIVERGENCE_ZH[reading.rsi_divergence]}"
    return [
        f"价格：近 {price_window} 日"
        f"{_PRICE_ZH.get(reading.price_state, reading.price_state)}"
        f" ({reading.price_return:+.1%})",
        f"资金流：CMF {reading.cmf:+.2f}"
        f"（{_FLOW_ZH.get(reading.flow_state, reading.flow_state)}）",
        f"动量：RSI {reading.rsi:.0f}"
        f"（{_ZONE_ZH.get(reading.rsi_zone, reading.rsi_zone)}{div_suffix}）",
    ]


def format_view_card(
    reading: MoneyFlowReading,
    signal: MoneyFlowSignal | None = None,
    *,
    price_window: int = 20,
) -> str:
    """On-demand (bot) card: always shows the three-axis read; adds the
    verdict + 多空 narration when a divergence fired, else states plainly
    that none did. ``signal`` (when present) carries kind/strength/bull/bear.
    """
    lines = [f"📊 <b>资金流分析 · {html.escape(reading.ticker)}</b>"]

    if signal is not None:
        confirming = {"accumulation": "bullish", "distribution": "bearish"}
        show_div = signal.rsi_divergence == confirming.get(signal.kind)
        lines.append(
            f"{_KIND_ZH.get(signal.kind, signal.kind)} · "
            f"{_STRENGTH_ZH.get(signal.strength, signal.strength)}"
        )
    else:
        show_div = False
        lines.append("当前无明显量价背离，以下为三轴读数")

    lines.append("")
    lines += _axes_lines(reading, price_window, show_div)

    if signal is not None:
        if signal.bull:
            lines += ["", f"💡 多头视角：{html.escape(signal.bull)}"]
        if signal.bear:
            lines.append(f"🔻 空头视角：{html.escape(signal.bear)}")

    lines += [
        "",
        "⚠️ 量价代理指标（CMF/RSI）推断，非逐笔主力资金；请结合基本面与机构持仓交叉验证。",
    ]
    return "\n".join(lines)
