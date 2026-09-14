"""Chart rendering for the money-flow divergence view.

Draws a 3-panel stacked chart — Price / CMF / RSI on a shared x-axis — so the
divergence between price and money flow, graded by RSI position, is visible at
a glance. Returns PNG bytes for Telegram ``reply_photo``.

Labels are intentionally ASCII (no CJK) so the chart never depends on a
Noto/CJK font being installed — the textual card already carries the Chinese.
"""

from __future__ import annotations

from io import BytesIO

import matplotlib

matplotlib.use("Agg")  # headless (VPS) — must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402

from v2.moneyflow.indicators import cmf_series, rsi_series  # noqa: E402
from v2.moneyflow.models import DivergenceConfig  # noqa: E402

plt.rcParams["axes.unicode_minus"] = False

_VERDICT_ASCII = {
    ("accumulation", "strong"): "Accumulation (strong)",
    ("accumulation", "moderate"): "Accumulation (moderate)",
    ("distribution", "strong"): "Distribution (strong)",
    ("distribution", "moderate"): "Distribution (moderate)",
}


def render_moneyflow_chart(
    ticker: str,
    prices,
    cfg: DivergenceConfig,
    *,
    signal=None,
    max_bars: int = 60,
) -> bytes | None:
    """Render Price / CMF / RSI as a stacked PNG. None if data is too short.

    ``prices`` is a chronological list of OHLCV bars (objects with
    .high/.low/.close/.volume). ``signal`` (a MoneyFlowSignal) is optional —
    when present its verdict is shown in the suptitle.
    """
    highs = [p.high for p in prices]
    lows = [p.low for p in prices]
    closes = [p.close for p in prices]
    volumes = [p.volume for p in prices]

    rsi_full = rsi_series(closes, cfg.rsi_window)
    cmf_full = cmf_series(highs, lows, closes, volumes, cfg.cmf_window)
    if not rsi_full or not cmf_full:
        return None

    # Common tail length where all three series line up on the same last bars.
    n = min(len(closes), len(rsi_full), len(cmf_full), max_bars)
    if n < 2:
        return None
    price = closes[-n:]
    rsi = rsi_full[-n:]
    cmf = [c if c is not None else float('nan') for c in cmf_full[-n:]]
    x = list(range(n))

    fig, (ax_p, ax_c, ax_r) = plt.subplots(
        3, 1, figsize=(8, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 2, 2], "hspace": 0.12},
    )

    # --- Panel 1: Price ---
    up = price[-1] >= price[0]
    pcolor = "#2ca02c" if up else "#d62728"
    ax_p.plot(x, price, linewidth=2, color=pcolor)
    ax_p.fill_between(x, price, price[0], alpha=0.12, color=pcolor)
    ax_p.axhline(price[0], color="#888", linewidth=0.8, linestyle="--", alpha=0.6)
    ax_p.set_ylabel("Price ($)")
    ax_p.grid(True, alpha=0.3)

    # --- Panel 2: CMF (money flow) ---
    ax_c.bar(
        x, cmf, width=0.9,
        color=["#2ca02c" if v >= 0 else "#d62728" for v in cmf],
        alpha=0.7,
    )
    ax_c.axhline(0, color="#333", linewidth=0.8)
    ax_c.axhline(cfg.cmf_inflow_threshold, color="#2ca02c", linewidth=0.7,
                 linestyle=":", alpha=0.7)
    ax_c.axhline(cfg.cmf_outflow_threshold, color="#d62728", linewidth=0.7,
                 linestyle=":", alpha=0.7)
    ax_c.set_ylabel("CMF")
    ax_c.grid(True, alpha=0.3)

    # --- Panel 3: RSI (momentum) ---
    ax_r.plot(x, rsi, linewidth=2, color="#7030a0")
    ax_r.axhspan(cfg.rsi_low, cfg.rsi_high, color="#999", alpha=0.12)  # neutral zone
    ax_r.axhline(cfg.rsi_overbought, color="#d62728", linewidth=0.8,
                 linestyle="--", alpha=0.7)
    ax_r.axhline(cfg.rsi_oversold, color="#2ca02c", linewidth=0.8,
                 linestyle="--", alpha=0.7)
    ax_r.set_ylim(0, 100)
    ax_r.set_yticks([0, 30, 50, 70, 100])
    ax_r.set_ylabel("RSI")
    ax_r.set_xlabel(f"last {n} trading days")
    ax_r.grid(True, alpha=0.3)
    # annotate current RSI value
    ax_r.annotate(
        f"{rsi[-1]:.0f}", xy=(x[-1], rsi[-1]),
        xytext=(4, 0), textcoords="offset points",
        va="center", fontsize=10, fontweight="bold", color="#7030a0",
    )

    for ax in (ax_p, ax_c, ax_r):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    verdict = _VERDICT_ASCII.get(
        (getattr(signal, "kind", None), getattr(signal, "strength", None)),
        "No clear divergence",
    )
    fig.suptitle(f"{ticker}  ·  Money Flow  ·  {verdict}",
                 fontsize=13, fontweight="bold")

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()
