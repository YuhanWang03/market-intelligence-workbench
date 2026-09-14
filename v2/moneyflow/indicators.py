"""Pure technical-indicator functions for the money-flow pipeline.

No side effects, no I/O — every function takes plain sequences and returns
plain numbers, so they're trivial to unit-test against hand-computed series.

Indicators:
    chaikin_money_flow — the "money flow" axis (volume-weighted, -1..1)
    rsi_series / rsi   — the momentum axis (Wilder, 0..100)
    window_return      — the price axis (simple return over N bars)
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def chaikin_money_flow(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    volumes: Sequence[float],
    window: int = 20,
) -> float | None:
    """Chaikin Money Flow over the last *window* bars.

    CMF = Σ(MFV, window) / Σ(volume, window), where the money-flow
    multiplier MFM = ((C-L) - (H-C)) / (H-L) and MFV = MFM × volume.
    Bars with H==L contribute 0 (undefined multiplier). Returns None when
    there aren't enough bars or total volume is zero.
    """
    if window <= 0 or len(closes) < window:
        return None
    h = np.asarray(highs[-window:], dtype=float)
    l = np.asarray(lows[-window:], dtype=float)
    c = np.asarray(closes[-window:], dtype=float)
    v = np.asarray(volumes[-window:], dtype=float)

    rng = h - l
    with np.errstate(divide="ignore", invalid="ignore"):
        mfm = np.where(rng > 0, ((c - l) - (h - c)) / rng, 0.0)
    total_vol = float(v.sum())
    if total_vol <= 0:
        return None
    return float((mfm * v).sum() / total_vol)


def cmf_series(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    volumes: Sequence[float],
    window: int = 20,
) -> list[float | None]:
    """Rolling Chaikin Money Flow, one value per bar from index ``window-1``.

    Returned list is aligned to ``closes[window-1:]`` — used for charting.
    An element is ``None`` when that window had zero total volume.
    """
    n = len(closes)
    if window <= 0 or n < window:
        return []
    out: list[float | None] = []
    for i in range(window - 1, n):
        s = slice(i - window + 1, i + 1)
        out.append(chaikin_money_flow(highs[s], lows[s], closes[s], volumes[s], window))
    return out


def rsi_series(closes: Sequence[float], window: int = 14) -> list[float]:
    """Wilder's RSI for every bar from index *window* onward.

    Returned list is aligned to ``closes[window:]`` — i.e. ``out[-1]`` is
    the RSI of the latest close. Empty list when there's not enough data.
    """
    if window <= 0 or len(closes) < window + 1:
        return []
    deltas = np.diff(np.asarray(closes, dtype=float))
    gains = np.clip(deltas, 0.0, None)
    losses = -np.clip(deltas, None, 0.0)

    avg_gain = float(gains[:window].mean())
    avg_loss = float(losses[:window].mean())

    def _rsi(ag: float, al: float) -> float:
        if al == 0.0:
            return 100.0 if ag > 0.0 else 50.0
        rs = ag / al
        return 100.0 - 100.0 / (1.0 + rs)

    out = [_rsi(avg_gain, avg_loss)]
    for i in range(window, len(deltas)):
        avg_gain = (avg_gain * (window - 1) + float(gains[i])) / window
        avg_loss = (avg_loss * (window - 1) + float(losses[i])) / window
        out.append(_rsi(avg_gain, avg_loss))
    return out


def rsi(closes: Sequence[float], window: int = 14) -> float | None:
    """Latest RSI value, or None if there isn't enough history."""
    series = rsi_series(closes, window)
    return series[-1] if series else None


def window_return(closes: Sequence[float], window: int) -> float | None:
    """Simple return of the latest close vs *window* bars ago."""
    if window <= 0 or len(closes) < window + 1:
        return None
    prev = float(closes[-1 - window])
    if prev <= 0:
        return None
    return (float(closes[-1]) - prev) / prev
