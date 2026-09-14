"""Timeframe-aware technical indicators and price-zone analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

Timeframe = Literal["30m", "1h", "1d", "1w"]


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _rolling_mean(values: list[float], period: int) -> list[float | None]:
    output: list[float | None] = [None] * len(values)
    running = 0.0
    for index, value in enumerate(values):
        running += value
        if index >= period:
            running -= values[index - period]
        if index + 1 >= period:
            output[index] = running / period
    return output


def _ema(values: list[float], period: int) -> list[float | None]:
    output: list[float | None] = [None] * len(values)
    if len(values) < period:
        return output
    current = sum(values[:period]) / period
    output[period - 1] = current
    alpha = 2 / (period + 1)
    for index in range(period, len(values)):
        current = values[index] * alpha + current * (1 - alpha)
        output[index] = current
    return output


def _rsi(values: list[float], period: int = 14) -> list[float | None]:
    gains = [0.0]
    losses = [0.0]
    for previous, current in zip(values, values[1:]):
        delta = current - previous
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))
    avg_gain = _rolling_mean(gains, period)
    avg_loss = _rolling_mean(losses, period)
    return [None if gain is None or loss is None else (100.0 if loss == 0 else 100 - 100 / (1 + gain / loss)) for gain, loss in zip(avg_gain, avg_loss)]


def _obv(closes: list[float], volumes: list[float]) -> list[float]:
    output = [0.0]
    for index in range(1, len(closes)):
        direction = 1 if closes[index] > closes[index - 1] else -1 if closes[index] < closes[index - 1] else 0
        output.append(output[-1] + direction * volumes[index])
    return output


def _cmf_series(bars: list[dict], period: int = 20) -> list[float | None]:
    flows = []
    volumes = []
    for bar in bars:
        spread = float(bar["high"]) - float(bar["low"])
        volume = float(bar["volume"])
        multiplier = 0.0 if spread <= 0 else ((float(bar["close"]) - float(bar["low"])) - (float(bar["high"]) - float(bar["close"]))) / spread
        flows.append(multiplier * volume)
        volumes.append(volume)
    flow_ma = _rolling_mean(flows, period)
    volume_ma = _rolling_mean(volumes, period)
    return [None if flow is None or volume in (None, 0) else flow / volume for flow, volume in zip(flow_ma, volume_ma)]


def _atr(bars: list[dict], period: int = 14) -> list[float | None]:
    true_ranges: list[float] = []
    for index, bar in enumerate(bars):
        previous_close = bars[index - 1]["close"] if index else bar["close"]
        true_ranges.append(max(
            bar["high"] - bar["low"],
            abs(bar["high"] - previous_close),
            abs(bar["low"] - previous_close),
        ))
    return _rolling_mean(true_ranges, period)


def enrich_bars(bars: list[dict], timeframe: Timeframe) -> list[dict]:
    """Attach timeframe-bound SMA, ATR and volume-MA values to every bar."""
    closes = [float(bar["close"]) for bar in bars]
    volumes = [float(bar["volume"]) for bar in bars]
    series = {
        "sma20": _rolling_mean(closes, 20),
        "sma50": _rolling_mean(closes, 50),
        "sma200": _rolling_mean(closes, 200),
        "atr14": _atr(bars, 14),
        "volumeMA20": _rolling_mean(volumes, 20),
        "ema20": _ema(closes, 20),
        "rsi14": _rsi(closes, 14),
        "obv": _obv(closes, volumes),
        "cmf20": _cmf_series(bars, 20),
    }
    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    macd_values = [None if a is None or b is None else a - b for a, b in zip(ema12, ema26)]
    compact_macd = [value if value is not None else 0.0 for value in macd_values]
    signal = _ema(compact_macd, 9)
    series["macd"] = macd_values
    series["macdSignal"] = [None if macd_values[index] is None else signal[index] for index in range(len(signal))]
    enriched: list[dict] = []
    for index, bar in enumerate(bars):
        indicators = {
            "timeframe": timeframe,
            **{
                key: round(value[index], 6) if value[index] is not None else None
                for key, value in series.items()
            },
        }
        volume_ma = indicators["volumeMA20"]
        indicators["volumeRatio"] = (
            round(bar["volume"] / volume_ma, 6)
            if volume_ma not in (None, 0)
            else None
        )
        bar_return = bar["close"] / bar["open"] - 1 if bar["open"] else None
        enriched.append({
            **bar,
            "barReturn": round(bar_return, 8) if bar_return is not None else None,
            "indicators": indicators,
        })
    return enriched


@dataclass
class _Pivot:
    kind: Literal["high", "low"]
    price: float
    index: int
    timestamp: str
    width: float
    reversal: float
    volume_ratio: float
    recency: float


def _pivot_window(timeframe: Timeframe) -> int:
    return {"30m": 3, "1h": 3, "1d": 5, "1w": 3}[timeframe]


def _detect_pivots(bars: list[dict], timeframe: Timeframe) -> list[_Pivot]:
    window = _pivot_window(timeframe)
    pivots: list[_Pivot] = []
    count = len(bars)
    if count < window * 2 + 1:
        return pivots
    half_life = max(count * .28, 30)
    for index in range(window, count - window):
        bar = bars[index]
        neighbors = bars[index - window:index] + bars[index + 1:index + window + 1]
        is_high = bar["high"] >= max(item["high"] for item in neighbors)
        is_low = bar["low"] <= min(item["low"] for item in neighbors)
        if not is_high and not is_low:
            continue
        atr = bar["indicators"].get("atr14") or max(bar["close"] * .01, .01)
        # The configured value is the total zone width; store a half-width
        # because zones are represented as center +/- width internally.
        width = max(bar["close"] * .003, atr * .3) * .5
        lookahead = bars[index + 1:min(count, index + window * 3 + 1)]
        volume_ratio = bar["indicators"].get("volumeRatio") or 1.0
        recency = math.exp(-(count - index - 1) / half_life)
        if is_high:
            future_low = min((item["low"] for item in lookahead), default=bar["low"])
            pivots.append(_Pivot(
                "high", bar["high"], index, bar["timestamp"], width,
                max((bar["high"] - future_low) / atr, 0), volume_ratio, recency,
            ))
        if is_low:
            future_high = max((item["high"] for item in lookahead), default=bar["high"])
            pivots.append(_Pivot(
                "low", bar["low"], index, bar["timestamp"], width,
                max((future_high - bar["low"]) / atr, 0), volume_ratio, recency,
            ))
    return pivots


def _cluster_pivots(pivots: list[_Pivot]) -> list[dict]:
    zones: list[dict] = []
    clusters: list[list[_Pivot]] = []
    for pivot in sorted(pivots, key=lambda item: item.price):
        target = next((cluster for cluster in clusters if abs(
            pivot.price - sum(item.price for item in cluster) / len(cluster)
        ) <= max(pivot.width, max(item.width for item in cluster))), None)
        if target is None:
            clusters.append([pivot])
        else:
            target.append(pivot)
    for cluster in clusters:
        weights = [1 + item.recency + min(item.volume_ratio, 3) * .25 for item in cluster]
        center = sum(item.price * weight for item, weight in zip(cluster, weights)) / sum(weights)
        width = max(max(item.width for item in cluster), center * .003 * .5)
        touch_count = len(cluster)
        recency = max(item.recency for item in cluster)
        reversal = sum(item.reversal for item in cluster) / touch_count
        volume_confirmation = sum(item.volume_ratio for item in cluster) / touch_count
        score = (
            touch_count * 2
            + min(reversal, 4) * 1.15
            + min(volume_confirmation, 3) * .65
            + recency * 2
        )
        latest = max(cluster, key=lambda item: item.index)
        role_weight = {
            kind: sum(weight for item, weight in zip(cluster, weights) if item.kind == kind)
            for kind in ("high", "low")
        }
        zones.append({
            "originalRole": max(role_weight, key=role_weight.get),
            "lower": center - width,
            "upper": center + width,
            "midpoint": center,
            "score": score,
            "recencyScore": recency,
            "touchCount": touch_count,
            "lastTouchedAt": latest.timestamp,
            "reversalMagnitudeAtr": reversal,
            "volumeConfirmation": volume_confirmation,
        })
    return zones


def _merge_nearby_zones(zones: list[dict], merge_distance: float) -> list[dict]:
    """Merge overlapping/nearby pivot zones before assigning S/R roles."""
    merged: list[dict] = []
    for zone in sorted(zones, key=lambda item: item["lower"]):
        if not merged:
            merged.append(dict(zone))
            continue
        previous = merged[-1]
        gap = zone["lower"] - previous["upper"]
        combined_span = max(previous["upper"], zone["upper"]) - min(previous["lower"], zone["lower"])
        allowed_span = max(
            previous["upper"] - previous["lower"],
            zone["upper"] - zone["lower"],
        ) + merge_distance * 2.5
        if gap > 0 and (gap > merge_distance or combined_span > allowed_span):
            merged.append(dict(zone))
            continue
        previous_touches = previous["touchCount"]
        new_touches = zone["touchCount"]
        total_touches = previous_touches + new_touches
        previous_score = previous["score"]
        new_score = zone["score"]
        total_score = max(previous_score + new_score + 1.0, previous_score, new_score)
        role = previous["originalRole"] if previous_score >= new_score else zone["originalRole"]
        midpoint = (
            previous["midpoint"] * previous_score + zone["midpoint"] * new_score
        ) / max(previous_score + new_score, .01)
        half_width = min(
            combined_span / 2,
            max(
                (previous["upper"] - previous["lower"]) / 2,
                (zone["upper"] - zone["lower"]) / 2,
            ) + merge_distance * .5,
        )
        previous.update({
            "originalRole": role,
            "lower": midpoint - half_width,
            "upper": midpoint + half_width,
            "midpoint": midpoint,
            "score": total_score,
            "recencyScore": max(previous["recencyScore"], zone["recencyScore"]),
            "touchCount": total_touches,
            "lastTouchedAt": max(previous["lastTouchedAt"], zone["lastTouchedAt"]),
            "reversalMagnitudeAtr": (
                previous["reversalMagnitudeAtr"] * previous_touches
                + zone["reversalMagnitudeAtr"] * new_touches
            ) / total_touches,
            "volumeConfirmation": (
                previous["volumeConfirmation"] * previous_touches
                + zone["volumeConfirmation"] * new_touches
            ) / total_touches,
        })
    return merged


def detect_price_zones(
    bars: list[dict],
    timeframe: Timeframe,
    current_price: float,
) -> tuple[list[dict], list[dict]]:
    """Return ranked active support/resistance zones for this timeframe."""
    working_bars = bars[-600:]
    zones = _cluster_pivots(_detect_pivots(working_bars, timeframe))
    latest_atr = working_bars[-1]["indicators"].get("atr14") if working_bars else None
    merge_distance = max(current_price * .0015, (latest_atr or current_price * .01) * .2)
    zones = _merge_nearby_zones(zones, merge_distance)
    supports: list[dict] = []
    resistances: list[dict] = []
    for zone in zones:
        if zone["upper"] < current_price:
            role = "support"
            distance = (current_price - zone["upper"]) / current_price
            role_reversal = zone["originalRole"] == "high"
            target = supports
        elif zone["lower"] > current_price:
            role = "resistance"
            distance = (zone["lower"] - current_price) / current_price
            role_reversal = zone["originalRole"] == "low"
            target = resistances
        else:
            continue
        target.append({
            **zone,
            "role": role,
            "timeframe": timeframe,
            "distancePct": distance,
            "roleReversal": role_reversal,
        })
    # Labels are spatial, not score ranks: S1/R1 must always be nearest.
    supports.sort(key=lambda item: (item["distancePct"], -item["score"]))
    resistances.sort(key=lambda item: (item["distancePct"], -item["score"]))
    for prefix, collection in (("S", supports), ("R", resistances)):
        for index, zone in enumerate(collection[:5], start=1):
            zone["id"] = f"{prefix}{index}"
            zone["major"] = index > 2 or zone["distancePct"] > .08
            for key in ("lower", "upper", "midpoint", "score", "recencyScore", "distancePct", "reversalMagnitudeAtr", "volumeConfirmation"):
                zone[key] = round(zone[key], 6)
    return supports[:5], resistances[:5]


def build_technical_analysis(
    bars: list[dict],
    timeframe: Timeframe,
    quote: dict,
    adjustment_mode: str,
) -> dict:
    latest = bars[-1]
    indicators = latest["indicators"]
    current_price = float(quote["price"] or latest["close"])
    supports, resistances = detect_price_zones(bars, timeframe, current_price)
    sma20 = _finite(indicators.get("sma20"))
    sma50 = _finite(indicators.get("sma50"))
    sma200 = _finite(indicators.get("sma200"))
    ordered_bullish = sma20 is not None and sma50 is not None and sma20 >= sma50 and current_price >= sma20
    ordered_bearish = sma20 is not None and sma50 is not None and sma20 <= sma50 and current_price <= sma20
    if sma200 is not None:
        ordered_bullish = ordered_bullish and sma50 is not None and sma50 >= sma200
        ordered_bearish = ordered_bearish and sma50 is not None and sma50 <= sma200
    trend = "BULLISH" if ordered_bullish else "BEARISH" if ordered_bearish else "MIXED"
    volume_ratio = _finite(indicators.get("volumeRatio"))
    previous_close = bars[-2]["close"] if len(bars) > 1 else latest["close"]
    broken_resistance = next((zone for zone in supports if zone["roleReversal"] and previous_close <= zone["upper"] < latest["close"]), None)
    broken_support = next((zone for zone in resistances if zone["roleReversal"] and previous_close >= zone["lower"] > latest["close"]), None)
    if broken_resistance:
        breakout_status = "CONFIRMED_BREAKOUT" if (volume_ratio or 0) >= 1.2 else "LOW_VOLUME_BREAKOUT"
    elif broken_support:
        breakout_status = "CONFIRMED_BREAKDOWN" if (volume_ratio or 0) >= 1.2 else "LOW_VOLUME_BREAKDOWN"
    else:
        breakout_status = "NO_BREAKOUT"
    return {
        "symbol": quote["symbol"],
        "timestamp": quote["timestamp"],
        "source": quote["source"],
        "currentPrice": round(current_price, 6),
        "regularClose": quote.get("regularClose"),
        "dailyChangePct": quote.get("dailyChangePct"),
        "dayReturn": quote.get("dayReturn", quote.get("dailyChangePct")),
        "adjustedDayReturn": quote.get("adjustedDayReturn"),
        "barReturn": latest.get("barReturn"),
        "previousRegularClose": quote.get("previousClose"),
        "timeframe": timeframe,
        "marketSession": quote["session"],
        "isDelayed": quote["isDelayed"],
        "adjustmentMode": adjustment_mode,
        "SMA20": sma20,
        "SMA50": sma50,
        "SMA200": sma200,
        "EMA20": _finite(indicators.get("ema20")),
        "RSI14": _finite(indicators.get("rsi14")),
        "MACD": _finite(indicators.get("macd")),
        "MACDSignal": _finite(indicators.get("macdSignal")),
        "CMF20": _finite(indicators.get("cmf20")),
        "OBV": _finite(indicators.get("obv")),
        "ATR14": _finite(indicators.get("atr14")),
        "volume": latest["volume"],
        "volumeMA20": _finite(indicators.get("volumeMA20")),
        "volumeRatio": volume_ratio,
        "supports": supports,
        "resistances": resistances,
        "trend": trend,
        "marketRegime": "TRENDING" if trend != "MIXED" else "RANGE_OR_TRANSITION",
        "breakoutStatus": breakout_status,
        "latestBarIsFinal": latest["isFinal"],
    }
