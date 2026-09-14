"""Structured market-performance and move-attribution adapters for Agent V2."""

from __future__ import annotations

import hashlib
import math
import re
from datetime import date, datetime, time, timedelta
from typing import Any, Callable
from zoneinfo import ZoneInfo

from v2.agent_v2.execution import CapabilityRegistry, ExecutionContext
from v2.agent_v2.models import EvidenceItem, ResultStatus, ToolEnvelope

_ET = ZoneInfo("America/New_York")
_REGULAR_OPEN = time(9, 30)
_REGULAR_CLOSE = time(16, 0)


def _now_et() -> datetime:
    return datetime.now(_ET)


def _as_et(now: datetime | None = None) -> datetime:
    current = now or _now_et()
    return current.replace(tzinfo=_ET) if current.tzinfo is None else current.astimezone(_ET)


def _observation_state(as_of: str, now: datetime | None = None) -> dict[str, Any]:
    current = _as_et(now)
    try:
        observation_date = date.fromisoformat(as_of[:10])
    except ValueError:
        observation_date = None
    intraday = bool(
        observation_date == current.date()
        and current.weekday() < 5
        and _REGULAR_OPEN <= current.time().replace(tzinfo=None) < _REGULAR_CLOSE
    )
    return {
        "market_session": "REGULAR" if intraday else "CLOSED",
        "is_intraday": intraday,
        "volume_is_final": not intraday,
        "observed_at": current.isoformat(timespec="minutes"),
        "observed_at_label": current.strftime("%Y-%m-%d %H:%M ET"),
    }


# Wording rules the verifier enforces on sentences that cite market evidence.
# They live here, next to the data that makes them necessary, and reach the
# verifier only through the generic ``constraints`` / ``citable`` metadata.
_REJECTED_CANDIDATE_NOTE = re.compile(r"无直接证据|未提及|关联弱|不匹配|Tier 3|长期预测", re.I)

# The rules are sentences, not patterns: the verifier hands each one with
# the text that cites the evidence to the claim judge (one model call per
# answer), so a paraphrase is caught as surely as the wording we once
# matched.  When the rule applies is still decided here, from the data.
_INTRADAY_PRICE_RULE = {"forbid_claim": "把盘中价格说成收盘价或完整交易日的口径（没有说明是盘中、截至查询时的价格）", "warning": "盘中价格被表述为完整收盘口径"}
_INTRADAY_VOLUME_RULE = {"forbid_claim": "用尚未收盘的盘中累计成交量断定放量、缩量、量能不足或走势能否持续", "warning": "未收盘成交量被用于判定放量、缩量或持续性"}
_CANDIDATE_RULE = {"forbid_claim": "把“证据”里这条候选解释本身说成已经确认的原因、主要原因或直接驱动（没有“可能”“候选”“尚未确认”这类限定）；回答里把别的、已确认的驱动说成原因不算", "warning": "候选归因被表述为已确认原因"}
# Soft: the judge read "三个下跌日都没有确认的高置信度驱动" as a leak and the
# whole answer fell back; wording advice is not grounds to discard an answer.
_COUNT_LEAK_RULE = {"forbid_claim": "把归因系统的内部统计口径原样写出来，比如“0 个高置信度驱动”“确认驱动数为 1”“候选驱动 2 个”这种带计数的系统术语；用自然语言说“还有两条线索”“最相关的一条线索”“没有确认的直接驱动”不算", "warning": "将内部归因计数直接暴露给用户", "soft": True}


def _pct(value: float | None) -> str:
    return "数据不足" if value is None else f"{float(value):+.2%}"


def _cite(item: EvidenceItem | None) -> str:
    return f"[{item.id}]" if item is not None else ""


def _scoped(evidence: list[EvidenceItem], scope: str) -> list[EvidenceItem]:
    return [item for item in evidence if item.metadata.get("evidence_scope") == scope]


def _usable(prices) -> list:
    """Bars with a finite, positive close, in order; a NaN placeholder row would poison every figure.

    Kept dependency-free: the offline eval runs this adapter without the
    data package, and the yfinance source drops such rows itself.
    """

    kept = []
    for bar in prices or []:
        try:
            close = float(bar.close)
        except (TypeError, ValueError, AttributeError):
            continue
        if not math.isfinite(close) or close <= 0:
            continue
        kept.append(bar)
    return kept


def _default_price_source():
    from v2.data.price_source import default_price_source

    return default_price_source()


def _default_move_provider(ticker: str):
    from v2.bot.responders import _build_query_anomaly
    from v2.data import CachedFDClient
    from v2.memory import AnomalyMemory
    from v2.monitoring import attribute

    with CachedFDClient() as fd:
        anomaly = _build_query_anomaly(ticker, fd)
        if anomaly is None:
            return None
        try:
            memory = AnomalyMemory()
        except Exception:
            memory = None
        return attribute(anomaly, fd_client=fd, memory=memory)


def _return(prices, window: int) -> float | None:
    if len(prices) <= window or float(prices[-1 - window].close) <= 0:
        return None
    return float(prices[-1].close) / float(prices[-1 - window].close) - 1


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _evidence_id(kind: str, ticker: str, as_of: str, claim: str) -> str:
    digest = hashlib.sha1(f"{kind}|{ticker}|{as_of}|{claim}".encode("utf-8")).hexdigest()[:16]
    return f"evidence-market-{kind}-{digest}"


def _item(
    kind: str,
    ticker: str,
    as_of: str,
    claim: str,
    context: ExecutionContext,
    *,
    metric: str = "",
    value: Any = None,
    confidence: float = 1.0,
    source_url: str = "",
    metadata: dict[str, Any] | None = None,
) -> EvidenceItem:
    return EvidenceItem(
        id=_evidence_id(kind, ticker, as_of, claim),
        entity=ticker,
        claim=claim,
        metric=metric,
        value=value,
        as_of=as_of,
        source_id="market_data" if kind not in {"driver", "candidate"} else "move_attribution",
        source_title="Daily/intraday OHLCV market data" if kind not in {"driver", "candidate"} else "Move attribution evidence",
        source_url=source_url,
        confidence=confidence,
        producer_run_id=context.run_id,
        metadata={"evidence_scope": kind, **(metadata or {})},
    )


def _performance_envelope(ticker: str, context: ExecutionContext, price_source, *, now: datetime | None = None) -> ToolEnvelope:
    current = _as_et(now)
    today = current.date()
    start = today - timedelta(days=430)
    prices = _usable(price_source.get_prices(ticker, start.isoformat(), today.isoformat()))
    if len(prices) < 2:
        return ToolEnvelope("market.performance", ResultStatus.FAILED, subject=ticker, errors=["no recent price history"])

    latest, previous = prices[-1], prices[-2]
    as_of = str(latest.time)[:10]
    observation = _observation_state(as_of, current)
    observation_metadata = {key: value for key, value in observation.items() if key != "observed_at_label"}
    close = float(latest.close)
    day_return = close / float(previous.close) - 1 if float(previous.close) > 0 else None
    windows = {"1d": day_return, "5d": _return(prices, 5), "1m": _return(prices, 21), "3m": _return(prices, 63), "1y": _return(prices, 252)}
    volumes = [float(row.volume) for row in prices[-31:-1] if _finite(row.volume) is not None]
    avg_volume_30d = sum(volumes) / len(volumes) if volumes else None
    volume_ratio = float(latest.volume) / avg_volume_30d if avg_volume_30d else None
    recent_returns = [float(prices[i].close) / float(prices[i - 1].close) - 1 for i in range(max(1, len(prices) - 21), len(prices)) if float(prices[i - 1].close) > 0]
    volatility_21d = (sum((value - sum(recent_returns) / len(recent_returns)) ** 2 for value in recent_returns) / max(1, len(recent_returns) - 1)) ** 0.5 * (252**0.5) if len(recent_returns) > 1 else None

    evidence: list[EvidenceItem] = []
    if observation["is_intraday"]:
        price_claim = f"{ticker} 截至 {observation['observed_at_label']} 的盘中价格为 {close:.2f} 美元，相对前一交易日收盘价 {day_return:+.2%}。"
    else:
        price_claim = f"{ticker} 截至 {as_of} 收盘价为 {close:.2f} 美元，单日涨跌幅为 {day_return:+.2%}。" if day_return is not None else f"{ticker} 截至 {as_of} 收盘价为 {close:.2f} 美元。"
    intraday_rules = {"constraints": [_INTRADAY_PRICE_RULE]} if observation["is_intraday"] else {}
    evidence.append(_item("price", ticker, as_of, price_claim, context, metric="close", value=close, metadata={"day_return": day_return, **observation_metadata, **intraday_rules}))
    available_windows = {key: value for key, value in windows.items() if value is not None}
    window_prefix = f"{ticker} 截至查询时的区间回报（含当前盘中价格）：" if observation["is_intraday"] else f"{ticker} 区间回报："
    window_claim = window_prefix + "，".join(f"{key} {value:+.2%}" for key, value in available_windows.items()) + "。"
    evidence.append(_item("returns", ticker, as_of, window_claim, context, metadata={"returns": available_windows, **observation_metadata}))
    if avg_volume_30d is not None and volume_ratio is not None:
        if observation["is_intraday"]:
            volume_claim = f"{ticker} 截至 {observation['observed_at_label']} 的盘中累计成交量为 {int(latest.volume)} 股，相当于 30 日完整交易日平均成交量 {avg_volume_30d:.0f} 股的 {volume_ratio:.2f} 倍；当日未收盘，不能据此判定是否放量或缩量。"
        else:
            volume_claim = f"{ticker} {as_of} 成交量为 {int(latest.volume)} 股，30 日平均成交量为 {avg_volume_30d:.0f} 股，量比为 {volume_ratio:.2f} 倍。"
        volume_rules = {"constraints": [_INTRADAY_VOLUME_RULE]} if observation["is_intraday"] else {}
        evidence.append(_item("volume", ticker, as_of, volume_claim, context, metric="volume_ratio", value=volume_ratio, metadata={"volume": int(latest.volume), "average_volume_30d": avg_volume_30d, **observation_metadata, **volume_rules}))
    if volatility_21d is not None:
        volatility_claim = f"{ticker} 基于截至查询时价格估算的近 21 个交易日年化实现波动率为 {volatility_21d:.2%}。" if observation["is_intraday"] else f"{ticker} 近 21 个交易日的实现波动率折算年化后为 {volatility_21d:.2%}。"
        evidence.append(_item("volatility", ticker, as_of, volatility_claim, context, metric="annualized_volatility_21d", value=volatility_21d, metadata=observation_metadata))

    from v2.universe import BENCHMARK_ETF, sector_etf_for

    sector_etf = sector_etf_for(ticker)
    relative: dict[str, dict[str, float | None]] = {}
    limitations: list[str] = []
    for benchmark in dict.fromkeys((sector_etf, BENCHMARK_ETF)):
        benchmark_prices = _usable(price_source.get_prices(benchmark, start.isoformat(), today.isoformat()))
        benchmark_windows = {key: _return(benchmark_prices, window) for key, window in (("1d", 1), ("5d", 5), ("1m", 21))}
        comparable = {key: value for key, value in benchmark_windows.items() if value is not None and windows.get(key) is not None}
        if not comparable:
            limitations.append(f"{benchmark} benchmark history unavailable")
            continue
        relative[benchmark] = {key: windows[key] - value for key, value in comparable.items()}
        claim_prefix = f"截至同一查询时点的盘中基准 {benchmark} 回报：" if observation["is_intraday"] else f"同期基准 {benchmark} 回报："
        claim = claim_prefix + "，".join(f"{key} {value:+.2%}（{ticker} 相对 {windows[key] - value:+.2%}）" for key, value in comparable.items()) + "。"
        evidence.append(_item("benchmark", ticker, as_of, claim, context, metadata={"benchmark": benchmark, "benchmark_returns": comparable, "relative_returns": relative[benchmark], **observation_metadata}))

    metrics = {
        "close": close,
        "returns": available_windows,
        "volume": int(latest.volume),
        "average_volume_30d": avg_volume_30d,
        "volume_ratio": volume_ratio,
        "annualized_volatility_21d": volatility_21d,
        "sector_benchmark": sector_etf,
        "relative_returns": relative,
        **observation_metadata,
    }
    summary = price_claim + " " + window_claim
    status = ResultStatus.COMPLETED if "1m" in available_windows and relative else ResultStatus.PARTIAL_DATA
    envelope = ToolEnvelope("market.performance", status, subject=ticker, as_of=as_of, summary=summary, metrics=metrics, evidence=evidence, limitations=limitations, metadata={"require_cited_numbers": True})
    envelope.metadata["narrative"] = _performance_narrative(envelope)
    return envelope


def _move_envelope(ticker: str, context: ExecutionContext, anomaly, *, now: datetime | None = None) -> ToolEnvelope:
    if anomaly is None:
        return ToolEnvelope("market.explain_move", ResultStatus.FAILED, subject=ticker, errors=["no recent move data"])
    as_of = str(anomaly.date)[:10]
    observation = _observation_state(as_of, now)
    observation_metadata = {key: value for key, value in observation.items() if key != "observed_at_label"}
    evidence: list[EvidenceItem] = []
    if observation["is_intraday"]:
        price_claim = f"{ticker} 截至 {observation['observed_at_label']} 盘中报 {float(anomaly.price):.2f} 美元，相对前一交易日收盘价 {float(anomaly.price_change_pct):+.2%}。"
        volume_claim = f"{ticker} 截至查询时的盘中累计成交量为 {int(anomaly.volume_today)} 股，相当于 30 日完整交易日均量 {float(anomaly.volume_avg_30d):.0f} 股的 {float(anomaly.volume_ratio):.2f} 倍；盘中成交量尚未定型，不能据此判定是否放量或缩量。"
    else:
        price_claim = f"{ticker} 在 {as_of} 收于 {float(anomaly.price):.2f} 美元，较前一交易日 {float(anomaly.price_change_pct):+.2%}。"
        volume_claim = f"{ticker} 当日成交量为 {int(anomaly.volume_today)} 股，30 日均量为 {float(anomaly.volume_avg_30d):.0f} 股，量比 {float(anomaly.volume_ratio):.2f} 倍。"
    price_rules = {"constraints": [_INTRADAY_PRICE_RULE]} if observation["is_intraday"] else {}
    volume_rules = {"constraints": [_INTRADAY_VOLUME_RULE]} if observation["is_intraday"] else {}
    evidence.append(_item("price", ticker, as_of, price_claim, context, metric="price_change_pct", value=float(anomaly.price_change_pct), metadata={"close": float(anomaly.price), **observation_metadata, **price_rules}))
    evidence.append(_item("volume", ticker, as_of, volume_claim, context, metric="volume_ratio", value=float(anomaly.volume_ratio), metadata={**observation_metadata, **volume_rules}))
    if anomaly.sector_etf and anomaly.sector_return_1d is not None:
        benchmark_prefix = "截至同一查询时点的盘中" if observation["is_intraday"] else "同期"
        benchmark_claim = f"{benchmark_prefix}行业基准 {anomaly.sector_etf} 单日回报为 {float(anomaly.sector_return_1d):+.2%}，{ticker} 相对回报为 {float(anomaly.relative_1d_pp or 0):+.2%}。"
        evidence.append(_item("benchmark", ticker, as_of, benchmark_claim, context, metadata={"benchmark": anomaly.sector_etf, "contrarian": bool(anomaly.contrarian), **observation_metadata}))

    findings: list[dict[str, Any]] = []
    source_rows = [source.model_dump() if hasattr(source, "model_dump") else dict(source) for source in anomaly.sources]
    high_confidence = 0
    confidence_value = {"高": 0.9, "中": 0.6, "低": 0.3}
    for reason in anomaly.reasons:
        level = str(reason.confidence)
        confirmed = level == "高"
        high_confidence += int(confirmed)
        role = "driver" if confirmed else "candidate"
        qualifier = "高置信度归因" if confirmed else f"{level}置信度候选解释"
        note = f"；校验备注：{reason.note}" if reason.note else ""
        claim = f"{ticker} {qualifier}：{reason.text}{note}。"
        # Attribution currently returns a shared source set, not a reason-to-source
        # mapping. Only expose a direct URL when the relationship is unambiguous.
        source = source_rows[0] if len(source_rows) == 1 else {}
        confidence = confidence_value.get(level, 0.3)
        note_text = str(reason.note or "")
        citable = confirmed or (confidence >= 0.5 and not _REJECTED_CANDIDATE_NOTE.search(note_text))
        metadata = {"claim_role": "confirmed_driver" if confirmed else "candidate_driver", "causal_confidence": level, "driver_text": reason.text, "note": reason.note, "supporting_sources": source_rows}
        if not confirmed:
            metadata["constraints"] = [_CANDIDATE_RULE]
        if not citable:
            metadata["citable"] = False
            metadata["uncitable_warning"] = "展示了缺乏直接支持的过弱异动线索"
        item = _item(role, ticker, as_of, claim, context, confidence=confidence, source_url=str(source.get("url") or ""), metadata=metadata)
        evidence.append(item)
        findings.append({"claim": reason.text, "causal_confidence": level, "confirmed": confirmed, "evidence_ids": [item.id]})

    limitations: list[str] = []
    if high_confidence == 0:
        limitations.append("没有高置信度的同日催化剂证据，具体触发原因尚未确认。")
    if not source_rows:
        limitations.append("没有可用于归因的已验证同日新闻来源。")
    metrics = {
        "price": float(anomaly.price),
        "price_change_pct": float(anomaly.price_change_pct),
        "volume_ratio": float(anomaly.volume_ratio),
        "sector_etf": anomaly.sector_etf,
        "sector_return_1d": anomaly.sector_return_1d,
        "relative_1d": anomaly.relative_1d_pp,
        "confirmed_driver_count": high_confidence,
        "candidate_driver_count": len(anomaly.reasons) - high_confidence,
        **observation_metadata,
    }
    assessment_claim = (
        f"{ticker} 异动归因中有 {high_confidence} 个高置信度直接驱动，"
        f"{len(anomaly.reasons) - high_confidence} 个候选解释。"
    )
    if high_confidence == 0:
        assessment_claim += " 现有证据不足以确认具体触发原因。"
    evidence.append(
        _item(
            "attribution",
            ticker,
            as_of,
            assessment_claim,
            context,
            metric="confirmed_driver_count",
            value=high_confidence,
            metadata={
                "claim_role": "attribution_assessment",
                "candidate_driver_count": len(anomaly.reasons) - high_confidence,
            },
        )
    )
    answer_constraints: list[dict[str, Any]] = [dict(_COUNT_LEAK_RULE)]
    if high_confidence == 0:
        answer_constraints.append({"max_cited": {"metadata": {"claim_role": "candidate_driver"}, "max": 1, "warning": "未确认直接驱动时展示了过多弱候选线索"}})
    envelope = ToolEnvelope(
        "market.explain_move",
        ResultStatus.COMPLETED if evidence else ResultStatus.PARTIAL_DATA,
        subject=ticker,
        as_of=as_of,
        summary=price_claim,
        metrics=metrics,
        findings=findings,
        evidence=evidence,
        limitations=limitations,
        metadata={
            "next_steps": list(anomaly.next_steps),
            "filtered_news_count": int(anomaly.filtered_count),
            "source_count": len(source_rows),
            "require_cited_numbers": True,
            "answer_constraints": answer_constraints,
        },
    )
    envelope.metadata["narrative"] = _move_narrative(envelope)
    return envelope


def _driver_text(item: EvidenceItem) -> str:
    text = str(item.metadata.get("driver_text") or "").strip()
    if text:
        return text.rstrip("。")
    text = item.claim.split("：", 1)[-1].split("；校验备注", 1)[0]
    return text.rstrip("。")


def _performance_narrative(market: ToolEnvelope) -> str:
    """Deterministic prose for a performance envelope; the generic synthesizer's fallback."""

    evidence = market.evidence
    ticker = market.subject
    price = next(iter(_scoped(evidence, "price")), None)
    volume = next(iter(_scoped(evidence, "volume")), None)
    benchmarks = _scoped(evidence, "benchmark")
    returns_item = next(iter(_scoped(evidence, "returns")), None)
    volatility = next(iter(_scoped(evidence, "volatility")), None)
    returns = market.metrics.get("returns") or {}
    close = market.metrics.get("close")
    day = returns.get("1d")
    direction = "上涨" if day is not None and day > 0 else "下跌" if day is not None and day < 0 else "基本持平"
    trend = "偏强" if (returns.get("5d") or 0) > 0 and (returns.get("1m") or 0) > 0 else "偏弱" if (returns.get("5d") or 0) < 0 and (returns.get("1m") or 0) < 0 else "分化"
    if market.metrics.get("is_intraday") and price is not None:
        first = f"{ticker} 最近的股价表现{trend}。{price.claim.rstrip('。')}；近 5 日回报 {_pct(returns.get('5d'))}，近 1 月回报 {_pct(returns.get('1m'))}{_cite(price)}{_cite(returns_item)}。"
    else:
        day_text = f"最新交易日{direction} {abs(float(day)):.2%}；" if day is not None else ""
        first = (
            f"{ticker} 最近的股价表现{trend}。截至 {market.as_of}，收盘价为 {float(close):.2f} 美元，"
            f"{day_text}近 5 日回报 {_pct(returns.get('5d'))}，近 1 月回报 {_pct(returns.get('1m'))}"
            f"{_cite(price)}{_cite(returns_item)}。"
        )
    relative = market.metrics.get("relative_returns") or {}
    relative_parts: list[str] = []
    for item in benchmarks:
        benchmark = str(item.metadata.get("benchmark") or "基准")
        values = relative.get(benchmark) or {}
        relative_parts.append(f"相对 {benchmark}，单日超额 {_pct(values.get('1d'))}，近 5 日 {_pct(values.get('5d'))}，近 1 月 {_pct(values.get('1m'))}{_cite(item)}")
    second = "；".join(relative_parts) + "。" if relative_parts else "行业与大盘基准数据暂时不足。"
    volume_ratio = market.metrics.get("volume_ratio")
    if market.metrics.get("is_intraday") and volume is not None:
        volume_text = f"{volume.claim.rstrip('。')}{_cite(volume)}。"
    elif volume is not None and volume_ratio is not None:
        volume_text = f"当日成交量约 {int(market.metrics.get('volume') or 0) / 10_000:.0f} 万股，是 30 日平均水平的 {float(volume_ratio):.2f} 倍{_cite(volume)}。"
    else:
        volume_text = "成交量对比数据暂时不足。"
    volatility_value = market.metrics.get("annualized_volatility_21d")
    risk_text = f"近 21 个交易日年化波动率约为 {float(volatility_value):.2%}{_cite(volatility)}，说明短线波动仍然较大。" if volatility is not None and volatility_value is not None else ""
    next_step = "待收盘后再判断量能，并观察相对行业的超额表现能否延续。" if market.metrics.get("is_intraday") else "接下来重点观察成交量能否跟上，以及相对行业的超额表现能否延续。"
    return "\n\n".join((first, second, f"{volume_text}{risk_text}{next_step}"))


def _move_narrative(market: ToolEnvelope) -> str:
    """Deterministic prose for a move-explanation envelope."""

    evidence = market.evidence
    ticker = market.subject
    price = next(iter(_scoped(evidence, "price")), None)
    volume = next(iter(_scoped(evidence, "volume")), None)
    benchmarks = _scoped(evidence, "benchmark")
    change = market.metrics.get("price_change_pct")
    close = market.metrics.get("price")
    direction = "上涨" if change is not None and change > 0 else "下跌" if change is not None and change < 0 else "基本持平"
    certainty = "确实" if change else ""
    first_parts = [f"{price.claim.rstrip('。')}{_cite(price)}。"] if price is not None else [f"{ticker} 在 {market.as_of}{certainty}{direction} {abs(float(change)):.2%}，价格为 {float(close):.2f} 美元。"]
    if volume is not None:
        first_parts.append(f"{volume.claim.rstrip('。')}{_cite(volume)}。")
    if benchmarks:
        first_parts.append(f"{benchmarks[0].claim.rstrip('。')}{_cite(benchmarks[0])}。")
    assessment = next(iter(_scoped(evidence, "attribution")), None)
    confirmed = _scoped(evidence, "driver")
    candidates = [item for item in _scoped(evidence, "candidate") if item.metadata.get("citable", True)]
    if confirmed:
        reason_text = "；".join(f"{_driver_text(item)}{_cite(item)}" for item in confirmed[:2])
        second = f"目前能直接支持的高置信度驱动是：{reason_text}。"
    else:
        second = f"但“为什么{direction}”目前还不能下定论：暂未找到可核实的同日催化剂，具体触发原因尚未确认{_cite(assessment)}。"
    if candidates:
        candidate = max(candidates, key=lambda item: float(item.confidence or 0))
        note = str(candidate.metadata.get("note") or "").strip().rstrip("。")
        note_text = f"；但{note}" if note else ""
        second += f"目前最相关的一条候选线索是“{_driver_text(candidate)}”{note_text}，因此它仍只能作为排查方向{_cite(candidate)}。"
    relative = market.metrics.get("relative_1d")
    if isinstance(relative, (int, float)) and abs(float(relative)) >= 0.0005:
        relation = f"股价{'跑赢' if float(relative) > 0 else '跑输'}行业基准"
    elif isinstance(relative, (int, float)):
        relation = "股价与行业基准基本同步"
    else:
        relation = "行业基准对比暂缺"
    move_word = "涨幅" if direction == "上涨" else "跌幅" if direction == "下跌" else "涨跌"
    if market.metrics.get("is_intraday"):
        third = f"从盘面看，{relation}{_cite(benchmarks[0] if benchmarks else None)}。但当前成交量仍是盘中累计值{_cite(volume)}，不能用它推断放量、缩量或{direction}的持续性；应待收盘后再判断量能，并等待公司公告或可核验的同日新闻确认催化剂。"
    else:
        third = f"从盘面看，{relation}{_cite(benchmarks[0] if benchmarks else None)}，但成交量未同比例放大{_cite(volume)}，因此不宜单凭{move_word}追认某个原因。接下来应观察放量延续性，并等待公司公告或可核验的同日新闻确认催化剂。"
    return "\n\n".join(("".join(first_parts), second, third))


_DRAWDOWN_WINDOWS = (("1m", 21, "1 月"), ("3m", 63, "3 月"), ("1y", 252, "1 年"))


def _pick_window(move_pct: float | None, window: str | None, returns: dict[str, float | None], *, direction: str = "down") -> str:
    """The window to inspect: the caller's, else the shortest one holding at least half of the move, else 1y."""

    if window in {key for key, _, _ in _DRAWDOWN_WINDOWS}:
        return str(window)
    sign = -1 if direction == "down" else 1
    if isinstance(move_pct, (int, float)) and move_pct * sign > 0:
        move = float(move_pct) / 100.0
        for key, _, _ in _DRAWDOWN_WINDOWS:
            value = returns.get(key)
            if value is not None and value * sign > 0 and value / move >= 0.5:
                return key
        return "1y"
    return "3m"


def _default_sector_etf(ticker: str) -> str:
    from v2.universe import sector_etf_for

    return sector_etf_for(ticker)


def _close_on_or_before(prices, day: str) -> float | None:
    """The last daily close dated ``day`` or earlier, or None when the history starts later."""

    close = None
    for bar in prices:
        if str(bar.time)[:10] > day:
            break
        close = float(bar.close)
    return close if close and close > 0 else None


def _extreme_stretch(span, *, up: bool) -> tuple[int, int, float | None]:
    """The largest drawdown (or run-up) inside ``span``: (peak index, trough index, move).

    A running high (or low) is carried along; the bar with the deepest fall
    from it (or the highest rise above it) closes the stretch.  ``move`` is
    None when no bar sits below a previous high (or above a previous low).
    """

    best_move: float | None = None
    best_pair = (0, 0)
    anchor = 0
    for index in range(len(span)):
        close = float(span[index].close)
        anchor_close = float(span[anchor].close)
        if (up and close < anchor_close) or (not up and close > anchor_close):
            anchor = index
            continue
        if index == anchor or anchor_close <= 0:
            continue
        move = close / anchor_close - 1
        if best_move is None or (move > best_move if up else move < best_move):
            best_move, best_pair = move, (anchor, index)
    if best_move is None or best_move == 0:
        return 0, 0, None
    start, end = best_pair
    return (end, start, best_move) if up else (start, end, best_move)


def _drawdown_envelope(ticker: str, context: ExecutionContext, price_source, *, loss_pct: float | None = None, window: str | None = None, top: int = 3, now: datetime | None = None, sector_for: Callable[[str], str] | None = None, direction: str = "down") -> ToolEnvelope:
    """Locate a stretch in time from daily closes: window, peak-to-trough (or trough-to-peak), the extreme days.

    ``direction`` "down" is a drawdown (a loss since purchase); "up" is the
    mirror image, a run-up (a gain since purchase): the window that holds
    most of the gain, the low to the high inside it, and the best days.
    """

    up = direction == "up"
    capability = "market.runup" if up else "market.drawdown"
    current = _as_et(now)
    today = current.date()
    start = today - timedelta(days=430)
    prices = _usable(price_source.get_prices(ticker, start.isoformat(), today.isoformat()))
    # A session still in progress is not a completed daily bar.
    if prices and _observation_state(str(prices[-1].time)[:10], current)["is_intraday"]:
        prices = prices[:-1]
    if len(prices) < 3:
        return ToolEnvelope(capability, ResultStatus.FAILED, subject=ticker, errors=["no recent price history"])
    returns = {key: _return(prices, bars) for key, bars, _ in _DRAWDOWN_WINDOWS}
    chosen = _pick_window(loss_pct, window, returns, direction=direction)
    bars, label = next((bars, label) for key, bars, label in _DRAWDOWN_WINDOWS if key == chosen)
    span = prices[-(bars + 1):]
    window_return = float(span[-1].close) / float(span[0].close) - 1 if float(span[0].close) > 0 else None
    as_of = str(span[-1].time)[:10]
    window_start = str(span[1].time)[:10] if len(span) > 1 else as_of
    evidence: list[EvidenceItem] = []
    metrics: dict[str, Any] = {"window": chosen, "window_start": window_start, "as_of": as_of, "window_return": window_return, "returns": {key: value for key, value in returns.items() if value is not None}}
    if window_return is not None:
        evidence.append(_item("window_return", ticker, as_of, f"{ticker} 近 {label}（{window_start} 至 {as_of}）区间回报 {window_return:+.2%}。", context, metric="window_return", value=window_return, metadata={"window": chosen}))
    daily = []
    for previous, bar in zip(span, span[1:]):
        if float(previous.close) > 0:
            daily.append((str(bar.time)[:10], float(bar.close) / float(previous.close) - 1, float(bar.close)))
    # The extreme days are the ones inside the stretch itself: a big day in an
    # earlier rise that fell back to the low is not part of this run-up.
    peak_index, trough_index, drawdown = _extreme_stretch(span, up=up)
    if drawdown is not None:
        first, last = sorted((str(span[peak_index].time)[:10], str(span[trough_index].time)[:10]))
        inside = [row for row in daily if first < row[0] <= last]
    else:
        inside = daily
    if up:
        worst = sorted((row for row in inside if row[1] > 0), key=lambda row: -row[1])[: max(1, min(int(top or 3), 5))]
    else:
        worst = sorted((row for row in inside if row[1] < 0), key=lambda row: row[1])[: max(1, min(int(top or 3), 5))]
    worst.sort(key=lambda row: row[0])
    day_scope, days_key = ("best_day", "best_days") if up else ("worst_day", "worst_days")
    for day, change, close in worst:
        evidence.append(_item(day_scope, ticker, day, f"{ticker} {day} 单日 {change:+.2%}，收盘 {close:.2f} 美元。", context, metric="daily_return", value=change, metadata={"date": day, "close": close}))
    metrics[days_key] = [{"date": day, "return": change, "close": close} for day, change, close in worst]
    peak, trough = span[peak_index], span[trough_index]
    if drawdown is not None:
        peak_day, trough_day = str(peak.time)[:10], str(trough.time)[:10]
        if up:
            claim = f"{ticker} 从 {trough_day} 的低点 {float(trough.close):.2f} 美元到 {peak_day} 的高点 {float(peak.close):.2f} 美元上涨 {drawdown:+.2%}。"
        else:
            claim = f"{ticker} 从 {peak_day} 的高点 {float(peak.close):.2f} 美元到 {trough_day} 的低点 {float(trough.close):.2f} 美元回撤 {drawdown:+.2%}。"
        evidence.append(_item("peak_trough", ticker, peak_day if up else trough_day, claim, context, metric="runup" if up else "drawdown", value=drawdown, metadata={"peak_date": peak_day, "trough_date": trough_day}))
        metrics["peak"] = {"date": peak_day, "close": float(peak.close)}
        metrics["trough"] = {"date": trough_day, "close": float(trough.close)}
        metrics["runup" if up else "drawdown"] = drawdown
        metrics["move"] = drawdown
        # The sector ETF over the same stretch: how much of the fall was the
        # sector's, and how much was this stock's own.
        benchmark = ""
        try:
            benchmark = str(sector_for(ticker) or "") if sector_for else ""
        except Exception:  # noqa: BLE001 — an unknown ticker has no sector; that is data, not a crash
            benchmark = ""
        span_start, span_end = (trough_day, peak_day) if up else (peak_day, trough_day)
        if benchmark and benchmark.upper() != ticker.upper():
            try:
                benchmark_prices = _usable(price_source.get_prices(benchmark, (date.fromisoformat(span_start) - timedelta(days=7)).isoformat(), span_end))
            except Exception as exc:  # noqa: BLE001
                benchmark_prices = []
                limitations_extra = f"{benchmark} 同期行情不可用：{type(exc).__name__}"
            else:
                limitations_extra = ""
            start_close, end_close = _close_on_or_before(benchmark_prices, span_start), _close_on_or_before(benchmark_prices, span_end)
            if start_close and end_close:
                benchmark_return = end_close / start_close - 1
                if up:
                    gap = (drawdown - benchmark_return) * 100  # positive: the stock rose more than its sector
                    more, less = "多涨", "少涨"
                else:
                    gap = (benchmark_return - drawdown) * 100  # positive: the stock fell more than its sector
                    more, less = "多跌", "少跌"
                if abs(gap) < 0.05:
                    relation = "与基准基本同步"
                else:
                    relation = f"{ticker} 比基准{more if gap > 0 else less} {abs(gap):.2f} 个百分点"
                evidence.append(_item("benchmark_span", ticker, span_end, f"同期行业基准 {benchmark} 从 {span_start} 到 {span_end} 回报 {benchmark_return:+.2%}，{relation}。", context, metric="benchmark_span_return", value=benchmark_return, metadata={"benchmark": benchmark, "peak_date": peak_day, "trough_date": trough_day, "gap_pp": gap}))
                metrics["benchmark_span"] = {"benchmark": benchmark, "return": benchmark_return, "gap_pp": gap}
            elif limitations_extra:
                metrics["benchmark_span_error"] = limitations_extra
    word = "涨幅" if up else "跌幅"
    limitations = [] if worst else [f"区间内没有{'上涨' if up else '下跌'}的交易日"]
    if metrics.get("benchmark_span_error"):
        limitations.append(str(metrics.pop("benchmark_span_error")))
    envelope = ToolEnvelope(
        capability,
        ResultStatus.COMPLETED if worst else ResultStatus.PARTIAL_DATA,
        subject=ticker,
        as_of=as_of,
        summary=f"{ticker} 近 {label} 区间回报 {_pct(window_return)}，{word}最大的交易日：" + ("、".join(f"{day} {change:+.2%}" for day, change, _ in worst) or "无"),
        metrics=metrics,
        evidence=evidence,
        limitations=limitations,
        metadata={
            "require_cited_numbers": True,
            "direction": direction,
            ("best_dates" if up else "worst_dates"): [day for day, _, _ in worst],
            "dates": [day for day, _, _ in worst],
            "queries": [f"why did {ticker} stock {'rise' if up else 'fall'} on {day}" for day, _, _ in worst],
        },
    )
    span_item = next(iter(_scoped(evidence, "benchmark_span")), None)
    if span_item is not None:
        # "Sector or stock" is the one comparison a stretch answer cannot skip.
        envelope.metadata["answer_constraints"] = [{"require_cited": {"metadata": {"evidence_scope": "benchmark_span"}, "warning": f"这段{word}的回答必须引用同期行业基准对比那条证据 [{span_item.id}]（同期 {span_item.metadata['benchmark']} 的回报和 {ticker} 比基准多/少{'涨' if up else '跌'}的百分点）"}}]
    envelope.metadata["narrative"] = _drawdown_narrative(envelope, label)
    return envelope


def _drawdown_narrative(result: ToolEnvelope, label: str) -> str:
    ticker = result.subject
    window_item = next(iter(_scoped(result.evidence, "window_return")), None)
    peak_item = next(iter(_scoped(result.evidence, "peak_trough")), None)
    up = result.metadata.get("direction") == "up"
    worst_items = _scoped(result.evidence, "best_day" if up else "worst_day")
    parts: list[str] = []
    if window_item is not None:
        parts.append(f"{window_item.claim.rstrip('。')}{_cite(window_item)}。")
    if peak_item is not None:
        parts.append(f"{peak_item.claim.rstrip('。')}{_cite(peak_item)}。")
    span_item = next(iter(_scoped(result.evidence, "benchmark_span")), None)
    if span_item is not None:
        parts.append(f"{span_item.claim.rstrip('。')}{_cite(span_item)}。")
    if worst_items:
        parts.append(f"{ticker} 近 {label}{'涨幅' if up else '跌幅'}最大的交易日：" + "；".join(f"{item.metadata['date']} {float(item.value):+.2%}{_cite(item)}" for item in worst_items) + "。")
    else:
        parts.append(f"{ticker} 近 {label}区间内没有{'上涨' if up else '下跌'}的交易日。")
    return "\n".join(parts)


def register_market_capabilities(
    registry: CapabilityRegistry,
    *,
    price_source_factory: Callable[[], Any] = _default_price_source,
    move_provider: Callable[[str], Any] = _default_move_provider,
    now_factory: Callable[[], datetime] = _now_et,
    sector_for: Callable[[str], str] | None = _default_sector_etf,
) -> None:
    def performance(arguments: dict[str, Any], context: ExecutionContext) -> ToolEnvelope:
        ticker = str(arguments.get("ticker") or "").upper()
        return _performance_envelope(ticker, context, price_source_factory(), now=now_factory())

    def explain(arguments: dict[str, Any], context: ExecutionContext) -> ToolEnvelope:
        ticker = str(arguments.get("ticker") or "").upper()
        return _move_envelope(ticker, context, move_provider(ticker), now=now_factory())

    def runup(arguments: dict[str, Any], context: ExecutionContext) -> ToolEnvelope:
        ticker = str(arguments.get("ticker") or "").upper()
        gain = arguments.get("gain_pct")
        return _drawdown_envelope(
            ticker,
            context,
            price_source_factory(),
            loss_pct=float(gain) if isinstance(gain, (int, float)) else None,
            window=arguments.get("window"),
            sector_for=sector_for,
            direction="up",
            top=int(arguments.get("top") or 3),
            now=now_factory(),
        )

    def drawdown(arguments: dict[str, Any], context: ExecutionContext) -> ToolEnvelope:
        ticker = str(arguments.get("ticker") or "").upper()
        loss = arguments.get("loss_pct")
        return _drawdown_envelope(
            ticker,
            context,
            price_source_factory(),
            loss_pct=float(loss) if isinstance(loss, (int, float)) else None,
            window=arguments.get("window"),
            sector_for=sector_for,
            top=int(arguments.get("top") or 3),
            now=now_factory(),
        )

    registry.register("market.performance", performance)
    registry.register("market.explain_move", explain)
    registry.register("market.drawdown", drawdown)
    registry.register("market.runup", runup)
