"""Anomaly attribution via Tavily search + DeepSeek synthesis.

Phase A additions:
- Entity filter: news items must mention the ticker or company name; the
  rest get hard-filtered as "noise" (count reported back to user).
- Next-step suggestions: alongside causal reasons, the LLM emits 1-2
  actionable follow-up observations (upcoming earnings, technical levels,
  related-ticker effects).
"""

from __future__ import annotations

import json
import logging
import os

from langchain_core.messages import HumanMessage, SystemMessage
from v2.data.metered import ChatDeepSeek
from v2.data.metered import TavilyClient

from v2.data.client import FDClient
from v2.monitoring.models import Anomaly, NewsSource, ScoredReason

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = (
    "你是一名股票异动归因分析师。给定一次异动事件和已通过实体校验的新闻片段，"
    "给出两类输出：\n"
    "\n"
    "(A) reasons：2-3 个最可能的原因\n"
    "  - 每条 ≤ 30 字\n"
    "  - 必须来自给定的新闻片段，不许引用未在片段中出现的事实\n"
    "  - 按相关性排序\n"
    "  - 如果新闻片段都与异动无关，输出空数组 []\n"
    "\n"
    "(B) next_steps：1-2 条 actionable 观察建议\n"
    "  - 形式如：即将发生的事件（财报、Fed、产品发布等）\n"
    "  - 或：关键技术位/支撑位监控\n"
    "  - 或：关联标的的连带效应（如供应链、同业竞争对手）\n"
    "  - 每条 ≤ 35 字，具体可执行（不是泛泛而谈）\n"
    "  - 基于异动类型和归因，合理推演\n"
    "\n"
    "硬约束：\n"
    "- 只输出 JSON，不要 markdown 代码块\n"
    "- 不许编造未在新闻中出现的具体数字\n"
    "\n"
    'JSON 格式：{"reasons": ["...", "..."], "next_steps": ["...", "..."]}'
)

_FLAG_HUMAN = {
    "volume_spike": "成交量爆量",
    "52w_high": "52 周新高",
    "52w_low": "52 周新低",
}

# English keywords for Tavily — its index is English news, Chinese hurts relevance.
_FLAG_SEARCH = {
    "volume_spike": "unusual volume",
    "52w_high": "new high",
    "52w_low": "decline drop down",
}


def attribute(
    anomaly: Anomaly,
    fd_client: FDClient | None = None,
    memory=None,
) -> Anomaly:
    """Fill in reasons / next_steps / sources / counters / history in place.

    *fd_client* (optional) — used to look up the official company name for
    entity-filter matching. Without it, only ticker-symbol match is used.

    *memory* (optional, v2.memory.AnomalyMemory) — Phase C: queries ChromaDB
    for similar past anomalies before storing the current one.
    """
    news = _search_news(anomaly)
    anomaly.tavily_calls = 1 if news is not None else 0

    if news:
        # Phase A: entity filter — drop news that don't mention ticker/name
        company_name = _lookup_company_name(anomaly.ticker, fd_client)
        matched, rejected = _entity_filter(news, anomaly.ticker, company_name)
        anomaly.filtered_count = len(rejected)

        if rejected:
            logger.info(
                "%s: entity-filter dropped %d/%d news items",
                anomaly.ticker, len(rejected), len(news),
            )

        if matched:
            parsed, gen_tokens = _synthesize(anomaly, matched)
            raw_reasons = parsed.get("reasons", [])
            anomaly.next_steps = parsed.get("next_steps", [])
            anomaly.llm_tokens = gen_tokens

            if raw_reasons:
                # Phase B ① C: Verifier scores each reason's causal confidence.
                scored, ver_tokens = _verify_reasons(anomaly, raw_reasons, matched)
                anomaly.reasons = scored
                anomaly.llm_tokens = gen_tokens + ver_tokens

                anomaly.sources = [
                    NewsSource(title=n.get("title", "")[:80], url=n.get("url", ""))
                    for n in matched[:3]
                    if n.get("url")
                ]
            else:
                logger.warning(
                    "%s: %d entity-matched results but Generator produced no reasons",
                    anomaly.ticker, len(matched),
                )
        else:
            logger.warning(
                "%s: all %d Tavily results rejected by entity filter",
                anomaly.ticker, len(news),
            )

    # Phase C: ALWAYS remember the anomaly (even if unattributed) — the bare
    # fact "ticker X had flag set Y on date Z" is itself valuable history.
    # recall happens before remember so we never match ourselves.
    if memory is not None:
        try:
            high_med_reasons = " ".join(
                r.text for r in anomaly.reasons
                if r.confidence in ("高", "中")
            )
            query_text = " ".join([
                anomaly.ticker,
                " ".join(anomaly.flags),
                high_med_reasons,
            ]).strip() or anomaly.ticker

            anomaly.historical_context = memory.recall(
                anomaly.ticker,
                query_text,
                lookback_days=30,
                n_results=3,
                exclude_date=anomaly.date,
            )
            memory.remember(anomaly)
        except Exception as exc:
            logger.warning("AnomalyMemory failed for %s: %s",
                           anomaly.ticker, exc)

    return anomaly


def _lookup_company_name(ticker: str, fd_client: FDClient | None) -> str | None:
    """Best-effort company name fetch — used to widen entity-filter matching.

    News articles often use the company name ("Apple", "Nvidia") rather than
    the ticker symbol, so matching on both improves recall.
    """
    if fd_client is None:
        return None
    try:
        facts = fd_client.get_company_facts(ticker)
        return facts.name if facts and facts.name else None
    except Exception as exc:
        logger.debug("company_name lookup for %s failed: %s", ticker, exc)
        return None


def _entity_filter(
    news: list[dict],
    ticker: str,
    company_name: str | None,
) -> tuple[list[dict], list[dict]]:
    """Split news into (matched, rejected) by whether they mention the entity.

    A news item passes if its title or content contains:
      - the ticker symbol (case-insensitive), OR
      - the company name (case-insensitive), OR
      - the first word of the company name (e.g. "Apple" from "Apple Inc")
    """
    ticker_lc = ticker.lower()
    name_lc = company_name.lower() if company_name else None
    name_head = name_lc.split()[0] if name_lc and len(name_lc.split()[0]) >= 4 else None

    matched: list[dict] = []
    rejected: list[dict] = []
    for item in news:
        text = ((item.get("title") or "") + " " + (item.get("content") or "")).lower()
        if (
            ticker_lc in text
            or (name_lc and name_lc in text)
            or (name_head and name_head in text)
        ):
            matched.append(item)
        else:
            rejected.append(item)
    return matched, rejected


def _search_news(anomaly: Anomaly) -> list[dict] | None:
    """Run a Tavily news search tailored to the anomaly type."""
    keywords = " ".join(_FLAG_SEARCH.get(f, f) for f in anomaly.flags)
    query = f"{anomaly.ticker} stock {keywords}"

    try:
        client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        response = client.search(
            query=query,
            max_results=5,
            topic="news",
            days=3,
            search_depth="basic",
        )
    except Exception as exc:
        logger.warning("Tavily search failed for %s: %s", anomaly.ticker, exc)
        return None

    return response.get("results", [])


def _synthesize(
    anomaly: Anomaly, news: list[dict],
) -> tuple[dict, int]:
    """Ask DeepSeek to extract reasons + next_steps from news snippets.

    Returns (parsed_dict, tokens). The dict has keys 'reasons' and 'next_steps',
    both lists. Returns ({}, 0) on any failure.
    """
    flag_text = " + ".join(_FLAG_HUMAN.get(f, f) for f in anomaly.flags)

    context_lines: list[str] = []
    for i, item in enumerate(news, 1):
        title = (item.get("title") or "").strip()
        content = (item.get("content") or "").strip()[:300]
        url = item.get("url") or ""
        context_lines.append(f"[{i}] {title}\n    {content}\n    来源: {url}")
    context = "\n\n".join(context_lines)

    user_prompt = (
        f"异动事件：{anomaly.ticker} 出现 {flag_text}\n"
        f"日期：{anomaly.date}\n"
        f"今日价格变化：{anomaly.price_change_pct:+.2%}\n"
        f"成交量倍数：{anomaly.volume_ratio:.1f}x\n\n"
        f"通过实体校验的新闻片段（已过滤无关项）：\n{context}\n\n"
        f"请同时给出 reasons（归因）和 next_steps（下一步观察建议）。"
    )

    try:
        llm = ChatDeepSeek(model="deepseek-chat", temperature=0.3)
        response = llm.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])
    except Exception as exc:
        logger.warning("DeepSeek attribution failed for %s: %s", anomaly.ticker, exc)
        return {}, 0

    content = _strip_code_fence(response.content)
    try:
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            return {}, 0
        # Sanitize: stringify and drop empties
        out = {
            "reasons":    [str(r) for r in parsed.get("reasons", []) if r],
            "next_steps": [str(s) for s in parsed.get("next_steps", []) if s],
        }
    except (json.JSONDecodeError, TypeError) as exc:
        logger.warning("DeepSeek returned non-JSON for %s: %s", anomaly.ticker, exc)
        return {}, 0

    tokens = 0
    meta = getattr(response, "usage_metadata", None) or {}
    if isinstance(meta, dict):
        tokens = int(meta.get("total_tokens", 0))

    return out, tokens


# ---------------------------------------------------------------------------
# Phase B ① C: Verifier LLM — second pass that scores causal confidence
# ---------------------------------------------------------------------------


_VERIFIER_PROMPT = (
    "你是一名严苛的金融分析师，负责评估归因理由的因果链强度。"
    "给定一次股票异动、候选归因、以及**带 source tier 标注的新闻片段**，"
    "对每条理由独立打分。\n"
    "\n"
    "【Source Tier 含义】\n"
    "  - Tier 1：监管机构（SEC）或顶级权威财媒（Reuters / Bloomberg / WSJ / FT / CNBC）\n"
    "  - Tier 2：主流财经媒体（Investors / SeekingAlpha / MarketWatch / Fool 等）\n"
    "  - Tier 3：一般来源、博客、聚合站、不知名媒体\n"
    "\n"
    "【打分规则 — 按证据 source 强度】\n"
    "  - 高 🟢：≥ 1 条 Tier 1 OR ≥ 2 条 Tier 2 直接支持本理由，且能解释异动幅度\n"
    "  - 中 🟡：1 条 Tier 2 支持，或 Tier 1 仅间接相关，或行业溢出/同业联动\n"
    "  - 低 🔴：仅 Tier 3 来源、未经证实传闻、单一来源、时间不匹配、与异动量级不符\n"
    "\n"
    "硬约束：\n"
    "1. 严格、保守——理由模糊或仅 Tier 3 支撑时强制'低'\n"
    "2. note 字段可选（≤ 20 字简评，建议在'中/低'时填写，最好点出 source 弱点）\n"
    "3. 只输出 JSON，不要 markdown\n"
    "\n"
    'JSON 格式：\n'
    '{"scores": [\n'
    '  {"text": "原理由原文", "confidence": "高", "note": ""},\n'
    '  {"text": "原理由原文", "confidence": "低", "note": "仅 Tier 3 单一来源"}\n'
    "]}\n"
    "\n"
    "顺序必须与输入一致，text 必须照抄输入。"
)


# ---------------------------------------------------------------------------
# Source-tier classification (Step 2 of resume polish)
# ---------------------------------------------------------------------------

_TIER1_DOMAINS = frozenset({
    "sec.gov",
    "reuters.com",
    "bloomberg.com",
    "wsj.com",
    "ft.com",
    "cnbc.com",
    "nytimes.com",
    "barrons.com",
    "economist.com",
})

_TIER2_DOMAINS = frozenset({
    "investors.com",
    "seekingalpha.com",
    "marketwatch.com",
    "fool.com",
    "businessinsider.com",
    "yahoo.com",
    "finance.yahoo.com",
    "247wallst.com",
    "morningstar.com",
    "forbes.com",
    "thestreet.com",
    "benzinga.com",
})


def _source_tier(url: str) -> int:
    """Map a news URL to a source-quality tier (1=highest, 3=lowest)."""
    if not url:
        return 3
    try:
        from urllib.parse import urlparse
        domain = urlparse(url).netloc.lower().lstrip(".")
        if domain.startswith("www."):
            domain = domain[4:]
    except Exception:
        return 3

    if any(domain == d or domain.endswith("." + d) for d in _TIER1_DOMAINS):
        return 1
    if any(domain == d or domain.endswith("." + d) for d in _TIER2_DOMAINS):
        return 2
    return 3


def _tier_label(tier: int) -> str:
    return {1: "Tier 1 权威", 2: "Tier 2 主流", 3: "Tier 3 一般"}.get(tier, "Tier 3")


def _verify_reasons(
    anomaly: Anomaly,
    reasons: list[str],
    news: list[dict],
) -> tuple[list[ScoredReason], int]:
    """Second LLM pass: score each Generator-produced reason's causal confidence."""
    flag_text = " + ".join(_FLAG_HUMAN.get(f, f) for f in anomaly.flags)

    # Annotate each news item with its source tier (Step 2: source-quality scoring)
    news_context: list[str] = []
    tier_counts = {1: 0, 2: 0, 3: 0}
    for i, item in enumerate(news[:5], 1):
        title = (item.get("title") or "").strip()
        content = (item.get("content") or "").strip()[:200]
        url = (item.get("url") or "").strip()
        tier = _source_tier(url)
        tier_counts[tier] += 1
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc or "?"
        except Exception:
            domain = "?"
        news_context.append(
            f"[{i}] [{_tier_label(tier)} · {domain}] {title}\n    {content}"
        )

    user_prompt = (
        f"异动：{anomaly.ticker} {flag_text}\n"
        f"价格变化：{anomaly.price_change_pct:+.2%}\n"
        f"成交量倍数：{anomaly.volume_ratio:.1f}x\n\n"
        f"候选归因（待评分）：\n"
        + "\n".join(f"{i}. {r}" for i, r in enumerate(reasons, 1))
        + "\n\n"
        f"支撑新闻片段（已分层）：\n"
        + "\n\n".join(news_context)
        + f"\n\n来源构成：Tier 1 ×{tier_counts[1]} · "
          f"Tier 2 ×{tier_counts[2]} · Tier 3 ×{tier_counts[3]}"
    )

    try:
        llm = ChatDeepSeek(model="deepseek-chat", temperature=0.2)  # lower T = stricter
        response = llm.invoke([
            SystemMessage(content=_VERIFIER_PROMPT),
            HumanMessage(content=user_prompt),
        ])
    except Exception as exc:
        logger.warning("Verifier LLM failed for %s: %s", anomaly.ticker, exc)
        # Fall back: keep reasons unscored (default confidence = "中")
        return [ScoredReason(text=r) for r in reasons], 0

    content = _strip_code_fence(response.content)
    try:
        parsed = json.loads(content)
        raw_scores = parsed.get("scores", []) if isinstance(parsed, dict) else []
    except json.JSONDecodeError as exc:
        logger.warning("Verifier non-JSON for %s: %s", anomaly.ticker, exc)
        return [ScoredReason(text=r) for r in reasons], 0

    # Map Verifier output back to input order. Match by text if possible,
    # else fall back to index alignment.
    scored: list[ScoredReason] = []
    by_text = {
        (s.get("text") or "").strip(): s
        for s in raw_scores if isinstance(s, dict)
    }
    for i, reason in enumerate(reasons):
        score = by_text.get(reason.strip())
        if score is None and i < len(raw_scores) and isinstance(raw_scores[i], dict):
            score = raw_scores[i]
        if score is None:
            scored.append(ScoredReason(text=reason))
            continue

        conf = str(score.get("confidence", "中")).strip()
        # Sanitize confidence (accept aliases / English / etc.)
        if conf not in ("高", "中", "低"):
            mapped = {
                "high": "高", "medium": "中", "med": "中", "low": "低",
                "h": "高", "m": "中", "l": "低",
                "强": "高", "弱": "低",
            }
            conf = mapped.get(conf.lower(), "中")
        note = str(score.get("note", "")).strip()[:40]
        scored.append(ScoredReason(text=reason, confidence=conf, note=note))

    tokens = 0
    meta = getattr(response, "usage_metadata", None) or {}
    if isinstance(meta, dict):
        tokens = int(meta.get("total_tokens", 0))

    return scored, tokens


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text
