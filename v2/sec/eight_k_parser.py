"""8-K item code extraction + priority classification.

Stage 0 deliverable — the priority rating table for every 8-K item code,
calibrated to the user's real 30-day universe activity.

Per Stage 0 decision #3 (real-data calibration): a single 8-K with N
items emits ONE card listing all items with their per-item priority,
not N cards. ``parse_eight_k_filing`` returns a single ``EightKEvent``
per filing.

Per Stage 0 decision #4: filings whose only material item is 2.02
(earnings results) are flagged ``has_earnings_overlap=True, is_2_02_only=True``
and the cron skips them — Phase 1 ⑧ Earnings Summaries already handles
that data, no duplicate notifications.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from v2.sec.models import EightKEvent, EightKItem, PriorityTier, SecFiling

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Priority table — single source of truth for 8-K item severity
# ---------------------------------------------------------------------------
# Derived from Stage 0 task 3 audit. Each item carries:
#   (tier, human-readable Chinese description)
#
# Items not in this table fall through to ("P3", "其他 item"). That's
# intentional — SEC adds new items occasionally (1.05 added in 2023 for
# cybersecurity), and we'd rather under-prioritize an unknown item than
# silently drop it.

_ITEM_TABLE: dict[str, tuple[PriorityTier, str]] = {
    # ---- P0: existential / fraud-suspicion / control change ----
    "1.03": ("P0", "破产 / 重整"),
    "1.05": ("P0", "重大网络安全事件"),
    "2.04": ("P0", "债务加速到期 (covenant 违约)"),
    "3.01": ("P0", "退市"),
    "4.02": ("P0", "前期财报不可依赖 (restatement)"),
    "5.01": ("P0", "控制权变更"),
    # 5.02 is special — defaults P1, escalates to P0 if LLM extracts
    # senior-exec departure. See pipeline.py for the escalation.
    "5.02": ("P1", "高管 / 董事会变动"),

    # ---- P1: material business events ----
    "1.01": ("P1", "重大商业合约 (新签)"),
    "1.02": ("P1", "重大商业合约 (终止)"),
    "2.01": ("P1", "M&A 完成"),
    "2.03": ("P1", "重大债务承担"),
    "2.05": ("P1", "重组 / 退出业务"),
    "2.06": ("P1", "资产减值"),
    "4.01": ("P1", "审计师变更"),

    # ---- P2: routine governance / disclosure ----
    "2.02": ("P2", "财报数据"),   # overlaps with ⑧ — pipeline skips when alone
    "3.02": ("P2", "未注册证券发行 (PIPE)"),
    "3.03": ("P2", "证券持有人权利变更"),
    "5.08": ("P2", "股东董事提名"),
    "7.01": ("P2", "Reg FD 自愿披露"),
    "8.01": ("P2", "其他事件"),

    # ---- P3: noise / administrative ----
    "1.04": ("P3", "矿山安全披露"),
    "5.03": ("P3", "公司章程修订"),
    "5.04": ("P3", "退休计划交易暂停"),
    "5.05": ("P3", "道德准则修订"),
    "5.07": ("P3", "股东大会投票结果"),
    "9.01": ("P3", "财务报表 / 附件"),  # always co-filed, never material alone
}

_DEFAULT_TIER: PriorityTier = "P3"
_DEFAULT_DESC = "其他 item"


# Match "ITEM 5.02" / "Item 5.02:" / "5.02 Departure of Directors" etc.
# edgartools returns items as strings of the form "ITEM 5.02:..." or
# "ITEM 5.02 Departure of Directors..."; this regex tolerates both.
_ITEM_CODE_RE = re.compile(
    r"(?:ITEM\s+)?(\d{1,2}\.\d{2})",
    re.IGNORECASE,
)


def _extract_code(raw: str) -> str | None:
    """Pull the ``X.YY`` numeric code out of an edgartools item string.

    Returns None if the string doesn't look like an item code at all
    (defensive — SDK occasionally returns explanatory blurbs in the
    list when a filing has malformed structure).
    """
    if not raw:
        return None
    m = _ITEM_CODE_RE.search(str(raw))
    return m.group(1) if m else None


def classify_item(code: str) -> tuple[PriorityTier, str]:
    """Return ``(priority_tier, description)`` for an 8-K item code.

    Unknown codes default to ``("P3", "其他 item")`` so new SEC items
    don't crash the cron.
    """
    return _ITEM_TABLE.get(code, (_DEFAULT_TIER, _DEFAULT_DESC))


# ---------------------------------------------------------------------------
# Stage 2 pickup — item-text extractor for the 5.02 LLM path
# ---------------------------------------------------------------------------

# Matches "Item 5.02" / "ITEM 5.02" / "Item 5.02:" / "Item 5.02 Departure of..."
# at the start of a section. The ``\b`` after the code prevents matching
# 5.02 inside a longer code (no real 8-K item starts with 5.020+ but the
# regex is defensive).
def _item_header_pattern(code: str) -> "re.Pattern[str]":
    return re.compile(
        rf"item\s+{re.escape(code)}\b",
        re.IGNORECASE,
    )


def get_item_text(eight_k_obj, code: str) -> str:
    """Extract the body text for one specific item from a parsed 8-K.

    edgartools' ``EightK`` exposes ``.text`` containing the full filing
    body. SEC 8-Ks structure each item as a section starting with
    "Item X.YY [Title]". This function locates the requested ``code``'s
    header and returns the text up to the next "Item X.YY" header (or
    end of document).

    Args:
        eight_k_obj: an edgartools ``EightK`` instance (from ``filing.obj()``).
        code: the item code to extract, e.g. ``"5.02"``.

    Returns:
        The item body text. Returns ``""`` on:
        - missing/empty ``.text`` attribute
        - item code not found in the document
        - any parsing exception

    Empty return triggers ner_5_02's conservative fallback (
    ``has_senior_exec=True``), so the cron still escalates to P0 on
    parser failure rather than silently dropping the LLM call.
    """
    try:
        text = getattr(eight_k_obj, "text", None) or ""
        if callable(text):
            # Newer edgartools exposes ``EightK.text()`` as a method; the
            # bound method used to reach the regex and fail every 8-K.
            text = text() or ""
        if not isinstance(text, str):
            text = str(text)
        if not text:
            return ""

        header_re = _item_header_pattern(code)
        match = header_re.search(text)
        if match is None:
            return ""

        start = match.start()

        # Find the next "Item X.YY" boundary. Search after the current
        # header's end (not start, to avoid matching itself).
        next_re = re.compile(
            r"item\s+\d{1,2}\.\d{2}\b",
            re.IGNORECASE,
        )
        next_match = next_re.search(text, match.end())
        end = next_match.start() if next_match else len(text)

        # Sanity cap: 8-K items are typically 200-2000 chars. If
        # something matched implausibly large (e.g. the next-boundary
        # regex failed because the SDK normalized whitespace), cap to
        # 4000 chars so the LLM context stays well below budget.
        section = text[start:end].strip()
        if len(section) > 4000:
            section = section[:4000]
        return section

    except Exception as exc:
        logger.warning("get_item_text(%s) failed: %s", code, exc)
        return ""


def parse_eight_k_filing(
    edgar_filing: Any,
    sec_filing: SecFiling,
) -> EightKEvent | None:
    """Convert an edgartools 8-K filing into our ``EightKEvent``.

    Args:
        edgar_filing: an edgartools ``Filing`` object (8-K form). Its
            ``.obj()`` returns an ``EightK`` instance with an ``.items``
            attribute (list of item-string titles).
        sec_filing: the pre-built ``SecFiling`` metadata for the same
            filing (constructed by the caller; we receive it instead of
            re-extracting accession_number etc.).

    Returns:
        ``EightKEvent`` with all items classified, or ``None`` if the
        filing's ``.obj()`` failed (logged as warning).
    """
    try:
        eight_k = edgar_filing.obj()
    except Exception as exc:
        logger.warning(
            "8-K .obj() failed for %s acc=%s: %s",
            sec_filing.ticker, sec_filing.accession_number, exc,
        )
        return None

    raw_items = getattr(eight_k, "items", None) or []

    classified: list[EightKItem] = []
    for raw in raw_items:
        code = _extract_code(raw)
        if code is None:
            continue
        tier, description = classify_item(code)
        classified.append(EightKItem(
            code=code,
            priority_tier=tier,
            description=description,
            extracted_meta={},     # 5.02 LLM extraction happens in pipeline
        ))

    event = EightKEvent(
        filing=sec_filing,
        items=classified,
    )
    # has_earnings_overlap is True if ANY item is 2.02; cron uses
    # is_2_02_only (defined on EightKEvent) to decide whether to skip
    # the whole filing or just annotate the 2.02 line.
    event.has_earnings_overlap = any(it.code == "2.02" for it in classified)
    return event
