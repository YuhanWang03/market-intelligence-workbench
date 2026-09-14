"""Notifier protocol and concrete implementations.

The Protocol defines what it means to "push a report somewhere".
Any class with these methods can be passed wherever a Notifier is expected.

Implementations:
    ConsoleNotifier  — prints to stdout, for local debugging
    TelegramNotifier — pushes to a Telegram chat via Bot API

To add a new channel (Feishu, Discord, Slack), just write a class with the
two methods — no inheritance needed.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from io import BytesIO
from typing import Iterable, Protocol, runtime_checkable

from v2.archive.store import trading_day_expiry_iso
from v2.reporting.priority import (
    PriorityResult,
    compute_importance,
    tier_emoji_prefix,
)

# Pushes are retained in archive.db for two calendar days. The cleanup job
# (v2.scheduler.jobs.archive_cleanup_job) sweeps expired rows once a day.
_RETENTION_DAYS = 2


def _default_p1() -> PriorityResult:
    """Fallback priority for callers that don't pass one explicitly.

    Scored "default" so we land in the P2 band by score, but constructed
    with tier="P1" so behavior matches the historical default (immediate
    Telegram + archive). Existing code paths that haven't been migrated
    to the new priority system still work.
    """
    base = compute_importance("default", {})
    # Force tier to P1 even if base score is P2 — backward-compat default.
    return PriorityResult(
        score=base.score, tier="P1",
        reasons=base.reasons + ["forced_p1_backward_compat"],
    )


@runtime_checkable
class Notifier(Protocol):
    """Anything that can push a text message and a photo."""

    def send_text(self, text: str) -> None: ...
    def send_photo(self, image: bytes, caption: str = "") -> None: ...


class ConsoleNotifier:
    """Local fallback — prints to stdout. Use for development."""

    def send_text(self, text: str) -> None:
        print("─" * 60)
        print(text)
        print("─" * 60)

    def send_photo(self, image: bytes, caption: str = "") -> None:
        print(f"[image: {len(image):,} bytes] {caption}")


class TelegramNotifier:
    """Push to a Telegram chat via Bot API.

    Reads TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID from env if not given.
    Uses HTML parse mode (simpler escaping than MarkdownV2).

    If an *archive* is given, every send_* call also writes a row to the
    local archive DB. Real-time callers may request Telegram-first ordering,
    so only successfully delivered alerts become visible in the web feed.
    """

    def __init__(
        self,
        token: str | None = None,
        chat_id: str | None = None,
        *,
        archive=None,
        archive_after_send: bool = False,
        retention_trading_days: int | None = None,
    ) -> None:
        self._token = token or os.environ["TELEGRAM_BOT_TOKEN"]
        self._chat_id = chat_id or os.environ["TELEGRAM_CHAT_ID"]
        self._archive = archive
        self._archive_after_send = archive_after_send
        self._retention_trading_days = retention_trading_days

    def send_text(
        self,
        text: str,
        *,
        trace=None,
        title: str | None = None,
        tickers: Iterable[str] = (),
        priority=None,
    ) -> None:
        """Push to Telegram + archive a row, with tier-aware behavior.

        priority: a v2.reporting.priority.PriorityResult. If omitted,
        the call defaults to P1 (immediate Telegram + archive) for
        backward compatibility with callers from before this change.

        Tier behavior:
          P0 → Telegram with 🚨🚨🚨 prefix + archive
          P1 → Telegram (no prefix) + archive
          P2 → archive only; daily digest cron rolls these up
          P3 → archive only; dashboard hides by default
        """
        priority = priority or _default_p1()
        decorated = tier_emoji_prefix(priority.tier) + text
        if priority.tier in ("P2", "P3"):
            self._archive_with_priority(
                kind="text",
                text=text, image=None, caption=None,
                trace=trace, title=title, tickers=tickers, priority=priority,
            )
            return    # archive-only; no Telegram
        if self._archive_after_send:
            asyncio.run(self._send_text(decorated))
            self._archive_with_priority(
                kind="text",
                text=text, image=None, caption=None,
                trace=trace, title=title, tickers=tickers, priority=priority,
            )
        else:
            self._archive_with_priority(
                kind="text",
                text=text, image=None, caption=None,
                trace=trace, title=title, tickers=tickers, priority=priority,
            )
            asyncio.run(self._send_text(decorated))

    def send_photo(
        self,
        image: bytes,
        caption: str = "",
        *,
        trace=None,
        title: str | None = None,
        tickers: Iterable[str] = (),
        priority=None,
    ) -> None:
        priority = priority or _default_p1()
        decorated = tier_emoji_prefix(priority.tier) + caption
        if priority.tier in ("P2", "P3"):
            self._archive_with_priority(
                kind="photo",
                text=None, image=image, caption=caption,
                trace=trace, title=title, tickers=tickers, priority=priority,
            )
            return
        if self._archive_after_send:
            asyncio.run(self._send_photo(image, decorated))
            self._archive_with_priority(
                kind="photo",
                text=None, image=image, caption=caption,
                trace=trace, title=title, tickers=tickers, priority=priority,
            )
        else:
            self._archive_with_priority(
                kind="photo",
                text=None, image=image, caption=caption,
                trace=trace, title=title, tickers=tickers, priority=priority,
            )
            asyncio.run(self._send_photo(image, decorated))

    def _archive_with_priority(
        self, *, kind: str, text, image, caption,
        trace, title, tickers, priority,
    ) -> None:
        """One archive write that carries the priority fields. Centralized
        so the text / photo paths can't drift apart."""
        if self._archive is None:
            return
        common = dict(
            tickers=tickers,
            trace_json=_trace_to_json(trace),
            title=title,
            expires_at=(
                trading_day_expiry_iso(self._retention_trading_days)
                if self._retention_trading_days else _expires_at()
            ),
            importance_score=priority.score,
            priority_tier=priority.tier,
            priority_reasons=",".join(priority.reasons),
        )
        if kind == "text":
            self._archive.save_text(text, **common)
        else:
            self._archive.save_photo(image, caption or "", **common)
        if self._retention_trading_days:
            self._archive.prune_realtime(self._retention_trading_days)

    async def _send_text(self, text: str) -> None:
        from telegram import Bot

        bot = Bot(token=self._token)
        async with bot:
            await bot.send_message(
                chat_id=self._chat_id, text=text, parse_mode="HTML",
            )

    async def _send_photo(self, image: bytes, caption: str) -> None:
        from telegram import Bot

        bot = Bot(token=self._token)
        async with bot:
            await bot.send_photo(
                chat_id=self._chat_id,
                photo=BytesIO(image),
                caption=caption,
                parse_mode="HTML",
            )


def _trace_to_json(trace) -> str | None:
    """Serialize a Trace's events list to JSON, or None if no trace bound."""
    if trace is None:
        return None
    events = getattr(trace, "events", None)
    if not events:
        return None
    try:
        return json.dumps(events, ensure_ascii=False)
    except (TypeError, ValueError):
        # Best-effort — never let archive serialization break a push.
        return None


def _expires_at() -> str:
    """ISO 8601 timestamp two calendar days from now (UTC)."""
    return (datetime.now(timezone.utc) + timedelta(days=_RETENTION_DAYS)).isoformat()
