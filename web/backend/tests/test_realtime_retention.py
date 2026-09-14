from __future__ import annotations

from datetime import datetime, timezone

import pytest

from v2.archive.store import (
    recent_trading_day_cutoff_iso,
    trading_day_expiry_iso,
)
from v2.reporting.notifier import TelegramNotifier


def test_two_trading_day_window_skips_weekend_and_market_holiday():
    # 2026-09-07 is Labor Day. On Tuesday the retained sessions are
    # Tuesday 09-08 and Friday 09-04.
    now = datetime(2026, 9, 8, 15, 0, tzinfo=timezone.utc)
    cutoff = datetime.fromisoformat(recent_trading_day_cutoff_iso(2, now=now))
    assert cutoff == datetime(2026, 9, 4, 4, 0, tzinfo=timezone.utc)


def test_realtime_expiry_starts_after_two_retained_sessions():
    # A Friday alert is retained for Friday and Tuesday when Monday is a
    # market holiday, then expires at Wednesday's ET midnight.
    now = datetime(2026, 9, 4, 15, 0, tzinfo=timezone.utc)
    expiry = datetime.fromisoformat(trading_day_expiry_iso(2, now=now))
    assert expiry == datetime(2026, 9, 9, 4, 0, tzinfo=timezone.utc)


class _FakeArchive:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def save_text(self, _text: str, **_kwargs) -> int:
        self.events.append("archive")
        return 1

    def prune_realtime(self, trading_days: int) -> int:
        assert trading_days == 2
        self.events.append("prune")
        return 0


def test_realtime_notifier_archives_only_after_telegram_success(monkeypatch):
    events: list[str] = []
    notifier = TelegramNotifier(
        token="test-token",
        chat_id="test-chat",
        archive=_FakeArchive(events),
        archive_after_send=True,
        retention_trading_days=2,
    )

    async def fake_send(_text: str) -> None:
        events.append("telegram")

    monkeypatch.setattr(notifier, "_send_text", fake_send)
    notifier.send_text("test alert")

    assert events == ["telegram", "archive", "prune"]


def test_realtime_notifier_does_not_archive_failed_delivery(monkeypatch):
    events: list[str] = []
    notifier = TelegramNotifier(
        token="test-token",
        chat_id="test-chat",
        archive=_FakeArchive(events),
        archive_after_send=True,
        retention_trading_days=2,
    )

    async def failed_send(_text: str) -> None:
        events.append("telegram")
        raise RuntimeError("delivery failed")

    monkeypatch.setattr(notifier, "_send_text", failed_send)
    with pytest.raises(RuntimeError, match="delivery failed"):
        notifier.send_text("test alert")

    assert events == ["telegram"]
