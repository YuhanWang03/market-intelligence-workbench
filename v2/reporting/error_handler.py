"""Wrap entry scripts so any unhandled exception triggers a Telegram alert.

Usage::

    from v2.reporting import notify_on_error

    @notify_on_error("Daily Screen")
    def main() -> None:
        ...

The exception is re-raised after notification so the local console still
shows the full traceback and the process exits with a non-zero code
(important for cron / scheduler integration).
"""

from __future__ import annotations

import functools
import html
import logging
import traceback
from typing import Callable

from v2.reporting.notifier import TelegramNotifier

logger = logging.getLogger(__name__)


def notify_on_error(script_name: str) -> Callable:
    """Decorator: push a Telegram alert if the wrapped function raises."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                _push(script_name, exc, traceback.format_exc())
                raise
        return wrapper

    return decorator


def _push(script_name: str, exc: Exception, tb: str) -> None:
    """Best-effort push — never raises (we're already in an error path)."""
    try:
        notifier = TelegramNotifier()
        notifier.send_text(
            f"<b>❌ {html.escape(script_name)} 失败</b>\n\n"
            f"<code>{html.escape(type(exc).__name__)}: "
            f"{html.escape(str(exc)[:200])}</code>\n\n"
            f"<pre>{html.escape(tb[-800:])}</pre>"
        )
    except Exception as inner:
        logger.warning("Failed to push error notification: %s", inner)
