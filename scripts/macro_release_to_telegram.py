"""Macro release scanner — Mon-Fri 09:00 ET.

The fifteenth scheduled agent. Gates internally on
``release_calendar.get_release_today(today_iso)``; non-release days
exit silently with no archive write.

On release days:

- **CPI / PCE / NFP / GDP / PPI** → ``build_release_event`` runs the
  full template-fill pipe: FRED series → numeric transforms →
  summarizer (Layer 1 prompt + Layer 2 regex reject of predictive
  verbs + numeric leak). The card surfaces Python-computed numbers
  alongside the LLM's qualitative labels.

- **FOMC** → routes through ``fomc_parser`` (Python statement diff +
  SEP dot-plot extract) + ``tavily_consensus`` (sell-side hawkish /
  dovish majority vote restricted to 8 trusted news domains). The
  LLM is NEVER asked for a hawkish / dovish verdict per Stage 0
  design ack.

Priority is computed per release using ``surprise_sigma`` magnitude
and the FOMC SEP shift; see ``v2/reporting/priority.py`` for the
ladder.

⑮b FOMC +6h follow-up (transcript-based re-push) is deferred to
Phase 4.5 — needs its own Stage 0 work to choose a transcript source
and chunking strategy. The main ⑮ card already carries the statement
diff + dot plot + sell-side aggregate, which is the substance.

Card formatter is inline here for Stage 2 (Stage 5 lifts to
``v2.reporting.format_macro_*``).
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from v2.archive import Archive
from v2.macro import build_release_event
from v2.macro.release_calendar import get_release_today
from v2.observability import capture_trace_with_framing, install_all
from v2.reporting import (
    TelegramNotifier,
    format_macro_fomc_card,
    format_macro_release_card,
    notify_on_error,
)
from v2.reporting.priority import compute_importance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


_TZ_ET = ZoneInfo("US/Eastern")


# ---------------------------------------------------------------------------
# Priority routing
# ---------------------------------------------------------------------------

def _release_kind_and_meta(release) -> tuple[str, dict]:
    """Pick the priority kind for a non-FOMC release based on
    surprise_label, plus the metadata the priority module reads."""
    sigma = release.surprise_sigma or 0.0
    abs_sigma = abs(sigma)
    md = {"surprise_sigma": sigma, "surprise_label": release.surprise_label}

    if abs_sigma >= 3.0:
        return "macro_release_p0", md
    if abs_sigma >= 1.0:
        return "macro_release_p1", md
    return "macro_release_p2", md


def _fomc_kind_and_meta(event) -> tuple[str, dict]:
    """FOMC routing: SEP shift → P0; sell-side hawkish unexpected →
    extra nudge; otherwise base P1 (FOMC is always at least P1)."""
    md = {
        "is_fomc": True,
        "sep_shift": event.sep_dot_plot_change,
        "sell_side_consensus": (
            "hawkish_unexpected"
            if event.sell_side_sentiment == "hawkish"
            else event.sell_side_sentiment
        ),
    }
    if event.sep_dot_plot_change in ("hawkish_shift", "dovish_shift"):
        return "macro_release_p0", md
    return "macro_release_p1", md


# ---------------------------------------------------------------------------
# Push helpers
# ---------------------------------------------------------------------------

def _push_release(notifier, trace, release) -> None:
    kind, md = _release_kind_and_meta(release)
    priority = compute_importance(kind, md)

    text = format_macro_release_card(release, tier=priority.tier)
    notifier.send_text(
        text,
        trace=trace,
        title=f"宏观 {release.release_type} · {release.period} · {priority.tier}",
        tickers=[],
        priority=priority,
    )


def _push_fomc(notifier, trace, fomc) -> None:
    kind, md = _fomc_kind_and_meta(fomc)
    priority = compute_importance(kind, md)

    text = format_macro_fomc_card(fomc, tier=priority.tier)
    notifier.send_text(
        text,
        trace=trace,
        title=f"FOMC · {fomc.meeting_date} · {priority.tier}",
        tickers=[],
        priority=priority,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@notify_on_error("Macro Release Scanner")
def main() -> int:
    load_dotenv()
    install_all()

    today_iso = datetime.now(_TZ_ET).date().isoformat()
    todays = get_release_today(today_iso)

    if not todays:
        logger.info("Macro release: no scheduled release on %s — silent exit",
                    today_iso)
        return 0

    archive = Archive("macro")

    with capture_trace_with_framing(
        agent="macro", intent="macro_release_view",
        text=f"(自动推送) 宏观 release scanner · {today_iso} · "
             f"{len(todays)} 个 release",
        responder_name="_r_macro_release",
    ) as trace:
        report = build_release_event(today_iso)
        trace.emit(
            "chat_message", role="bot",
            text=(
                f"宏观 release · {today_iso} · "
                f"{len(report.today_releases)} releases · "
                f"FOMC={report.fomc_event is not None} · "
                f"warnings={len(report.warnings)}"
            ),
        )

        notifier = TelegramNotifier(archive=archive)

        # FOMC first if present (it's always the highest-priority event
        # of any FOMC day).
        if report.fomc_event is not None:
            try:
                _push_fomc(notifier, trace, report.fomc_event)
            except Exception as exc:
                logger.warning("FOMC push failed: %s", exc)

        for release in report.today_releases:
            try:
                _push_release(notifier, trace, release)
            except Exception as exc:
                logger.warning("Release push failed for %s: %s",
                               release.release_type, exc)

    logger.info(
        "Macro release complete: %d releases / FOMC=%s / warnings=%d",
        len(report.today_releases),
        report.fomc_event is not None,
        len(report.warnings),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
