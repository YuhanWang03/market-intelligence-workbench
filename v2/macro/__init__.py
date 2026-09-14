"""Phase 4 — Macro Agent (FOMC / CPI / PCE / NFP via FRED + Tavily).

Separate from v2/earnings/ (corporate earnings) because the data shape,
release cadence, and hallucination-defense posture are fundamentally
different: macro releases come from BLS/BEA/Fed schedules, the FOMC
has its own statement-diff path that does NOT use the LLM for the
hawkish/dovish judgement, and the prompt layer forbids forward
prediction at multiple checkpoints.

Public API — Stage 1 surface:
- :func:`build_macro_snapshot` — ⑭ daily close snapshot
- :func:`build_release_event` — ⑮ release-day report (CPI / PCE / NFP / GDP / PPI / FOMC)
- :func:`build_claims_event`  — ⑯ Thursday weekly Claims
- :func:`build_weekly_recap`  — ⑰ Friday weekly summary
- :data:`MacroSnapshot`, :data:`MacroRelease`, :data:`FOMCEvent`,
  :data:`MacroReport`, :data:`ReleaseVintage`

Stages 2-7 build on this:
- Stage 2: 4 cron scripts + priority kinds + 14→18 scheduler jobs
- Stage 3: LLM fingerprint registered (already done in this stage)
- Stage 4: bot commands /macro + /fomc + 2 NL intents
- Stage 5: formatter lift to v2/reporting/format_macro_*
- Stage 6: cron integration tests
- Stage 7: README + final-check
"""

from v2.macro.models import (
    FOMCEvent,
    MacroRelease,
    MacroReport,
    MacroSnapshot,
    ReleaseVintage,
)
from v2.macro.pipeline import (
    build_claims_event,
    build_macro_snapshot,
    build_release_event,
    build_weekly_recap,
)

__all__ = [
    "FOMCEvent",
    "MacroRelease",
    "MacroReport",
    "MacroSnapshot",
    "ReleaseVintage",
    "build_claims_event",
    "build_macro_snapshot",
    "build_release_event",
    "build_weekly_recap",
]
