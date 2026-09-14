"""Phase 3 — SEC monitoring agent (8-K + Form 4).

Separate from v2/institutional/ (13F-HR) because 8-K item parsing and
Form 4 transaction parsing share zero logic with quarterly position
snapshots. See Stage 0 task 1 for the architectural decision.

10-Q parsing is deferred to Phase 3.5 (not in Stage 1 scope).
"""

from v2.sec.client import get_recent_filings
from v2.sec.cluster import find_clusters
from v2.sec.eight_k_parser import classify_item, get_item_text, parse_eight_k_filing
from v2.sec.form4_parser import parse_form4_filing
from v2.sec.insider_role import lookup_insider_role
from v2.sec.models import (
    NOISE_TRANSACTION_CODES,
    SIGNAL_TRANSACTION_CODES,
    EightKEvent,
    EightKItem,
    Form4Cluster,
    Form4Transaction,
    PriorityTier,
    SecFiling,
    SecScanResult,
)
from v2.sec.ner_5_02 import extract_5_02
from v2.sec.pipeline import run_sec_scan

__all__ = [
    "NOISE_TRANSACTION_CODES",
    "SIGNAL_TRANSACTION_CODES",
    "EightKEvent",
    "EightKItem",
    "Form4Cluster",
    "Form4Transaction",
    "PriorityTier",
    "SecFiling",
    "SecScanResult",
    "classify_item",
    "extract_5_02",
    "find_clusters",
    "get_item_text",
    "get_recent_filings",
    "lookup_insider_role",
    "parse_eight_k_filing",
    "parse_form4_filing",
    "run_sec_scan",
]
