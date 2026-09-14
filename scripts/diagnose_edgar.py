"""Probe edgartools to see what's actually returned for 13F filings.

Usage:
    poetry run python scripts/diagnose_edgar.py
"""

from __future__ import annotations

import os

from edgar import Company, set_identity

set_identity(os.environ.get("EDGAR_IDENTITY", "Market Intelligence Workbench contact@example.com"))

co = Company("1067983")  # Berkshire — known to have many 13F filings
filings = co.get_filings(form="13F-HR")

print(f"Type: {type(filings).__name__}")
print(f"Length: {len(filings) if hasattr(filings, '__len__') else 'unknown'}")

print("\n--- First 5 (as iterated) ---")
for i, f in enumerate(list(filings)[:5]):
    period = getattr(f, "period_of_report", None)
    filed = getattr(f, "filing_date", None)
    print(f"  [{i}] period={period}  filed={filed}")

print("\n--- Last 5 (as iterated) ---")
all_f = list(filings)
for i, f in enumerate(all_f[-5:]):
    period = getattr(f, "period_of_report", None)
    filed = getattr(f, "filing_date", None)
    print(f"  [{len(all_f) - 5 + i}] period={period}  filed={filed}")

print("\n--- Trying .latest(2) ---")
try:
    latest = filings.latest(2)
    print(f"  Type: {type(latest).__name__}")
    if hasattr(latest, "__iter__") and not isinstance(latest, str):
        for f in latest:
            print(f"  period={getattr(f, 'period_of_report', '?')}")
    else:
        print(f"  Single object: period={getattr(latest, 'period_of_report', '?')}")
except Exception as e:
    print(f"  ERROR: {e}")

print("\n--- Parsing the latest filing's holdings ---")
# Pick whatever single one looks newest
candidates = sorted(
    all_f,
    key=lambda f: str(getattr(f, "period_of_report", "0000")),
    reverse=True,
)
latest_one = candidates[0]
print(f"  Chose: period={latest_one.period_of_report}, accession={latest_one.accession_no}")

try:
    thirteenF = latest_one.obj()
    print(f"  obj type: {type(thirteenF).__name__}")
    print(f"  attrs sample: {[a for a in dir(thirteenF) if not a.startswith('_')][:25]}")

    # Try multiple property names
    for prop in ("infotable", "holdings", "info_table"):
        val = getattr(thirteenF, prop, None)
        if val is not None:
            print(f"  .{prop}: type={type(val).__name__}, "
                  f"len={len(val) if hasattr(val, '__len__') else '?'}")
            if hasattr(val, "columns"):
                print(f"    columns: {list(val.columns)}")
            if hasattr(val, "iterrows"):
                print(f"    --- First 2 rows ---")
                for i, (_, row) in enumerate(val.iterrows()):
                    if i >= 2:
                        break
                    print(f"    [{i}] {dict(row)}")
            break
except Exception as e:
    import traceback
    print(f"  ERROR: {e}")
    traceback.print_exc()
