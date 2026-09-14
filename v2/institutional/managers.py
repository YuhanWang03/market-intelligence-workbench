"""Hardcoded list of institutional managers to track via 13F.

Each entry: (CIK as string, display name). CIKs can be verified by browsing
https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=<cik>

Updating this list: just add (cik, name) tuples. The orchestrator handles
the rest.
"""

# 10 managers covering value, quant, activist, and growth styles
MANAGERS: list[tuple[str, str]] = [
    ("1067983", "Berkshire Hathaway"),         # Warren Buffett
    ("1649339", "Scion Asset Mgmt"),            # Michael Burry
    ("1336528", "Pershing Square Capital"),     # Bill Ackman
    ("1079114", "Greenlight Capital"),          # David Einhorn
    ("1037389", "Renaissance Technologies"),
    ("1179392", "Two Sigma Investments"),
    ("1009207", "D.E. Shaw & Co"),
    ("1423053", "Citadel Advisors"),
    ("1135730", "Coatue Management"),
    ("1697748", "ARK Investment Mgmt"),         # Cathie Wood
]
