"""Named stock universes for the Lab screener.

Three index lists ship with the repo as a dated snapshot; ``load_universe``
prefers a fresher copy in ``data/universes.json`` when one exists.  Refresh
that file on a machine with open internet (the VPS) with::

    python -m v2.screening.universes --refresh            # all three, from Wikipedia
    python -m v2.screening.universes --show sp500          # print what will be used

A stale snapshot is harmless for screening: a delisted ticker simply fails
the metrics fetch and is skipped, a newcomer is missing until the next
refresh.  The snapshot date is reported alongside every screening run.

Backtests need more: the S&P 500 page also carries a table of every addition
and removal with its effective date.  ``--refresh`` stores it, and
:func:`members_at` rewinds today's list through those changes to give the
constituents on any past date (point-in-time membership).  Without that a
momentum backtest run on today's list "knows" which names were later added
because they rallied — a look-ahead bias that flatters the result.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import urllib.request
from datetime import date, datetime
from typing import Callable
from html.parser import HTMLParser
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "universes.json"

#: date of the bundled snapshot below
BUNDLED_AS_OF = "2025-12-31"

DOW30: list[str] = [
    "AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "GS", "HD", "HON", "IBM", "JNJ",
    "JPM", "KO", "MCD", "MMM", "MRK", "MSFT", "NKE", "NVDA", "PG", "SHW", "TRV", "UNH", "V", "VZ", "WMT",
]

NASDAQ100: list[str] = [
    "AAPL", "ABNB", "ADBE", "ADI", "ADP", "ADSK", "AEP", "AMAT", "AMD", "AMGN", "AMZN", "ANSS", "APP", "ARM", "ASML",
    "AVGO", "AXON", "AZN", "BIIB", "BKNG", "BKR", "CCEP", "CDNS", "CDW", "CEG", "CHTR", "CMCSA", "COST", "CPRT",
    "CRWD", "CSCO", "CSGP", "CSX", "CTAS", "CTSH", "DASH", "DDOG", "DXCM", "EA", "EXC", "FANG", "FAST", "FTNT",
    "GEHC", "GFS", "GILD", "GOOG", "GOOGL", "HON", "IDXX", "INTC", "INTU", "ISRG", "KDP", "KHC", "KLAC", "LIN",
    "LRCX", "LULU", "MAR", "MCHP", "MDLZ", "MELI", "META", "MNST", "MRVL", "MSFT", "MSTR", "MU", "NFLX", "NVDA",
    "NXPI", "ODFL", "ON", "ORLY", "PANW", "PAYX", "PCAR", "PDD", "PEP", "PLTR", "PYPL", "QCOM", "REGN", "ROP",
    "ROST", "SBUX", "SNPS", "TEAM", "TMUS", "TSLA", "TTD", "TTWO", "TXN", "VRSK", "VRTX", "WBD", "WDAY", "XEL", "ZS",
]

SP500: list[str] = [
    # Information technology
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "ADBE", "AMD", "CSCO", "ACN", "IBM", "INTU", "NOW", "QCOM", "TXN",
    "AMAT", "PANW", "ANET", "MU", "ADI", "LRCX", "KLAC", "APH", "CRWD", "CDNS", "SNPS", "MSI", "ADSK", "FTNT", "ROP",
    "WDAY", "NXPI", "MCHP", "TEL", "IT", "GLW", "CTSH", "FICO", "HPQ", "DELL", "MPWR", "KEYS", "ON", "CDW", "HPE",
    "TYL", "NTAP", "PTC", "TDY", "WDC", "STX", "ZBRA", "GDDY", "FSLR", "TER", "TRMB", "JBL", "AKAM", "SWKS", "GEN",
    "FFIV", "VRSN", "JNPR", "ENPH", "EPAM", "QRVO", "SMCI", "PLTR",
    # Communication services
    "GOOGL", "GOOG", "META", "NFLX", "DIS", "TMUS", "CMCSA", "VZ", "T", "CHTR", "EA", "WBD", "TTWO", "OMC", "LYV",
    "IPG", "FOXA", "FOX", "NWSA", "NWS", "MTCH", "PARA",
    # Consumer discretionary
    "AMZN", "TSLA", "HD", "MCD", "BKNG", "LOW", "TJX", "SBUX", "NKE", "ORLY", "CMG", "MAR", "GM", "HLT", "ABNB",
    "AZO", "ROST", "F", "DHI", "RCL", "YUM", "LEN", "TSCO", "EBAY", "GRMN", "DECK", "NVR", "PHM", "ULTA", "EXPE",
    "LULU", "DRI", "APTV", "CCL", "BBY", "POOL", "TPR", "KMX", "LVS", "DPZ", "GPC", "LKQ", "RL", "HAS", "MGM",
    "NCLH", "WYNN", "CZR", "MHK", "BWA", "ETSY",
    # Consumer staples
    "WMT", "COST", "PG", "KO", "PEP", "PM", "MDLZ", "MO", "CL", "TGT", "KMB", "MNST", "KDP", "KVUE", "GIS", "SYY",
    "STZ", "KHC", "HSY", "ADM", "KR", "CHD", "MKC", "K", "DG", "DLTR", "CLX", "EL", "TSN", "HRL", "CAG", "SJM", "CPB",
    "TAP", "BG", "LW", "BF.B", "WBA",
    # Health care
    "LLY", "UNH", "JNJ", "ABBV", "MRK", "TMO", "ABT", "ISRG", "AMGN", "DHR", "PFE", "SYK", "BSX", "VRTX", "GILD",
    "MDT", "BMY", "ELV", "CI", "REGN", "ZTS", "CVS", "BDX", "MCK", "HCA", "COR", "EW", "A", "IDXX", "IQV", "RMD",
    "GEHC", "HUM", "CNC", "DXCM", "MTD", "CAH", "BIIB", "WST", "ZBH", "STE", "WAT", "LH", "HOLX", "BAX", "DGX", "ALGN",
    "PODD", "MOH", "COO", "VTRS", "RVTY", "TECH", "INCY", "CRL", "UHS", "HSIC", "DVA", "SOLV", "MRNA",
    # Financials
    "BRK.B", "JPM", "V", "MA", "BAC", "WFC", "GS", "AXP", "MS", "SPGI", "BLK", "C", "SCHW", "PGR", "MMC", "CB", "FI",
    "ICE", "CME", "KKR", "BX", "PYPL", "AON", "MCO", "USB", "PNC", "AJG", "COF", "TFC", "APO", "TRV", "AFL", "BK",
    "AMP", "ALL", "MET", "AIG", "MSCI", "PRU", "ACGL", "HIG", "NDAQ", "FIS", "DFS", "WTW", "MTB", "STT", "BRO", "FITB",
    "TROW", "RJF", "GPN", "HBAN", "SYF", "CINF", "NTRS", "RF", "CFG", "CPAY", "WRB", "CBOE", "PFG", "KEY", "FDS", "L",
    "EG", "JKHY", "AIZ", "GL", "ERIE", "IVZ", "BEN", "MKTX",
    # Industrials
    "GE", "CAT", "RTX", "UNP", "HON", "ETN", "BA", "DE", "LMT", "ADP", "UPS", "GEV", "TT", "PH", "WM", "GD", "CTAS",
    "MMM", "ITW", "NOC", "TDG", "CSX", "EMR", "FDX", "CARR", "NSC", "PCAR", "URI", "JCI", "CPRT", "GWW", "PWR", "LHX",
    "CMI", "FAST", "PAYX", "AME", "ODFL", "VRSK", "IR", "RSG", "OTIS", "EFX", "DAL", "XYL", "WAB", "AXON", "HWM", "DOV",
    "ROK", "BR", "UAL", "FTV", "LDOS", "VLTO", "HUBB", "BLDR", "EXPD", "MAS", "J", "TXT", "IEX", "SNA", "LUV", "SWK",
    "PNR", "NDSN", "CHRW", "JBHT", "ALLE", "ROL", "DAY", "GNRC", "PAYC", "AOS", "HII",
    # Energy
    "XOM", "CVX", "COP", "EOG", "WMB", "SLB", "PSX", "MPC", "OKE", "KMI", "VLO", "HES", "OXY", "FANG", "BKR", "TRGP",
    "HAL", "DVN", "CTRA", "EQT", "APA",
    # Materials
    "LIN", "SHW", "APD", "ECL", "FCX", "NEM", "CTVA", "DD", "MLM", "VMC", "NUE", "DOW", "PPG", "IFF", "LYB", "BALL",
    "AVY", "PKG", "STLD", "CF", "IP", "AMCR", "ALB", "EMN", "CE", "MOS",
    # Real estate
    "PLD", "AMT", "EQIX", "WELL", "SPG", "DLR", "PSA", "O", "CCI", "CBRE", "EXR", "VICI", "IRM", "CSGP", "AVB", "VTR",
    "SBAC", "EQR", "WY", "INVH", "ESS", "MAA", "ARE", "KIM", "DOC", "UDR", "HST", "CPT", "REG", "BXP", "FRT",
    # Utilities
    "NEE", "SO", "DUK", "CEG", "SRE", "AEP", "VST", "D", "PCG", "PEG", "EXC", "XEL", "ED", "ETR", "WEC", "EIX", "DTE",
    "PPL", "AWK", "ES", "FE", "AEE", "CNP", "ATO", "CMS", "NRG", "NI", "LNT", "EVRG", "PNW", "AES",
]

_BUNDLED: dict[str, list[str]] = {"sp500": SP500, "nasdaq100": NASDAQ100, "dow30": DOW30}
UNIVERSE_LABELS: dict[str, str] = {"sp500": "标普 500", "nasdaq100": "纳斯达克 100", "dow30": "道琼斯 30"}

#: candidate pages per index, tried in order — Wikipedia moves constituent
#: tables between the index article and a "List of … companies" article.
_HEADERS = ("symbol", "ticker", "ticker symbol", "stock symbol")
_WIKI: dict[str, list[str]] = {
    "sp500": ["https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"],
    "nasdaq100": [
        "https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies",
        "https://en.wikipedia.org/wiki/List_of_Nasdaq-100_companies",
        "https://en.wikipedia.org/wiki/Nasdaq-100",
    ],
    "dow30": [
        "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
        "https://en.wikipedia.org/wiki/List_of_Dow_Jones_Industrial_Average_companies",
        "https://en.wikipedia.org/wiki/Historical_components_of_the_Dow_Jones_Industrial_Average",
    ],
}


def _dedupe(tickers: list[str]) -> list[str]:
    out: list[str] = []
    for t in tickers:
        t = t.strip().upper().replace(".", ".")
        if t and t not in out:
            out.append(t)
    return out


def load_universe(name: str) -> tuple[list[str], str]:
    """``(tickers, as_of)`` — the refreshed file when present, else the bundled snapshot."""
    if name not in _BUNDLED:
        raise KeyError(f"unknown universe {name!r}; known: {', '.join(_BUNDLED)}")
    if DATA_PATH.exists():
        try:
            data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
            entry = data.get(name)
            if entry and entry.get("tickers"):
                return _dedupe(list(entry["tickers"])), str(entry.get("as_of") or "")
        except (OSError, ValueError) as exc:
            logger.warning("universes.json unreadable (%s); using bundled snapshot", exc)
    return _dedupe(_BUNDLED[name]), BUNDLED_AS_OF


def universe_status() -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for name in _BUNDLED:
        tickers, as_of = load_universe(name)
        changes = load_changes(name)
        out[name] = {"size": len(tickers), "as_of": as_of, "label": UNIVERSE_LABELS[name],
                     "changes": len(changes), "history_from": changes[-1]["date"] if changes else None,
                     "membership": membership_mode(name), "date_added": len(load_date_added(name))}
    return out


# ------------------------------------------------------- point-in-time membership

def load_changes(name: str) -> list[dict[str, str | None]]:
    """Recorded additions / removals for ``name`` (newest first), or [] when never refreshed."""
    if not DATA_PATH.exists():
        return []
    try:
        entry = json.loads(DATA_PATH.read_text(encoding="utf-8")).get(name) or {}
    except (OSError, ValueError):
        return []
    rows = [c for c in (entry.get("changes") or []) if c.get("date")]
    return sorted(rows, key=lambda c: c["date"], reverse=True)


def load_date_added(name: str) -> dict[str, str]:
    """``ticker → date it joined`` for the current constituents (from the list's "Date added" column)."""
    if not DATA_PATH.exists():
        return {}
    try:
        entry = json.loads(DATA_PATH.read_text(encoding="utf-8")).get(name) or {}
    except (OSError, ValueError):
        return {}
    return {t: d for t, d in (entry.get("date_added") or {}).items() if d}


def membership_mode(name: str) -> str:
    """How well past membership can be reconstructed.

    ``"full"``      – an additions/removals history is stored: exact membership on any date.
    ``"additions"`` – only each current member's join date is known: names that joined after a
                      date are excluded (removes the look-ahead), but names removed since cannot
                      be restored (some survivorship remains).
    ``"none"``      – today's list only.
    """
    if load_changes(name):
        return "full"
    if load_date_added(name):
        return "additions"
    return "none"


def members_at(name: str, on: str | date) -> tuple[list[str], bool]:
    """``(tickers, point_in_time)`` — the constituents on ``on``.

    With a stored change history, today's list is rewound through every change
    dated after ``on`` (drop what was added, restore what was removed). With
    only join dates, members that joined after ``on`` are dropped. Otherwise
    today's list is returned with ``point_in_time`` False.
    """
    on_iso = on.isoformat() if isinstance(on, date) else str(on)[:10]
    current, as_of = load_universe(name)
    mode = membership_mode(name)
    if mode == "none":
        return current, False
    if as_of and on_iso >= as_of:
        return current, True
    if mode == "additions":
        joined = load_date_added(name)
        return [t for t in current if not joined.get(t) or joined[t] <= on_iso], True
    members = set(current)
    for c in load_changes(name):  # newest first
        if c["date"] <= on_iso:
            break
        if c.get("added"):
            members.discard(c["added"])
        if c.get("removed"):
            members.add(c["removed"])
    return sorted(members), True


def membership_lookup(name: str) -> Callable[[str], list[str]] | None:
    """A ``date → tickers`` function for strategies, or None when nothing about past membership is stored."""
    if membership_mode(name) == "none":
        return None
    return lambda on: members_at(name, on)[0]


# ----------------------------------------------------------------- refresh from Wikipedia

class _Tables(HTMLParser):
    """Collect every table on the page as rows of cell text (nesting-safe)."""

    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self._stack: list[list[list[str]]] = []
        self._saved: list[tuple[list[str] | None, list[str] | None]] = []  # outer row/cell while inside a nested table
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self._stack.append([])
            self._saved.append((self._row, self._cell))
            self._row, self._cell = None, None
        elif tag == "tr" and self._stack:
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append(re.sub(r"\s+", " ", "".join(self._cell)).strip())
            self._cell = None
        elif tag == "tr" and self._row is not None and self._stack:
            if self._row:
                self._stack[-1].append(self._row)
            self._row = None
        elif tag == "table" and self._stack:
            self.tables.append(self._stack.pop())
            self._row, self._cell = self._saved.pop() if self._saved else (None, None)

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


_TICKER_RE = re.compile(r"[A-Z][A-Z0-9.\-]{0,7}")


def _symbols_from_table(rows: list[list[str]], header_names: tuple[str, ...]) -> list[str]:
    head = " ".join(" ".join(r) for r in rows[:2]).lower()
    if "added" in head and "removed" in head:
        return []  # the additions/removals history, not a constituent list (see parse_changes)
    for h, header in enumerate(rows[:3]):  # header may follow a caption row
        cols = [c.strip().lower() for c in header]
        col = next((i for i, c in enumerate(cols) if c in header_names), None)
        if col is None:
            continue
        out = []
        for row in rows[h + 1:]:
            if len(row) > col:
                cell = row[col].strip()
                # "NYSE: MMM" / "NASDAQ: AAPL" → keep the symbol after the exchange prefix
                if ":" in cell:
                    cell = cell.rsplit(":", 1)[-1]
                cell = cell.replace("\u200b", "").replace(" ", "").strip()
                if _TICKER_RE.fullmatch(cell):
                    out.append(cell)
        return out
    return []


def parse_constituents(html: str, header_names: tuple[str, ...]) -> list[str]:
    """Symbols from whichever table has a matching header and the most valid tickers."""
    return [t for t, _ in parse_constituents_with_dates(html, header_names)]


def parse_constituents_with_dates(html: str, header_names: tuple[str, ...]) -> list[tuple[str, str | None]]:
    """``(symbol, date_added)`` pairs; the date comes from a "Date added" column when the table has one."""
    parser = _Tables()
    parser.feed(html)
    best: list[tuple[str, str | None]] = []
    for rows in parser.tables:
        found = _symbols_from_table(rows, header_names)
        if len(found) <= len(best):
            continue
        dates: dict[str, str | None] = {}
        for h, header in enumerate(rows[:3]):
            cols = [c.strip().lower() for c in header]
            sym_col = next((i for i, c in enumerate(cols) if c in header_names), None)
            date_col = next((i for i, c in enumerate(cols) if c.startswith("date added") or c.startswith("date first added")), None)
            if sym_col is None or date_col is None:
                continue
            for row in rows[h + 1:]:
                if len(row) > max(sym_col, date_col):
                    sym = _clean_symbol(row[sym_col])
                    if sym:
                        dates[sym] = _parse_date(row[date_col])
            break
        seen: set[str] = set()
        best = []
        for t in found:
            if t not in seen:
                seen.add(t)
                best.append((t, dates.get(t)))
    return best


_DATE_FORMATS = ("%B %d, %Y", "%b %d, %Y", "%Y-%m-%d", "%d %B %Y", "%d %b %Y", "%b. %d, %Y")
_DATE_IN_TEXT = re.compile(r"(\d{4}-\d{2}-\d{2}|[A-Z][a-z]+\.? \d{1,2}, \d{4}|\d{1,2} [A-Z][a-z]+ \d{4})")


def _parse_date(text: str) -> str | None:
    """ISO date from a table cell: tolerates footnotes, sort keys, odd spaces."""
    text = re.sub(r"\[.*?\]", "", text).replace("\u200b", "").replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip().rstrip(".")
    candidates = [text] + _DATE_IN_TEXT.findall(text)
    for cand in candidates:
        for fmt in _DATE_FORMATS:
            try:
                return datetime.strptime(cand, fmt).date().isoformat()
            except ValueError:
                continue
    return None


def _changes_tables(html: str) -> list[list[list[str]]]:
    """Every table whose first rows mention additions and removals."""
    parser = _Tables()
    parser.feed(html)
    out = []
    for rows in parser.tables:
        head = " ".join(" ".join(r) for r in rows[:3]).lower()
        if any(w in head for w in ("added", "addition")) and any(w in head for w in ("removed", "removal", "deleted", "deletion")):
            out.append(rows)
    return out


#: where the S&P 500 additions/removals table has lived; the list page itself dropped it in 2026
_CHANGES_URLS = [
    "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies_changes",
    "https://en.wikipedia.org/wiki/Changes_to_the_S%26P_500",
    "https://en.wikipedia.org/wiki/List_of_changes_to_the_S%26P_500",
    "https://en.wikipedia.org/wiki/S%26P_500_component_changes",
    "https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500",
]
_SEARCH_API = "https://en.wikipedia.org/w/api.php?action=query&list=search&format=json&srlimit=15&srsearch="


def discover_changes_pages(timeout: float = 30.0) -> list[str]:
    """Article URLs the search API returns for S&P 500 component changes (titles containing 'S&P 500')."""
    import urllib.parse

    urls: list[str] = []
    for query in ("S&P 500 components changes added removed", "S&P 500 list changes history"):
        try:
            payload = json.loads(_fetch(_SEARCH_API + urllib.parse.quote(query), timeout))
        except Exception as exc:  # noqa: BLE001
            logger.warning("wikipedia search failed: %s", exc)
            continue
        for hit in payload.get("query", {}).get("search", []):
            title = hit.get("title", "")
            if "s&p 500" in title.lower() or "s&p500" in title.lower():
                url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))
                if url not in urls:
                    urls.append(url)
    return urls


def fetch_changes(timeout: float = 30.0) -> tuple[list[dict[str, str | None]], str | None, list[str]]:
    """``(changes, source_url, attempts)`` — first candidate page whose table parses to ≥ 50 rows."""
    attempts: list[str] = []
    candidates = list(_CHANGES_URLS)
    for url in discover_changes_pages(timeout):
        if url not in candidates:
            candidates.append(url)
    for url in candidates:
        try:
            rows = parse_changes(_fetch(url, timeout))
        except Exception as exc:  # noqa: BLE001
            attempts.append(f"{url}: {type(exc).__name__}: {str(exc)[:60]}")
            continue
        if len(rows) >= 50:
            return rows, url, attempts
        attempts.append(f"{url}: {len(rows)} rows")
    return [], None, attempts


def _clean_symbol(cell: str) -> str | None:
    cell = cell.replace("\u200b", "").strip()
    if ":" in cell:
        cell = cell.rsplit(":", 1)[-1]
    cell = cell.replace(" ", "")
    return cell if cell and _TICKER_RE.fullmatch(cell) else None


def parse_changes(html: str) -> list[dict[str, str | None]]:
    """Additions / removals from the S&P 500 page's "Selected changes" table.

    The table has a two-row header (Effective Date | Added | Removed | Reason,
    then Ticker | Security | Ticker | Security). A date cell spanning several
    rows (rowspan) shows up here as rows with one cell fewer; those inherit the
    previous date. Either ticker may be empty. Newest first, like the page.
    """
    best: list[dict[str, str | None]] = []
    for rows in _changes_tables(html):
        out: list[dict[str, str | None]] = []
        last_date: str | None = None
        for row in rows:
            cells = [c.strip() for c in row]
            joined = " ".join(cells).lower()
            if not cells or ("added" in joined and "removed" in joined) or joined.startswith("ticker security"):
                continue  # header rows
            maybe = _parse_date(cells[0])
            if maybe:
                last_date, body = maybe, cells[1:]
            elif len(cells) in (4, 5) and last_date:
                body = cells  # rowspan continuation: date omitted
            else:
                continue
            if len(body) < 3:
                continue
            added = _clean_symbol(body[0])
            removed = _clean_symbol(body[2])
            if added or removed:
                out.append({"date": last_date, "added": added, "removed": removed,
                            "added_name": body[1] or None, "removed_name": body[3] if len(body) > 3 and body[3] else None})
        if len(out) > len(best):
            best = out
    return best


def describe_changes(html: str, limit: int = 8) -> list[str]:
    """Raw first rows of every candidate changes table — for ``--dump-changes`` debugging."""
    lines = []
    for i, rows in enumerate(_changes_tables(html)):
        lines.append(f"candidate table {i}: {len(rows)} rows, parsed {len(parse_changes(html)) if i == 0 else '-'}")
        for r in rows[:limit]:
            lines.append("   " + " | ".join(r))
    return lines or ["no table mentions both 'Added' and 'Removed' in its first rows"]


def describe_tables(html: str) -> list[str]:
    """One line per table: row count and header cells — for ``--dump`` debugging."""
    parser = _Tables()
    parser.feed(html)
    return [f"table {i}: {len(rows)} rows · header {rows[0][:8] if rows else []}" for i, rows in enumerate(parser.tables)]


def _fetch(url: str, timeout: float) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "ai-hedge-fund-altdata/2026 (+https://github.com/YuhanWang03/ai-hedge-fund-altdata)"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", "replace")


def refresh_from_wikipedia(names: list[str] | None = None, *, path: Path = DATA_PATH, timeout: float = 30.0) -> dict[str, object]:
    """Fetch current constituents and write ``data/universes.json``.

    Each index is independent: a page whose layout defeats the parser is
    reported under ``errors`` and its previous entry (or the bundled list)
    stays in force; the others are still written.
    """
    names = names or list(_WIKI)
    data: dict[str, dict[str, object]] = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            data = {}
    sizes: dict[str, int] = {}
    errors: dict[str, str] = {}
    minimum = {"sp500": 480, "nasdaq100": 90, "dow30": 28}
    for name in names:
        attempts: list[str] = []
        picked: tuple[list[str], str] | None = None
        dated: list[tuple[str, str | None]] = []
        for url in _WIKI[name]:
            try:
                dated = parse_constituents_with_dates(_fetch(url, timeout), _HEADERS)
                tickers = _dedupe([t for t, _ in dated])
            except Exception as exc:  # noqa: BLE001 — try the next candidate page
                attempts.append(f"{url}: {type(exc).__name__}: {str(exc)[:80]}")
                continue
            if len(tickers) >= minimum[name]:
                picked = (tickers, url)
                break
            attempts.append(f"{url}: parsed {len(tickers)} symbols")
        if picked is None:
            errors[name] = f"no candidate page yielded ≥ {minimum[name]} symbols — " + " | ".join(attempts) + f" — run --dump {name}"
            logger.warning("universe refresh %s failed: %s", name, errors[name])
            continue
        tickers, url = picked
        entry: dict[str, object] = {"tickers": tickers, "as_of": date.today().isoformat(), "source": url}
        date_added = {t: d for t, d in dated if d}
        if date_added:
            entry["date_added"] = date_added
            sizes[f"{name}_date_added"] = len(date_added)
        if name == "sp500":
            changes, source, attempts_c = fetch_changes(timeout)
            if changes:
                entry["changes"] = changes
                entry["changes_source"] = source
                sizes[f"{name}_changes"] = len(changes)
            else:
                previous = (data.get(name) or {}).get("changes")
                if previous:
                    entry["changes"] = previous  # keep what we had rather than losing history
                    entry["changes_source"] = (data.get(name) or {}).get("changes_source")
                errors[f"{name}_changes"] = ("no page yielded an additions/removals table — " + " | ".join(attempts_c)
                                             + (f" — kept previous {len(previous)}" if previous else " — membership falls back to join dates (additions only)"))
        data[name] = entry
        sizes[name] = len(tickers)
    if sizes:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"written": str(path) if sizes else None, "sizes": sizes, "errors": errors}


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m v2.screening.universes")
    parser.add_argument("--refresh", action="store_true", help="fetch current constituents from Wikipedia into data/universes.json")
    parser.add_argument("--show", choices=sorted(_BUNDLED), help="print the tickers that will be used for one universe")
    parser.add_argument("--dump", choices=sorted(_BUNDLED), help="print every table header found on the Wikipedia page (parser debugging)")
    parser.add_argument("--show-at", nargs=2, metavar=("UNIVERSE", "DATE"), help="print the constituents on a past date, e.g. --show-at sp500 2024-09-10")
    parser.add_argument("--dump-changes", action="store_true", help="print the raw first rows of the S&P 500 additions/removals table (parser debugging)")
    args = parser.parse_args(argv)
    if args.dump_changes:
        for url in _CHANGES_URLS + discover_changes_pages(30.0):
            print(f"== {url}")
            try:
                for line in describe_changes(_fetch(url, 30.0)):
                    print("  " + line)
            except Exception as exc:  # noqa: BLE001
                print(f"  fetch failed: {type(exc).__name__}: {exc}")
    if args.show_at:
        name, on = args.show_at
        tickers, pit = members_at(name, on)
        mode = membership_mode(name)
        label = {"full": "point-in-time (additions/removals history)", "additions": "additions-only (members that joined later are excluded; removed names cannot be restored)",
                 "none": "NO HISTORY STORED — this is today’s list; run --refresh"}[mode]
        print(f"{name} on {on} · {len(tickers)} tickers · {label}")
        print(" ".join(tickers))
    if args.dump:
        for url in _WIKI[args.dump]:
            print(f"== {url}")
            try:
                for line in describe_tables(_fetch(url, 30.0)):
                    print("  " + line)
            except Exception as exc:  # noqa: BLE001
                print(f"  fetch failed: {type(exc).__name__}: {exc}")
    if args.refresh:
        report = refresh_from_wikipedia()
        print(json.dumps(report, ensure_ascii=False, indent=1))
        if report["errors"]:
            return 1
    if args.show:
        tickers, as_of = load_universe(args.show)
        print(f"{args.show} · {len(tickers)} tickers · as of {as_of}")
        print(" ".join(tickers))
    if not (args.refresh or args.show or args.dump or args.show_at or args.dump_changes):
        print(json.dumps(universe_status(), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
