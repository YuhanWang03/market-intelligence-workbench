"""Entity (ticker) extraction shared by routing and the eval fixtures.

Three recognisers run over the request text, in order of decreasing precision:

1. Chinese company aliases (``英伟达`` → ``NVDA``).
2. Known symbols from the project universe, matched case-insensitively so
   ``nvda`` and single-letter tickers such as ``V`` or ``T`` resolve.  Lowercase
   matches are only trusted inside CJK text, where a bare English token is a
   symbol rather than a word.
3. Bare uppercase tokens of 2–8 letters, minus common finance acronyms.

Results keep the order of first appearance and are de-duplicated.
"""

from __future__ import annotations

import re

_CJK = re.compile(r"[一-鿿]")
_UPPER_TICKER = re.compile(r"(?<![A-Za-z0-9])[A-Z]{2,8}(?![A-Za-z0-9])")
_KNOWN_TICKER = re.compile(r"(?<![A-Za-z0-9.])[A-Za-z]{1,5}(?:\.[A-Za-z])?(?![A-Za-z0-9])")

NOT_TICKERS = frozenset(
    {
        "AI",
        "API",
        "CEO",
        "CFO",
        "CPI",
        "ETF",
        "EPS",
        "FD",
        "FOMC",
        "GDP",
        "LLM",
        "NFP",
        "PCE",
        "PPI",
        "ROE",
        "ROIC",
        "RSI",
        "CMF",
        "SEC",
        "USD",
        "VS",
        "PEAD",
        "TAM",
    }
)

# Chinese aliases for names the bot prompt already teaches the model to map.
# Longer aliases are listed before their prefixes so ``阿里巴巴`` wins over ``阿里``.
CHINESE_ALIASES: tuple[tuple[str, str], ...] = (
    ("英伟达", "NVDA"),
    ("苹果公司", "AAPL"),
    ("苹果", "AAPL"),
    ("微软", "MSFT"),
    ("谷歌", "GOOGL"),
    ("特斯拉", "TSLA"),
    ("亚马逊", "AMZN"),
    ("超微半导体", "AMD"),
    ("英特尔", "INTC"),
    ("博通", "AVGO"),
    ("甲骨文", "ORCL"),
    ("奈飞", "NFLX"),
    ("网飞", "NFLX"),
    ("高通", "QCOM"),
    ("美光", "MU"),
    ("阿里巴巴", "BABA"),
    ("阿里", "BABA"),
    ("拼多多", "PDD"),
    ("京东", "JD"),
    ("百度", "BIDU"),
    ("蔚来", "NIO"),
    ("腾讯", "TCEHY"),
    ("迪士尼", "DIS"),
    ("可口可乐", "KO"),
    ("沃尔玛", "WMT"),
    ("波音", "BA"),
    ("摩根大通", "JPM"),
    ("高盛", "GS"),
    ("伯克希尔", "BRK.B"),
    ("礼来", "LLY"),
    ("辉瑞", "PFE"),
    ("强生", "JNJ"),
    ("埃克森美孚", "XOM"),
    ("台积电", "TSM"),
    ("超微电脑", "SMCI"),
    ("赛富时", "CRM"),
    ("优步", "UBER"),
)


def _known_symbols() -> frozenset[str]:
    try:
        from v2.universe import BROAD_MARKET_ETFS, SECTOR_ETFS, TICKER_TO_SECTOR
    except Exception:  # noqa: BLE001 — the universe module is optional for the core
        return frozenset(ticker for _, ticker in CHINESE_ALIASES)
    return frozenset({*TICKER_TO_SECTOR, *BROAD_MARKET_ETFS, *SECTOR_ETFS, *(ticker for _, ticker in CHINESE_ALIASES)})


KNOWN_SYMBOLS = _known_symbols()


def extract_entities(text: str) -> tuple[str, ...]:
    """Return tickers mentioned in ``text`` in order of first appearance."""

    source = text or ""
    found: list[tuple[int, str]] = []
    claimed: list[tuple[int, int]] = []

    def claim(start: int, end: int, ticker: str) -> None:
        if any(start < b and end > a for a, b in claimed):
            return
        claimed.append((start, end))
        found.append((start, ticker))

    for alias, ticker in CHINESE_ALIASES:
        for match in re.finditer(re.escape(alias), source):
            claim(match.start(), match.end(), ticker)

    cjk_text = bool(_CJK.search(source))
    for match in _KNOWN_TICKER.finditer(source):
        token = match.group(0)
        upper = token.upper()
        if upper not in KNOWN_SYMBOLS or upper in NOT_TICKERS:
            continue
        if token != upper and (not cjk_text or len(token) == 1):
            continue
        claim(match.start(), match.end(), upper)

    for match in _UPPER_TICKER.finditer(source):
        token = match.group(0)
        if token in NOT_TICKERS:
            continue
        claim(match.start(), match.end(), token)

    ordered: list[str] = []
    for _, ticker in sorted(found):
        if ticker not in ordered:
            ordered.append(ticker)
    return tuple(ordered)
