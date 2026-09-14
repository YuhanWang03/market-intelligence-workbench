"""Short-lived conversation state — enough to resolve "它" and "那只".

The bot is stateless per message: ``intent.classify`` sees one sentence, so
"NVDA 怎么样" followed by "那它财报呢" leaves the second question with no
subject and lands on ``unknown``.

Resolution happens **before** classification, which is the point: rewriting
"那它财报呢" to "NVDA 财报呢" lets the *fast* path answer it. Conversation memory
is not an agent feature here — it removes work from the agent by making more
queries answerable in one hop.

Scope is deliberately small. This is a single-user bot (chat-ID filtered), so
state lives in memory and is lost on restart; that costs one repeated question
after a deploy. Persisting it would mean a new table in ``state.db`` and a
migration for something with a 30-minute useful life.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field

#: Pronouns and demonstratives that need an antecedent.
#: Leading 这/那 is absorbed so "那它财报呢" becomes "NVDA 财报呢" rather than
#: "那NVDA财报呢" — the classifier tolerates the latter, but the rewritten text
#: is shown to the user, so it needs to read naturally.
#: The measure word must not be followed by a time unit. 「这个月」 is not a
#: demonstrative pointing at a ticker, and substituting one destroys the
#: question: 「我这个月比上个月表现好还是差？」 came back as 「我NVDA月比上个月
#: 表现好还是差？」 and the agent dutifully answered about NVDA. Same class of
#: bug as 最近／最新 in the router's superlative signal — a substring that looks
#: like the thing but is part of a time expression.
_TIME_UNIT = "月周年天日季星礼小分"

_PRONOUN = re.compile(
    rf"[这那]?(?:它|他|她)|[这那][只支家个](?![{_TIME_UNIT}])|\bit\b|\bthey\b",
    re.IGNORECASE,
)

#: A whole message that is nothing but a follow-up question — no subject at all,
#: not even a pronoun to substitute. "我的持仓最近整体在跌还是涨？" then "为什么？"
#: is the natural way to ask, and it arrived on the first live session: the
#: classifier saw a contextless "为什么？", returned unknown, and the agent spent
#: a full multi-step budget concluding it needed to ask what the user meant.
#:
#: The whole message must match, so "为什么 NVDA 涨" is untouched — this fires
#: only when there is genuinely nothing else in the message to go on.
_ELLIPTICAL = re.compile(
    r"^(?:那|所以|然后|但)?\s*(?:这是?)?\s*"
    r"(?:为什么|为啥|为何|怎么会|怎么回事|什么原因|原因呢|原因是什么|why)"
    r"[\s?？。.!！~～]*$",
    re.IGNORECASE,
)

_TICKER_LIKE = re.compile(r"\b[A-Z]{2,5}\b")
_NOT_TICKERS = frozenset({
    "AI", "US", "USD", "CEO", "CFO", "SEC", "ETF", "IPO", "EPS", "GDP", "CPI",
    "PCE", "NFP", "PPI", "FOMC", "FED", "RSI", "CMF", "OK", "VS", "PE", "ROE",
})

DEFAULT_TTL_SECONDS = 1800.0
DEFAULT_MAX_TURNS = 6


def extract_tickers(text: str) -> list[str]:
    seen: list[str] = []
    for token in _TICKER_LIKE.findall(text or ""):
        if token not in _NOT_TICKERS and token not in seen:
            seen.append(token)
    return seen


@dataclass(frozen=True)
class Turn:
    """One exchange, reduced to what a later turn might need to refer back to."""

    query: str
    tickers: tuple[str, ...] = ()
    tools_used: tuple[str, ...] = ()
    answer_digest: str = ""
    path: str = ""
    ts: float = field(default_factory=time.time)


@dataclass(frozen=True)
class Resolution:
    """Outcome of pronoun resolution — kept explicit so the UI can disclose it."""

    text: str
    rewritten: bool = False
    antecedent: str = ""
    note: str = ""


class SessionStore:
    """Per-chat ring buffer of recent turns, with TTL expiry.

    Thread-safe because the bot resolves and records from an executor thread
    while the polling loop keeps running.
    """

    def __init__(self, ttl_seconds: float = DEFAULT_TTL_SECONDS,
                 max_turns: int = DEFAULT_MAX_TURNS) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_turns = max_turns
        self._turns: dict[int, list[Turn]] = {}
        self._lock = threading.Lock()

    def recent(self, chat_id: int, n: int = 3) -> list[Turn]:
        cutoff = time.time() - self.ttl_seconds
        with self._lock:
            turns = [t for t in self._turns.get(chat_id, []) if t.ts >= cutoff]
            self._turns[chat_id] = turns
        return turns[-n:]

    def record(self, chat_id: int, turn: Turn) -> None:
        with self._lock:
            turns = self._turns.setdefault(chat_id, [])
            turns.append(turn)
            del turns[:-self.max_turns]

    def clear(self, chat_id: int | None = None) -> None:
        with self._lock:
            if chat_id is None:
                self._turns.clear()
            else:
                self._turns.pop(chat_id, None)

    def last_ticker(self, chat_id: int) -> str:
        for turn in reversed(self.recent(chat_id, n=self.max_turns)):
            if turn.tickers:
                return turn.tickers[0]
        return ""

    def last_query(self, chat_id: int) -> str:
        """The previous question, for a follow-up that carries no subject."""
        turns = self.recent(chat_id, n=self.max_turns)
        return turns[-1].query if turns else ""

    def resolve(self, chat_id: int, text: str) -> Resolution:
        """Substitute a pronoun with the most recently discussed ticker.

        Three guards, each preventing a way this could make things worse:
        no pronoun -> untouched; the message already names a ticker -> untouched
        (the pronoun refers to something else); nothing to refer back to ->
        untouched, and the query fails the same way it does today.
        """
        raw = (text or "").strip()
        if not raw:
            return Resolution(raw)

        # A bare follow-up has no pronoun to substitute — the *question* is what
        # is missing, so the previous one is restored in front of it.
        if _ELLIPTICAL.match(raw):
            previous = self.last_query(chat_id)
            if not previous or _ELLIPTICAL.match(previous):
                return Resolution(raw)
            rewritten = f"{previous.rstrip('？?。. ')}——{raw}"
            return Resolution(
                text=rewritten,
                rewritten=True,
                antecedent=previous,
                note=f"「{raw}」按上文补全为「{rewritten}」",
            )

        if not _PRONOUN.search(raw):
            return Resolution(raw)
        if extract_tickers(raw):
            return Resolution(raw)

        antecedent = self.last_ticker(chat_id)
        if not antecedent:
            return Resolution(raw)

        rewritten = _PRONOUN.sub(antecedent, raw, count=1)
        return Resolution(
            text=rewritten,
            rewritten=True,
            antecedent=antecedent,
            note=f"「{raw}」按上文补全为「{rewritten}」",
        )


#: Process-wide default. The bot has one chat; tests construct their own.
STORE = SessionStore()
