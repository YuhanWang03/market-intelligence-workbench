"""Data shapes shared by every persona.

Everything here is a plain dataclass or a thin attribute-access view so the
persona modules stay free of framework imports.  A persona receives a
:class:`~v2.personas.snapshot.PersonaSnapshot`, returns a
:class:`PersonaSignal`, and never touches the network itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal, Mapping

Signal = Literal["bullish", "bearish", "neutral"]


class Record:
    """Attribute-access view over a row of API data; unknown fields read as ``None``.

    The upstream agents were written against pydantic models where a missing
    line item raised ``AttributeError`` while a present-but-null one read as
    ``None``.  Every scoring rule already guards with ``is not None``, so
    collapsing both cases to ``None`` lets the same rules run over whatever
    subset of fields a data provider actually returns.
    """

    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, Any] | None = None, **extra: Any) -> None:
        merged = dict(data or {})
        merged.update(extra)
        object.__setattr__(self, "_data", merged)

    def __getattr__(self, name: str) -> Any:
        # Only reached when normal attribute lookup fails (i.e. not a slot).
        if name.startswith("__"):
            raise AttributeError(name)
        return self._data.get(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self._data[name] = value

    def __getitem__(self, name: str) -> Any:
        return self._data[name]

    def __contains__(self, name: str) -> bool:
        return name in self._data

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Record) and other._data == self._data

    def __repr__(self) -> str:
        head = ", ".join(f"{k}={v!r}" for k, v in list(self._data.items())[:4])
        more = "" if len(self._data) <= 4 else f", +{len(self._data) - 4}"
        return f"Record({head}{more})"

    def get(self, name: str, default: Any = None) -> Any:
        return self._data.get(name, default)

    def to_dict(self) -> dict[str, Any]:
        return dict(self._data)

    # pydantic-compatible spelling, used by a couple of upstream helpers.
    def model_dump(self) -> dict[str, Any]:
        return self.to_dict()


def as_records(rows: Any) -> list[Record]:
    """Coerce a list of dicts / pydantic models / dataclasses into Records."""
    out: list[Record] = []
    for row in rows or []:
        if isinstance(row, Record):
            out.append(row)
        elif isinstance(row, Mapping):
            out.append(Record(row))
        elif hasattr(row, "model_dump"):
            out.append(Record(row.model_dump()))
        elif hasattr(row, "__dataclass_fields__"):
            out.append(Record({k: getattr(row, k) for k in row.__dataclass_fields__}))
        elif hasattr(row, "__dict__"):
            out.append(Record({k: v for k, v in vars(row).items() if not k.startswith("_")}))
        else:
            raise TypeError(f"cannot coerce {type(row).__name__} into Record")
    return out


@dataclass
class SubScore:
    """One dimension of a persona's checklist (e.g. Buffett's moat)."""

    name: str
    score: float
    max_score: float
    details: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "score": self.score,
            "max_score": self.max_score,
            "details": self.details,
            "data": _jsonable(self.data),
        }


@dataclass
class Evaluation:
    """Deterministic output of a persona's rules, before any verdict is drawn."""

    parts: list[SubScore] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)
    margin_of_safety: float | None = None
    insufficient: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        return float(sum(p.score for p in self.parts))

    @property
    def max_score(self) -> float:
        return float(sum(p.max_score for p in self.parts))

    @property
    def ratio(self) -> float:
        return self.score / self.max_score if self.max_score > 0 else 0.0

    def part(self, name: str) -> SubScore | None:
        for p in self.parts:
            if p.name == name:
                return p
        return None


@dataclass
class PersonaSignal:
    """What one simulated investor says about one ticker on one date."""

    persona: str
    ticker: str
    as_of: str
    signal: Signal
    confidence: int
    score: float
    max_score: float
    parts: list[SubScore] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)
    margin_of_safety: float | None = None
    reasoning: str = ""
    narrative: str | None = None
    narrative_grounded: bool | None = None
    abstained: bool = False
    data_gaps: list[str] = field(default_factory=list)
    snapshot_hash: str = ""

    @property
    def ratio(self) -> float:
        return self.score / self.max_score if self.max_score > 0 else 0.0

    @property
    def direction(self) -> int:
        return {"bullish": 1, "bearish": -1}.get(self.signal, 0)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PersonaSignal":
        parts = [
            SubScore(name=p["name"], score=float(p["score"]), max_score=float(p["max_score"]), details=p.get("details", ""), data=dict(p.get("data") or {}))
            for p in data.get("parts") or []
        ]
        return cls(
            persona=data["persona"], ticker=data["ticker"], as_of=data.get("as_of", ""),
            signal=data.get("signal", "neutral"), confidence=int(data.get("confidence", 0)),
            score=float(data.get("score", 0.0)), max_score=float(data.get("max_score", 0.0)),
            parts=parts, facts=dict(data.get("facts") or {}), margin_of_safety=data.get("margin_of_safety"),
            reasoning=data.get("reasoning", ""), narrative=data.get("narrative"),
            narrative_grounded=data.get("narrative_grounded"), abstained=bool(data.get("abstained", False)),
            data_gaps=list(data.get("data_gaps") or []), snapshot_hash=data.get("snapshot_hash", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "persona": self.persona,
            "ticker": self.ticker,
            "as_of": self.as_of,
            "signal": self.signal,
            "confidence": self.confidence,
            "score": self.score,
            "max_score": self.max_score,
            "ratio": round(self.ratio, 4),
            "parts": [p.to_dict() for p in self.parts],
            "facts": _jsonable(self.facts),
            "margin_of_safety": self.margin_of_safety,
            "reasoning": self.reasoning,
            "narrative": self.narrative,
            "narrative_grounded": self.narrative_grounded,
            "abstained": self.abstained,
            "data_gaps": list(self.data_gaps),
            "snapshot_hash": self.snapshot_hash,
        }


def _jsonable(value: Any) -> Any:
    """Round-trip through JSON so numpy scalars and Records serialize cleanly."""
    return json.loads(json.dumps(value, default=_default))


def _default(value: Any) -> Any:
    if isinstance(value, Record):
        return value.to_dict()
    if hasattr(value, "item"):  # numpy scalar
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            pass
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    return str(value)
