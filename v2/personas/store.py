"""SQLite persistence for committee runs, persona signals and snapshot cache.

Three tables, one file (``data/personas.db`` by default, next to the other
v2 databases):

* ``committee_runs`` — one row per run: who was asked, about what, the
  full result JSON.  This is what the Lab's run log reads.
* ``persona_signals`` — one row per (run, ticker, persona).  Flat columns
  for signal / confidence / score so a forward-return backfill and a
  per-persona hit-rate query are plain SQL later; ``fwd_*`` start NULL.
* ``snapshots`` — the fundamentals snapshot per (ticker, as_of), keyed by
  content hash, so a second run on the same day costs no API calls.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from v2.personas.snapshot import PersonaSnapshot

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = PROJECT_ROOT / "data" / "personas.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS committee_runs (
  id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL,
  as_of TEXT NOT NULL,
  source TEXT NOT NULL,
  tickers_json TEXT NOT NULL,
  personas_json TEXT NOT NULL,
  n_tickers INTEGER NOT NULL,
  elapsed_s REAL NOT NULL DEFAULT 0,
  result_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_committee_runs_created ON committee_runs(created_at DESC);
CREATE TABLE IF NOT EXISTS persona_signals (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL,
  created_at TEXT NOT NULL,
  as_of TEXT NOT NULL,
  ticker TEXT NOT NULL,
  persona TEXT NOT NULL,
  signal TEXT NOT NULL,
  confidence INTEGER NOT NULL,
  score REAL NOT NULL,
  max_score REAL NOT NULL,
  margin_of_safety REAL,
  abstained INTEGER NOT NULL DEFAULT 0,
  snapshot_hash TEXT,
  reasoning TEXT,
  facts_json TEXT,
  price_at REAL,
  fwd_1m REAL,
  fwd_3m REAL,
  UNIQUE(run_id, ticker, persona)
);
CREATE INDEX IF NOT EXISTS idx_persona_signals_ticker ON persona_signals(ticker, as_of DESC);
CREATE INDEX IF NOT EXISTS idx_persona_signals_persona ON persona_signals(persona, as_of DESC);
CREATE TABLE IF NOT EXISTS snapshots (
  ticker TEXT NOT NULL,
  as_of TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  fetched_at TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  PRIMARY KEY (ticker, as_of)
);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class PersonaStore:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else DEFAULT_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            yield conn
            conn.commit()
        finally:
            conn.close()

    # -- runs ---------------------------------------------------------------------

    def save_run(self, result: dict[str, Any], *, source: str, run_id: str | None = None) -> str:
        """Persist a ``CommitteeResult.to_dict()`` (plus any extras) and its signals."""
        run_id = run_id or uuid.uuid4().hex[:12]
        now = utc_now()
        verdicts = result.get("verdicts") or []
        tickers = [v["ticker"] for v in verdicts]
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO committee_runs (id, created_at, as_of, source, tickers_json, personas_json, n_tickers, elapsed_s, result_json) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    run_id, now, result.get("as_of", ""), source, json.dumps(tickers),
                    json.dumps(result.get("personas") or []), len(tickers), float(result.get("elapsed_s") or 0.0),
                    json.dumps({**result, "run_id": run_id, "source": source, "created_at": now}, ensure_ascii=False, default=str),
                ),
            )
            rows = []
            for v in verdicts:
                for s in v.get("signals") or []:
                    rows.append((
                        run_id, now, s.get("as_of", ""), v["ticker"], s["persona"], s["signal"], int(s["confidence"]),
                        float(s["score"]), float(s["max_score"]), s.get("margin_of_safety"), 1 if s.get("abstained") else 0,
                        s.get("snapshot_hash"), s.get("reasoning"), json.dumps(s.get("facts") or {}, ensure_ascii=False, default=str),
                        v.get("price"),
                    ))
            conn.executemany(
                "INSERT OR REPLACE INTO persona_signals (run_id, created_at, as_of, ticker, persona, signal, confidence, score, max_score, margin_of_safety, abstained, snapshot_hash, reasoning, facts_json, price_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                rows,
            )
        return run_id

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute("SELECT result_json FROM committee_runs WHERE id = ?", (run_id,)).fetchone()
        return json.loads(row["result_json"]) if row else None

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, created_at, as_of, source, tickers_json, personas_json, n_tickers, elapsed_s FROM committee_runs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "run_id": r["id"], "created_at": r["created_at"], "as_of": r["as_of"], "source": r["source"],
                "tickers": json.loads(r["tickers_json"]), "personas": json.loads(r["personas_json"]),
                "n_tickers": r["n_tickers"], "elapsed_s": r["elapsed_s"],
            }
            for r in rows
        ]

    def update_signal_narrative(self, run_id: str, ticker: str, persona: str, narrative: str, grounded: bool | None) -> bool:
        """Write an LLM narrative into a stored run's result JSON. Returns False if not found."""
        with self._conn() as conn:
            row = conn.execute("SELECT result_json FROM committee_runs WHERE id = ?", (run_id,)).fetchone()
            if not row:
                return False
            payload = json.loads(row["result_json"])
            hit = False
            for v in payload.get("verdicts") or []:
                if v.get("ticker") != ticker:
                    continue
                for sig in v.get("signals") or []:
                    if sig.get("persona") == persona:
                        sig["narrative"] = narrative
                        sig["narrative_grounded"] = grounded
                        hit = True
            if hit:
                conn.execute("UPDATE committee_runs SET result_json = ? WHERE id = ?", (json.dumps(payload, ensure_ascii=False, default=str), run_id))
            return hit

    # -- signals ------------------------------------------------------------------

    def latest_signals(self, ticker: str, limit: int = 13) -> list[dict[str, Any]]:
        """Most recent signal per persona for one ticker."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT persona, signal, confidence, score, max_score, margin_of_safety, abstained, as_of, run_id, reasoning
                   FROM persona_signals WHERE ticker = ? AND id IN (
                     SELECT MAX(id) FROM persona_signals WHERE ticker = ? GROUP BY persona
                   ) ORDER BY persona LIMIT ?""",
                (ticker.upper(), ticker.upper(), limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def signals_awaiting_forward_returns(self, *, older_than_days: int, column: str = "fwd_1m", today: date | None = None) -> list[dict[str, Any]]:
        """Signals old enough to be scored, whose ``column`` is still NULL."""
        if column not in ("fwd_1m", "fwd_3m"):
            raise ValueError("column must be fwd_1m or fwd_3m")
        base = today or datetime.now(timezone.utc).date()
        cutoff = (base - timedelta(days=older_than_days)).isoformat()
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT id, ticker, as_of, persona, signal, price_at FROM persona_signals WHERE {column} IS NULL AND abstained = 0 AND as_of <= ? ORDER BY as_of",
                (cutoff,),
            ).fetchall()
        return [dict(r) for r in rows]

    def set_forward_return(self, signal_id: int, *, column: str, value: float) -> None:
        if column not in ("fwd_1m", "fwd_3m"):
            raise ValueError("column must be fwd_1m or fwd_3m")
        with self._conn() as conn:
            conn.execute(f"UPDATE persona_signals SET {column} = ? WHERE id = ?", (value, signal_id))

    def persona_scoreboard(self) -> list[dict[str, Any]]:
        """Per-persona hit rates over scored, non-neutral votes, for both horizons.

        Besides the raw hit rate each row carries a 95 % Wilson interval on the
        1-month rate and the persona's *own* baseline: the hit rate its mix of
        bullish / bearish votes would have earned on the same tickers by chance
        (``share_bullish × P(up) + share_bearish × P(down)``). ``edge_1m`` is
        the rate minus that baseline — the number that means something.
        """
        base = self.scoreboard_baseline()
        up_1m, up_3m = base.get("up_rate_1m"), base.get("up_rate_3m")
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT persona,
                          SUM(CASE WHEN fwd_1m IS NOT NULL AND signal != 'neutral' THEN 1 ELSE 0 END) AS n,
                          SUM(CASE WHEN (signal = 'bullish' AND fwd_1m > 0) OR (signal = 'bearish' AND fwd_1m < 0) THEN 1 ELSE 0 END) AS hits,
                          SUM(CASE WHEN fwd_1m IS NOT NULL AND signal = 'bullish' THEN 1 ELSE 0 END) AS bull_1m,
                          AVG(CASE WHEN fwd_1m IS NULL THEN NULL WHEN signal = 'bullish' THEN fwd_1m WHEN signal = 'bearish' THEN -fwd_1m END) AS avg_directional_1m,
                          SUM(CASE WHEN fwd_3m IS NOT NULL AND signal != 'neutral' THEN 1 ELSE 0 END) AS n_3m,
                          SUM(CASE WHEN (signal = 'bullish' AND fwd_3m > 0) OR (signal = 'bearish' AND fwd_3m < 0) THEN 1 ELSE 0 END) AS hits_3m,
                          SUM(CASE WHEN fwd_3m IS NOT NULL AND signal = 'bullish' THEN 1 ELSE 0 END) AS bull_3m,
                          AVG(CASE WHEN fwd_3m IS NULL THEN NULL WHEN signal = 'bullish' THEN fwd_3m WHEN signal = 'bearish' THEN -fwd_3m END) AS avg_directional_3m,
                          SUM(CASE WHEN abstained = 0 AND signal = 'neutral' THEN 1 ELSE 0 END) AS neutral,
                          SUM(CASE WHEN abstained = 1 THEN 1 ELSE 0 END) AS abstained,
                          SUM(CASE WHEN abstained = 0 THEN 1 ELSE 0 END) AS votes
                   FROM persona_signals GROUP BY persona"""
            ).fetchall()
        out = []
        for r in rows:
            n, hits, n3, hits3 = int(r["n"] or 0), int(r["hits"] or 0), int(r["n_3m"] or 0), int(r["hits_3m"] or 0)
            if n == 0 and n3 == 0:
                continue  # nothing scored yet
            rate = hits / n if n else None
            rate3 = hits3 / n3 if n3 else None
            base_1m = _mix_baseline(int(r["bull_1m"] or 0), n, up_1m)
            base_3m = _mix_baseline(int(r["bull_3m"] or 0), n3, up_3m)
            lo, hi = _wilson(hits, n)
            out.append({
                "persona": r["persona"], "n": n, "hits": hits, "hit_rate": rate, "avg_directional_1m": r["avg_directional_1m"],
                "ci_low": lo, "ci_high": hi, "baseline_1m": base_1m, "edge_1m": (rate - base_1m) if rate is not None and base_1m is not None else None,
                "n_3m": n3, "hits_3m": hits3, "hit_rate_3m": rate3, "avg_directional_3m": r["avg_directional_3m"],
                "baseline_3m": base_3m, "edge_3m": (rate3 - base_3m) if rate3 is not None and base_3m is not None else None,
                "neutral": int(r["neutral"] or 0), "abstained": int(r["abstained"] or 0), "votes": int(r["votes"] or 0),
            })
        out.sort(key=lambda x: (-x["n"], -(x["hit_rate"] or 0)))
        return out

    def scoreboard_baseline(self) -> dict[str, Any]:
        """What the scored tickers did on their own: one observation per (ticker, as_of).

        ``up_rate_1m`` is the hit rate an always-bullish voter would have had;
        ``1 - up_rate_1m`` the always-bearish one. Forward returns are the same
        for every persona voting on a (ticker, as_of), hence the GROUP BY.
        """
        with self._conn() as conn:
            r1 = conn.execute(
                """SELECT COUNT(*) AS n, AVG(CASE WHEN f > 0 THEN 1.0 ELSE 0.0 END) AS up, AVG(f) AS avg
                   FROM (SELECT MAX(fwd_1m) AS f FROM persona_signals WHERE fwd_1m IS NOT NULL GROUP BY ticker, as_of)"""
            ).fetchone()
            r3 = conn.execute(
                """SELECT COUNT(*) AS n, AVG(CASE WHEN f > 0 THEN 1.0 ELSE 0.0 END) AS up, AVG(f) AS avg
                   FROM (SELECT MAX(fwd_3m) AS f FROM persona_signals WHERE fwd_3m IS NOT NULL GROUP BY ticker, as_of)"""
            ).fetchone()
        return {"n_1m": int(r1["n"] or 0), "up_rate_1m": r1["up"], "avg_return_1m": r1["avg"],
                "n_3m": int(r3["n"] or 0), "up_rate_3m": r3["up"], "avg_return_3m": r3["avg"]}

    def signal_counts(self) -> dict[str, int]:
        """Vote bookkeeping for the scoreboard: totals, scored, pending per horizon."""
        today = datetime.now(timezone.utc).date()
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM persona_signals WHERE abstained = 0").fetchone()[0]
            scored_1m = conn.execute("SELECT COUNT(*) FROM persona_signals WHERE fwd_1m IS NOT NULL").fetchone()[0]
            scored_3m = conn.execute("SELECT COUNT(*) FROM persona_signals WHERE fwd_3m IS NOT NULL").fetchone()[0]
            runs = conn.execute("SELECT COUNT(*) FROM committee_runs").fetchone()[0]
            tickers = conn.execute("SELECT COUNT(DISTINCT ticker) FROM persona_signals").fetchone()[0]
        return {
            "runs": runs, "tickers": tickers, "votes": total,
            "scored_1m": scored_1m, "scored_3m": scored_3m,
            "due_1m": len(self.signals_awaiting_forward_returns(older_than_days=30, column="fwd_1m", today=today)),
            "due_3m": len(self.signals_awaiting_forward_returns(older_than_days=91, column="fwd_3m", today=today)),
        }

    # -- snapshot cache -----------------------------------------------------------

    def cached_snapshot(self, ticker: str, as_of: str, *, max_age_hours: float = 24.0) -> PersonaSnapshot | None:
        with self._conn() as conn:
            row = conn.execute("SELECT fetched_at, payload_json FROM snapshots WHERE ticker = ? AND as_of = ?", (ticker.upper(), as_of)).fetchone()
        if not row:
            return None
        try:
            fetched = datetime.fromisoformat(row["fetched_at"])
        except ValueError:
            return None
        if fetched.tzinfo is None:
            fetched = fetched.replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) - fetched > timedelta(hours=max_age_hours):
            return None
        snap = PersonaSnapshot.from_dict(json.loads(row["payload_json"]))
        if any(g.startswith(("metrics_", "line_items_")) for g in snap.gaps):
            return None  # cached before the no-gaps rule existed; refetch
        return snap

    def save_snapshot(self, snap: PersonaSnapshot) -> None:
        if not snap.has_fundamentals or any(g.startswith(("metrics_", "line_items_")) for g in snap.gaps):
            return  # never cache a fetch whose core inputs failed; the next run should retry
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO snapshots (ticker, as_of, content_hash, fetched_at, payload_json) VALUES (?,?,?,?,?)",
                (snap.ticker, snap.as_of, snap.content_hash, snap.fetched_at or utc_now(), json.dumps(snap.to_dict(), default=str)),
            )


def _wilson(hits: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    """95 % Wilson score interval for a proportion; (None, None) without observations."""
    if n <= 0:
        return None, None
    p = hits / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def _mix_baseline(bullish: int, n: int, up_rate: float | None) -> float | None:
    """Hit rate a persona's bullish/bearish mix earns by chance given how often tickers rose."""
    if n <= 0 or up_rate is None:
        return None
    return (bullish * up_rate + (n - bullish) * (1 - up_rate)) / n
