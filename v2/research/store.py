"""SQLite persistence for research runs, snapshots, caches and relationships."""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = PROJECT_ROOT / "data" / "research_cache.db"
ENGINE_VERSION = "research-v1.0"
FEATURE_FREEZE = True


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


SCHEMA = """
CREATE TABLE IF NOT EXISTS research_runs (
  id TEXT PRIMARY KEY, ticker TEXT NOT NULL, status TEXT NOT NULL,
  trigger_type TEXT NOT NULL, requested_modules TEXT NOT NULL,
  started_at TEXT, completed_at TEXT, created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL, engine_version TEXT NOT NULL, error_summary TEXT
);
CREATE INDEX IF NOT EXISTS idx_research_job_ticker ON research_runs(ticker, created_at DESC);
CREATE TABLE IF NOT EXISTS research_module_runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, ticker TEXT NOT NULL,
  module TEXT NOT NULL, status TEXT NOT NULL, started_at TEXT, completed_at TEXT,
  duration_ms INTEGER, error_type TEXT, error_message TEXT, missing_fields TEXT,
  source_count INTEGER DEFAULT 0, verified_source_count INTEGER DEFAULT 0,
  confidence REAL DEFAULT 0, completeness REAL DEFAULT 0,
  cache_hit INTEGER DEFAULT 0, cache_expires_at TEXT, source_freshness TEXT,
  result_json TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
  UNIQUE(run_id, module)
);
CREATE INDEX IF NOT EXISTS idx_module_run ON research_module_runs(run_id, module);
CREATE TABLE IF NOT EXISTS research_snapshots (
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT UNIQUE NOT NULL,
  ticker TEXT NOT NULL, generated_at TEXT NOT NULL, engine_version TEXT NOT NULL,
  overall_status TEXT NOT NULL, result_json TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_snapshot_ticker ON research_snapshots(ticker, generated_at DESC);
CREATE TABLE IF NOT EXISTS latest_research_results (
  ticker TEXT PRIMARY KEY, snapshot_id INTEGER NOT NULL, result_json TEXT NOT NULL,
  generated_at TEXT NOT NULL, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS research_module_cache (
  cache_key TEXT PRIMARY KEY, ticker TEXT NOT NULL, module TEXT NOT NULL,
  period TEXT NOT NULL, engine_version TEXT NOT NULL, result_json TEXT NOT NULL,
  generated_at REAL NOT NULL, expires_at REAL NOT NULL, source_freshness TEXT
);
CREATE INDEX IF NOT EXISTS idx_module_cache_expiry ON research_module_cache(expires_at);
CREATE TABLE IF NOT EXISTS research_data_cache (
  cache_key TEXT PRIMARY KEY, ticker TEXT NOT NULL, data_type TEXT NOT NULL,
  period TEXT NOT NULL, version TEXT NOT NULL, payload_json TEXT NOT NULL,
  generated_at REAL NOT NULL, expires_at REAL NOT NULL, diagnostics_json TEXT
);
CREATE TABLE IF NOT EXISTS company_relationships (
  id INTEGER PRIMARY KEY AUTOINCREMENT, source_ticker TEXT NOT NULL,
  target_ticker TEXT NOT NULL, target_name TEXT, relationship_type TEXT NOT NULL,
  status TEXT NOT NULL, confidence REAL NOT NULL, description TEXT,
  first_seen_at TEXT NOT NULL, last_seen_at TEXT NOT NULL, last_verified_at TEXT,
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
  UNIQUE(source_ticker, target_ticker, relationship_type)
);
CREATE INDEX IF NOT EXISTS idx_relationship_source ON company_relationships(source_ticker, status);
CREATE TABLE IF NOT EXISTS relationship_sources (
  id INTEGER PRIMARY KEY AUTOINCREMENT, relationship_id INTEGER NOT NULL,
  provider TEXT, title TEXT, url TEXT, published_at TEXT, fetched_at TEXT,
  evidence_text TEXT, evidence_summary TEXT,
  UNIQUE(relationship_id, url), FOREIGN KEY(relationship_id) REFERENCES company_relationships(id)
);
CREATE TABLE IF NOT EXISTS research_peer_preferences (
  ticker TEXT NOT NULL, peer_ticker TEXT NOT NULL, action TEXT NOT NULL,
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
  PRIMARY KEY(ticker, peer_ticker)
);
"""


class ResearchStore:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else DEFAULT_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            legacy = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='research_runs'").fetchone()
            if legacy:
                columns = {row[1] for row in conn.execute("PRAGMA table_info(research_runs)").fetchall()}
                if "id" not in columns:
                    if not conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='legacy_research_runs'").fetchone():
                        conn.execute("ALTER TABLE research_runs RENAME TO legacy_research_runs")
                    else:
                        conn.execute("ALTER TABLE research_runs RENAME TO legacy_research_runs_2")
            conn.executescript(SCHEMA)
            self._migrate_legacy_results(conn)

    def _migrate_legacy_results(self, conn: sqlite3.Connection) -> None:
        """Import the former ticker-level cache once without deleting it."""
        if not conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='legacy_research_runs'").fetchone():
            return
        for row in conn.execute("SELECT ticker,payload_json,generated_at FROM legacy_research_runs").fetchall():
            payload = _loads(row["payload_json"], {})
            ticker = str(payload.get("ticker") or str(row["ticker"]).split(":")[-1]).upper()
            if not ticker or not payload:
                continue
            generated = payload.get("generated_at") or datetime.fromtimestamp(float(row["generated_at"]), timezone.utc).isoformat()
            run_id = payload.get("run_id") or f"legacy-{ticker}-{int(float(row['generated_at']))}"
            payload.setdefault("run_id", run_id)
            payload.setdefault("ticker", ticker)
            payload.setdefault("engine_version", "research-v2")
            payload.setdefault("status", "COMPLETED")
            conn.execute(
                """INSERT OR IGNORE INTO research_snapshots
                (run_id,ticker,generated_at,engine_version,overall_status,result_json,created_at) VALUES (?,?,?,?,?,?,?)""",
                (run_id, ticker, generated, payload["engine_version"], payload["status"], _json(payload), utc_now()),
            )
            snapshot = conn.execute("SELECT id FROM research_snapshots WHERE run_id=?", (run_id,)).fetchone()
            if snapshot:
                conn.execute(
                    """INSERT OR IGNORE INTO latest_research_results
                    (ticker,snapshot_id,result_json,generated_at,updated_at) VALUES (?,?,?,?,?)""",
                    (ticker, snapshot["id"], _json(payload), generated, utc_now()),
                )
            type_map = {"supplier": "SUPPLIER", "customer": "CUSTOMER", "smaller_peer": "COMPETITOR", "beneficiary": "BENEFICIARY", "partner": "PARTNER", "platform": "PLATFORM", "substitute": "SUBSTITUTE"}
            relationships = payload.get("modules", {}).get("supply_chain", {}).get("details", {}).get("relationships", [])
            for relation in relationships:
                target = relation.get("target_company") or relation.get("target_ticker")
                if not target:
                    continue
                rel_type = type_map.get(str(relation.get("relationship_type", "")).lower(), "PARTNER")
                verified = bool(relation.get("verified"))
                conn.execute(
                    """INSERT OR IGNORE INTO company_relationships
                    (source_ticker,target_ticker,target_name,relationship_type,status,confidence,description,first_seen_at,last_seen_at,last_verified_at,created_at,updated_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (ticker, str(target).upper(), relation.get("target_name"), rel_type, "VERIFIED" if verified else "DISCOVERED", float(relation.get("confidence") or (.9 if verified else .45)), relation.get("description"), generated, generated, generated if verified else None, generated, generated),
                )
                relation_id = conn.execute("SELECT id FROM company_relationships WHERE source_ticker=? AND target_ticker=? AND relationship_type=?", (ticker, str(target).upper(), rel_type)).fetchone()["id"]
                if relation.get("source"):
                    conn.execute("INSERT OR IGNORE INTO relationship_sources (relationship_id,provider,url,fetched_at,evidence_summary) VALUES (?,?,?,?,?)", (relation_id, "Tavily", relation["source"], generated, relation.get("description")))

    def recover_incomplete_runs(self) -> int:
        """Mark jobs left active by a previous backend process as failed."""
        now = utc_now()
        with self._conn() as conn:
            cursor = conn.execute(
                """UPDATE research_runs SET status='FAILED', completed_at=?, updated_at=?,
                   error_summary=COALESCE(error_summary, 'Backend restarted while run was active')
                   WHERE status IN ('PENDING','RUNNING')""",
                (now, now),
            )
            conn.execute(
                """UPDATE research_module_runs SET status='FAILED', completed_at=?, updated_at=?,
                   error_type='BACKEND_RESTART', error_message='Backend restarted while module was active'
                   WHERE status IN ('PENDING','RUNNING')""", (now, now),
            )
            return cursor.rowcount

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self.path), timeout=20.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def create_run(self, run_id: str, ticker: str, modules: list[str], trigger_type: str) -> dict:
        now = utc_now()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO research_runs VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (run_id, ticker, "PENDING", trigger_type, _json(modules), None, None, now, now, ENGINE_VERSION, None),
            )
            for module in modules:
                conn.execute(
                    """INSERT INTO research_module_runs
                    (run_id,ticker,module,status,created_at,updated_at) VALUES (?,?,?,?,?,?)""",
                    (run_id, ticker, module, "PENDING", now, now),
                )
        return self.get_run(run_id) or {}

    def update_run(self, run_id: str, status: str, *, error: str | None = None) -> None:
        now = utc_now()
        started = now if status == "RUNNING" else None
        completed = now if status in {"PARTIAL_DATA", "PARTIAL_ERROR", "COMPLETED", "FAILED"} else None
        with self._conn() as conn:
            conn.execute(
                """UPDATE research_runs SET status=?, updated_at=?,
                started_at=COALESCE(started_at,?), completed_at=COALESCE(?,completed_at), error_summary=? WHERE id=?""",
                (status, now, started, completed, error, run_id),
            )

    def update_module(self, run_id: str, module: str, status: str, result: dict | None = None) -> None:
        now = utc_now()
        result = result or {}
        started = now if status == "RUNNING" else None
        completed = now if status not in {"PENDING", "RUNNING"} else None
        error = result.get("error") or ((result.get("errors") or [None])[0])
        provider_error = ((result.get("provider_errors") or [None])[0])
        recorded_error_type = (
            provider_error.get("type") if isinstance(provider_error, dict)
            else type(error).__name__ if error and not isinstance(error, str)
            else "EXECUTION_ERROR" if error
            else None
        )
        recorded_error_message = (
            provider_error.get("message") if isinstance(provider_error, dict)
            else str(error) if error
            else None
        )
        with self._conn() as conn:
            conn.execute(
                """UPDATE research_module_runs SET status=?, updated_at=?,
                started_at=COALESCE(started_at,?), completed_at=COALESCE(?,completed_at),
                duration_ms=CASE WHEN ? IS NOT NULL AND started_at IS NOT NULL
                  THEN CAST((julianday(?) - julianday(started_at))*86400000 AS INTEGER) ELSE duration_ms END,
                error_type=?, error_message=?, missing_fields=?, source_count=?, verified_source_count=?,
                confidence=?, completeness=?, cache_hit=?, cache_expires_at=?, source_freshness=?, result_json=?
                WHERE run_id=? AND module=?""",
                (status, now, started, completed, completed, now,
                 recorded_error_type, recorded_error_message, _json(result.get("missing_fields", [])),
                 int(result.get("source_count", 0) or 0), int(result.get("verified_source_count", 0) or 0),
                 float(result.get("confidence", 0) or 0), float(result.get("completeness", 0) or 0),
                 int(bool(result.get("cache_hit"))), result.get("cache_expires_at"),
                 _json(result.get("source_freshness", {})), _json(result) if result else None, run_id, module),
            )

    def get_run(self, run_id: str) -> dict | None:
        with self._conn() as conn:
            run = conn.execute("SELECT * FROM research_runs WHERE id=?", (run_id,)).fetchone()
            if not run:
                return None
            rows = conn.execute("SELECT * FROM research_module_runs WHERE run_id=? ORDER BY id", (run_id,)).fetchall()
            snapshot = conn.execute("SELECT result_json FROM research_snapshots WHERE run_id=?", (run_id,)).fetchone()
        data = dict(run)
        data["job_id"] = data.pop("id")
        data["requested_modules"] = _loads(data["requested_modules"], [])
        data["module_status"] = {row["module"]: row["status"] for row in rows}
        data["module_runs"] = [{**dict(row), "missing_fields": _loads(row["missing_fields"], []), "source_freshness": _loads(row["source_freshness"], {}), "result": _loads(row["result_json"], None)} for row in rows]
        data["result"] = _loads(snapshot[0], None) if snapshot else None
        data["error"] = data.get("error_summary")
        data["deduplicated"] = False
        return data

    def active_run(self, ticker: str, modules: list[str]) -> dict | None:
        encoded = _json(modules)
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id FROM research_runs WHERE ticker=? AND requested_modules=? AND status IN ('PENDING','RUNNING') ORDER BY created_at DESC LIMIT 1",
                (ticker, encoded),
            ).fetchone()
        return self.get_run(row[0]) if row else None

    def save_snapshot(self, run_id: str, result: dict) -> int:
        now = utc_now()
        generated_at = result.get("generated_at") or now
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT OR IGNORE INTO research_snapshots
                (run_id,ticker,generated_at,engine_version,overall_status,result_json,created_at)
                VALUES (?,?,?,?,?,?,?)""",
                (run_id, result["ticker"], generated_at, result.get("engine_version", ENGINE_VERSION), result["status"], _json(result), now),
            )
            snapshot_id = cur.lastrowid
            if not snapshot_id:
                snapshot_id = conn.execute("SELECT id FROM research_snapshots WHERE run_id=?", (run_id,)).fetchone()[0]
            result["snapshot_id"] = int(snapshot_id)
            for evidence in result.get("evidence_index", []):
                evidence["snapshot_id"] = int(snapshot_id)
            conn.execute("UPDATE research_snapshots SET result_json=? WHERE id=?", (_json(result), snapshot_id))
            conn.execute(
                """INSERT INTO latest_research_results VALUES (?,?,?,?,?)
                ON CONFLICT(ticker) DO UPDATE SET snapshot_id=excluded.snapshot_id,
                result_json=excluded.result_json,generated_at=excluded.generated_at,updated_at=excluded.updated_at""",
                (result["ticker"], snapshot_id, _json(result), generated_at, now),
            )
        return int(snapshot_id)

    def latest(self, ticker: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT result_json FROM latest_research_results WHERE ticker=?", (ticker,)).fetchone()
        return _loads(row[0], None) if row else None

    def history(self, ticker: str, limit: int = 50) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT run_id,generated_at,overall_status,engine_version,result_json FROM research_snapshots WHERE ticker=? ORDER BY generated_at DESC LIMIT ?",
                (ticker, limit),
            ).fetchall()
        output = []
        for row in rows:
            result = _loads(row["result_json"], {})
            modules = result.get("modules", {})
            output.append({"run_id": row["run_id"], "generated_at": row["generated_at"], "status": row["overall_status"], "engine_version": row["engine_version"], "score_summary": result.get("scores", {}), "completeness_summary": {name: item.get("completeness", 0) for name, item in modules.items()}})
        return output

    def compare_latest(self, ticker: str) -> dict | None:
        with self._conn() as conn:
            rows = conn.execute("SELECT run_id,result_json FROM research_snapshots WHERE ticker=? ORDER BY generated_at DESC LIMIT 2", (ticker,)).fetchall()
        if len(rows) < 2:
            return None
        new, old = _loads(rows[0]["result_json"], {}), _loads(rows[1]["result_json"], {})
        def changes(section: str) -> dict:
            before, after = old.get(section, {}), new.get(section, {})
            keys = set(before) | set(after)
            return {key: {"old": before.get(key), "new": after.get(key)} for key in keys if before.get(key) != after.get(key)}
        old_modules, new_modules = old.get("modules", {}), new.get("modules", {})
        def indexed(rows: list[dict], key: str = "id") -> dict:
            return {str(row.get(key) or row.get("title")): row for row in rows if isinstance(row, dict)}
        def delta(old_rows: list[dict], new_rows: list[dict], key: str = "id") -> dict:
            before, after = indexed(old_rows, key), indexed(new_rows, key)
            return {
                "added": [after[item] for item in after.keys() - before.keys()],
                "removed": [before[item] for item in before.keys() - after.keys()],
                "changed": [{"old": before[item], "new": after[item]} for item in before.keys() & after.keys() if before[item] != after[item]],
            }
        old_drivers, new_drivers = old.get("key_drivers", {}), new.get("key_drivers", {})
        old_scenarios, new_scenarios = old.get("scenarios", {}), new.get("scenarios", {})
        return {
            "ticker": ticker, "old_run_id": rows[1]["run_id"], "new_run_id": rows[0]["run_id"],
            "score_changes": changes("scores"),
            "risk_changes": {"level": {"old": old.get("risk_level"), "new": new.get("risk_level")} if old.get("risk_level") != new.get("risk_level") else {},
                             "findings": delta([row for row in old.get("research_findings", []) if row.get("category") == "risk"], [row for row in new.get("research_findings", []) if row.get("category") == "risk"], "title")},
            "expectation_changes": {"old": old_modules.get("expectations", {}).get("metrics", {}), "new": new_modules.get("expectations", {}).get("metrics", {})},
            "catalyst_changes": delta(old.get("key_catalysts", []), new.get("key_catalysts", []), "title"),
            "fundamental_changes": {"old": old_modules.get("fundamental", {}).get("metrics", {}), "new": new_modules.get("fundamental", {}).get("metrics", {})},
            "thesis_changes": {"old": old.get("core_thesis") or old.get("investment_thesis"), "new": new.get("core_thesis") or new.get("investment_thesis"),
                               "changed": (old.get("core_thesis") or old.get("investment_thesis")) != (new.get("core_thesis") or new.get("investment_thesis"))},
            "driver_changes": {"positive": delta(old_drivers.get("positive", []), new_drivers.get("positive", []), "title"),
                               "negative": delta(old_drivers.get("negative", []), new_drivers.get("negative", []), "title")},
            "scenario_changes": {name: {"old": old_scenarios.get(name, {}).get("assumptions", []), "new": new_scenarios.get(name, {}).get("assumptions", [])}
                                 for name in ("BULL", "BASE", "BEAR") if old_scenarios.get(name, {}).get("assumptions", []) != new_scenarios.get(name, {}).get("assumptions", [])},
        }

    @staticmethod
    def cache_key(ticker: str, name: str, period: str, version: str = ENGINE_VERSION) -> str:
        return f"{version}:{ticker}:{name}:{period}"

    def get_module_cache(self, ticker: str, module: str, period: str = "default") -> dict | None:
        key = self.cache_key(ticker, module, period)
        with self._conn() as conn:
            row = conn.execute("SELECT result_json,expires_at FROM research_module_cache WHERE cache_key=?", (key,)).fetchone()
        if not row or row["expires_at"] <= time.time():
            return None
        result = _loads(row["result_json"], None)
        if result:
            result["cache_hit"] = True
            result["cache_expires_at"] = datetime.fromtimestamp(row["expires_at"], timezone.utc).isoformat()
        return result

    def delete_module_cache(self, ticker: str, module: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM research_module_cache WHERE ticker=? AND module=?", (ticker.upper(), module))

    def peer_preferences(self, ticker: str) -> dict[str, list[str]]:
        with self._conn() as conn:
            rows = conn.execute("SELECT peer_ticker,action FROM research_peer_preferences WHERE ticker=? ORDER BY peer_ticker", (ticker.upper(),)).fetchall()
        return {
            "added": [row["peer_ticker"] for row in rows if row["action"] == "ADD"],
            "removed": [row["peer_ticker"] for row in rows if row["action"] == "REMOVE"],
        }

    def set_peer_preference(self, ticker: str, peer_ticker: str, action: str) -> dict[str, list[str]]:
        ticker, peer_ticker, action = ticker.upper(), peer_ticker.upper(), action.upper()
        if action not in {"ADD", "REMOVE"}:
            raise ValueError("peer action must be ADD or REMOVE")
        now = utc_now()
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO research_peer_preferences VALUES (?,?,?,?,?)
                ON CONFLICT(ticker,peer_ticker) DO UPDATE SET action=excluded.action,updated_at=excluded.updated_at""",
                (ticker, peer_ticker, action, now, now),
            )
        self.delete_module_cache(ticker, "valuation")
        return self.peer_preferences(ticker)

    def put_module_cache(self, ticker: str, module: str, result: dict, ttl_seconds: int, period: str = "default") -> None:
        now = time.time()
        expires = now + ttl_seconds
        result.update({"cache_hit": False, "cache_expires_at": datetime.fromtimestamp(expires, timezone.utc).isoformat()})
        value = dict(result)
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO research_module_cache VALUES (?,?,?,?,?,?,?,?,?)",
                (self.cache_key(ticker, module, period), ticker, module, period, ENGINE_VERSION, _json(value), now, expires, _json(value.get("source_freshness", {}))),
            )

    def get_data_cache(self, ticker: str, data_type: str, period: str = "default") -> tuple[Any, dict] | None:
        key = self.cache_key(ticker, data_type, period)
        with self._conn() as conn:
            row = conn.execute("SELECT payload_json,diagnostics_json,expires_at FROM research_data_cache WHERE cache_key=?", (key,)).fetchone()
        if not row or row["expires_at"] <= time.time():
            return None
        return _loads(row[0], []), _loads(row[1], {})

    def put_data_cache(self, ticker: str, data_type: str, payload: Any, ttl_seconds: int, diagnostics: dict, period: str = "default") -> None:
        now = time.time()
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO research_data_cache VALUES (?,?,?,?,?,?,?,?,?)",
                (self.cache_key(ticker, data_type, period), ticker, data_type, period, ENGINE_VERSION, _json(payload), now, now + ttl_seconds, _json(diagnostics)),
            )

    def upsert_relationship(self, relationship: dict, sources: list[dict] | None = None) -> int:
        now = utc_now()
        source = str(relationship["source_ticker"]).upper()
        target = str(relationship["target_ticker"]).upper()
        rel_type = str(relationship["relationship_type"]).upper()
        verified_at = now if relationship.get("status") == "VERIFIED" else relationship.get("last_verified_at")
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO company_relationships
                (source_ticker,target_ticker,target_name,relationship_type,status,confidence,description,first_seen_at,last_seen_at,last_verified_at,created_at,updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(source_ticker,target_ticker,relationship_type) DO UPDATE SET
                target_name=excluded.target_name,status=excluded.status,confidence=excluded.confidence,
                description=excluded.description,last_seen_at=excluded.last_seen_at,
                last_verified_at=COALESCE(excluded.last_verified_at,company_relationships.last_verified_at),updated_at=excluded.updated_at""",
                (source, target, relationship.get("target_name"), rel_type, relationship.get("status", "DISCOVERED"), float(relationship.get("confidence", 0)), relationship.get("description"), relationship.get("first_seen_at", now), now, verified_at, now, now),
            )
            relation_id = conn.execute("SELECT id FROM company_relationships WHERE source_ticker=? AND target_ticker=? AND relationship_type=?", (source, target, rel_type)).fetchone()[0]
            for item in sources or []:
                conn.execute(
                    """INSERT OR IGNORE INTO relationship_sources
                    (relationship_id,provider,title,url,published_at,fetched_at,evidence_text,evidence_summary) VALUES (?,?,?,?,?,?,?,?)
                    ON CONFLICT(relationship_id,url) DO UPDATE SET provider=excluded.provider,title=excluded.title,
                    fetched_at=excluded.fetched_at,evidence_text=excluded.evidence_text,evidence_summary=excluded.evidence_summary""",
                    (relation_id, item.get("provider"), item.get("title"), item.get("url"), item.get("published_at"), item.get("fetched_at", now), item.get("evidence_text"), item.get("evidence_summary")),
                )
        return int(relation_id)

    def relationships(self, ticker: str) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM company_relationships WHERE source_ticker=? ORDER BY confidence DESC,updated_at DESC", (ticker.upper(),)).fetchall()
            sources = conn.execute("SELECT s.* FROM relationship_sources s JOIN company_relationships r ON r.id=s.relationship_id WHERE r.source_ticker=? ORDER BY s.fetched_at DESC,s.id DESC", (ticker.upper(),)).fetchall()
        return [{**dict(row), 'sources': [dict(s) for s in sources if s['relationship_id'] == row['id']]} for row in rows]

    def relationship(self, relation_id: int) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM company_relationships WHERE id=?", (relation_id,)).fetchone()
            sources = conn.execute("SELECT * FROM relationship_sources WHERE relationship_id=? ORDER BY fetched_at,id", (relation_id,)).fetchall() if row else []
        return {**dict(row), "sources": [dict(item) for item in sources]} if row else None
