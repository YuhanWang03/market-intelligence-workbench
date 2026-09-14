"""Read-only monitor archive. Chronological retrieval, not semantic similarity."""
from contextlib import closing
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from pathlib import Path
import sqlite3
from types import SimpleNamespace

from v2.agent_v2.models import EvidenceItem, ResultStatus, ToolEnvelope


class _Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
    def handle_data(self, data):
        self.parts.append(data)


def evidence(row):
    parser = _Text()
    parser.feed(row["text_html"] or "")
    body = " ".join(parser.parts)
    return EvidenceItem(
        id=f"monitor-{row['id']}", entity=row["tickers"] or "monitor",
        claim=f"监控档案在 {row['ts']} 记录了以下内容（记录时间不等于事件发生时间，记录中的归因未独立证实）：{body[:6000]}",
        as_of=row["ts"], source_id=f"monitor_archive:{row['id']}",
        source_title=f"监控原始记录 #{row['id']}",
        metadata={"record_id": row["id"], "recorded_at": row["ts"], "evidence_scope": "monitor_record", "truncated": len(body) > 6000},
    )


class MonitorArchive:
    def __init__(self, path=None, now=None):
        self.path = Path(path or Path(__file__).resolve().parents[2] / "data/archive.db")
        self.now = now or (lambda: datetime.now(timezone.utc))

    def _read(self, clause, parameters, limit):
        # mode=ro refuses missing databases and prevents migrations or writes.
        with closing(sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True, timeout=3)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT id,ts,agent,msg_type,tickers,text_html FROM pushes WHERE "
                "(agent IN ('intraday_anomaly','anomaly','alert') OR msg_type='intraday_anomaly') AND "
                + clause + " ORDER BY julianday(ts) DESC,id DESC LIMIT ?", (*parameters, limit)).fetchall()
            return [dict(row) for row in rows]

    def get(self, record_id, ticker=""):
        if not str(record_id).isascii() or not str(record_id).isdigit() or len(str(record_id)) > 18:
            raise ValueError("Invalid monitor record ID")
        rows = self._read("id=?", (int(record_id),), 1)
        if not rows:
            raise LookupError("Monitor record not found or no longer retained")
        row = rows[0]
        symbols = [value.strip().upper() for value in (row["tickers"] or "").split(",")]
        if ticker and ticker.upper() not in symbols:
            raise ValueError("Selected ticker does not match archived record")
        return row

    def recall(self, ticker, query, days):
        if not ticker or len(ticker) > 32:
            raise ValueError("Ticker required")
        days = max(1, min(int(days), 365))
        now = self.now()
        rows = self._read(
            "instr(',' || replace(upper(COALESCE(tickers,'')), ' ', '') || ',', ?) > 0 "
            "AND julianday(ts)>=julianday(?) AND julianday(ts)<=julianday(?)",
            (f",{ticker.upper()},", (now-timedelta(days=days)).isoformat(), now.isoformat()), 6)
        return [SimpleNamespace(date=row["ts"][:10], flags=row["msg_type"], doc=evidence(row).claim,
                                metadata={"monitor_record": row}) for row in rows]


def register_monitor_history(registry, archive):
    def history(arguments, context):
        rows = archive.recall(arguments["ticker"], arguments.get("query", ""), arguments.get("lookback_days", 90))
        return ToolEnvelope("market.anomaly_history", ResultStatus.COMPLETED, subject=arguments["ticker"],
                            evidence=[evidence(row.metadata["monitor_record"]) for row in rows],
                            metrics={"count": len(rows)},
                            limitations=["按股票和记录时间倒序读取最多六条已保留档案，不是语义相似检索；无记录不代表没有发生异动；记录不独立证明因果。"])
    registry.register("market.anomaly_history", history)
