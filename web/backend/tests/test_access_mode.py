from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app import auth, chart_snapshots, public_snapshots
from app.auth import require_owner
from app.main import enforce_site_access
from app.routers import access


def _settings(tmp_path: Path):
    return SimpleNamespace(
        owner_token="owner-secret-value",
        admin_username="owner",
        admin_password_hash="",
        session_secret="separate-session-secret",
        session_ttl_seconds=3600,
        cookie_secure=False,
        guest_enabled=True,
        public_snapshot_db_path=tmp_path / "public.sqlite",
        repo_root=tmp_path,
    )


def _client(monkeypatch, tmp_path: Path) -> TestClient:
    settings = _settings(tmp_path)
    monkeypatch.setattr(auth, "SETTINGS", settings)
    monkeypatch.setattr(access, "SETTINGS", settings)
    monkeypatch.setattr(public_snapshots, "SETTINGS", settings)
    access._FAILURES.clear()
    app = FastAPI()
    app.include_router(access.router)

    @app.get("/api/protected", dependencies=[Depends(require_owner)])
    async def protected():
        return {"ok": True}

    return TestClient(app)


def test_owner_login_guest_session_and_logout(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    assert client.get("/api/auth/status").json()["role"] == "anonymous"
    assert client.post("/api/auth/login", json={"username": "owner", "password": "wrong"}).status_code == 401

    guest = client.post("/api/auth/guest", json={})
    assert guest.status_code == 200 and guest.json()["role"] == "guest"
    assert client.get("/api/protected").status_code == 403

    assert client.post("/api/auth/logout", json={}).status_code == 200
    owner = client.post("/api/auth/login", json={"username": "owner", "password": "owner-secret-value"})
    assert owner.status_code == 200 and owner.json()["role"] == "owner"
    assert client.get("/api/protected").json() == {"ok": True}
    assert client.post("/api/auth/logout", json={}).status_code == 200
    assert client.get("/api/protected").status_code == 401


def test_guest_only_reads_owner_published_allowlisted_snapshot(monkeypatch, tmp_path):
    owner = _client(monkeypatch, tmp_path)
    assert owner.post("/api/auth/login", json={"username": "owner", "password": "owner-secret-value"}).status_code == 200
    saved = owner.post("/api/public/snapshot", json={"path": "/api/portfolio", "payload": {"paper": True, "value": 100}})
    assert saved.status_code == 200
    assert owner.post("/api/public/snapshot", json={"path": "/api/agent-v2/ask", "payload": {}}).status_code == 400

    guest = _client(monkeypatch, tmp_path)
    assert guest.post("/api/auth/guest", json={}).status_code == 200
    response = guest.get("/api/public/snapshot", params={"path": "/api/portfolio"})
    assert response.status_code == 200
    assert response.json()["payload"] == {"paper": True, "value": 100}
    assert guest.post("/api/public/snapshot", json={"path": "/api/portfolio", "payload": {}}).status_code == 403
    assert guest.get("/api/public/snapshot", params={"path": "/api/agent-v2/ask"}).status_code == 403


def test_password_hash_and_tampered_session(monkeypatch, tmp_path):
    settings = _settings(tmp_path)
    password = "a-long-owner-password"
    settings.admin_password_hash = auth.make_password_hash(password)
    monkeypatch.setattr(auth, "SETTINGS", settings)
    assert auth.verify_admin_credentials("owner", password)
    assert not auth.verify_admin_credentials("owner", "wrong-password")

    token = auth.create_session_token("owner", "owner")
    assert auth.decode_session_token(token).role == "owner"
    assert auth.decode_session_token(token + "x") is None


def test_global_firewall_blocks_guest_from_accidentally_unprotected_api(monkeypatch, tmp_path):
    settings = _settings(tmp_path)
    monkeypatch.setattr(auth, "SETTINGS", settings)
    monkeypatch.setattr(access, "SETTINGS", settings)
    monkeypatch.setattr(public_snapshots, "SETTINGS", settings)

    app = FastAPI()
    app.middleware("http")(enforce_site_access)
    app.include_router(access.router)

    @app.get("/api/future-provider-route")
    async def accidentally_unprotected():
        return {"provider_called": True}

    client = TestClient(app)
    assert client.get("/api/future-provider-route").status_code == 401
    assert client.post("/api/auth/guest", json={}).status_code == 200
    assert client.get("/api/future-provider-route").status_code == 403
    assert client.post("/api/auth/logout", json={}).status_code == 200
    assert client.post("/api/auth/login", json={"username": "owner", "password": "owner-secret-value"}).status_code == 200
    assert client.get("/api/future-provider-route").json() == {"provider_called": True}


def test_owner_can_queue_holding_chart_snapshots_but_guest_cannot(monkeypatch, tmp_path):
    client = _client(monkeypatch, tmp_path)
    calls = []
    monkeypatch.setattr(access, "reserve_chart_snapshot_refresh", lambda symbols, ranges: (True, {
        "status": "running", "total": len(symbols) * len(ranges), "published": 0, "failed": 0,
    }))
    monkeypatch.setattr(access, "publish_holding_chart_snapshots", lambda symbols, ranges: calls.append((symbols, ranges)))
    assert client.post("/api/auth/guest", json={}).status_code == 200
    assert client.post("/api/public/snapshot/price-history/refresh", json={"symbols": ["IVV"]}).status_code == 403
    assert client.post("/api/auth/logout", json={}).status_code == 200
    assert client.post("/api/auth/login", json={"username": "owner", "password": "owner-secret-value"}).status_code == 200
    response = client.post("/api/public/snapshot/price-history/refresh", json={"symbols": ["ivv", "IVV"], "ranges": ["3m"]})
    assert response.status_code == 200
    assert response.json()["started"] is True
    assert calls == [(["IVV"], ["3M"])]


def test_chart_snapshot_worker_publishes_each_symbol_range(monkeypatch, tmp_path):
    from app.routers import portfolio

    settings = _settings(tmp_path)
    monkeypatch.setattr(public_snapshots, "SETTINGS", settings)
    monkeypatch.setattr(portfolio, "_fetch_price_history", lambda symbol, range_key, extended: {
        "symbol": symbol, "range": range_key, "includesExtendedHours": extended,
    })
    chart_snapshots.reserve_chart_snapshot_refresh(["ivv"], ["1m", "3m"])
    chart_snapshots.publish_holding_chart_snapshots(["ivv"], ["1m", "3m"])
    one_month = public_snapshots.read_snapshot("/api/price-history/IVV?extended=false&range=1M")
    three_month = public_snapshots.read_snapshot("/api/price-history/IVV?range=3M&extended=false")
    assert one_month and one_month["payload"]["range"] == "1M"
    assert three_month and three_month["payload"]["range"] == "3M"
    assert chart_snapshots.chart_snapshot_status()["published"] == 2
