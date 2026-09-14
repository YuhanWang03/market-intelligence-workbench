"""FastAPI entry for the new web backend (thin layer over v2 modules).

Personal, single-user operational panel + chat. Rebuilt from scratch; the
old dashboard/ is frozen. Run:

    cd web/backend
    WEB_OWNER_TOKEN=dev PYTHONPATH=.:../.. \
        uvicorn app.main:app --reload --port 8100
"""

from __future__ import annotations

import sys
import asyncio
import logging
from importlib.util import find_spec
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import SETTINGS
repo_root = str(SETTINGS.repo_root.resolve())
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from app.routers import agent_v2, chat, committee, dashboard, health, portfolio, research, workspace  # noqa: E402

@asynccontextmanager
async def lifespan(app):
    async def poll_prices():
        from v2.data.price_sync import sync_prices
        from v2.data.usage_ledger import reconcile_tavily_flat_rate, rollup_and_prune
        cycle = 0
        while True:
            try:
                await asyncio.to_thread(rollup_and_prune, 24)
                if cycle % 6 == 0:
                    await asyncio.to_thread(sync_prices)
                    await asyncio.to_thread(reconcile_tavily_flat_rate)
            except Exception:
                logging.getLogger(__name__).warning('Usage retention or official price sync unavailable')
            cycle += 1
            await asyncio.sleep(3600)
    task = asyncio.create_task(poll_prices())
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


app = FastAPI(title="AI Hedge Fund · Web", version="0.1.0", lifespan=lifespan)


@app.middleware('http')
async def billing_channel(request, call_next):
    from v2.usage_context import usage_channel
    with usage_channel('web'):
        return await call_next(request)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(SETTINGS.cors_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(agent_v2.router)
# The production V3 service has its own environment and reverse-proxy route.
# Keep the V2/legacy web environment importable without LangGraph installed.
if find_spec("langgraph") is not None:
    from app.routers import agent_v3
    app.include_router(agent_v3.router)
app.include_router(chat.router)
app.include_router(portfolio.router)
app.include_router(dashboard.router)
app.include_router(workspace.router)
app.include_router(research.router)
app.include_router(committee.router)
