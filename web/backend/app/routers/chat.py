"""POST /api/chat — free-form NL query → same intents the bot supports.

Classify (DeepSeek, strict-enum) → dispatch to a v2 responder → return the
HTML card (+ optional base64 chart). Both steps are blocking, so they run in
a threadpool to keep the event loop free.
"""

from __future__ import annotations

import asyncio
import html as _html
import logging

from fastapi import APIRouter, Depends
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from app.auth import require_owner

router = APIRouter(prefix="/api", tags=["chat"])
logger = logging.getLogger("web.chat")

# A responder that reaches a slow/unreachable external API (Tavily / OpenAI
# embeddings) could otherwise hang the request forever. Cap it.
_CLASSIFY_TIMEOUT = 25
_DISPATCH_TIMEOUT = 45


class ChatIn(BaseModel):
    text: str


@router.get("/chat/jobs/{job_id}", dependencies=[Depends(require_owner)])
async def chat_job(job_id: str) -> dict:
    """Compatibility poller backed by the single Research Job registry."""
    from app.routers.research import get_research_run
    from v2.research.compat import format_chain_result

    job = await get_research_run(job_id)
    if job["status"] in {"COMPLETED", "PARTIAL"} and job.get("result"):
        formatted, data = format_chain_result(job["result"])
        return {"job_id": job_id, "status": "completed", "intent": "chain", "html": formatted, "data": data}
    if job["status"] == "FAILED":
        return {"job_id": job_id, "status": "failed", "intent": "chain",
                "error": job.get("error", "产业链研究失败")}
    return {"job_id": job_id, "status": "running", "intent": "chain",
            "ticker": job.get("ticker", "")}


@router.post("/chat", dependencies=[Depends(require_owner)])
async def chat(body: ChatIn) -> dict:
    from app.dispatch import dispatch, parse_slash

    try:
        # Slash commands (/flow AAPL, ...) skip the LLM classifier;
        # free-form text goes through it.
        parsed = parse_slash(body.text)
        if parsed is None:
            from v2.bot.intent import classify
            parsed = await asyncio.wait_for(
                run_in_threadpool(classify, body.text), timeout=_CLASSIFY_TIMEOUT,
            )

        if parsed.get("intent") == "chain":
            from app.routers.research import ResearchRunInput, create_research_run

            job = await create_research_run(ResearchRunInput(ticker=parsed.get("ticker", "")))
            return {"intent": "chain", "status": "running", "job_id": job["job_id"],
                    "deduplicated": bool(job.get("deduplicated"))}

        result = await asyncio.wait_for(
            run_in_threadpool(dispatch, parsed), timeout=_DISPATCH_TIMEOUT,
        )
        return {"intent": parsed.get("intent"), "args": parsed, **result}
    except asyncio.TimeoutError:
        logger.warning("chat timed out for %r", body.text)
        return {
            "intent": "timeout",
            "html": (
                "⏱ 查询超时（某个外部数据源响应过慢）。<br>"
                "「为什么涨/跌」需要连 Tavily / OpenAI，本机网络可能受阻；"
                "可改用 <code>/flow</code>、左侧面板，或把网页部署到 VPS。"
            ),
        }
    except Exception as exc:
        # Surface the real error instead of an opaque 500 (full traceback
        # goes to the uvicorn console).
        logger.exception("chat failed for %r", body.text)
        return {
            "intent": "error",
            "html": f"❌ 出错: <code>{_html.escape(str(exc))}</code>",
            "error_type": type(exc).__name__,
        }
