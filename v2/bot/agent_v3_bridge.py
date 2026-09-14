"""Explicit Telegram V3 entry, through the isolated V3 HTTP service."""
import asyncio
import os
import httpx

_locks = {}


def parse_flags(text):
    parts = text.split(maxsplit=1)
    if parts and parts[0] in {"--web","--noweb"}:
        return (parts[1] if len(parts)>1 else ""), parts[0] == "--web"
    return text, False


async def handle_agent_v3(update, context):
    question, allow_web = parse_flags(" ".join(context.args or []))
    if not question:
        await update.message.reply_text("用法：/ask_v3 [--web] 问题；/ask_v3 --web 分析MU")
        return
    from dotenv import dotenv_values
    token = os.environ.get("WEB_OWNER_TOKEN") or dotenv_values("/etc/hedge-fund/web.env").get("WEB_OWNER_TOKEN")
    if not token:
        await update.message.reply_text("V3 服务鉴权尚未配置。")
        return
    session = f"telegram-v3:{update.effective_chat.id}:{update.effective_user.id}"
    lock = _locks.setdefault(session,asyncio.Lock())
    async with lock:
        await update.message.reply_text("Agent V3 正在处理…")
        try:
            async with httpx.AsyncClient(timeout=240) as client:
                response = await client.post("http://127.0.0.1:8104/api/agent-v3/ask",headers={"X-Owner-Token":token},json={"text":question,"session_id":session,"allow_web":allow_web,"background":False})
                response.raise_for_status()
                payload = response.json()
            answer = payload.get("answer") or "本次没有生成回答。"
            sources = [f"{e['id']}: {e['source_url']}" for e in payload.get("evidence",[]) if e.get("source_url")]
            text = f"Agent V3 · {payload.get('status','unknown')}\n{answer}"
            if sources:
                text += "\n来源：\n" + "\n".join(dict.fromkeys(sources))
            for start in range(0,len(text),3500):
                await update.message.reply_text(text[start:start+3500],disable_web_page_preview=True)
        except (httpx.HTTPError, ValueError):
            await update.message.reply_text("V3 服务请求失败，请稍后重试。")
