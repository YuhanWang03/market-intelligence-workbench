"""Version-independent Telegram message delivery."""
import logging
from typing import Any
from v2.agent_common import presentation

logger = logging.getLogger(__name__)

#: Telegram rejects any message body longer than this.
TELEGRAM_LIMIT = 4096
#: Headroom for the routing chip, the disclosure line and a continuation mark.
CHUNK_LIMIT = 3600


def split_for_telegram(text: str, limit: int = CHUNK_LIMIT) -> list[str]:
    """Cut an answer into pieces Telegram will accept, at paragraph breaks.

    An agent answer can run past 4096 characters — the multi-step ones routinely
    do — and ``editMessageText`` then fails outright. The old code caught that
    exception, named it in a comment ("message unchanged / deleted / too long")
    and passed, so a complete, correct, expensive answer was discarded and the
    placeholder kept saying 「分析中…」.
    """
    body = (text or "").strip()
    if len(body) <= limit:
        return [body]

    chunks: list[str] = []
    rest = body
    while len(rest) > limit:
        window = rest[:limit]
        # Prefer a paragraph break, then a line break; only cut mid-line when
        # the alternative is a chunk barely worth sending.
        cut = max(window.rfind("\n\n"), window.rfind("\n"))
        if cut < limit // 3:
            cut = limit
        chunks.append(rest[:cut].rstrip())
        rest = rest[cut:].lstrip("\n")
    if rest.strip():
        chunks.append(rest.strip())
    return chunks


async def _deliver(placeholder: Any, text: str) -> None:
    """Put the whole answer in front of the user, however long it is."""
    chunks = split_for_telegram(text)
    await _edit(placeholder, chunks[0])
    reply = getattr(placeholder, "reply_text", None)
    for index, chunk in enumerate(chunks[1:], start=2):
        marked = f"<i>（续 {index}/{len(chunks)}）</i>\n\n{chunk}"
        if reply is None:
            break
        try:
            await reply(marked, parse_mode="HTML", disable_web_page_preview=True)
        except Exception:  # noqa: BLE001
            try:
                await reply(presentation.to_plain_text(chunk),
                            disable_web_page_preview=True)
            except Exception:  # noqa: BLE001
                logger.warning("续段 %d/%d 发送失败", index, len(chunks))
                break


async def _edit(placeholder: Any, text: str) -> None:
    """Edit a Telegram message, falling back to plain text on malformed HTML.

    Responder cards are hand-built HTML and safe; an agent's answer is written by
    the model and can contain a stray '<'. The bot has been bitten by exactly
    this before (see main._error_handler), and losing a correct answer to a
    parse error is the worst possible outcome, so the fallback is unconditional.
    """
    if len(text) > TELEGRAM_LIMIT:
        text = text[: TELEGRAM_LIMIT - 24].rstrip() + "\n…（已截断）"
    try:
        await placeholder.edit_text(text, parse_mode="HTML",
                                    disable_web_page_preview=True)
    except Exception:  # noqa: BLE001 — telegram.error.BadRequest and friends
        try:
            # Plain text, with the markup removed rather than shown: a reader
            # who gets "<b>结论</b>" is worse off than one who gets "结论".
            await placeholder.edit_text(presentation.to_plain_text(text),
                                        disable_web_page_preview=True)
        except Exception:  # noqa: BLE001 — message unchanged / deleted / gone
            # Never silent again: this is how a finished run ends up looking
            # like a hung one, and the log is the only place it can be seen.
            logger.warning("无法写入占位消息（%d 字符）", len(text or ""))
