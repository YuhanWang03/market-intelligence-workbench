"""What the user actually sees — the answer minus the model talking to itself.

Two leaks showed up in the first live Telegram session, both of them the model
narrating its own process into the final message:

    你说得对，我犯了张冠李戴的错误。让我重新核对每个数字的真实归属……
    ## 结论：持仓里最危险的是 ARM

    用户问「为什么？」，但没有上下文指明具体针对什么……让我直接询问澄清。
    ---
    你的问题「为什么？」缺少上下文，我无法确定你想问的是哪件事。

The first is a reply addressed to the grounding checker, which the user never
saw complain. The second is deliberation the model itself separated from the
answer with a horizontal rule — it knew which half was the answer.

The tempting fix is a prompt line: "don't narrate your reasoning". This project
has already measured what those are worth — three rounds of instructing the
model to show its arithmetic changed nothing, and making the *checker* accept
shown arithmetic changed behaviour immediately. So this is a rule the code
enforces, not one the prompt requests.

Deliberately narrow. It fires only when the model itself marked the boundary
with a rule or a heading, and only when what sits before that boundary reads as
deliberation. Prose that runs straight into its answer with no marker is left
alone: guessing where an unmarked answer begins would eventually eat one.
"""

from __future__ import annotations

import re

#: A horizontal rule on its own line — the model's own "the answer starts here".
_RULE = re.compile(r"^[ \t]*(?:-{3,}|_{3,}|\*{3,})[ \t]*$", re.M)

#: A markdown heading, the other marker the model uses for the same purpose.
_HEADING = re.compile(r"^[ \t]*#{1,6}[ \t]+\S", re.M)

#: First-person process talk. 让我们 is excluded on purpose — "让我们看看这几只"
#: is a normal way to open an answer, while 让我 alone is the model narrating.
_DELIBERATION = re.compile(
    r"让我(?!们)|我需要|我应该|我先|我来|用户(?:问|想问|说|的问题)|你说得对"
    r"|我犯了|我此前|我不该|重新核对|直接询问|澄清一下|让我直接"
    r"|上一条回复|上一版|之前的回答|我收回|随口编|张冠李戴了|初稿")

#: A repair round talking about the answer it is replacing. The user never saw
#: that draft, so a paragraph about it carries nothing for them — and it
#: carries the rejected figure: 「上一条回复里的 400 是我随口编的，我收回」
#: re-states the very number the check refused, and the rewrite fails again on
#: it. Dropped when it is the opening paragraph and a real answer follows.
_RETRACTION = re.compile(r"上一条回复|上一版|之前的回答|我收回|随口编|张冠李戴了|初稿")
_PARAGRAPH_END = re.compile(r"\n[ \t]*\n")

#: A single opening sentence of process narration, with no rule or heading after
#: it. The marked case is handled above; this is the unmarked one, and it kept
#: arriving anyway — 「你说得对，我犯了把 ARKK 的数据安到 ARKQ 头上的错误。」 and
#: 「我来对照你的持仓和 NVDA 供应链信息。」 both reached users.
#:
#: Bounded hard: one sentence, short, and only when substantial content follows.
#: Guessing where an unmarked answer begins would eventually eat a real
#: paragraph, so this only ever removes the first clause of the first line.
MAX_OPENER = 60
#: There must be a real answer left behind. A short opener followed by a short
#: remainder is more likely one sentence of substance than narration plus body.
MIN_REMAINDER = 40
_OPENER_END = re.compile(r"[。！？\n]")

#: How much text may be discarded as preamble. A bound rather than a judgment:
#: if the marker sits this far in, whatever precedes it is too substantial to
#: throw away on a heuristic, however much it sounds like thinking out loud.
MAX_PREAMBLE = 500


def strip_deliberation(text: str) -> str:
    """Drop a leading block of process narration the model marked off itself.

    Returns ``text`` unchanged unless every condition holds: there is a rule or
    heading, what precedes it reads as deliberation, it is under MAX_PREAMBLE
    characters, and something is left afterwards.
    """
    body = text or ""
    for pattern, keep_marker in ((_RULE, False), (_HEADING, True)):
        match = pattern.search(body)
        if not match:
            continue
        preamble = body[: match.start()]
        remainder = body[match.start() if keep_marker else match.end():].strip()
        if (preamble.strip() and remainder
                and len(preamble) <= MAX_PREAMBLE
                and _DELIBERATION.search(preamble)):
            return remainder

    match = _PARAGRAPH_END.search(body)
    if match:
        first, rest = body[: match.start()], body[match.end():].strip()
        if (len(first) <= MAX_PREAMBLE and _RETRACTION.search(first)
                and len(rest) >= MIN_REMAINDER):
            return rest

    match = _OPENER_END.search(body)
    if match:
        opener, rest = body[: match.end()], body[match.end():].strip()
        if (len(opener) <= MAX_OPENER and _DELIBERATION.search(opener)
                and len(rest) >= MIN_REMAINDER):
            return rest
    return body.strip()


# ---------------------------------------------------------------------------
# Markdown → Telegram HTML
# ---------------------------------------------------------------------------
#
# The bot sends with parse_mode="HTML" because every existing responder card is
# hand-built HTML. The agent's answer is not: it is written by a model, and
# models write Markdown. So the first live answers arrived with their markup
# showing — "# 结论：", "## 一、整体表现", "**小幅上涨**", and pipe tables drawn
# character by character.
#
# Telegram's HTML is a short list — b, i, u, s, code, pre, a, blockquote — with
# no headings and no tables, so this is a translation, not a pass-through:
# headings become bold lines and tables become padded <pre>, which is the only
# way columns stay lined up in a chat client.

import html as _html

_FENCE = re.compile(r"```[^\n]*\n(.*?)```", re.S)
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_RULE = re.compile(r"^\s*\|[\s:|-]+\|\s*$")
_HEADING = re.compile(r"^[ \t]*(#{1,6})[ \t]+(.+?)[ \t]*#*$", re.M)
_BULLET = re.compile(r"^([ \t]*)[-*+][ \t]+", re.M)
_BOLD = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*|__(?=\S)(.+?)(?<=\S)__", re.S)
_CODE = re.compile(r"`([^`\n]+)`")
#: Single-asterisk italics only between non-word characters — a bare * inside a
#: figure or a footnote marker must not turn half the answer sideways.
_ITALIC = re.compile(r"(?<![\w*])\*(?=\S)([^*\n]+?)(?<=\S)\*(?![\w*])")


def _display_width(text: str) -> int:
    """CJK glyphs occupy two cells; column padding is wrong without this."""
    return sum(2 if ord(ch) > 0x2E7F else 1 for ch in text)


#: How wide a <pre> block may get before it stops being readable. A phone shows
#: roughly this many monospace cells; past it the block scrolls sideways and the
#: reader has to drag a 140-character row across the screen to see its last
#: column. Seen live, on a two-column table whose second cell held nine tickers.
MAX_PRE_WIDTH = 46


def _render_table(rows: list[str]) -> str:
    """A Markdown table as a padded <pre> block — Telegram has no table tag.

    Padding only works while the widest row still fits the screen. When it does
    not, columns are abandoned for one wrapped line per row: text that wraps is
    always more readable on a phone than a grid that scrolls.
    """
    cells = [[c.strip() for c in row.strip().strip("|").split("|")]
             for row in rows if not _TABLE_RULE.match(row)]
    if not cells:
        return ""
    columns = max(len(r) for r in cells)
    cells = [r + [""] * (columns - len(r)) for r in cells]
    widths = [max(_display_width(r[i]) for r in cells) for i in range(columns)]

    if sum(widths) + 2 * (columns - 1) > MAX_PRE_WIDTH:
        return _render_table_as_lines(cells)

    lines = [
        "  ".join(cell + " " * (widths[i] - _display_width(cell))
                  for i, cell in enumerate(row)).rstrip()
        for row in cells
    ]
    return "<pre>" + "\n".join(lines) + "</pre>"


#: Cells that carry no information and should not be read out in the line form.
_EMPTY_CELL = frozenset({"", "—", "-", "–", "n/a", "N/A", "无"})


def _render_table_as_lines(cells: list[list[str]]) -> str:
    """A table too wide to align, as one wrapping bullet per row."""
    header, *rows = cells
    if not rows:
        rows, header = [header], [""] * len(header)
    out: list[str] = []
    for row in rows:
        rest = [f"{header[i]} {row[i]}".strip()
                for i in range(1, len(row)) if row[i] not in _EMPTY_CELL]
        lead = row[0] if row[0] not in _EMPTY_CELL else ""
        out.append("· " + " · ".join([p for p in [lead, *rest] if p]))
    return "\n".join(out)


def to_telegram_html(text: str) -> str:
    """Translate the model's Markdown into the subset Telegram renders."""
    body = (text or "").strip()
    if not body:
        return ""

    blocks: list[str] = []

    def _stash(rendered: str) -> str:
        blocks.append(rendered)
        return f"\x00{len(blocks) - 1}\x00"

    body = _FENCE.sub(
        lambda m: _stash("<pre>" + _html.escape(m.group(1).strip()) + "</pre>"), body)

    # Tables are consumed line-block by line-block before anything is escaped,
    # so the padding is computed on the text the reader will actually see.
    out: list[str] = []
    pending: list[str] = []
    for line in body.split("\n"):
        if _TABLE_ROW.match(line):
            pending.append(_html.escape(line))
            continue
        if pending:
            out.append(_stash(_render_table(pending)))
            pending = []
        out.append(line)
    if pending:
        out.append(_stash(_render_table(pending)))
    body = "\n".join(out)

    body = _html.escape(body)
    body = _BULLET.sub(r"\1· ", body)
    body = _CODE.sub(lambda m: f"<code>{m.group(1)}</code>", body)
    body = _BOLD.sub(lambda m: f"<b>{m.group(1) or m.group(2)}</b>", body)
    body = _ITALIC.sub(lambda m: f"<i>{m.group(1)}</i>", body)
    # Headings last, and any bold inside one is dropped rather than nested:
    # "# 结论：整体在**小幅上涨**" would otherwise emit <b>…<b>…</b></b>.
    body = _HEADING.sub(
        lambda m: "<b>" + m.group(2).replace("<b>", "").replace("</b>", "") + "</b>",
        body)

    for index, rendered in enumerate(blocks):
        body = body.replace(f"\x00{index}\x00", rendered)
    return body


_TAG = re.compile(r"<[^>]+>")


def to_plain_text(text: str) -> str:
    """Last resort when Telegram rejects the markup: readable, not raw tags."""
    return _html.unescape(_TAG.sub("", text or "")).strip()
