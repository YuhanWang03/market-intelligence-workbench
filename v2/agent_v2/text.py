"""Small text helpers shared by adapters and the deterministic synthesizer."""

from __future__ import annotations

import html
import re

_TAG = re.compile(r"<[^>]+>")
_BLANK_RUN = re.compile(r"[ \t]{2,}")


def plain_text(value: str) -> str:
    """Strip HTML tags and entities from a formatted card, keeping its line structure."""

    text = html.unescape(_TAG.sub("", value or ""))
    lines = [_BLANK_RUN.sub(" ", line).strip() for line in text.split("\n")]
    return "\n".join(lines).strip()
