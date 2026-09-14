"""LLM client for the agent loop — OpenAI-compatible, stdlib only.

Why not LangChain
-----------------
The bot uses ``ChatDeepSeek`` for a single classification call, which is a fine
fit for one-shot work. An agent loop needs the opposite of an abstraction: exact
control over the message list, the tool-call payloads, and the token accounting,
because those three things *are* the engineering problem. So this speaks the
OpenAI ``/chat/completions`` wire format directly over ``urllib`` — no new
dependency, no framework upgrade risk, and every byte on the wire is inspectable.

That choice also buys provider portability for free. Anything OpenAI-compatible
works by pointing ``AGENT_LLM_BASE_URL`` at it:

    DeepSeek   https://api.deepseek.com/v1          (default)
    Groq       https://api.groq.com/openai/v1       (free tier)
    Gemini     https://generativelanguage.googleapis.com/v1beta/openai
    Ollama     http://localhost:11434/v1            (local, zero cost)
    OpenAI     https://api.openai.com/v1
"""

from __future__ import annotations

import json
import re
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ToolCall:
    """One tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str = ""
    parse_error: str = ""


@dataclass
class LLMResponse:
    """One assistant turn: prose, tool calls, or both."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = ""
    latency_ms: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMClient(Protocol):
    """Everything the loop needs from a model."""

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse: ...


#: How many tries a 429 gets, and how long between them: 2, 4, 8, 16, 30, 30 s
#: — a tokens-per-minute window has to actually roll over.
RATE_LIMIT_ATTEMPTS = 6
RATE_LIMIT_CAP_SECONDS = 30.0


def backoff_seconds(attempt: int, *, rate_limited: bool) -> float:
    """Sleep before retry number ``attempt`` (1-based)."""
    if rate_limited:
        return min(RATE_LIMIT_CAP_SECONDS, 2.0 ** attempt)
    return 2.0 ** (attempt - 1)


def _redact(text: str, secret: str) -> str:
    """Replace a credential inside an error message with a stub."""
    if secret and secret in text:
        text = text.replace(secret, f"{secret[:6]}…{secret[-4:]}")
    return re.sub(r"Bearer\s+\S+", "Bearer ***", text)


class LLMError(RuntimeError):
    """Transport or protocol failure the loop cannot turn into an observation."""


def _parse_arguments(raw: str) -> tuple[dict[str, Any], str]:
    """Parse a tool-call argument blob, tolerating the usual model output quirks."""
    text = (raw or "").strip()
    if not text:
        return {}, ""
    if text.startswith("```"):
        lines = text.split("\n")[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return {}, f"invalid JSON arguments ({exc.msg})"
    if not isinstance(parsed, dict):
        return {}, "arguments must be a JSON object"
    return parsed, ""


class OpenAICompatLLM:
    """Chat-completions client with tool calling, retries and token accounting."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        timeout: float = 90.0,
        max_retries: int = 3,
    ) -> None:
        self.model = model or os.environ.get("AGENT_LLM_MODEL", "deepseek-v4-flash")
        self.base_url = (base_url or os.environ.get("AGENT_LLM_BASE_URL")
                         or "https://api.deepseek.com/v1").rstrip("/")
        # .strip(): a key copied out of a CRLF .env carries a trailing \r, and
        # http.client rejects the header — after the sweep has already run
        # every case three times. Seen once; cost 387 s and printed the key.
        self.api_key = (api_key or os.environ.get("AGENT_LLM_API_KEY")
                        or os.environ.get("DEEPSEEK_API_KEY")
                        or os.environ.get("OPENAI_API_KEY") or "").strip()
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if tools:
            payload["tools"] = tools
            # ``tool_choice`` may name one function ({"type": "function", "function": {"name": ...}}) to force it.
            payload["tool_choice"] = tool_choice or "auto"

        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        started = time.time()
        last_error: Exception | None = None
        attempts = self.max_retries
        attempt = 0
        while attempt < attempts:
            accounted = False
            try:
                request = urllib.request.Request(
                    f"{self.base_url}/chat/completions",
                    data=body, headers=headers, method="POST",
                )
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                from urllib.parse import urlparse
                from v2.data.usage_ledger import record_llm
                from v2.usage_context import current_source
                record_llm(data, self.model, 'DeepSeek' if urlparse(self.base_url).hostname == 'api.deepseek.com' else 'Other LLM', current_source('agent.chat'))
                accounted = True
                return self._to_response(data, int((time.time() - started) * 1000))
            except urllib.error.HTTPError as exc:
                self._record_unknown_attempt()
                detail = exc.read().decode("utf-8", "replace")[:400]
                last_error = LLMError(f"HTTP {exc.code} from {self.base_url}: {detail}")
                # 4xx other than rate-limit will not fix themselves on retry.
                if exc.code < 500 and exc.code != 429:
                    raise last_error from exc
                if exc.code == 429:
                    # A tokens-per-minute limit clears on its own; 1 s, 2 s was
                    # not enough for it to. A 200k-TPM account turned a full
                    # sweep into 27 `error` flakes and a 63% agent score.
                    attempts = max(attempts, RATE_LIMIT_ATTEMPTS)
            except Exception as exc:  # noqa: BLE001 — network flakiness
                if not accounted:
                    self._record_unknown_attempt()
                # Never let the credential into a message that ends up in an
                # eval report or a chat log.
                last_error = LLMError(_redact(str(exc), self.api_key))
            attempt += 1
            if attempt < attempts:
                time.sleep(backoff_seconds(attempt, rate_limited=attempts > self.max_retries))

        raise LLMError(f"LLM call failed after {attempts} attempts: {last_error}")

    def _record_unknown_attempt(self):
        from urllib.parse import urlparse
        from v2.data.usage_ledger import record
        from v2.usage_context import current_source
        record('llm', 'DeepSeek' if urlparse(self.base_url).hostname == 'api.deepseek.com' else 'Other LLM',
               self.model, {}, source=current_source('agent.chat'), state='failed', usage_basis='unknown')

    @staticmethod
    def _to_response(data: dict[str, Any], latency_ms: int) -> LLMResponse:
        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"no choices in response: {str(data)[:300]}")
        message = choices[0].get("message") or {}
        usage = data.get("usage") or {}

        calls: list[ToolCall] = []
        for index, item in enumerate(message.get("tool_calls") or []):
            function = item.get("function") or {}
            raw = function.get("arguments", "") or ""
            parsed, error = _parse_arguments(raw)
            calls.append(ToolCall(
                id=item.get("id") or f"call_{index}",
                name=function.get("name", ""),
                arguments=parsed,
                raw_arguments=raw,
                parse_error=error,
            ))

        return LLMResponse(
            text=(message.get("content") or "").strip(),
            tool_calls=calls,
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            finish_reason=choices[0].get("finish_reason", "") or "",
            latency_ms=latency_ms,
        )


class ScriptedLLM:
    """Deterministic stand-in that replays a fixed list of responses.

    Tests for an agent loop must not depend on a model's mood. Scripting the
    model turns 'does the loop recover from a tool error' into an assertion
    instead of an anecdote.
    """

    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[list[dict[str, Any]]] = []
        #: The tool_choice each call carried, when any.
        self.tool_choices: list[Any] = []

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
    ) -> LLMResponse:
        self.calls.append(list(messages))
        self.tool_choices.append(tool_choice)
        if not self.responses:
            return LLMResponse(text="(scripted LLM exhausted)", finish_reason="stop")
        return self.responses.pop(0)


def build_llm(**kwargs: Any) -> LLMClient:
    """Construct the configured client. Env-driven so ops can swap providers."""
    return OpenAICompatLLM(**kwargs)


def describe_provider() -> str:
    client = OpenAICompatLLM()
    key_state = "set" if client.api_key else "MISSING"
    return f"{client.model} @ {client.base_url} (api key: {key_state})"
