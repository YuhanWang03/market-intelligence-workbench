"""Monkey-patch v2 modules so external calls emit trace events.

Design:
- Nothing is patched at import time. The dashboard backend explicitly calls
  install_all() during startup.
- Each individual patch is wrapped in try/except. A missing optional module
  (e.g. v2.data on a machine where only the bot is installed) reduces the
  hook count but never raises.
- install_all() is idempotent: a sentinel attribute on the patched function
  prevents double-wrapping if called twice (which would silently double-emit
  every event).
- All wrappers check current_trace() first. If no trace is bound (production
  path), the wrapper is a thin pass-through that runs the original function
  unchanged.

Returns a list[str] of hook names actually installed so callers can log /
assert in tests.
"""

from __future__ import annotations

import functools
import importlib
import logging
import time
from typing import Any, Callable

from v2.observability import pricing
from v2.observability.trace import current_trace, emit

logger = logging.getLogger(__name__)

_INSTALLED_SENTINEL = "__v2_observability_wrapped__"
_installed: list[str] = []


def installed_hooks() -> list[str]:
    """Names of hooks that were installed on the most recent install_all()."""
    return list(_installed)


def _already_wrapped(fn: Callable) -> bool:
    return getattr(fn, _INSTALLED_SENTINEL, False)


def _mark_wrapped(wrapper: Callable, original: Callable) -> Callable:
    setattr(wrapper, _INSTALLED_SENTINEL, True)
    setattr(wrapper, "__wrapped_original__", original)
    return wrapper


def _patch_method(cls: type, name: str, make_wrapper: Callable[[Callable], Callable]) -> bool:
    """Replace cls.name with make_wrapper(original). Returns True if patched."""
    original = getattr(cls, name, None)
    if original is None:
        return False
    if _already_wrapped(original):
        return False
    wrapper = _mark_wrapped(make_wrapper(original), original)
    setattr(cls, name, wrapper)
    return True


# ---------------------------------------------------------------------------
# FD client (CachedFDClient.get_* methods)
# ---------------------------------------------------------------------------

def _wrap_fd_method(endpoint_name: str):
    def make_wrapper(original):
        @functools.wraps(original)
        def wrapper(self, *args, **kwargs):
            trace = current_trace()
            if trace is None:
                return original(self, *args, **kwargs)

            t0 = time.perf_counter()
            cache_state = "unknown"
            num_results: int | None = None
            error: str | None = None
            try:
                result = original(self, *args, **kwargs)
                cache_state = "hit" if getattr(self, "_last_was_cache_hit", False) else "miss"
                # Best-effort row count — helps debug "empty return" cases
                # like "FD returned [] in 3ms ⇒ responder says No price data".
                try:
                    if result is None:
                        num_results = 0
                    elif hasattr(result, "__len__"):
                        num_results = len(result)
                except Exception:
                    pass
                return result
            except Exception as exc:
                # Capture the exception type + message so the trace shows
                # WHY the call returned nothing instead of swallowing it.
                error = f"{type(exc).__name__}: {str(exc)[:200]}"
                raise
            finally:
                elapsed_ms = int((time.perf_counter() - t0) * 1000)
                ticker = (
                    args[0] if args else kwargs.get("ticker") or kwargs.get("symbol")
                )
                # Surface a human-readable hint when the call returned
                # nothing — by far the most common cause is FD coverage
                # lag (end_date past the coverage window → HTTP 400 →
                # empty list). The dashboard's TraceEvent renders this
                # as an amber warning row.
                # Phase 4.5-mini removed the old "FD 返回空 ⇒ likely
                # date-lag" hint: prices now go through
                # v2.data.price_source (yfinance), so empty FD results
                # only come from get_earnings / get_financial_metrics /
                # get_insider_trades where date-lag isn't the dominant
                # cause. The amber-warning trace row still fires below;
                # the dashboard surfaces num_results=0 without a
                # misleading explanation.
                emit(
                    "api_call",
                    provider="fd",
                    endpoint=endpoint_name,
                    ticker=ticker,
                    cache=cache_state,
                    num_results=num_results,
                    error=error,
                    hint=None,
                    elapsed_ms=elapsed_ms,
                )
        return wrapper
    return make_wrapper


def _patch_fd_client() -> int:
    """Patch every public get_* / fetch_* method on CachedFDClient.

    The README documents v2.data as the FD client home but actual class
    layout may evolve; we discover methods by prefix rather than hardcoding.
    """
    try:
        mod = importlib.import_module("v2.data")
    except Exception as exc:
        logger.debug("v2.data not importable, skipping FD hooks: %s", exc)
        return 0

    count = 0
    for cls_name in ("CachedFDClient", "FDClient"):
        cls = getattr(mod, cls_name, None)
        if cls is None:
            continue
        for attr in dir(cls):
            if not (attr.startswith("get_") or attr.startswith("fetch_")):
                continue
            method = getattr(cls, attr, None)
            if not callable(method):
                continue
            if _patch_method(cls, attr, _wrap_fd_method(f"{cls_name}.{attr}")):
                count += 1
    return count


# ---------------------------------------------------------------------------
# DeepSeek LLM (langchain_deepseek.ChatDeepSeek.invoke / .ainvoke)
# ---------------------------------------------------------------------------

# Single source of truth for LLM-role fingerprints. Keyed by a unique
# substring from each v2/ prompt; the matching role drives both the
# dashboard's PipelineBar pill highlight (verifier / generator / etc.)
# and the 📖 解析 disclosure lookup.
#
# Both fired-at-emit-time (this file) and fired-at-load-time (the
# dashboard's auto_push push_trace endpoint, via event_explanations.py)
# import from here so cron-captured trace_json carries .role too.
LLM_ROLE_FINGERPRINTS: tuple[tuple[str, str], ...] = (
    # Most distinctive substring first — earlier matches win.
    ("意图分类器",         "intent_classifier"),
    ("严苛的金融分析师",   "verifier"),
    ("归因理由的因果链",   "verifier"),
    ("股票异动归因分析师", "generator"),
    ("机构持仓分析师",     "interpret_changes"),
    # v2/lateral/discover.py — "...给定一组种子股票..."
    ("种子股票",           "proposer"),
    # v2/screening/narrator.py — "...给出 bull + bear 各一句..."
    ("bull + bear",        "narrator"),
    # v2/earnings/summarizer.py — post-release bull/bear/narrative
    ("财报发布总结分析师", "earnings_summarizer"),
    # v2/sec/ner_5_02.py — 8-K item 5.02 NER (CEO/CFO departures + appointments)
    ("8-K 5.02 高管变动信息抽取器", "sec_5_02_extractor"),
    # v2/macro/summarizer.py — Phase 4 macro release LLM template-fill
    ("宏观数据解读分析师",         "macro_summarizer"),
)


def detect_llm_role(prompt: str | None) -> str | None:
    """Return the v2 role name (intent_classifier / verifier / ...) for
    a prompt, or None if no fingerprint matches.

    Exposed as a public function so event_explanations.py and any
    future consumer can share the same fingerprint list.
    """
    if not prompt:
        return None
    for needle, role in LLM_ROLE_FINGERPRINTS:
        if needle in prompt:
            return role
    return None

# Prompts and responses are forwarded to the dashboard verbatim up to this
# many characters. The hedge-fund agents' real prompts run 1.5–4 KB; the
# largest (Generator with Tier-1/2 evidence) reaches ~10 KB. 16 KB leaves
# headroom for outliers without risking SSE buffer pressure.
_PREVIEW_CAP = 16 * 1024


def _cap_str(s: str, limit: int = _PREVIEW_CAP) -> str:
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n... (truncated, {len(s) - limit} more chars)"


def _llm_summarize(messages: Any) -> str:
    """Best-effort textual rendering of a LangChain message list / string."""
    try:
        if isinstance(messages, str):
            return _cap_str(messages)
        parts = []
        for m in messages if isinstance(messages, (list, tuple)) else [messages]:
            content = getattr(m, "content", None) or str(m)
            parts.append(str(content))
        return _cap_str("\n".join(parts))
    except Exception:
        return "<unprintable>"


def _llm_response_preview(result: Any) -> tuple[str, int, int]:
    """Extract (preview, input_tokens, output_tokens) from a LangChain result."""
    preview = ""
    in_tok = 0
    out_tok = 0
    try:
        content = getattr(result, "content", None)
        if content is not None:
            preview = _cap_str(str(content))
        else:
            preview = _cap_str(str(result))
        meta = getattr(result, "usage_metadata", None) or {}
        in_tok = int(meta.get("input_tokens", 0) or 0)
        out_tok = int(meta.get("output_tokens", 0) or 0)
        if in_tok == 0 and out_tok == 0:
            # Older LangChain: response_metadata.token_usage
            rm = getattr(result, "response_metadata", None) or {}
            usage = rm.get("token_usage") or rm.get("usage") or {}
            in_tok = int(usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0) or 0)
            out_tok = int(usage.get("completion_tokens", 0) or usage.get("output_tokens", 0) or 0)
    except Exception:
        pass
    return preview, in_tok, out_tok


def _wrap_llm_invoke(original):
    @functools.wraps(original)
    def wrapper(self, messages, *args, **kwargs):
        trace = current_trace()
        if trace is None:
            return original(self, messages, *args, **kwargs)

        prompt_preview = _llm_summarize(messages)
        model = getattr(self, "model_name", None) or getattr(self, "model", "deepseek-chat")
        t0 = time.perf_counter()
        result = original(self, messages, *args, **kwargs)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        response_preview, in_tok, out_tok = _llm_response_preview(result)
        cost = pricing.deepseek_cost(in_tok, out_tok)
        # Attach role here (rather than in the dashboard executor's sink)
        # so cron paths going through capture_trace_with_framing also get
        # role-tagged llm_call events — the executor.sink no longer has
        # a chance to enrich them in that flow.
        role = detect_llm_role(prompt_preview)
        emit(
            "llm_call",
            provider="deepseek",
            model=str(model),
            prompt_preview=prompt_preview,
            response_preview=response_preview,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=round(cost, 6),
            elapsed_ms=elapsed_ms,
            role=role,
        )
        return result
    return wrapper


def _patch_deepseek() -> int:
    try:
        mod = importlib.import_module("langchain_deepseek")
    except Exception as exc:
        logger.debug("langchain_deepseek not importable: %s", exc)
        return 0

    cls = getattr(mod, "ChatDeepSeek", None)
    if cls is None:
        return 0
    count = 0
    if _patch_method(cls, "invoke", lambda orig: _wrap_llm_invoke(orig)):
        count += 1
    return count


# ---------------------------------------------------------------------------
# Tavily search (tavily.TavilyClient.search)
# ---------------------------------------------------------------------------

def _wrap_tavily_search(original):
    @functools.wraps(original)
    def wrapper(self, query, *args, **kwargs):
        trace = current_trace()
        if trace is None:
            return original(self, query, *args, **kwargs)
        t0 = time.perf_counter()
        num_results: int | None = None
        try:
            result = original(self, query, *args, **kwargs)
            if isinstance(result, dict):
                items = result.get("results")
                if isinstance(items, list):
                    num_results = len(items)
            return result
        finally:
            emit(
                "api_call",
                provider="tavily",
                endpoint="search",
                query=str(query)[:200],
                num_results=num_results,
                cost_usd=round(pricing.tavily_cost(1), 6),
                elapsed_ms=int((time.perf_counter() - t0) * 1000),
            )
    return wrapper


def _patch_tavily() -> int:
    try:
        mod = importlib.import_module("tavily")
    except Exception as exc:
        logger.debug("tavily not importable: %s", exc)
        return 0
    cls = getattr(mod, "TavilyClient", None)
    if cls is None:
        return 0
    return 1 if _patch_method(cls, "search", lambda orig: _wrap_tavily_search(orig)) else 0


# ---------------------------------------------------------------------------
# Intent classifier (v2.bot.intent.classify)
# ---------------------------------------------------------------------------

def _patch_intent_classifier() -> int:
    try:
        mod = importlib.import_module("v2.bot.intent")
    except Exception as exc:
        logger.debug("v2.bot.intent not importable: %s", exc)
        return 0

    # Real entry point is `classify`; older drafts named it `classify_intent`.
    # Patch whichever exists.
    fn_name = "classify" if hasattr(mod, "classify") else "classify_intent"
    fn = getattr(mod, fn_name, None)
    if fn is None or _already_wrapped(fn):
        return 0

    @functools.wraps(fn)
    def wrapper(text, *args, **kwargs):
        trace = current_trace()
        if trace is None:
            return fn(text, *args, **kwargs)
        t0 = time.perf_counter()
        result = fn(text, *args, **kwargs)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        # The production classifier returns a dict like
        # {"intent": "thirteen_f", "ticker": "...", "manager": "brk", ...}.
        # Older code used dataclass attribute access; support both.
        intent_name = None
        intent_args: Any = None
        if isinstance(result, dict):
            intent_name = result.get("intent") or result.get("name")
            # Surface every non-empty arg field except intent itself.
            intent_args = {
                k: v for k, v in result.items()
                if k not in {"intent", "name", "raw"} and v not in (None, "", 0, 0.0)
            }
        else:
            intent_name = getattr(result, "intent", None) or getattr(result, "name", None)
            intent_args = getattr(result, "args", None)

        emit(
            "intent_classified",
            input_text=_cap_str(text, 2048),
            intent=intent_name,
            args=intent_args,
            elapsed_ms=elapsed_ms,
        )
        return result

    setattr(mod, fn_name, _mark_wrapped(wrapper, fn))
    return 1


# ---------------------------------------------------------------------------
# EDGAR (edgartools.Company / .get_filings)
# ---------------------------------------------------------------------------

def _wrap_edgar_method(class_name: str, method_name: str):
    def make_wrapper(original):
        @functools.wraps(original)
        def wrapper(self, *args, **kwargs):
            trace = current_trace()
            if trace is None:
                return original(self, *args, **kwargs)
            t0 = time.perf_counter()
            ident = getattr(self, "cik", None) or getattr(self, "name", None) or "?"
            try:
                return original(self, *args, **kwargs)
            finally:
                emit(
                    "api_call",
                    provider="edgar",
                    endpoint=f"{class_name}.{method_name}",
                    ticker=str(ident)[:40],
                    elapsed_ms=int((time.perf_counter() - t0) * 1000),
                )
        return wrapper
    return make_wrapper


def _patch_edgar() -> int:
    try:
        mod = importlib.import_module("edgar")
    except Exception as exc:
        logger.debug("edgartools not importable: %s", exc)
        return 0
    count = 0
    cls = getattr(mod, "Company", None)
    if cls is not None:
        for attr in ("get_filings", "get_facts", "get_financials", "latest"):
            if _patch_method(cls, attr, _wrap_edgar_method("Company", attr)):
                count += 1
    cls = getattr(mod, "Filing", None)
    if cls is not None:
        for attr in ("obj", "xbrl", "html", "text"):
            if _patch_method(cls, attr, _wrap_edgar_method("Filing", attr)):
                count += 1
    return count


# ---------------------------------------------------------------------------
# Alpaca TradingClient (alpaca-py)
# ---------------------------------------------------------------------------

def _wrap_alpaca_method(method_name: str):
    def make_wrapper(original):
        @functools.wraps(original)
        def wrapper(self, *args, **kwargs):
            trace = current_trace()
            if trace is None:
                return original(self, *args, **kwargs)
            t0 = time.perf_counter()
            try:
                return original(self, *args, **kwargs)
            finally:
                emit(
                    "api_call",
                    provider="alpaca",
                    endpoint=method_name,
                    elapsed_ms=int((time.perf_counter() - t0) * 1000),
                )
        return wrapper
    return make_wrapper


def _patch_alpaca() -> int:
    try:
        mod = importlib.import_module("alpaca.trading.client")
    except Exception as exc:
        logger.debug("alpaca-py not importable: %s", exc)
        return 0
    cls = getattr(mod, "TradingClient", None)
    if cls is None:
        return 0
    count = 0
    for attr in ("get_account", "get_all_positions", "get_open_position", "get_orders"):
        if _patch_method(cls, attr, _wrap_alpaca_method(attr)):
            count += 1
    return count


# ---------------------------------------------------------------------------
# Archive store (v2.archive.store.log_*  write functions)
# ---------------------------------------------------------------------------

def _wrap_db_write(db_name: str):
    def make_wrapper(original):
        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            trace = current_trace()
            if trace is None:
                return original(*args, **kwargs)
            t0 = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                emit(
                    "db_write",
                    db=db_name,
                    fn=getattr(original, "__name__", "<unknown>"),
                    elapsed_ms=int((time.perf_counter() - t0) * 1000),
                )
        return wrapper
    return make_wrapper


def _patch_module_db_writes(module_path: str, db_label: str, prefixes: tuple[str, ...]) -> int:
    """Patch every function in `module_path` whose name starts with one of
    the prefixes (typically save_, log_, insert_, write_, store_, remember).
    """
    try:
        mod = importlib.import_module(module_path)
    except Exception as exc:
        logger.debug("%s not importable: %s", module_path, exc)
        return 0
    count = 0
    for attr in dir(mod):
        if not attr.startswith(prefixes):
            continue
        fn = getattr(mod, attr)
        if not callable(fn) or _already_wrapped(fn):
            continue
        wrapper = _mark_wrapped(_wrap_db_write(db_label)(fn), fn)
        setattr(mod, attr, wrapper)
        count += 1
    return count


def _patch_class_db_methods(
    module_path: str, class_name: str, db_label: str, methods: tuple[str, ...]
) -> int:
    try:
        mod = importlib.import_module(module_path)
    except Exception as exc:
        logger.debug("%s not importable: %s", module_path, exc)
        return 0
    cls = getattr(mod, class_name, None)
    if cls is None:
        return 0
    count = 0
    for name in methods:
        if _patch_method(cls, name, _wrap_db_write(db_label)):
            count += 1
    return count


def _patch_archive_store() -> int:
    write_prefixes = ("log_", "insert_", "write_", "save_", "store_", "remember")
    n = 0
    n += _patch_module_db_writes("v2.archive.store", "archive", write_prefixes)
    n += _patch_module_db_writes("v2.institutional.tracker", "edgar.db", write_prefixes)
    n += _patch_module_db_writes("v2.etf.tracker", "etf.db", write_prefixes)
    n += _patch_module_db_writes("v2.bot.state", "bot_state.db", write_prefixes)
    n += _patch_module_db_writes("v2.streamer.tracker", "options.db", write_prefixes)
    n += _patch_module_db_writes("v2.screening.cache", "screening_cache.db", write_prefixes)
    # ChromaDB-backed memory uses a class with .remember / .add etc.
    n += _patch_class_db_methods(
        "v2.memory.store", "AnomalyMemory", "chroma", ("remember", "add", "store"),
    )
    return n


# ---------------------------------------------------------------------------
# Public installer
# ---------------------------------------------------------------------------

def install_all() -> list[str]:
    """Install all monkey-patches. Idempotent. Returns list of hook tags."""
    global _installed
    _installed = []

    matrix = [
        ("fd", _patch_fd_client),
        ("deepseek", _patch_deepseek),
        ("tavily", _patch_tavily),
        ("edgar", _patch_edgar),
        ("alpaca", _patch_alpaca),
        ("intent", _patch_intent_classifier),
        ("db_writes", _patch_archive_store),
    ]
    for tag, fn in matrix:
        try:
            n = fn()
        except Exception as exc:
            logger.warning("observability hook %s failed: %s", tag, exc)
            n = 0
        if n > 0:
            _installed.append(f"{tag}:{n}")

    logger.info("observability hooks installed: %s", _installed or "<none>")
    return list(_installed)
