"""Import the heavy scientific libraries once, in the main thread, before any worker thread needs them.

``v2.data.price_source`` imports yfinance lazily on the first price request.
When the first requests of a process arrive on two executor threads at once
(a fan-out over the holdings, a quality run with ``--parallel 2``), both
threads import yfinance, pandas and numpy together and one of them sees a
partially initialised numpy ("module 'numpy' has no attribute 'matrix'",
"cannot import name 'NDArray' from partially initialized module
'numpy._typing'"), so every market call of that run fails.  Importing them
here first makes the later imports cache hits.
"""

from __future__ import annotations

import importlib
import logging
import threading

logger = logging.getLogger(__name__)

HEAVY_MODULES = ("numpy", "pandas", "scipy", "yfinance", "edgar")
_lock = threading.Lock()
_done: set[str] = set()


def warm_imports(modules: tuple[str, ...] = HEAVY_MODULES) -> list[str]:
    """Import ``modules`` now (best effort, once); returns the names that could not be imported."""

    missing: list[str] = []
    with _lock:
        for name in modules:
            if name in _done:
                continue
            try:
                importlib.import_module(name)
            except Exception as exc:  # noqa: BLE001 — an optional library is simply absent
                missing.append(name)
                logger.debug("warm-up import of %s skipped: %s", name, exc)
            _done.add(name)
    return missing
