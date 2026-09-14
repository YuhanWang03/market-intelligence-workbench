"""Load .env for v2 tests so FINANCIAL_DATASETS_API_KEY is available.

Also installs a minimal ``edgar`` stub when the real edgartools package
is missing (sandbox / CI without the dep), so test collection inside
``v2/sec/`` doesn't blow up on ``from edgar import ...``. Production
VPS ships the real package and is unaffected.
"""

import sys
import types

from dotenv import load_dotenv

load_dotenv()

from v2.testing_stubs import install_data_stubs  # noqa: E402

install_data_stubs()  # no-op on the VPS, where the real v2/data package exists

try:
    import edgar  # noqa: F401
except ImportError:
    _edgar = types.ModuleType("edgar")
    _edgar.Company = type("Company", (), {})
    _edgar.set_identity = lambda *a, **kw: None
    sys.modules["edgar"] = _edgar

try:
    import langchain_deepseek  # noqa: F401
except ImportError:
    _ld = types.ModuleType("langchain_deepseek")
    _ld.ChatDeepSeek = type("ChatDeepSeek", (), {
        "__init__": lambda self, *a, **kw: None,
        "invoke": lambda self, *a, **kw: types.SimpleNamespace(content="{}"),
    })
    sys.modules["langchain_deepseek"] = _ld

try:
    import tavily  # noqa: F401
except ImportError:
    _tv = types.ModuleType("tavily")
    _tv.TavilyClient = type("TavilyClient", (), {
        "__init__": lambda self, *a, **kw: None,
        "search": lambda self, *a, **kw: {"results": []},
    })
    sys.modules["tavily"] = _tv
