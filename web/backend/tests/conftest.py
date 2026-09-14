"""Make the web backend importable where the production-only ``v2/data``
package is absent (sandbox / CI checkout).

``v2/data/client.py`` and friends ship on the VPS and are git-ignored (see
the root ``.gitignore``).  Several v2 modules import them at module level,
so without them ``app.main`` cannot even be imported.  When — and only
when — the real package is missing, install minimal stand-ins exposing the
names those modules import.  Tests that need real data behaviour
monkeypatch the call sites; nothing here performs I/O.  On a machine with
the real package this file is a no-op.
"""

from __future__ import annotations

from v2.testing_stubs import install_data_stubs  # noqa: E402

install_data_stubs()

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_lab_stores(tmp_path, monkeypatch):
    """Every test gets fresh lab.db / personas.db instead of the repo's data/."""
    monkeypatch.setenv("WEB_LAB_DB", str(tmp_path / "lab.db"))
    monkeypatch.setenv("WEB_PERSONAS_DB", str(tmp_path / "personas.db"))
    from v2.data import cost_ledger
    monkeypatch.setattr(cost_ledger, '_DB_PATH', tmp_path / 'costs.db')
    from app.routers import committee, workspace

    monkeypatch.setattr(workspace, "_LAB_STORE", None)
    monkeypatch.setattr(committee, "_STORE", None)
    yield
