"""Keep every test off the real home directory: caches and staged data land in a temp dir."""

from __future__ import annotations

import os
import tempfile

_HOME = tempfile.mkdtemp(prefix="aihydro-ws-tests-")
# Set before any aihydro_watershed module is imported (some create caches at import time).
os.environ["HOME"] = _HOME
os.environ["AIHYDRO_HOME"] = os.path.join(_HOME, ".aihydro")
os.environ.pop("AIHYDRO_MERIT_DIR", None)
os.environ.pop("MERIT_DATA_DIR", None)
os.environ.pop("AIHYDRO_MERIT_BASE_URL", None)


import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _no_bulk_downloads(monkeypatch):
    """Tests must never fetch MERIT data: the default opener raises; tests inject their own."""
    from aihydro_watershed.merit import verified_fetch as vf

    def _blocked(url: str):
        raise AssertionError(f"test attempted a network download: {url}")

    monkeypatch.setattr(vf, "_default_opener", _blocked)
