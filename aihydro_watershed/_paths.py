"""Filesystem roots shared across aihydro-watershed (one place that honours ``AIHYDRO_HOME``)."""

from __future__ import annotations

import os
from pathlib import Path


def aihydro_home() -> Path:
    """Root for caches and staged data: ``$AIHYDRO_HOME`` if set, else ``~/.aihydro``."""
    env = os.environ.get("AIHYDRO_HOME")
    return Path(env) if env else Path.home() / ".aihydro"
