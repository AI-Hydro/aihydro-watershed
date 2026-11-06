"""
Deprecated compatibility shim for the removed Google Drive / ``gdown`` downloader.

The old module installed an unhashed shapefile set from a public Drive folder (with a runtime
``pip install gdown`` and a ``git clone`` of delineator). It is replaced by pinned, SHA-256-verified
files; see :mod:`aihydro_watershed.merit.verified_fetch` and :class:`MeritDataManager`. These three
functions keep the names that ``aihydro-tools`` (``scripts/install_merit_minimal.py``) still imports,
stage the verified files instead, and return ``True`` when the file is present and verified.
The ``dest_dir`` / ``level2_dir`` arguments are ignored: files always go under ``$AIHYDRO_HOME/merit``.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

log = logging.getLogger(__name__)


def _warn() -> None:
    warnings.warn(
        "aihydro_watershed.merit.merit_download is deprecated; use MeritDataManager.ensure_basins_region "
        "(pinned, SHA-256-verified files).",
        DeprecationWarning,
        stacklevel=3,
    )


def ensure_level2_index(level2_dir: Path | None = None, *, clone_delineator: bool = True) -> bool:
    """Stage the verified megabasin index (``megabasins.db``). ``clone_delineator`` is ignored."""
    _warn()
    from aihydro_watershed.merit.merit_manager import MeritDataManager

    return MeritDataManager().ensure_level2_index(download=True) is not None


def _stage_region_db(kind: str, pfaf: str) -> bool:
    from aihydro_watershed.merit import verified_fetch as vf
    from aihydro_watershed.merit.merit_manager import MeritDataManager

    mgr = MeritDataManager()
    got = mgr.verified_asset(kind, pfaf)
    if got is None:
        dest = mgr.basins_db_path(pfaf) if kind == "basins" else mgr.rivers_db_path(pfaf)
        got = mgr._stage(vf.region_asset_spec(kind, pfaf), dest)
    return got is not None


def download_river_shapefile(pfaf: str, dest_dir: Path | None = None) -> bool:
    """Stage the verified ``rivers##.db`` for the region (the shapefile layout is no longer used)."""
    _warn()
    return _stage_region_db("rivers", pfaf)


def download_catchment_shapefile(pfaf: str, dest_dir: Path | None = None) -> bool:
    """Stage the verified ``basins##.db`` for the region (the shapefile layout is no longer used)."""
    _warn()
    return _stage_region_db("basins", pfaf)
