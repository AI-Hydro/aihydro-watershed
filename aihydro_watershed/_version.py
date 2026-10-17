"""Version recorded into result provenance.

A result states the identity of the code that produced it: this package's
imported ``__version__``, never distribution metadata. An editable install's
``dist-info`` goes stale when the source is bumped, and a sealed record would
then state a version the code never was (Wave U proof 3, E7).

This package states only its own version. It does not look at, import or
guess the version of the tools layer above it: the serving host's run record
states that itself (``ai_hydro.__version__``) and keeps this one as the
producer version. Distribution metadata (``aihydro-watershed`` only) is read
solely to detect disagreement: when it differs from the module version both
are reported with ``version_mismatch``.
"""

from __future__ import annotations

from typing import Any, Dict

_DIST = "aihydro-watershed"


def _dist_version() -> "str | None":
    try:
        from importlib.metadata import version
        return version(_DIST)
    except Exception:
        return None


def version_info() -> Dict[str, Any]:
    """``{package, version, version_source, module_version, distribution_version, version_mismatch}``."""
    try:
        import aihydro_watershed
        module_v = getattr(aihydro_watershed, "__version__", None)
    except Exception:
        module_v = None
    if not (isinstance(module_v, str) and module_v and module_v != "unknown"):
        module_v = None
    dist_v = _dist_version()
    return {
        "package": _DIST,
        "version": module_v or "unknown",
        "version_source": "module" if module_v else "unavailable",
        "module_version": module_v,
        "distribution_version": dist_v,
        "version_mismatch": module_v is not None and dist_v is not None and dist_v != module_v,
    }


def tool_version() -> str:
    """The version to stamp into ``HydroMeta.version``."""
    return version_info()["version"]


def version_params() -> Dict[str, Any]:
    """Extra ``HydroMeta.params`` entries: non-empty only on a version mismatch."""
    info = version_info()
    return {"version_check": info} if info["version_mismatch"] else {}
