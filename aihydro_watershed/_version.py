"""Version recorded into result provenance.

The version stamped on a result is that of the code that ran: an imported
module's ``__version__``, never distribution metadata. An editable install's
``dist-info`` goes stale when the source is bumped, and a sealed record would
then state a version the code never was (Wave U proof 3, E7).

Results are attributed to the ``aihydro-tools`` release that serves them. This
package must not import the tools layer above it, so it only *looks* at
``sys.modules["ai_hydro"]`` (present when hosted by the tools MCP server; never
imported here). If the host is not loaded, this package's own ``__version__``
is stamped and ``version_source`` says so. Distribution metadata is read only
to detect disagreement: when it differs from the module version both are
reported with ``version_mismatch``.
"""


from __future__ import annotations

import sys
from typing import Any, Dict

_OWN_DIST = "aihydro-watershed"
_HOST_MODULE, _HOST_DIST = "ai_hydro", "aihydro-tools"


def _dist_version(dist: str) -> "str | None":
    try:
        from importlib.metadata import version
        return version(dist)
    except Exception:
        return None


def _str_or_none(v: Any) -> "str | None":
    return v if isinstance(v, str) and v and v != "unknown" else None


def version_info() -> Dict[str, Any]:
    """``{package, version, version_source, module_version, distribution_version, version_mismatch}``."""
    host_v = _str_or_none(getattr(sys.modules.get(_HOST_MODULE), "__version__", None))
    if host_v is not None:
        package, module_v, dist = _HOST_DIST, host_v, _HOST_DIST
        source = "host_module"
    else:
        try:
            import aihydro_watershed
            module_v = _str_or_none(getattr(aihydro_watershed, "__version__", None))
        except Exception:
            module_v = None
        package, dist = _OWN_DIST, _OWN_DIST
        source = "module" if module_v else "unavailable"
    dist_v = _dist_version(dist)
    return {
        "package": package,
        "version": module_v or "unknown",
        "version_source": source,
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
