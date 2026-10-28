"""
aihydro-watershed — watershed delineation + characterization
============================================================

A standalone, pip-installable package for delineating watersheds anywhere on
Earth and characterizing them — without pulling in the full AI-Hydro MCP
hydrology pack. Carved out of ``aihydro-tools`` per ADR-002.

Subpackages
-----------
- ``delineation`` — global pour-point / gauge delineation. NLDI (CONUS) +
  MERIT-Hydro/pyflwdir (global) + MERIT-Basins hybrid backends, with a router
  that picks the right method automatically.
- ``merit``        — MERIT-Hydro data management (download, regional cache, map
  layers, region presets).
- ``characterize`` — watershed characterization: geomorphic parameters,
  watershed attributes, topographic wetness index.
- ``terrain``      — terrain / runoff intelligence: curve number, event runoff,
  erosion.
- ``signatures``   — watershed-scale hydrological signatures (baseflow index,
  flow-duration curve, flood frequency, drought indices). Consumes streamflow
  from ``aihydro-data``.

Layering
--------
Depends DOWN on ``aihydro-core`` (the ``HydroResult`` contract) and
``aihydro-data`` (data acquisition). Never imports the ``ai_hydro`` tools pack.

NOTE: The public API below is wired in Wave A2 (module migration). Until then
this package exposes only ``__version__``; importing the subpackages directly
is the interim path.
"""

from __future__ import annotations

from aihydro_watershed._numpy_compat import ensure_in1d as ensure_numpy_compat

__version__ = "0.2.1"

# pysheds (<= 0.5) calls np.in1d, missing from NumPy 2.4+. Applying the idempotent
# shim at import time covers every process that imports this package, including
# callers that import pysheds directly (e.g. the tools pack). See _numpy_compat.
ensure_numpy_compat()

# Public API — populated in Wave A2 as modules land in their subpackages.
# Kept intentionally lazy (no eager submodule imports) so `import aihydro_watershed`
# stays cheap and does not require the heavy delineation extras until a real
# entry point is called.
__all__: list[str] = ["__version__", "ensure_numpy_compat"]
