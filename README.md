# aihydro-watershed

**Watershed delineation + characterization for any point on Earth — as a standalone package.**

Delineate a watershed from a USGS gauge or any global lat/lon pour point, then
characterize it (geomorphics, topographic wetness index, terrain/runoff,
watershed-scale hydrological signatures) — without installing the full AI-Hydro
MCP hydrology stack.

Part of the [AI-Hydro](https://github.com/AI-Hydro) ecosystem. Carved out of
`aihydro-tools` so delineation has a clean, focused install (see ADR-002 and
`MCP/docs/ECOSYSTEM_ROADMAP.md`).

## Install

```bash
# Delineation (fast tier: cloud DEM + pysheds + MERIT-Hydro/pyflwdir)
pip install "aihydro-watershed[delineation]"

# Characterization + signatures
pip install "aihydro-watershed[analysis]"

# Everything
pip install "aihydro-watershed[all]"
```

On Python 3.13 the `analysis` extra omits `xrspatial` (no wheel); use
`analysis-legacy` on Python 3.10–3.12 if you need TWI via xarray-spatial.

The accurate MERIT-Basins hybrid tier additionally needs `upstream-delineator`
from GitHub — see install notes (Wave A5).

## What's inside

| Subpackage | Capability |
|---|---|
| `delineation` | Global pour-point / gauge delineation. NLDI (CONUS) + MERIT-Hydro/pyflwdir (global) + MERIT-Basins hybrid; auto-routing. |
| `merit` | MERIT-Hydro data management (regional cache, map layers, region presets). |
| `characterize` | Geomorphic parameters, watershed attributes, topographic wetness index. |
| `terrain` | Curve number, event runoff, erosion. |
| `signatures` | Baseflow index, flow-duration curve, flood frequency, drought indices (consumes streamflow from `aihydro-data`). |

The flood-inundation suite is **not** part of this package (different subsystem).

## Layering

Depends only **downward**: `aihydro-core` (the `HydroResult` contract) and
`aihydro-data` (data acquisition). It never imports the `ai_hydro` tools pack —
enforced by `tests/test_layering.py` and `import-linter`.

## Status

Alpha — extraction in progress (Wave A of the ecosystem roadmap). Public API
lands in Wave A2.
