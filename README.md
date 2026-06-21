# aihydro-watershed

**Watershed delineation + characterization for any point on Earth — as a standalone package.**

Delineate a watershed from a USGS gauge or any global lat/lon pour point, then
characterize it (geomorphics, topographic wetness index, terrain/runoff,
watershed-scale hydrological signatures) — without installing the full AI-Hydro
MCP hydrology stack.

Part of the [AI-Hydro](https://github.com/AI-Hydro) ecosystem. Carved out of
`aihydro-tools` in Wave A of the [ecosystem roadmap](../docs/ECOSYSTEM_ROADMAP.md).

## Install

```bash
# Delineation (fast tier: cloud DEM + pysheds + MERIT-Hydro/pyflwdir)
pip install "aihydro-watershed[delineation]"

# Characterization + signatures
pip install "aihydro-watershed[analysis]"

# Everything
pip install "aihydro-watershed[all]"
```

On Python 3.13, the `analysis` extra omits `xrspatial` (no wheel); use
`analysis-legacy` on Python 3.10–3.12 if you need TWI via xarray-spatial.

## Quick start

```python
from aihydro_watershed.delineation.router import delineate_from_point

# CONUS pour point — uses NLDI automatically
result = delineate_from_point(40.7128, -74.0060, method="auto")
print(result.data["area_km2"])          # 1234.5
print(result.data["method_used"])       # "nldi_comid"

# Global pour point (Rhine at Cologne) — falls back to fast DEM tier
result = delineate_from_point(50.932, 6.970, method="auto")
print(result.data["area_km2"])

# Expected drainage area known? Pass it to improve COMID selection
result = delineate_from_point(39.27, -77.54, expected_area_km2=25_000)
```

Results are typed `HydroResult` (from `aihydro-core`) — provenance-stamped with
tool path, version, parameters, and data sources.

## What's inside

| Subpackage | Capability |
|---|---|
| `delineation` | Global pour-point / gauge delineation. Three tiers: NLDI (CONUS NHD-indexed), MERIT-Hydro/pyflwdir (local flowdir cache), MERIT-Basins hybrid (vector topology + raster refinement). Auto-routing chooses the best available tier. |
| `merit` | MERIT-Hydro data management: regional basin cache, flowdir rasters, map layers, region presets. |
| `characterize` | Geomorphic parameters (28 indices), watershed attributes, topographic wetness index. |
| `terrain` | Curve number, event runoff (SCS-CN), RUSLE erosion. |
| `signatures` | Baseflow index (Lyne-Hollick), flow-duration curve, flood frequency (Gumbel/GEV), drought indices (SPI, SPEI, Palmer). |

The flood-inundation suite is **not** part of this package.

## Delineation tiers

```
1. nldi        — USGS NLDI NHD-indexed basin (CONUS only, < 1 s)
2. merit_gee   — MERIT-Hydro flowdir from Google Earth Engine (global, cloud)
3. local_merit — MERIT-Hydro flowdir from local cache (global, offline after download)
4. fast        — pysheds on cloud-fetched DEM tile (global, no MERIT cache needed)
auto           — tries tiers in order of accuracy; degrades gracefully on failures
```

## Layering

Depends only **downward**:

```
aihydro-core  ←  contract (HydroResult / HydroMeta / ToolError)
aihydro-data  ←  data acquisition (streamflow, DEM, land cover, soil)
     ↑
aihydro-watershed   (this package)
```

The layering contract is enforced offline by `tests/test_layering.py` (AST walk,
zero deps) and by `import-linter` for anyone who installs the `dev` extra.

## Running tests

```bash
# Offline (layering guard — always passes, no network needed)
pytest tests/ -m "not live" -q

# Full suite including live API calls (requires network)
pytest tests/ -v

# Parity checks only (Wave A4 gate)
pytest tests/test_parity.py -m live -v
```

## Status

v0.1.0. Wave A extraction complete (2026-06-21). All five subpackages
(`delineation`, `merit`, `characterize`, `terrain`, `signatures`) are live.
Compatibility shims remain in `ai_hydro/analysis/` and `ai_hydro/data/` of
`aihydro-tools` for one release cycle; see `MIGRATION.md`.

## Migration from `aihydro-tools`

If you previously imported from `ai_hydro.analysis.*` or `ai_hydro.data.*`,
see [MIGRATION.md](MIGRATION.md) for the one-line import changes.
