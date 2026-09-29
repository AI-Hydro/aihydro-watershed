# aihydro-watershed

<p align="center">
  <a href="https://pypi.org/project/aihydro-watershed/"><img src="https://img.shields.io/pypi/v/aihydro-watershed?color=3775a9&label=PyPI" alt="PyPI" /></a>
  &nbsp;
  <a href="https://doi.org/10.5281/zenodo.20823440"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.20823440.svg" alt="DOI" /></a>
  &nbsp;
  <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License" /></a>
</p>

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

# Road culvert, catchment ~1 km2 (CONUS): USGS 3DEP 10 m + embankment notch
result = delineate_from_point(40.2497, -86.3877, method="small_catchment")
print(result.data["area_km2"], result.data["quality_flags"])
```

Results are typed `HydroResult` (from `aihydro-core`) — provenance-stamped with
tool path, version, parameters, and data sources.

## What's inside

| Subpackage | Capability |
|---|---|
| `delineation` | Global pour-point / gauge delineation. Three tiers: NLDI (CONUS NHD-indexed), MERIT-Hydro/pyflwdir (local flowdir cache), MERIT-Basins hybrid (vector topology + raster refinement). Auto-routing chooses the best available tier. |
| `merit` | MERIT-Hydro data management: regional basin cache, flowdir rasters, map layers, region presets. |
| `characterize` | Geomorphic parameters (28 indices), watershed attributes, topographic wetness index. |
| `terrain` | Curve number (TR-55, AMC II), event runoff (SCS-CN), RUSLE erosion. SSURGO soil attributes in CONUS (recorded hydrologic group and erodibility Kw via gNATSGO + Soil Data Access), POLARIS/SoilGrids elsewhere. |
| `signatures` | Baseflow index (Lyne-Hollick), flow-duration curve, flood frequency (Gumbel/GEV), drought indices (SPI, SPEI, Palmer). |

The flood-inundation suite is **not** part of this package.

## Delineation tiers

```
1. nldi        — USGS NLDI NHD-indexed basin (CONUS only, < 1 s)
2. merit_gee   — MERIT-Hydro flowdir from Google Earth Engine (global, cloud)
3. local_merit — MERIT-Hydro flowdir from local cache (global, offline after download)
4. fast        — pysheds on cloud-fetched DEM tile (global, no MERIT cache needed)
5. small_catchment (alias 3dep): CONUS catchments under ~5 km2 (road
                 culverts). USGS 3DEP 10 m bare-earth DEM, a notch carved
                 through the road embankment at the pour point, snap to the
                 largest drainage within 40 m
auto           — tries tiers in order of accuracy; degrades gracefully on failures
                 (in CONUS, expected_area_km2 < 5 sends it to small_catchment first)
```

Why a separate small-catchment tier: NLDI returns whole NHDPlus catchments
(and the router rejects anything under 1 km2), MERIT-Hydro is 90 m, and the
`fast` tier reads Copernicus GLO-30, a surface model that keeps canopy and
buildings. A road embankment also acts as a dam on any DEM unless it is
notched. On 67 Indiana culverts (INDOT SPR-4926) the small-catchment tier
matched engineer-computed drainage areas within a factor of 2 at 76 % of
sites (median ratio 1.01, Spearman 0.80). Results under 0.05 km2 were always
roadside-ditch snaps and are flagged `LIKELY_DITCH_SNAP`.

USGS StreamStats is not used as a tier: the old
`streamstats.usgs.gov/streamstatsservices` API is retired (404), and the
current `/ss-delineate/v1/delineate/sshydro/{STATE}?lat=&lon=` service does not
snap points onto its stream grid for very small channels.

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
# Offline (layering, geometry/router envelopes, and pure signature kernels)
pytest tests/ -m "not live" -q

# Full suite including live API calls (requires network)
pytest tests/ -v

# Parity checks only (Wave A4 gate)
pytest tests/test_parity.py -m live -v
```

The offline suite covers import layering, delineation failure/escalation
envelopes, geometry validity/GeoJSON conversion, and pure hydrologic signature
kernels. Live tests are reserved for NLDI/GEE/STAC/3DEP/API parity checks.

## Status

v0.1.0. Wave A extraction complete (2026-06-21). All five subpackages
(`delineation`, `merit`, `characterize`, `terrain`, `signatures`) are live.
Compatibility shims remain in `ai_hydro/analysis/` and `ai_hydro/data/` of
`aihydro-tools` for one release cycle; see `MIGRATION.md`.

## Migration from `aihydro-tools`

If you previously imported from `ai_hydro.analysis.*` or `ai_hydro.data.*`,
see [MIGRATION.md](MIGRATION.md) for the one-line import changes.

## Citation

If you use `aihydro-watershed` in your research, please cite:

```bibtex
@software{aihydro_watershed_2026,
  title   = {aihydro-watershed: Global Watershed Delineation and Characterization},
  author  = {Galib, Mohammad and Merwade, Venkatesh},
  year    = {2026},
  version = {0.1.0},
  doi     = {10.5281/zenodo.20823440},
  url     = {https://doi.org/10.5281/zenodo.20823440}
}
```
