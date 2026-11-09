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

### Command line (offline)

One console script, `aihydro-watershed`, runs the offline slice of the package: the
CAMELS-style signatures from a daily streamflow CSV you already hold. It makes no
network request and never asks for precipitation (so `runoff_ratio` and
`stream_elas` are null). Delineation needs DEM/NLDI/MERIT services and stays in the
Python API; the CLI takes the polygon it produced as a GeoJSON file.

```bash
aihydro-watershed signatures --flow-csv q.csv --area-km2 250 --geojson basin.geojson
# CSV columns default to `date` and `q_cms` (m3/s); override with --date-col / --flow-col.
# Negative discharge is refused; missing-data codes go in --na-values (default -999,-9999).
# The JSON has a `cli` block: missing/duplicate rows, absent days, warnings (also on stderr).
```

## What's inside

| Subpackage | Capability |
|---|---|
| `delineation` | Global pour-point / gauge delineation. Tiers: NLDI (CONUS NHD-indexed), MERIT-Basins hybrid (default outside CONUS: vector assembly of unit catchments + raster refinement of the terminal catchment), MERIT-Hydro/pyflwdir (local flowdir cache, Earth Engine). Auto-routing chooses the best available tier. |
| `merit` | MERIT-Hydro / MERIT-Basins data management: per-region vectors and rasters pinned by SHA-256, map layers, region presets. |
| `tiles` | Local river PMTiles built from staged MERIT-Basins `rivers##.db` (basemap slice B2). See [River tiles](#river-tiles-merit-basins-pmtiles). |
| `characterize` | Geomorphic parameters (28 indices), watershed attributes, topographic wetness index. |
| `terrain` | Curve number (TR-55, AMC II), event runoff (SCS-CN), RUSLE erosion. SSURGO soil attributes in CONUS (recorded hydrologic group and erodibility Kw via gNATSGO + Soil Data Access), POLARIS/SoilGrids elsewhere. |
| `signatures` | Baseflow index (Lyne-Hollick, Ladson 2013 recipe as used by CAMELS), flow-duration curve, flood frequency (Gumbel/GEV), drought indices (SPI, SPEI, Palmer). |

The flood-inundation suite is **not** part of this package.

## Delineation tiers

```
1. nldi        — USGS NLDI NHD-indexed basin (CONUS only, < 1 s)
2. merit_hybrid (alias merit_basins) — MERIT-Basins unit catchments assembled by
                 upstream traversal; only the terminal unit catchment is refined on
                 a small raster window. DEFAULT global tier (auto outside CONUS);
                 force it anywhere, including CONUS, with method="merit_hybrid".
3. merit_gee   — MERIT-Hydro flowdir from Google Earth Engine (small-basin fallback)
3. local_merit — MERIT-Hydro flowdir from local cache (global, offline after download)
4. fast        — pysheds on cloud-fetched DEM tile (global, no MERIT cache needed)
5. small_catchment (alias 3dep): CONUS catchments under ~5 km2 (road
                 culverts). USGS 3DEP 10 m bare-earth DEM, a notch carved
                 through the road embankment at the pour point, snap to the
                 largest drainage within 40 m
auto           — tries tiers in order of accuracy; degrades gracefully on failures
                 (in CONUS, expected_area_km2 < 5 sends it to small_catchment first)
```

### MERIT-Basins hybrid: data, licence, versions

- **Data.** Per-megabasin files published for Heberger's `delineator` 2.2.x
  (`basins##.db`, `rivers##.db`, optional `flowdir##.tif` / `accum##.tif`,
  plus the 52.9 MB level-2 index `megabasins.db`), pinned by SHA-256 and size in
  `aihydro_watershed/merit/delineator_data_manifest.json`. A file whose digest does
  not match is deleted before install and, if already on disk, refused
  (`MERIT_ASSET_INTEGRITY_FAILED`); nothing falls back to unverified data.
  Vectors are staged on demand under the tiered policy (`AIHYDRO_MERIT_AUTO_STAGE=never|tiered|always`,
  `merit_auto_stage_max_gb`, default 2 GB). Rasters are optional and staged explicitly. Everything lives under
  `$AIHYDRO_HOME/merit` (default `~/.aihydro/merit`); `AIHYDRO_MERIT_DIR` overrides.
- **First-call cost.** The first delineation in a region downloads that region's `basins##.db` + `rivers##.db` (SHA-256
  verified) under the tiered policy (the default, one constant: `DEFAULT_AUTO_STAGE_POLICY` in `merit/merit_manager.py`;
  set it to `"never"` for confirm-first, or use `AIHYDRO_MERIT_AUTO_STAGE=never`). From the pinned manifest, over all 60 regions:
  **min 14 MB (region 54), median 530 MB, max 2015 MB (region 56, Australia, above the 2 GB auto-stage limit so it needs an
  explicit `merit_ensure_basins_region`)**; North America (71-78, 81-86): min 60 MB, median 502 MB, max 916 MB. The very first call
  ever also fetches the 52.9 MB level-2 index. Staged rasters are optional and explicit (flowdir + accum per region: min 12 MB,
  median 696 MB, max 1931 MB). Staging needs 2x the pending bytes free on disk. Every call that downloads says so: quality flag
  `MERIT_AUTO_STAGED`, and `data_provenance.downloaded` lists `{name, bytes, url, sha256}`, with `data_provenance.acquisition_policy`.
  A file that fails verification is deleted and not retried automatically for 24 h (`MERIT_ASSET_INTEGRITY_FAILED`, with the
  observed digest); interrupted downloads resume with an HTTP Range request when the server supports it.
- **Licence.** Non-commercial and share-alike. mghydro.com releases the files under CC BY-NC-SA 4.0; MERIT Hydro is
  CC BY-NC 4.0 or ODbL 1.0; the sources disagree on the exact variant. The result carries the
  licence (`data_provenance.share_alike: true`) and the SHA-256 of every file used (`data_provenance`).
  Legacy shapefile layouts (`shp/merit_catchments`, `shp/merit_rivers`) are not read; `basins_region_cache` says so.
- **Version consistency.** The vectors are MERIT-Basins v0.7/v1.0_bugfix1. With the regional
  rasters staged, snapping and refinement use the same provider's files. Without them the tier uses
  Earth Engine (MERIT Hydro v1.0.1) and flags `MERIT_VERSION_MIX`; `data_provenance.version_consistency`
  records which dataset each step used. Refinement via Earth Engine is limited to a 40 km window and is
  skipped (flagged) beyond it.

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

## River tiles (MERIT-Basins PMTiles)

Turns a staged `rivers##.db` into a single-file vector tile archive a map can stream with HTTP range requests. Offline: it reads data that is already staged and never downloads.

```python
from aihydro_watershed.tiles import build_merit_river_tiles
rec = build_merit_river_tiles(72)          # -> $AIHYDRO_HOME/merit/tiles/rivers72.pmtiles (+ .provenance.json)
# or:  python -m aihydro_watershed.tiles 72 [--force] [--max-zoom 12]
```

- **Zoom by upstream area, not Strahler order (rule Z1).** One MVT layer per area band: `rivers_z0` (>= 10,000 km2, from z0), `rivers_z5` (>= 2,500), `rivers_z7` (>= 500), `rivers_z9` (>= 100), `rivers_z11` (everything else, from z11). A client draws all five layers; each is empty below its min zoom. Tiles stop at z12 (the source is 90 m); clients overzoom beyond. Reaches with no `uparea` go in the last band and are counted in the sidecar. The finest zoom holds every reach except degenerate sub-pixel ones, which GDAL drops; the sidecar counts them (`reaches_absent_at_max_zoom`, with up to 100 comids listed).
- **Attributes:** `comid`, `uparea` (km2), `sorder` (Strahler order), `lengthkm`. The source's primary key `comid` joins the tiles to delineation outputs.
- **Provenance:** `rivers##.pmtiles.provenance.json` records the source db sha256 (taken from the staging `.provenance.json`, and verified against the pinned hash first; `source_pinned` and `verified_at` come from that record, and `sha256_origin` says whether the hash was verified against the pin, trusted from an unverified sidecar, or computed), GDAL / pyogrio / geopandas versions, build options, output sha256 and size, reach counts per layer, licence and citation. The build is skipped when the sidecar matches the source sha, the options and the output file.
- **Licence:** derived from MERIT-Basins (Lin et al. 2019). Non-commercial and share-alike, with the exact variant unresolved: the files are attributed per mghydro.com as CC BY-NC-SA 4.0, MERIT Hydro is CC BY-NC 4.0 or ODbL 1.0, and share-alike obligations for derived data may apply (see [Licence](#merit-basins-hybrid-data-licence-versions) above). The sidecar and the `.pmtiles` metadata both carry a `licence_note` with that text. Keep the tiles local, attribute them ("Rivers: MERIT-Basins, Lin et al. 2019, CC BY-NC-SA 4.0 (attributed per mghydro.com; exact variant unresolved)"), and do not ship them in a commercially licensed bundle.
- **Dependency:** `pyogrio>=0.8.0` (declared; its wheels bundle GDAL 3.8.5, where the PMTiles driver first appears). Tiles are written through its bundled GDAL MVT/PMTiles driver; the build raises a clear error if the driver is missing. GDAL can neither set a zoom per feature nor append layers, so each band is written as its own archive and merged by tile id (`tiles/pmtiles_merge.py`; an MVT tile is a concatenation of layer messages, so the merge never decodes geometry).
- Coarse zooms drop sub-pixel reaches (GDAL simplification), so a low zoom shows fewer reaches than the band's threshold admits, never more.

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
