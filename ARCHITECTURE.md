# aihydro-watershed — Architecture

Global watershed delineation engine + characterization + terrain analysis +
hydrological signatures.  Carved out of the `aihydro-tools` monolith (Wave A);
designed to be usable standalone with `pip install aihydro-watershed`.

---

## Position in the ecosystem

```
                aihydro-core
                (HydroResult, ToolError, science protocols)
                      ▲
                      │
   ┌──────────────────┼──────────────────────┐
   │                  │                      │
aihydro-data    pygeoglim              aihydro-watershed  ← THIS PACKAGE
(data router)  (geology)              (delineation + analysis)
   │                  │                      │
   └──────────────────┴──────────────────────┘
                       ▲
                  aihydro-lsh
                  (CAMELS recipes, uses delineation + signatures)
```

---

## Delineation tier system

```
Pour point (lat, lon)
       │
       ▼
  delineation/router.py
  delineate_from_point(lat, lon)
       │
       ├─── Tier 0: small_catchment (CONUS, < ~5 km2, e.g. road culverts)
       │      small_catchment.py: USGS 3DEP 10 m via py3dep → EPSG:5070
       │        notch the road embankment (cells within 30 m → min within 60 m)
       │        pyflwdir.from_dem → snap to max upstream area within 40 m
       │        grow window 2.5 → 5 → 10 km until basin clears the edge
       │      Runs first in auto when expected_area_km2 < 5; also the CONUS
       │      fallback when NLDI returns < 1 km2 or fails.
       │
       ├─── Tier 1: NLDI (CONUS, NHDPlus-indexed)
       │      nldi_point.py: USGS NLDI via pynhd (api.water.usgs.gov/nldi)
       │      Returns NHDPlus COMID basin polygons (not StreamStats)
       │      ↓ fail (non-CONUS, network error, basin < 1 km2)
       │
       ├─── Tier 2: MERIT-Hydro DEM (global, preferred outside CONUS)
       │      merit_flowdir_pipeline.py:
       │        merit_manager.py → download 5° tiles → local cache
       │        merit_snap.py    → snap pour point to nearest channel
       │        pyflwdir.FlwdirRaster.delineate_basins() → polygon
       │      ↓ fail (MERIT tile missing, resolution too coarse)
       │
       └─── Tier 3: pysheds on GLO30/SRTM DEM (fallback, any resolution)
              pysheds_pipeline.py:
                dem_fetch.py → GLO30 via py3dep / SRTM via elevation pkg
                dem_conditioning.py → fill pits, resolve flats, burn streams
                pysheds.Grid → flow direction → accumulation → delineation
```

---

## Module map

```
aihydro_watershed/
│
├── __init__.py              Public API + version 0.1.0
│
├── delineation/             WATERSHED BOUNDARY
│   ├── router.py            delineate_from_point() — tier orchestrator
│   ├── nldi_point.py        NLDI (NHDPlus COMID) CONUS delineation via pynhd
│   ├── small_catchment.py   3DEP 10 m + embankment notch, culvert-scale catchments
│   ├── merit_flowdir_pipeline.py  MERIT global flow-direction pipeline
│   ├── merit_snap.py        Pour-point snapping to nearest MERIT channel
│   ├── pysheds_pipeline.py  pysheds DEM-based fallback delineation
│   ├── dem_fetch.py         GLO30 / SRTM DEM acquisition
│   ├── dem_conditioning.py  Pit filling, flat resolution, stream burning
│   ├── types.py             DelineationResult, DelineationMethod enum
│   └── utils.py             Geometry helpers, CRS normalisation
│
├── merit/                   MERIT-HYDRO DATA MANAGEMENT
│   ├── merit_manager.py     Tile inventory, download, local cache (~/.aihydro/merit/)
│   ├── verified_fetch.py    SHA-256-pinned download + verification of MERIT region files
│   ├── basins_store.py      basins##.db reader (unit-catchment topology, geometries, uparea)
│   ├── delineator_data_manifest.json  pinned file digests, sizes, licence (delineator 2.2.4)
│   ├── merit_map_layers.py  GeoDataFrame overlays (river network, basins)
│   ├── merit_manifest.yaml  Tile URL catalogue (5° tiles, global coverage)
│   ├── region_presets.py    Named region bboxes for batch prefetch
│   └── wbd_layers.py        WBD (Watershed Boundary Dataset) CONUS layers
│
├── characterize/            WATERSHED SHAPE & MORPHOLOGY
│   ├── watershed.py         area, perimeter, compactness index
│   ├── geomorphic.py        slope, hypsometric curve, drainage density
│   ├── twi.py               Topographic Wetness Index (TWI)
│   └── _dem.py              Internal DEM preparation helpers
│
├── terrain/                 TERRAIN & LAND-SURFACE PROCESSES
│   ├── curve_number.py      SCS CN computation (TR-55; soil + NLCD/ESA land cover)
│   ├── event_runoff.py      SCS CN-II rainfall-runoff model
│   ├── erosion.py           RUSLE soil erosion estimation
│   ├── _landcover.py        Land cover fetch (aihydro-data first; ESA STAC fallback)
│   ├── _soil.py             Soil router: SSURGO in CONUS, POLARIS fallback, SoilGrids global
│   └── ssurgo.py            SSURGO: gNATSGO mukey raster + Soil Data Access (hydgrp, Kw)
│
└── signatures/              HYDROLOGICAL SIGNATURES
    ├── signatures.py        extract_hydrological_signatures() — orchestrator
    │                          Auto-fetches streamflow:
    │                            gauge_id  → NWIS (CONUS observed)
    │                            gauge_id=None → GEOGLOWS v2 (global modelled)
    │                          Auto-fetches precipitation via aihydro-data
    │                            GRIDMET (CONUS) → ERA5-Land → CHIRPS_IRI (global)
    ├── baseflow.py          Baseflow separation (Eckhardt digital filter)
    │                          BFI (Baseflow Index)
    ├── flow_duration.py     FDC slope, q_mean, q5, q95 percentiles
    ├── flood_frequency.py   GEV/log-Pearson III fitting, return period flows
    ├── drought_indices.py   SMDI, SRI (Streamflow Drought Index)
    └── _streamflow_internal.py  Internal streamflow aggregation helpers
```

---

## Delineation data flow

```
delineate_from_point(lat=39.27, lon=-77.54)
          │
          ├─ Tier 1: NLDI
          │    pynhd.NLDI: comid_byloc → basin polygon for the COMID
          │    → GeoDataFrame (EPSG:4326 polygon)     ← typical CONUS return
          │    success → DelineationResult(data=gdf, method="nldi", area_km2=...)
          │
          ├─ Tier 2: MERIT (on NLDI failure)
          │    merit_snap.py: find nearest channel cell in flow-accumulation grid
          │    merit_flowdir_pipeline.py:
          │      tile download → load flow-direction raster → pyflwdir.delineate_basins()
          │      → polygon → vectorise (rasterio.features.shapes) → GeoDataFrame
          │    success → DelineationResult(method="merit", ...)
          │
          └─ Tier 3: pysheds (on MERIT failure)
               dem_fetch.py → dem_conditioning.py
               pysheds.Grid.catchment(pour_point) → mask → shapes → GeoDataFrame
               success → DelineationResult(method="pysheds", ...)
```

---

## Hydrological signatures flow

```
extract_hydrological_signatures(
    gauge_id  = "01638500",     # optional USGS gauge
    watershed_gdf = gdf,         # delineated polygon
    start_date    = "1990-01-01",
    end_date      = "2020-12-31",
)
          │
          ├─ Streamflow fetch
          │    gauge_id provided → USGS Water Data API daily values (aihydro_data.sources.usgs_waterdata.get_daily)
          │    gauge_id = None   → GEOGLOWS v2 S3 Zarr (anonymous, 1940-present)
          │
          ├─ Precipitation fetch
          │    aihydro_data.fetch("precipitation", watershed_gdf, ...)
          │    CONUS → GRIDMET; global → ERA5-Land → CHIRPS_IRI (no auth)
          │
          ├─ baseflow.py    → q_mean, BFI, q_baseflow
          ├─ flow_duration.py → fdc_slope, q5, q95, q_mean
          ├─ flood_frequency.py → q2, q5, q10, q50, q100 (GEV fit)
          ├─ drought_indices.py → SMDI, SRI_6m
          │
          └─ HydroResult(data={17 signatures}, meta=HydroMeta(sources=[...]))
```

---

## Global coverage

| Sub-module | CONUS | Global | Notes |
|---|---|---|---|
| Delineation (NLDI) | ✅ | — | NHD-only |
| Delineation (MERIT) | ✅ | ✅ | 5° tiles downloaded on demand (~180 GB full world) |
| Delineation (pysheds) | ✅ | ✅ | GLO30 (30 m) or SRTM (90 m) |
| Terrain / characterize | ✅ | ✅ | DEM-based; GLO30 globally |
| Land cover | ✅ | ✅ | ESA WorldCover STAC fallback (no auth) |
| Soil | ✅ | ✅ | POLARIS (CONUS) or SoilGrids (global) |
| Streamflow signatures | ✅ | ⚠️ | NWIS observed; GEOGLOWS modelled (NSE ~0.4) |
| Precipitation → runoff ratio | ✅ | ✅ | CHIRPS_IRI final fallback (no auth) |

---

## Layering guard

`tests/test_layering.py` walks the AST of every `.py` file in
`aihydro_watershed/` and asserts zero imports of `ai_hydro` (the tools
monolith package).  This runs offline with no external deps.

```
# What IS allowed                # What is FORBIDDEN
aihydro_core.*         ✅        ai_hydro.*          ✗
aihydro_data.*         ✅        (anything above watershed)
pygeoglim.*            ✅
```

---

## DelineationResult contract

```python
@dataclass
class DelineationResult:
    data:      gpd.GeoDataFrame   # watershed polygon in EPSG:4326
    method:    str                 # "nldi" | "merit" | "pysheds"
    area_km2:  float
    pour_lat:  float
    pour_lon:  float
    snap_lat:  float | None        # snapped point (MERIT)
    snap_lon:  float | None
    warnings:  list[str]
    meta:      dict                # method-specific provenance
```

---

## Place identity (BasinRef) — ADR-003, slice 3 P3

`aihydro_watershed/identity.py` is the **only minter** of
`aihydro_core.records.place.BasinRef` (spec and `aihydro.geom/1` live in core).
Every delineation result carries `data["basin_ref"]` (a `BasinRef.to_dict()`,
or `None` if minting failed, in which case basin-scoped claim promotion fails
closed downstream). Identity is the **network anchor**; the polygon digest names
one geometry realisation only.

| Method (`method_used`) | Anchor kind | network / version | element |
|---|---|---|---|
| NLDI `get_basins(<site>)` (`delineate_watershed`) | `gauge_index` | `nhdplusv2` / `unversioned` | `usgs:<site>` |
| NLDI COMID (point route, or gauge COMID fallback) | `network_element` | `nhdplusv2` / `unversioned` | `<comid>` |
| `merit_basins_hybrid` (terminal id known) | `network_element` | `merit-basins` / vector dataset version | terminal catchment id |
| `merit_gee_pyflwdir`, `local_merit_pyflwdir`, hybrid without terminal id | `grid_cell` | `merit-hydro` / raster dataset version | `EPSG:4326\|<n>as\|ix\|iy` |
| `dem_raw_fallback` | `grid_cell` | `raw-dem` / `unversioned` | `EPSG:4326\|<n>as\|ix\|iy` |
| `small_catchment_3dep` | `grid_cell` | `3dep-dem` / `unversioned` | `EPSG:5070\|10m\|ix\|iy` |

- Unknown product versions become `unversioned` plus flag `network_version_unverified`.
- Grid anchors use ONLY the snapped pour point and grid the tier reports
  (`snapped_outlet_lon/lat`, `snap_grid_crs`, `snap_grid_resolution_m`; pysheds
  reports the DEM-snapped cell centre on its UTM grid). The requested outlet is
  never used: with no snapped point `basin_ref` is `None` and
  `data["basin_ref_unavailable"]` says why (`no_snapped_pour_point`,
  `unsupported_method`, `mint_failed:<code>`). So two requests snapping to one
  cell share an id, and one request cell with two snapped streams does not.
  A centre lying exactly on a cell edge is subject to float rounding. In
  EPSG:4326 the step is `round(res/30.87)` arc-seconds (min 1"), so products
  finer than 30 m are quantised coarser on that CRS; use a projected CRS for them.
- Different methods give different ids. Sameness across methods is asserted only
  by a shared `usgs:` alias or `compare_realisations` (area ratio, IoU, outlet
  distance); ids are never merged.
- Gauge COMID fallback: `delineation_path="comid_fallback"`, flag
  `gauge_basin_comid_fallback`, usgs alias relation `fallback_of`.
- Geoconnex: `.../usgs/monitoring-location/<site>` is added with `verified=False`;
  a `ref/gages/<n>` PID comes only from `resolve_geoconnex_gage` (one GET, 5 s,
  cached at `$AIHYDRO_HOME|~/.aihydro/cache/place/geoconnex.json`) and is never
  constructed. No PID available: flag `geoconnex_unresolved`. Minting does no I/O.

## NLDI request form and outlet ambiguity (camels-delineation-1, 2026-11-05)

Decision record. Evidence: `docs/vision-2040/evidence/camels-delineation-1/` (outside this repo).

- **Request form.** We do not call pynhd's `comid_byloc` for the nearest-flowline lookup. pynhd 0.19.4 `NLDI.__byloc` formats `POINT({lon:.6f} {lat:.6f})` and its
  HTTP stack puts `+` on the wire; the USGS NLDI service returns 502 for that form at some points (08155200: 6/6 502 vs 8/8 200 with `%20`). `nldi_point.hydrolocation_comid` sends `%20`.
  Separately the service has a transient upstream read-timeout fault that returns 502 in either form at some points (02298608, about half of requests), handled by retry/back-off.
  Upstream issue (not filed anywhere, by instruction): pynhd should percent-encode the WKT space, and the NLDI service should accept `+` or return 400, not 502. Installed site-packages are not patched.
- **Outlet ambiguity.** The pour point is the caller's. A nearest reach draining < 80 km2 may be a tributary reach or the head of a larger stem; the tool no longer decides.
  It returns the nearest reach with `NLDI_OUTLET_AMBIGUOUS` and the candidate COMIDs (areas included). Moving downstream needs `expected_area_km2` or `allow_downstream_expansion=True`, and only to a candidate larger than the nearest reach.
  `NLDI_OUTLET_AMBIGUOUS` (quality flag): the nearest reach drains < 80 km2, so the outlet may be a tributary reach or a larger stem; the returned basin is the nearest reach's and `outlet_ambiguity` lists the alternatives.
  Side effect: map-click quick delineation now returns the small reach's basin plus the flag, not the former larger stem.
- **Fallback honesty (CONUS `auto`).** When NLDI raises (429, 5xx, no response, the fail-fast memo, or the throttled-reply shapes pynhd surfaces: `Service returned no features`,
  `string indices must be integers`), `auto` no longer hands the point to the 3DEP small-catchment tier. Order now: MERIT-Basins hybrid, then cached local MERIT / `merit_gee`
  (a failing cached-flowdir tier is recorded on the trail and the chain continues), then the 3DEP tier **only if its result carries neither `BASIN_TOUCHES_MAX_WINDOW` nor `LIKELY_DITCH_SNAP`**
  (otherwise `rejected` on the trail), then raw DEM. If all of those fail, the rejected 3DEP polygon is served as a last resort, flagged `LAST_RESORT_UNVERIFIED` plus its own flags
  (trail outcome `succeeded_unverified`). Forced methods and NLDI success are unchanged; 3DEP stays first when `expected_area_km2 < 5` and when NLDI answered with a sub-km2 reach (no fallback flag there: by design).
  The fallback can auto-stage MERIT-Basins vectors (about 0.1-0.9 GB per Pfaf region, under the 2 GB tiered cap) on first call; `AIHYDRO_MERIT_AUTO_STAGE=never` turns that off.
  Evidence (CAMELS, 27 `auto` gauges served by 3DEP after a failed NLDI): 19 wrong basins; the forced hybrid on the same gauges had 2 wrong basins and 24 of 27 within 5% of `area_gages2`.
  Recomputed by `docs/vision-2040/evidence/camels-delineation-1/release_2.8.9/rescore_autofb/rescore.py` (table sha256 `66837f3a...cb49`). **Limits:** n = 27, in-sample (the gauges motivated the rule),
  Wilson 95% interval for 2/27 is 2.1%-23.4%; the proxy hybrid used Earth Engine snapping (`snap_source gee`), so the figure holds for GEE-authenticated users and the unsnapped-outlet
  (`offline_unsnapped`) hybrid is unevaluated; the `expected_area_km2` path has no evidence arm; the proxy is the forced hybrid arm, not a re-run of this router.
  Quality flags: `FALLBACK_AFTER_TRANSIENT_NLDI` (NLDI failed transiently and another tier served), `FALLBACK_AFTER_NLDI_NOT_USABLE` (NLDI answered or failed non-transiently and its result was rejected),
  `SOURCE_DISAGREEMENT` (an earlier tier's partial area, NLDI upstream area or MERIT `uparea`, differs from the served polygon by more than 25%; details in `source_disagreement`).
  The failed NLDI row carries `transient: true` in this package's `fallback_history`; the aihydro-tools attempt-trail reader does not carry that key yet.
- **Unsnapped outlets (hybrid, no Earth Engine).** With no Earth Engine and no staged `accum##.tif`, the hybrid used to return the whole terminal unit catchment plus everything upstream
  for the raw coordinate (`snap_quality = offline_unsnapped`). Controlled test (camels-delineation-1 Amendment 4, condition B, n = 671): 82 wrong basins, 45.8% of those under 100 km2.
  Now, in order: (1) the outlet is snapped on the staged `flowdir##.tif` **inside the home unit catchment** (`merit_hybrid.snap_on_terminal_flowdir`; `snap_quality = offline_flowdir_home_catchment`,
  or `area_gated` when `expected_area_km2` is given, the same narrow 1 km / +/-10% gate as the accumulation path). Local upstream area comes from routing the terminal catchment's cells; the upstream unit
  catchments' summed area is added to each cell that at least 120 cells of theirs drain through (`INFLOW_RECEIVES_MIN_CELLS`, a choice made at first test, not swept). The snap, the off-reach test and
  the terminal refinement are then the existing code. The raster is staged first under the existing tiered policy (`auto_stage`, `AIHYDRO_MERIT_AUTO_STAGE`, 2 GB cap; the router's own flowdir staging no longer skips CONUS).
  CONUS `flowdir##.tif` sizes: the seven regions 71-75, 77, 78 are 99-304 MB each, decimal (manifest `size_bytes`; `accum##.tif` is 249-795 MB). A coordinate that falls in a neighbouring unit catchment than its river cannot reach that river
  (home-catchment search only; 3 of 304 gauges in the first test). (2) Only when no snap is possible (raster not staged and not stageable, snap error) the outlet stays `offline_unsnapped` and the result is
  marked at the **top level**: `reliability` (`"unreliable"` or `"provisional"`; `None` when the outlet was snapped), `reliability_reason`, `remedy` (ordered steps), `estimated_area_km2_on_reach`, `reliability_checks`,
  plus flags `OUTLET_UNSNAPPED_UNRELIABLE` / `OUTLET_UNSNAPPED`. The polygon is still returned. `unreliable` when returned area / along-reach area estimate > 1.25, or the point is > 300 m from the terminal
  catchment's river line (`rivers##.db` geometry, read by `BasinsStore.river_line`), or the line is missing so the test cannot run; otherwise `provisional`. The estimate is inflow + (1 - fraction along the reach)
  x terminal area (reach lines run downstream to upstream; local area taken to accrue evenly). **The 1.25 and 300 m thresholds are in-sample** (CAMELS, 671 gauges: flags 201, catches 72 of 82 wrong basins,
  1 of 135 basins >= 1,000 km2 flagged; ten-fold CV sensitivity 0.84 / specificity 0.89 for a wrong basin); finding `unsnapped-outlet-2026-11-07`. **`expected_area_km2` does not change an unsnapped outlet**;
  it only raises `AREA_DIFFERS_FROM_EXPECTED` after the polygon is built, and the remedy text says so. It steers the snap only once the flowdir snap runs. `local_merit_flowdir_pyflwdir` (flowdir-only tier, no vectors)
  cannot run the reach test: an unsnapped result there is marked `unreliable` outright, with a remedy. `scripts/eval/condition_bprime.py` is the evaluation driver for condition B' (B with the flowdir snap).
- **Not changed.** CAMELS 14158500's WRONG_BASIN is a reference defect (CAMELS' own list gives a 48% size error; the NLDI area is within 3% of the USGS area), not a tool fault.
  Using the NLDI `nwissite/USGS-<id>` feature when a gauge id is known is already the path of `delineate_watershed(gauge_id)`; `delineate_from_point` takes no gauge id and was not extended here.

