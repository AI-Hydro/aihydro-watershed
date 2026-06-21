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
       ├─── Tier 1: NLDI (CONUS streamstats)
       │      nldi_point.py: streamstats.waterservices.usgs.gov/nss/...
       │      Returns NHD catchment polygon — fast, authoritative for CONUS
       │      ↓ fail (non-CONUS, network error, small catchment)
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
│   ├── nldi_point.py        NLDI CONUS streamstats delineation
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
│   ├── merit_download.py    HydroSHEDS / MERIT-Hydro tile downloader
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
│   ├── curve_number.py      SCS CN computation (soil + NLCD/ESA land cover)
│   ├── event_runoff.py      SCS CN-II rainfall-runoff model
│   ├── erosion.py           RUSLE soil erosion estimation
│   ├── _landcover.py        Land cover fetch (aihydro-data first; ESA STAC fallback)
│   └── _soil.py             Soil data fetch (POLARIS CONUS; SoilGrids global)
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
          │    POST /streamstats/delineate?lat=39.27&lon=-77.54
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
          │    gauge_id provided → NWIS (dataretrieval.nwis.get_dv)
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
