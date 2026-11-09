# Changelog — aihydro-watershed

## [Unreleased]

### Added: local river PMTiles from staged MERIT-Basins (basemap slice B2, tile-building half)

- `aihydro_watershed.tiles.build_merit_river_tiles(region, out_dir=None, *, source_db=None, max_zoom=12, force=False, verify_source=True)` and `python -m aihydro_watershed.tiles <region>...` write `<AIHYDRO_HOME>/merit/tiles/rivers##.pmtiles` and a `.provenance.json` sidecar (source db sha256 from the staging sidecar, GDAL/pyogrio versions, build options, output sha256 and size, licence with a `licence_note` (non-commercial, share-alike, variant unresolved; also in the `.pmtiles` metadata), `source_pinned`/`verified_at`, the count of reaches GDAL drops at the finest zoom, citation Lin et al. 2019). Offline, idempotent. Declares `pyogrio>=0.8.0` (imported directly; its wheels bundle GDAL 3.8.5, where the PMTiles driver was added); the build raises a clear error if the driver is missing.
- Reaches are split into five MVT layers by upstream area (rule Z1 of `hydrology-basemap-2026-11-08`): `rivers_z0` (>= 10,000 km2), `rivers_z5` (>= 2,500), `rivers_z7` (>= 500), `rivers_z9` (>= 100), `rivers_z11` (rest); attributes `comid`, `uparea`, `sorder`, `lengthkm`. Tiles go to z12. GDAL cannot set a zoom per feature or append layers, so the bands are merged by tile id in `tiles/pmtiles_merge.py`.

### Fixed (review of the PMTiles builder)

- Merged archive metadata `bounds`/`center` are the union of all input bands (they came from the `rivers_z0` band only and disagreed with the header); a missing or empty `vector_layers` in an input is refused.
- Licence stated as unresolved rather than a flat CC BY-NC-SA 4.0 (`licence_note` in sidecar and metadata; attribution keeps "CC BY-NC-SA").
- `pyogrio>=0.8.0` declared; `sha256_origin` distinguishes verified-against-pin, unverified sidecar and computed; sidecar written atomically; `TILE_SCHEMA_VERSION` 2 (old archives rebuild).
- The finest zoom is documented and tested as holding every reach except degenerate sub-pixel ones, which are counted in the sidecar.

## [0.4.3] - 2026-11-08

### Fixed: an outlet that cannot be snapped is no longer returned silently (finding `unsnapped-outlet-2026-11-07` P1, P2)

With NLDI failing and no Earth Engine, `auto` served the MERIT-Basins hybrid with `snap_quality = offline_unsnapped`: the whole terminal unit catchment plus everything upstream. In the controlled
test (camels-delineation-1 Amendment 4, condition B, 671 CAMELS gauges) that was the wrong basin for 82, 45.8% of basins under 100 km2, with nothing at the top of the result saying so.

**Added**

- **Offline flow-direction snap (P2).** `merit_hybrid.snap_on_terminal_flowdir` snaps on the staged `flowdir##.tif` inside the home unit catchment, adding back the upstream catchments' area to the
  cells they drain through, with the package's own `snap_outlet_on_merit` (or `snap_outlet_area_gated` when `expected_area_km2` is given). `snap_quality = "offline_flowdir_home_catchment"`,
  `snap_source = "local_flowdir_home_catchment"`, flag `OFFLINE_FLOWDIR_SNAP_USED`; refinement then runs as for any snapped outlet and no `MERIT_VERSION_MIX` is raised. `merit_basins_hybrid_delineate`
  tries it whenever the outlet would otherwise be unsnapped, before falling back. It stages `flowdir##.tif` first under the existing tiered policy when `auto_stage` is on (progress is logged; the 2 GB cap
  and `AIHYDRO_MERIT_AUTO_STAGE=never` apply; 99-304 MB for the CONUS regions); the staging row (`flowdir_staging`: `succeeded` / `not_staged` / `confirm_required`) is in the trail. Not run for
  `auto_stage=False` unless the raster is already staged. Home-catchment search only (P3, looking past the border, is not done). First test, 304 CAMELS gauges in Pfaf 74/77: 16 wrong against
  37 unsnapped and 13 with Earth Engine (finding); the evaluation on all 671 is pre-registered separately.
- **Top-level reliability mark for what cannot be snapped (P1).** `HydroResult.data` now has `reliability` (`"unreliable"`, `"provisional"`, or `None` when the outlet was snapped), `reliability_reason`,
  `remedy` (ordered list), `estimated_area_km2_on_reach` and `reliability_checks`, set by the hybrid (`merit_hybrid.assess_unsnapped_outlet`) and carried by the router for `auto` and forced methods; it is also on the
  GeoJSON feature properties when set. `unreliable` when returned area / along-reach area estimate > 1.25, the point is > 300 m from the terminal catchment's river line, or the line is not available;
  else `provisional`. Flags `OUTLET_UNSNAPPED_UNRELIABLE` / `OUTLET_UNSNAPPED`; the polygon is unchanged. The thresholds are **in-sample** on CAMELS (see ARCHITECTURE.md). The remedy text is exact for the code: it does
  not offer `expected_area_km2` as a fix for an unsnapped outlet (it only checks the area). The flowdir-only tier (`local_merit_flowdir_pyflwdir`) marks an unsnapped result `unreliable` without a screen.
- **What `reliability` means.** `None` = no concern detected; it is not a validation. It is left only for an Earth Engine or accumulation-raster snap, or a flow-direction snap whose area gate matched the caller's `expected_area_km2`. `"provisional"` = the
  polygon is returned and a limit applies: a snap on flow direction alone, not validated against an accumulation raster (`OFFLINE_FLOWDIR_SNAP_UNVALIDATED`; in the 304-gauge smoke about 19% of basins under 100 km2 were still wrong), or an unsnapped outlet that passed the reach screen.
  `"unreliable"` = unsnapped and failed (or could not run) the screen.
- When the flowdir-only local tier returns `offline_unsnapped`, the router now hands the point to the hybrid (once per call) so the flowdir snap can run; if the hybrid cannot serve, the local result is kept with its `unreliable` mark.
- An unsnapped outlet no longer reports a fabricated `snap_distance_m: 0.0` / `snap_validation.ok: true`: `MeritSnap.distance_m` and `snap_distance_m` are `None`, `snap_validation.ok` is `None` with a `reason`.
- `BasinsStore.river_line(comid)`: the reach line (EPSG:4326, downstream vertex first) from `rivers##.db`, `None` when absent.
- `scripts/eval/condition_bprime.py`: offline driver and scorer for condition B' (B with the flowdir snap) against the A and B rows of the controlled test.

**Changed**

- The router's flowdir auto-stage branch (local MERIT tier) no longer skips CONUS points; the tiered policy still decides.
- Test fixture `merit_fixture` writes the `rivers` layer with reach lines (it had only the `uparea` table).

## [0.4.2] - 2026-11-06

### Fixed: hybrid polygon agrees with its own snapped outlet (PR #17; finding `small-basin-delineation-2026-11-06` P1, P2)

**Fixed**


- **`merit_basins_hybrid_delineate` no longer unions upstream unit catchments onto an outlet that is not on their reach.** When the snapped outlet's
  MERIT drainage area is below 0.9x the total area of the upstream unit catchments, the outlet sits on a tributary inside the terminal catchment and
  none of that inflow drains to it. The result is now the local (refined) terminal piece only, flagged `OUTLET_OFF_TERMINAL_REACH` and
  `HYBRID_LOCAL_ONLY`. The main-stem alternative (area from the MERIT-Basins `uparea`, the union's assembly area, and a note) is in
  `area_validation["outlet_assembly"]["mainstem_alternative"]` and `data_provenance["outlet_assembly"]`. With `expected_area_km2`, whichever of the
  two areas is closer in |ln(A/E)| is kept, so a gauge truly on the main stem whose snap fell on a tributary still gets the main-stem basin
  (`mode: "union_upstream_kept_by_expected_area"`; the polygon is then the whole assembly, upstream catchments plus the whole terminal catchment, i.e.
  `polygon_area == vector_assembly_area`; `HYBRID_AREA_VALIDATION_MISMATCH` then still fires). The 0.9 rule only catches clear tributary cases: an
  outlet on a tributary whose area is 0.9x the inflow or more is treated as on-reach. `HYBRID_AREA_VALIDATION_MISMATCH` is unchanged.
- **Terminal refinement failure no longer returns the whole terminal catchment when the outlet drains only part of it** (off-reach, or a headwater
  terminal with no upstream catchments, snapped area below 0.9x the terminal area). The basin is delineated on an Earth Engine `dir` window centred on
  the outlet (half-side `max(3 km, 2*sqrt(upa))`, grown 1.6x while the basin touches the edge, capped at 20 km) and flagged
  `TERMINAL_CATCHMENT_LOCAL_WINDOW_USED`. If that fails too the previous behaviour stands (whole terminal) and the failure is in the trail.
  `TERMINAL_CATCHMENT_REFINEMENT_FAILED` is kept either way.

**Added**

- **Area-gated snap** (`snap_outlet_area_gated`, `merit_hybrid.snap_on_local_accum(area_gate=True)`, `merit_get_snap_reference(area_gate=True)`): with
  an `expected_area_km2`, the hybrid snaps to the nearest stream cell within 1 km whose upstream area is within +/-10% of it (`snap_quality`
  `"area_gated"`). The gate result is recorded in `area_validation["outlet_assembly"]["area_gate"]` (`gate_applied` is false, with no flag or trail row, when the caller
  supplied a snap reference made without the gate). A gated cell more than 500 m from the gauge raises `AREA_GATE_SNAP_DISTANT`. If none qualifies the
  plain no-area snap is kept (its `snap_quality` gets the suffix `_area_gate_no_match`) and `AREA_GATE_NO_MATCH` (plus `SNAP_UPA_DIFFERS_FROM_EXPECTED` when the kept snap is
  more than 10% off) is raised; the search is never widened. The 30 km area-targeted scorer is untouched for the other tiers and is no longer used by
  the hybrid.
- The hybrid's own tier trail is exposed in the result as `data_provenance["hybrid_fallback_history"]` (a copy of `fallback_history`, with new
  `outlet_assembly`, `outlet_centred_window` and `area_gated_snap` rows). The router appends these rows to its trail as `merit_basins_hybrid/<step>`
  (same `method`/`outcome`/`reason` shape; a step's `succeeded` is written `completed` so it is not read as a served tier), ahead of the served
  `merit_basins_hybrid` row.
- `scripts/eval/rerun_hybrid.py` and `scripts/eval/score_rerun.py`: offline (sockets blocked) re-run and scoring of the hybrid over the
  camels-delineation-1 gauges from the campaign's cached Earth Engine windows and MERIT-Basins vectors.
- Live check of the outlet-centred window (the offline re-run had no cached window for it): 01195100 and 14092750, both wrong offline for that reason, are
  MATCHED live (IoU 0.72 and 0.85 against the CAMELS polygon; 14.1 vs 14.84 and 58.1 vs 57.48 km2 `area_gages2`), with and without the expected area.
  Evidence `camels-delineation-1/release_2.8.9/rescore_hybridpoly/live_check/` (two gauges only, not a live re-run of the 671).

**Not done**

- Passing a gauge's published area automatically (the finding's P2 "pass it for gauges") lives where gauge resolution does, outside this package.
- The snap-quality band record (High/Medium/Low/Caution), P3-P5 of the finding.

**Changed (router)**

- `_prepare_merit_context` asks `merit_get_snap_reference` for the area gate whenever an `expected_area_km2` is given (the small window is kept; the wide
  scorer is no longer used for that reference). The default global path lets the hybrid fetch its own reference, which is gated the same way.

## [0.4.1] - 2026-11-06

### Fixed

- **CONUS `method="auto"` no longer falls back silently to the 3DEP small-catchment tier when NLDI is throttled or down.** In the CAMELS evaluation (aihydro-tools 2.8.9 / watershed 0.4.0)
  27 `auto` gauges were served by `small_catchment_3dep` after a failed NLDI call; 19 were the wrong basin.
  That tier is built for culvert-scale catchments: with a 10 km half-width window it returns a truncated basin (`BASIN_TOUCHES_MAX_WINDOW`) or a roadside-ditch snap (`LIKELY_DITCH_SNAP`) on river-scale basins,
  and the router served it regardless. Now, when NLDI raises, `auto` tries the MERIT-Basins hybrid first (also when `expected_area_km2` is given, ahead of `merit_gee`), then the other MERIT tiers
  (a cached-flowdir tier that raises no longer ends the chain), then the 3DEP tier only if its result carries neither flag (otherwise the trail row is `rejected`), then raw DEM.
  If every later tier also fails, the rejected 3DEP polygon is served as a last resort with `LAST_RESORT_UNVERIFIED` plus its own flags; its trail row is `succeeded_unverified`
  (reads as served in the tools attempt trail). 3DEP is unchanged where it fits: `expected_area_km2 < 5`, and NLDI answering with a sub-km2 reach.
- Transient-failure classifier: HTTP status codes are matched as `HTTP <code>` / `status <code>`, a leading code (`502, message=...`, `500 Server Error: ...` as aiohttp and requests word it) or an exception's `status` / `response.status_code`, never as bare numbers elsewhere in the text (coordinates, distances and areas no longer match); timeouts and connection errors
  are recognised by exception type (and cause chain) as well as by message, so an empty-message `TimeoutError` counts.

### Evidence and limits

- Offline re-score of the 27 gauges with the forced-hybrid arm as proxy: wrong-basin 19 -> 2, within 5% of `area_gages2` 3 -> 24. Recomputed by
  `docs/vision-2040/evidence/camels-delineation-1/release_2.8.9/rescore_autofb/rescore.py` (output `rescore.json` next to it; input `results/table.csv`,
  sha256 `66837f3a3d0e33a35855e8c82384c43039bc031fabbe721bbebec3d69484cb49`; evidence repo commit d9a0ef5).
- Limits: n = 27; **in-sample** (these gauges motivated the rule); the 2/27 wrong-basin rate has a Wilson 95% interval of 2.1%-23.4%. The proxy arm is the forced hybrid, not a re-run of this router.
  All 27 proxy runs snapped the outlet with Earth Engine (`snap_source gee`), so the number holds for GEE-authenticated users; without Earth Engine the hybrid uses an unsnapped outlet
  (`offline_unsnapped`), which has **not been evaluated**. The `expected_area_km2` path (hybrid ahead of `merit_gee`) has no evidence arm.
- The 158-gauge `auto` figures from the same evaluation come from an incomplete, still-moving snapshot and are not cited.
- The fallback can auto-stage MERIT-Basins vectors for the point's Pfaf region on first call (about 0.1-0.9 GB per region, under the 2 GB tiered cap; progress goes to the log).
  `AIHYDRO_MERIT_AUTO_STAGE=never` disables it, and the hybrid then fails over to the next tier.

### Added

- Quality flags `FALLBACK_AFTER_TRANSIENT_NLDI` and `FALLBACK_AFTER_NLDI_NOT_USABLE` on any CONUS `auto` result not served by NLDI (not raised for the by-design sub-km2 culvert route); the serving trail row's `reason` names the NLDI failure.
  The failed NLDI row also carries `transient: true` in this package's `fallback_history`; the aihydro-tools attempt-trail reader does not carry that key yet, so it is not in the recorded trail.
- Quality flag `LAST_RESORT_UNVERIFIED` (see Fixed).
- Quality flag `SOURCE_DISAGREEMENT` and `source_disagreement` in the result: an earlier tier's partial area (NLDI upstream area, MERIT `uparea` of a rejected GEE result, a rejected 3DEP area) differs from the served polygon's area by more than 25%.

## [0.4.0] - 2026-11-06

**Defaults changed in this release.** (1) Outside CONUS, `delineate_from_point(method="auto")` now starts with the MERIT-Basins hybrid
(`merit_basins_hybrid`) instead of `merit_gee`; the first call per region downloads pinned, SHA-256-verified data (see the disclosure notes below;
`merit_manager.DEFAULT_AUTO_STAGE_POLICY = "never"` makes it confirm-first). (2) NLDI no longer moves the outlet downstream on its own: with no
`expected_area_km2` and a nearest reach draining < 80 km2, it returns that reach's basin with `outlet_ambiguity` and the `NLDI_OUTLET_AMBIGUOUS` flag
(also written to the NLDI row of `fallback_history`); pass `allow_downstream_expansion=True` to opt in to the old behaviour. Other behaviour changes below:
`slope_from_dem` fails closed, and the MERIT shapefile layout is no longer read.

This release merges aihydro-watershed PRs #11, #12, #7, #10, #8, #9 and #13. The sections below keep each PR's notes, in merge order.

### Changed

- **MERIT-Basins hybrid is the default global delineation tier.** `delineate_from_point(method="auto")` outside CONUS tries
  `merit_basins_hybrid` first (before `merit_gee` / `local_merit` / raw DEM, which remain as fallbacks). `method="merit_hybrid"`
  (alias `merit_basins`) forces it anywhere, including CONUS, so the CAMELS evaluation can score it; in CONUS `auto` still starts with NLDI.
  The tier is attempted once per call; the trail row is `merit_basins_hybrid` (`succeeded` / `failed`) and `method_used` reads `merit_basins_hybrid`.
- Data source replaced: the unhashed Google Drive download, the runtime `pip install gdown` and the `git clone` of delineator into `/tmp` are gone
  (`merit/merit_download.py` reduced to a deprecated shim). Per-region files come from mghydro.com's `delineator` 2.2.4 / data v3 set, pinned by SHA-256 and size in
  `merit/delineator_data_manifest.json`, streamed to `.part` and installed only if the digest matches. A mismatch raises
  `MERIT_ASSET_INTEGRITY_FAILED` (fail closed); an on-disk file that no longer matches is refused. `megabasins.db` (not served by mghydro.com) is pinned by
  the git blob SHA-1 at the v2.2.4 tag, and its SHA-256 is recorded on first verification and enforced afterwards.
  The old shapefile layout (`cat_pfaf_##` / `riv_pfaf_##`) is no longer read; re-stage with `merit_ensure_basins_region`.
- Vector assembly reads `basins##.db` (SQLite `nextdown` topology, `where`-filtered geometries) instead of loading a whole regional shapefile;
  the assembled area is checked against MERIT-Basins' own `uparea` (`VECTOR_ASSEMBLY_UPAREA_MISMATCH` above 2%).
- Terminal-catchment refinement reads a small window: the staged `flowdir##.tif` if present, otherwise one Earth Engine `dir` request limited to 40 km
  (larger terminals keep the whole unit catchment and raise `TERMINAL_REFINEMENT_WINDOW_TOO_LARGE`). An outlet that could not be snapped is no longer
  refined from an off-channel point (`TERMINAL_CATCHMENT_REFINEMENT_SKIPPED_UNSNAPPED`).
- Snapping uses the staged `accum##.tif` when present. Otherwise the Earth Engine reference (MERIT Hydro v1.0.1, against v0.7-based vectors) is used and
  the result is flagged `MERIT_VERSION_MIX`; `data_provenance.version_consistency` records the dataset of each step.
- `MeritDataManager` flowdir staging (`ensure_routing_region`, `auto_stage_flowdir`) now uses the same verified download (the old mghydro `raster/` URL was dead).
- `AIHYDRO_HOME` is honoured for the MERIT data root and the auto-stage config (`config.json`); `AIHYDRO_MERIT_AUTO_STAGE` overrides the policy.
  The Earth Engine window cache path in `merit_flowdir_pipeline` is unchanged here.

**Changed: review round 1 of PR #11**

- **Disclosure.** A call that downloads data flags `MERIT_AUTO_STAGED` and lists `data_provenance.downloaded` (`name`, `bytes`, `url`, `sha256`) and
  `acquisition_policy`. First-call cost per region (basins + rivers, pinned sizes): min 14 MB, median 530 MB, max 2015 MB over 60 regions
  (North America: 60 / 502 / 916 MB); +52.9 MB for the level-2 index on the first call ever. The default policy is the single constant
  `merit_manager.DEFAULT_AUTO_STAGE_POLICY = "tiered"` (`"never"` = confirm-first). Vector staging checks for 2x the pending bytes free on disk.
- **Download robustness.** An integrity mismatch writes a negative-cache marker (file, observed digest, time) and refuses automatic retry for 24 h
  with `MERIT_ASSET_INTEGRITY_FAILED` (surfaced through staging, never as `MERIT_BASINS_NOT_STAGED`; `force_retry=True` overrides). `.part` files are
  resumed with HTTP Range when the server answers 206. Each destination has a lock file and a `mkstemp` partial file so concurrent first downloads cannot interleave.
- **Pins.** `scripts/gen_delineator_manifest.py` regenerates the manifest deterministically from `constants.py` at the v2.2.4 tag commit
  (`c56095e8625d1ee3534217b0462a8ed43027c77c`; `source.tag_commit`, `source.constants_blob_sha1`, `source.hashes_file` is the commit URL). An unmodified
  copy of that file (MIT, with its LICENSE) is vendored under `tests/fixtures/delineator_v2.2.4/`; a test checks the manifest against it.
- **Sidecar.** The provenance fast path also checks `st_ino` and `ctime_ns`; `megabasins.db` and every `.db` under 200 MB are always re-hashed.
  A tamper that restores size and mtime is caught.
- `BasinsStore.locate` resolves several hits deterministically (nearest, then max `uparea`, then lowest comid). `geometries()` reports ids without a polygon
  (`VECTOR_ASSEMBLY_INCOMPLETE`).
- The licence string and `data_provenance.share_alike` state "share-alike" explicitly.
- `basins_region_cache` says "legacy layout found, not read; re-stage with ..." when the shapefile layout exists. `configure_delineator_env` is removed;
  `delineator_ready` now means verified `basins##.db` + `rivers##.db` (aihydro-tools `merit_ensure_basin` still calls it).
  `merit/merit_download.py` is kept only as a deprecated shim (`ensure_level2_index`, `download_river_shapefile`, `download_catchment_shapefile`) because
  aihydro-tools `scripts/install_merit_minimal.py` and a tools test import it; it stages the verified files.
- `tests/nonet.py`: a `-p nonet` pytest plugin that blocks non-loopback network access (`PYTHONPATH=tests pytest -p nonet ...`).

- `fetch_merit_gee_bbox`: `scale_m` is deprecated and ignored (a `DeprecationWarning` if passed); the returned window is `bbox` rounded out to whole MERIT cells and clamped to the
  asset's extent (lon -180..180, lat -60..85; no antimeridian wrap). New `native_window()`.
- Area, from the live run above (14 gauges, 14158500 excluded for the CAMELS fabric/USGS inconsistency): MERIT polygon vs `area_gages2` mean +0.32%, median -0.06%; vs the CAMELS
  polygon mean -1.78%, with the error correlating -0.85 with the fabric/`area_gages2` ratio. The polygon area is 0.08-0.19% above MERIT's own UPA at the snapped cell, and equals the pixel-area sum
  and the WGS84 geodesic area to under 0.1% (unit test). No bias correction was added.

- `delineate_from_point`: a `ToolError` raised after one or more tiers were tried now carries `details["fallback_history"]`, the same
  list a successful result returns (it reaches `to_dict()` and so the error envelope). Before, a total failure kept only the last
  tier's message. A history already set on the error is not overwritten; a failure before any tier ran (invalid coordinates) is unchanged.
  The NLDI tier now appends a row for failed, skipped and served, and the raw-DEM tier a `served` (or `rejected`/`failed`) row, so a result
  names the tier that served it; a result returned early by NLDI carries the history. An unexpected (non-`ToolError`) exception escaping the
  body keeps the trail as `exc.fallback_history`.
  The body moved into a private `_delineate_from_point`; the public signature and behaviour on success are unchanged.

**Changed: BREAKING-BEHAVIOUR): `slope_from_dem` fails closed**

Audit C8 latent sibling (`docs/vision-2040/findings/slope-mean-diagnosis-2026-11-05.md`, section 6).

- `aihydro_watershed.characterize._dem.slope_from_dem` now raises the new `SlopeUnavailableError` (a `ValueError`) unless the DEM is in a projected CRS with a readable pixel size. It refuses a missing CRS, a geographic CRS (spacing in degrees) and an unreadable or non-positive resolution. Before, a DEM still in EPSG:4326 (the callers keep the native CRS when the EPSG:5070/UTM reprojection fails, with only a warning) produced a "slope" from metres of rise per degree of run, close to 90 degrees everywhere, and a resolution that could not be read silently became a spacing of 1.0. Pixel size in a non-metre projected CRS (feet) is now converted to metres.
- Valid input (a metric grid) gives the same values as before. The returned DataArray still carries `units="degrees"`.
- `compute_twi` (`characterize/twi.py`): the two "keeping native CRS" fallbacks (EPSG:5070 and UTM reprojection failure) now raise (`SlopeUnavailableError`, surfaced as `ValueError("Failed to compute TWI: ...")`) instead of running flow accumulation, slope and the specific-catchment-area term on a native degree grid.
- **`geomorphic._compute_relief_metrics` (and so `Ls_pct`, `Rh`, `Rp`) was NOT unaffected, contrary to the earlier diagnosis.** When the UTM reprojection failed it kept the native CRS and, when the resolution could not be read, defaulted the cell size to 30.0 m, so a 7.2 % slope read `mean_slope_pct = 564648.6` and `Rh`/`Rp` were in degrees. Now it requires a metric grid via `_metric_spacing` (no 30 m default) and returns NaN for the relief metrics with a `flag` of the form `relief_unavailable:<reason>` when reprojection or spacing fails. On a working metric grid the values are unchanged. The earlier statement "lsh `Ls_pct` ... is a correct percent" holds only when the reprojection succeeded.
- **Propagation.** `_compute_stream_metrics_approximate` turned a NaN `H_m` into a drainage density of 0.5 (`max(0.5, min(nan, 5.0))`) and so into made-up `D_km_per_km2`, `Rn`, `C_km2_per_km`, `Rf`, `Cf_per_km2`; they (and `Cs`, `Slope_10_85`) are now NaN when relief is not finite (`MCh_km` depends on `Lb` only). The relief `flag` is surfaced as `relief_flag` in the output of `extract_geomorphic_parameters` and `..._result` (key present only when relief was unavailable), so "relief unavailable" can be told from "not computed".
- **UTM zone number** is now `% 60 + 1` (`geomorphic._metric_crs_for_geom`, `_compute_relief_metrics`, `twi.py`): longitude +180 gave zone 61 (invalid EPSG).
- Tests: `tests/test_relief_fail_closed.py` (the reviewer's repro now gives NaN + flag; the metric path still gives the right slope; unreadable resolution does not default to 30 m; zone wraps at 60); `tests/test_twi_fail_closed.py` (both branches raise, cause is `SlopeUnavailableError`); `tests/test_slope_from_dem.py` (analytic plane gives 1000 t within 1 % at 10, 30, 73 and 90 m; resampling invariance; UTM at two latitudes; feet CRS; geographic, unknown-CRS and unreadable-resolution inputs raise; units metadata).

- **Removed names** (replaced by the SHA-256-verified `merit/verified_fetch.py`, `merit/basins_store.py` and `delineation/merit_hybrid.py`).
  From `delineation/merit_flowdir_pipeline.py`: `merit_load_catchment_topology`, `merit_identify_terminal_catchment`, `merit_traverse_upstream_catchments`
  (and the shapefile-field constants they used). From `MeritDataManager`: `_try_download`, `_try_download_with_progress`, `_estimate_flowdir_size_bytes`
  (module-level helper), `configure_delineator_env`. Callers importing them must move to the hybrid / verified-fetch API
  (aihydro-tools `merit_ensure_basin` still imports the deprecated `merit/merit_download.py` shim).
- **`auto` order, global:** MERIT-Basins hybrid, then local MERIT routing when a regional flowdir is already cached (it runs before Earth Engine), else `merit_gee`
  (after a `merit_gee` failure, local MERIT is retried, staging the flowdir first outside CONUS), then raw DEM.

### Added

- `merit/verified_fetch.py`, `merit/basins_store.py`, `delineation/merit_hybrid.py`; `MeritDataManager.auto_stage_vectors`, `ensure_level2_index`, `verified_asset`.
- Result field `data_provenance` (file name, SHA-256, size, URL of every file used; licence; pinned manifest version; version consistency).
  The result `license` is now the non-commercial MERIT-Basins/mghydro/MERIT Hydro statement.
- `tests/test_merit_hybrid_default.py` (35 offline tests on a synthetic fixture); `tests/conftest.py` sends `HOME`/`AIHYDRO_HOME` to a temp dir and blocks downloads.

Live evidence for this section: `docs/evidence/merit-gee-native-grid-2026-11/` (README lists what was not recorded).

- Quality flag `NLDI_OUTLET_AMBIGUOUS`: the nearest NHDPlus reach drains < 80 km2, so the point may sit on a tributary reach or on a larger stem; the basin returned is the nearest reach's, candidates are in `outlet_ambiguity`.
- `requests>=2.31.0` declared as a dependency.
- `nldi_point.hydrolocation_url`, `hydrolocation_comid`, `HydrolocationHit`, `quick_comid_resolution`.

- `aihydro-watershed` console script with one subcommand, `signatures`: the CAMELS-style hydrological signatures from a local daily
  streamflow CSV, a basin polygon (GeoJSON) and a drainage area, offline (`precipitation="skip"`; `runoff_ratio` / `stream_elas` are null).
  A new entry point ships only with a release (bump needed). Delineation stays in the Python API. Input handling is explicit: negative discharge is refused, `--na-values` (default `-999,-9999`) are read as missing,
  duplicate dates keep the first row, and the JSON result carries a `cli` block (`n_missing`, `n_duplicate_dates`, `duplicate_dates`, absent calendar days, `warnings`).
- `install-alone` CI workflow: builds the wheel, installs only it plus its PyPI dependencies in a fresh venv outside the checkout, and runs
  that CLI task with every socket connection refused (AGU ecosystem audit C1).

### Fixed

- `method="merit_gee"` failed for 13 of 20 CAMELS pilot gauges with Earth Engine HTTP 400 "User memory limit exceeded" (12) or HTTP 503
  "memory capacity exceeded" (1). The raster request was `getDownloadURL` with `crs="EPSG:4326"`, `scale=92.77` and `.clip(rectangle)`, which makes Earth Engine
  resample onto a grid 0.004% off MERIT's native 3 arc-second cell; it failed at the first window expansion (about 32 km half-side). The request is now
  MERIT's own grid (`crs_transform` + `dimensions`, whole cells, no clip, no region). Live: that request succeeded on gauge 01547700 at 32, 51, 64 and 82 km
  half-side (4.1M cells) where the old one failed from 32 km; on the 16 pilot gauges 15 return a polygon (05495000 hits the window cap). Largest window verified live: 4.1M cells.
- Earth Engine is initialised once per process, thread-safely (was every fetch; the API discovery document was 64% of pilot traffic). `reset_ee()` forces re-initialisation after a credentials or project change.
- A basin needing more than `MERIT_GEE_MAX_WINDOW_CELLS` (4.5M cells) raises `MeritGeeWindowLimit` ("MERIT_GEE_WINDOW_LIMIT: ...") before any request instead of an Earth Engine
  memory error; the router records `MERIT_GEE_WINDOW_LIMIT` as the fallback reason. The 4.5M value was not run live (4.1M is the largest verified; above that is untested).
- `snap_outlet_on_merit` normalised its "nearest stream" score by the whole window's maximum UPA, so the snapped cell could change when the adaptive window grew. It now normalises
  by the candidates inside the search radius.
- Every returned (and every cached) GeoTIFF is checked against the requested native window (shape and transform); a mismatch raises on fetch and discards a cache entry.
  The cache key carries `"grid": "merit-native-v1"` and no longer includes `scale_m`, so old resampled entries are not reused.
- `merit_flowdir_pipeline` created and used `~/.aihydro/cache/...` at import, ignoring `AIHYDRO_HOME`; `merit_manager` read `~/.aihydro/config.json` and defaulted its
  root to `~/.aihydro/merit` the same way. Both now use `$AIHYDRO_HOME` (default `~/.aihydro`); the cache directory is created on first use. Both use the single shared helper `aihydro_watershed/_paths.py:aihydro_home` (from #11).

- **NLDI `hydrolocation` request form (CAMELS pilot, 11/20 gauges got HTTP 502).** pynhd 0.19.4 (`NLDI.comid_byloc`) sends `coords=POINT(lon lat)`
  with the space encoded as `+`. The USGS service answers some points in that form with 502 (CAMELS 08155200: 502 in 6/6 `+` requests, 200 in 8/8
  `%20` requests, same session; body: "Error executing process ... Read timed out (read timeout=5)", `upstream_status` 400). `nldi_point.hydrolocation_comid`
  now builds the `%20` request itself (used by the snap, COMID resolution, point delineation and the gauge COMID fallback), retries 5xx / 429 /
  connection errors with back-off (3, 8, 15 s; honours `Retry-After`) and raises `NLDI_UNAVAILABLE` carrying the HTTP status and host, or
  `NLDI_NOT_FOUND`, instead of a geopandas "no active geometry" error. The service also has a separate transient fault (upstream 5 s read timeout, about half of requests
  for 02298608 even in the `%20` form), which the retries absorb but cannot remove. See ARCHITECTURE.md "NLDI request form and outlet ambiguity".
- **Latency bound.** Our NLDI request loop has a total deadline of 90 s per router NLDI tier call (`nldi_deadline`), and an outage is remembered for
  60 s once two consecutive lookups have exhausted their retries on transient failures (no response, 429, 5xx; a 4xx, a deadline spent before any request, or any HTTP 200 never counts or resets the count), during which `hydrolocation_comid` fails immediately (`NLDI_UNAVAILABLE`) so repeated calls fall through to the next tier. Worst case: one NLDI tier call
  spends at most about 90 s in hydrolocation requests (a request started near the limit is capped to the time left, minimum 1 s) before `NLDI_UNAVAILABLE`; `delineate_from_point` has up to two NLDI tiers
  (expected-area, then quick), so about 180 s worst case, and the second is skipped by the memo after the first failure. Without the deadline the worst case was 4 x 30 s + 26 s back-off = 146 s per call.
  A lookup made outside a router tier (worker threads, snap) gets the same 90 s cap. Every `get_basins` wait in the NLDI delineation path is abandoned at the deadline (`NLDI_UNAVAILABLE`; remaining candidates skipped) and runs on at most 4 shared daemon threads, failing fast with `NLDI_UNAVAILABLE` when all are hung. A lookup counts toward the outage streak only if its last attempt was transient, and a 4xx resets the streak. pynhd's navigation calls keep pynhd's timeouts.
- **No silent downstream jump (CAMELS 10336660: 29.8 km2 gauge returned a 1414 km2 basin, x43.6).** With no `expected_area_km2` and a nearest reach draining
  < 80 km2, NLDI point delineation used to walk 30 km downstream and keep the largest basin. It now returns the nearest reach's basin, adds the
  `NLDI_OUTLET_AMBIGUOUS` quality flag and `outlet_ambiguity` (selected COMID, candidate downstream COMIDs with areas, how to resolve). The outlet moves downstream only with
  `allow_downstream_expansion=True` (new argument of `delineate_from_point` and `delineate_nldi_at_point`; the result then says `outlet_moved_downstream`).
  `resolve_comid_for_quick` keeps its signature (new keyword) and its default behaviour changes accordingly.

- The precipitation fetch in `extract_hydrological_signatures` now has a wall-clock bound (default 600 s, `AIHYDRO_PRECIP_FETCH_TIMEOUT_S`
  to override). Before, an unreachable network blocked the caller indefinitely inside the data layer's HTTP client; now the fetch fails with
  "precipitation fetch did not complete within N s (network unreachable or data service not responding)" in the precipitation provenance record,
  and the signatures that need precipitation are reported as unavailable, as for any other fetch failure.
- `tests/test_signatures_offline.py`: the two tests that hung without a network now pass `precipitation="skip"`, and an autouse fixture
  makes any precipitation fetch in that module fail loudly.

## [0.3.0] - 2026-10-30 BREAKING-VALUE

Version 0.3.0, not 0.2.2: a `<0.3` pin on an installed 2.8.4 would otherwise take these changed values.

Tool audit T1-P (D08, D09, D26 part) and the methods review: period statistics over incomplete periods.

### Added

- `signatures.periods.complete_periods(dates, values, freq, min_fraction, window, calendar, known_absent, year_start_month)`:
  the one period-completeness rule. It returns the complete periods, the excluded ones (observed/expected days), the periods whose
  expected days were reduced (`calendar_adjusted`) and a mask over the input. Conventions: `month`, `calendar_year`, `water_year`
  (1 Oct-30 Sep, labelled by the year it ends; Bulletin 17C Appendix 10, p. 104), `climatic_year` (1 Apr-31 Mar, label = the calendar year in which it BEGINS;
  USGS SIR 2006-5035: a climatic year is "designated by the calendar year in which it begins"; SIR 2008-5126 section 3 uses April 1 to
  March 31 for 7-day minimums), `custom_year` (+ `year_start_month`, for Southern Hemisphere and other regimes).
- **Days a source never holds.** `calendar="no_leap_dec31"` (GridMET via pygridmet, Daymet have no 31 Dec in leap years) or an explicit
  `known_absent` list takes those days out of a period's expected days, so a leap-year December is not dropped (without it SPI-12 lost
  24% of its values). Reported as `months_calendar_adjusted` / `years_calendar_adjusted`; a day absent for any other reason still excludes.
- Thresholds: months 1.0 (every day) is SOURCED (WMO-No. 1203 (2017) s4.4.1(c), p.7: "A monthly value for a sum parameter (for example, total precipitation) can only be
  calculated if there are complete data over the month", with exceptions for estimated or accumulated data within the 11-day and
  5-consecutive-day limits; WMO-TD 341 (1989), PDF p.6: "Monthly totals should be based on a full month's data"; the SPI User Guide WMO-No. 1090 marks missing as -99, asks for 20-30 years and has no within-month rule).
  Years 0.95 is an engineering choice (the package's water-balance fraction), not sourced. Both are arguments.

### Changed

- `spi` / `spei`: a month is used only if complete. A record ending on the 3rd no longer reads that month as drought (SPI-1 -3.09 on the
  audit series); an interior gap or NaN day gives NaN months, not zero precipitation (2010-07 -1.83 -> None); a window containing an
  incomplete month is not evaluated. `n_months` now counts months WITH an index value (`n_months_in_span` is the old count);
  `n_per_calendar_month` reports the windows behind each calendar month; a record in which no month can be standardised (fewer than 5
  complete windows per calendar month) raises a `ValueError` that says so. New: `months_excluded`, `n_months_complete`,
  `min_month_fraction`, `month_convention`, arguments `min_month_fraction` and `calendar`.
- `annual_maxima`: water year (was calendar year), years under 95% of days excluded and listed (`excluded_years`), `statistic` states
  "annual maximum of daily mean discharge (not the instantaneous peak)". Arguments `year_type`, `min_fraction`, `window`, `calendar`,
  `year_start_month`. On the audit series Q100 988 -> 688 and Q2 292 -> 335 (locked fixture: `test_locked_q100_fixture_partial_years_inflate_the_design_flood` in aihydro-tools).
- `seven_q10`: climatic year by default (the docstring said water year; the code used calendar year); 7-day mean formed on the
  calendar axis (no splicing across a gap). A year is used only if at least 95% of its days AND 95% of its 7-day windows are
  evaluable: one missing day removes 7 windows, and 18 scattered missing days left a third of the windows and read +108% high.
  New keys `years`, `year_type`, `year_convention`, `min_fraction`, `min_window_fraction`, `excluded_years` (with `reason`),
  `windows_per_year`. The 0.95 window fraction is an engineering choice, flagged.
- Timezone-aware dates keep their local calendar day (`tz_localize(None)`), in all three statistics.

## [0.2.1] - 2026-10-28

### Changed

- Reworded the slope_fdc definition provenance to describe the definition neutrally; values unchanged. (The earlier wording remains in git history; that is accepted.)
- Corrected the label of the packaged/compared attribute file `camels_hydro.txt`: it is CAMELS attributes v2.0 ("CAMELS-US v1.2" is the time-series version). Wording only; values unchanged.

## [0.2.0] - 2026-10-22 (Science Wave 3)

Evidence: `docs/vision-2040/evidence/signature-definitions-1` (scripts, CAMELS R sources, sha256, results).

### BREAKING-VALUE

- **`slope_fdc` changes meaning.** It is now `(Q66 - Q33) / (mean(q) * 0.33)`: non-exceedance percentiles (R type 7),
  zero-flow days kept, defined only when mean(q) > 0, no logarithm. Before 0.2.0 it was the log-space Sawicz et al.
  (2011) slope (e.g. 2.697 on 01013500; now 1.528). The new value reproduces the CAMELS attributes v2.0 file at 8.7e-9
  (01013500) and 2.3e-7 (01031500) relative. The old value is kept as **`slope_fdc_sawicz2011`**. The same change is
  made in `flow_duration.flow_duration_curve`. Provenance (carried as `signature_definitions["slope_fdc"]`):
  "slope of the flow duration curve between the 33rd and 66th percentiles, mean-normalised, as in the CAMELS attributes v2.0 distributed file `camels_hydro.txt` (naddor/camels@88e7a0d2d2 hydro/hydro_signatures.R:97). Reproduces the file at <=2.3e-7 relative on 01013500 and 01031500. Not proven to be the script that produced the file." This is not "the CAMELS
  definition" in general: later naddor/camels code (used for CAMELS-GB) emits the log form under the same name.
  Consumers that compare `slope_fdc` across datasets or versions must read `signature_definitions`.
- **Baseflow index kernel replaced; `separate_baseflow` BFI changes (e.g. 0.36 -> 0.445 on 01031500).** One canonical kernel,
  `baseflow.lyne_hollick_ladson2013` (alpha 0.925, 3 passes forward-backward-forward, 30-day reflection, quick flow
  subtracted only where positive, final clip at zero; the Ladson et al. 2013 recipe CAMELS used), now serves
  `extract_hydrological_signatures`, `baseflow.lyne_hollick` (what the `separate_baseflow` tool calls) and the BFI
  bootstrap. It reproduces CAMELS attributes v2.0 at 1.1e-7 (01013500: 0.585226) and 1.9e-6 (01031500: 0.445091) relative.
  Replaced: `baseflow.lyne_hollick` clamped the baseflow at every step and fed the clamped value back into the recursion
  (5-19% low, matches no reference implementation; kept, deprecated, as `lyne_hollick_legacy_clamp`), and the signatures
  filter lacked the 30-day reflection (+3e-4 relative). The two were the same filter in different algebraic forms; the
  difference was constraint handling. `BASEFLOW_SEPARATION_METHOD` is now `lyne_hollick_ladson2013`, so the
  `baseflow_method` param and the `baseflow_method=...` quality flag change with it.
- **`runoff_ratio` and `stream_elas` use complete water years only.** A water year counts when at least 95% of its days
  have both streamflow and precipitation (engineering choice, source to verify; `WATER_YEAR_MIN_PAIRED_FRACTION`).
  Incomplete years no longer enter either signature, so values can change where precipitation had holes.
- **Confidence intervals are water-year grouped, and the daily block bootstrap is gone.** `_uncertainty` entries now
  come from aihydro-core `grouped_bootstrap` (percentile, B = 2000, 90%, whole water years resampled), `method`
  `bootstrap_group` (was `bootstrap_block`); the BFI entry no longer has `block_size`. Fewer than 10 water years: no
  interval (`ci_refused: true`, `ci_low`/`ci_high` null, reason given) instead of a block-bootstrap number.
  **Requires aihydro-core >= 0.2.11** (pin raised; CI needs core 0.2.11 on PyPI).
- **`BFI_BOOTSTRAP_BLOCK_DAYS` is removed** (with the moving-block BFI bootstrap); `_baseflow_index_uncertainty` now takes the
  daily series and returns the grouped interval.
- **Group counts are complete water years.** The 10-group minimum and the reported `n_groups` count only water years
  holding >= 95% of their days (`n_groups_complete`, `n_partial_groups`, `n_groups_all` on every entry); partial edge
  years stay in the point estimate and the resample. A 2000-01-01..2009-12-31 window has 9 complete years and is
  refused. Refusals carry the same keys as computed entries (`coverage_note`, `low_n`, `B`, ...).
- **`stream_elas` anomalies use the day-weighted long-term mean** (exactly CAMELS `e_qp_sanka`; the first 0.2.0 draft used
  the mean of annual means, ~3e-5 different), in the point estimate and the CI alike.
- **Short series in `lyne_hollick`:** 30 days or fewer raises `ValueError` (as BFI.R); 31-364 days is computed with a
  `WarmupBiasWarning` (the index is biased low by the start-up transient; `warmup_bias_disclosure(n)` gives the
  record-able flag).
- Leading/trailing water years of the requested window with no paired day are reported in `water_years_excluded`.
- `reference_comparison` reads the used period and the precipitation completeness from the record, not from the caller:
  new RR/elasticity blockers `water_years_excluded`, `precipitation_used_with_gaps` (a skipped leap-year Dec 31 --
  `client_calendar_assumption` -- alone does not block) and `streamflow_period_differs`.
- **`zero_q_freq` counts days with q <= 0** (was `== 0`; CAMELS `sum(q<=0)/length`).

### Added

- `signature_definitions` (plain key): formula, quantile type, zero handling and source for slope_fdc,
  slope_fdc_sawicz2011, baseflow_index, q5, q95, high/low/zero_q_freq, runoff_ratio, stream_elas.
- `_precipitation` gains `n_expected_days`, `completeness`, `missing_ranges` (run-compressed `{start, end, days,
  reason}`; reason codes are passed through from aihydro-data when it supplies them, otherwise `absent_in_response`),
  `data_coverage`, `water_year_rule`, `water_years_used`, `water_years_excluded`, `n_paired_days_used`.
- `precipitation_status` / `_precipitation.status` gains **`used_with_gaps`** (computed, but not from a complete
  record; the reason lists the gaps and excluded years) and `rejected` for "no water year is complete".
  Consumers that test `status == "used"` must also accept `used_with_gaps`.
- `baseflow.lyne_hollick_ladson2013`, `flow_duration.fdc_slope_camels_file`, `flow_duration.fdc_slope_sawicz2011`.
- Grouped CIs for `q_mean`, `q_median`, `q5`, `q95`, `flow_variability`, `baseflow_index` (every valid streamflow day),
  `runoff_ratio` (sum q / sum p over the paired days, complete water years) and `stream_elas` (indicative; anomalies
  recomputed per resample, duplicated years contribute duplicated ratios; `n_years_near_zero_dP` reported and
  `ci_degraded` set when any year has |dP/mean P| < 0.02, an engineering threshold). The point estimate of each
  is computed from the same item set as its CI. Each entry carries core's `scope` (`varies` / `assumes` /
  `excludes`), `ci_degraded`, `ci_reason`, `valid_frac`, `low_n` (< 30 groups), `coverage_note` (percentile intervals
  under-cover at small group counts), `n_groups`, and `annual_lag1_autocorrelation` (between-year persistence the
  bootstrap does not capture).
- `_served_series` (dates, q_cms, qualifier and approval codes) so callers can bind validators to the series used.
- `qualifier_exposure`: per signature, the share of estimated / other-qualified / provisional days among the days that
  determine it (named in the output; tail and baseflow sets are engineering definitions). Present for fetched USGS
  series only.
- `signatures.comparison.reference_comparison(sigs, ref_row, area_used, area_ref, period_used, period_ref)`: pure
  function; per signature flags `area_dependent` (with the value rescaled to the reference area), `definition_comparable`,
  `forcing_differs`, `period_differs`, the resulting `blockers` and `can_claim_difference`; plus `climate_context`
  (frac_snow, aridity, p_seasonality).
- Water-year labels use aihydro-core `water_year` everywhere (one convention).
- Test fixture `served_streamflow_01031500.csv` (USGS NWIS, public domain) with provenance.

### Fixed

- `high_q_freq` / `low_q_freq` are days/year (already so in `SIGNATURE_UNITS`; stale wording removed).

## [0.1.6] - 2026-10-20 (CI fix: NumPy compatibility)

### Fixed

- **Delineation no longer fails on the newest NumPy (`AttributeError: module 'numpy' has no attribute 'in1d'`).**
  `pysheds` 0.5 (its latest release, so there is no upstream fix to pin to) calls `np.in1d` in
  `Grid.accumulation`; NumPy 2.4.6 and 2.5.4 no longer have it (2.2.6 does; the removal version was observed, not looked up in release notes). New
  `aihydro_watershed/_numpy_compat.py` restores `np.in1d` (via `np.isin(...).ravel()`, which keeps the flat
  1-D result) only when it is missing, and is applied where pysheds is imported (`delineation/pysheds_pipeline.py`,
  `characterize/twi.py`). A no-op on NumPy versions that still ship `in1d`. Reproduced with NumPy 2.5.4 (CI saw it on 3.11 / NumPy 2.4.6 and 3.12 / 2.5.4) + pysheds 0.5:
  `test_pysheds_snap_out_reports_snapped_cell_on_grid` failed before, passes after. The shim is also applied at `import aihydro_watershed` time and exported as `aihydro_watershed.ensure_numpy_compat()`, so callers that import pysheds directly (e.g. the tools pack) are covered. CI gains a step (Python 3.11/3.12, whose resolved NumPy lacks `in1d`; 3.10 resolves 2.2.6) that asserts `in1d` is absent and runs the compat + place-identity tests. Version bumped because runtime
  behaviour changed for users on the newest NumPy.

## [0.1.5] - 2026-10-20 (session study 01031500, S1 / M5)

### Notes

- `slope_fdc` uses the natural log over the 33-66% exceedance span. It differs from CAMELS' convention and is not comparable to the CAMELS column; a convention reconciliation is tracked separately.

### Added

- **`units` map on every signature result** (`SIGNATURE_UNITS`, derived from the computing code). The flow
  statistics `q_mean`, `q_std`, `q5`, `q95`, `q_median` are basin-mean depth rates in **mm/day**, not discharge
  (a real session read `q_mean = 2.02` as m3/s, a ~9x error). `zero_q_freq` is a fraction (not days/year);
  `hfd_mean` is day of water year (1 = 1 Oct); `high/low_q_freq` are days/year; durations are days.
  The map is a plain key so it survives the tools layer's `_`-key drop and reaches the sealed record.
- **Streamflow data digest.** The series the signatures were computed from (m3/s) is now digested like
  precipitation: `streamflow_digest`, `streamflow_n_days`, `streamflow_n_valid_days` (finite values), `streamflow_start`, `streamflow_end` (plain keys)
  and `_streamflow` (digest scheme, units). Absent when no usable streamflow was obtained.

## [0.1.4] - 2026-10-17 (Wave U patch, proof 3 finding E7)

### Fixed

- **Result provenance no longer takes its version from distribution metadata.** `HydroMeta.version` in
  `extract_hydrological_signatures`, `extract_geomorphic_parameters` and `delineate_watershed` was read
  with `importlib.metadata.version("aihydro-tools")`, which reads the `dist-info` of an editable install and
  goes stale when the source is bumped (sealed records stated 2.3.7 while 2.4.0 ran). It is now the
  imported `aihydro_watershed.__version__` (new `aihydro_watershed/_version.py`). This package always states its own
  identity and never looks at the tools layer; the tools run record states the tools version and keeps this one as
  the producer version. When distribution metadata disagrees, `meta.params["version_check"]` carries `module_version`, `distribution_version` and
  `version_mismatch: true`; on agreement nothing is added.

## [0.1.3] - 2026-10-17 (Wave U / U2)

### Changed

- **USGS streamflow and gauge metadata now come from the Water Data API through the
  aihydro-data adapter** (`aihydro_data.sources.usgs_waterdata`); the legacy
  `waterservices.usgs.gov` path (decommissioned 2027-02-22) is gone from this package.
  `dataretrieval` and `pygeohydro.NWIS` are no longer imported anywhere in watershed
  (requires `aihydro-data>=0.2.3`).
- `_fetch_streamflow_internal` returns `{q_cms, meta, quality, flags, upstream, site}`:
  the adapter's quality summary (approval and qualifier counts, provisional span, gaps), a
  per-day `approval_code` / `qualifier_code` frame aligned with `q_cms`, and the request
  provenance (URLs, body sha256). On failure it still returns `None` but leaves the
  adapter's error code and whether it is terminal in `_fetch_diag()`.
- `extract_hydrological_signatures`: `_streamflow_source` additionally carries `site`,
  `quality_summary` and `upstream` for an observed gauge, and `error` when no streamflow was
  obtained. A refused or unreachable gauge is reported as `observation: "none"`; no modelled
  product is substituted for a gauge request. Citation URL updated.
- `delineate_watershed` takes name, position and HUC from the API's monitoring-location
  record and adds `gauge_hydrologic_unit_code` (12-digit; legacy was 8-digit),
  `gauge_horizontal_datum` (the API's `original_horizontal_datum`, e.g. NAD83),
  `gauge_coordinate_note` (the API does not state the geometry CRS) and `gauge_site_id`.
  `gauge_name` capitalisation follows the API ("near", legacy "Near"). The NLDI citation
  host is `api.water.usgs.gov/nldi`.

- **Gauge-vs-polygon area check (ADR-003 A2 R0, explicit id).** For a fetched gauge the
  supplied `area_km2` is compared with the gauge's total drainage area (adapter
  `resolve_gauge(gauge_id=...)`, mi2 x 2.589988110336; pass when |ln ratio| <= ln 1.10).
  Recorded in `_streamflow_source["site_resolution"]` and mirrored in the non-underscore keys
  `gauge_area_check` (pass | fail | unavailable), `gauge_area_ratio`, `gauge_area_km2` and
  `streamflow_warnings`. `fail` warns (log + `streamflow_warnings`) and never refuses; the
  runoff depth still uses the supplied area. No check for caller-supplied series.

### Precision note (not a defect fix)

- Discharge is converted once, with the exact `0.028316846592` (1 ft = 0.3048 m). Values
  from the former watershed constant `0.0283168` differ by up to 1.6e-6 relative; values
  that came through aihydro-data's `0.028316847` differ by 1.44e-8. Scale-free signatures
  (baseflow index, event frequencies and durations, timing) are equal to floating-point
  precision (`slope_fdc` differs by 1 ulp); scale-bearing ones (q_mean, q5, q95, ...) move
  by that relative amount. Proof-1 gauge 01013500 WY1990-2009: `baseflow_index`
  0.5853970180143581 and `high_q_freq` 6.1 are bit-identical.
- A record shorter than 365 days (fetched or caller-supplied) is labelled `observation:
  "none"` with `error: {code: "INSUFFICIENT_RECORD", n_days, terminal: false}`; `site` and
  `quality_summary` are kept when the record was fetched.
- An unexpected exception inside the streamflow fetch maps to the non-terminal
  `FETCH_FAILED` code (only adapter errors carry their own code and terminal flag).
- `delineate_watershed` reports a missing adapter as `DEPENDENCY_ERROR` naming
  `aihydro-data>=0.2.3`, separately from the pynhd / geopandas check.

## [0.1.1] - 2026-10-03

### Fixed

- **`q5` and `q95` signatures follow CAMELS (defect P2-D0).** `compute_flow_stats_camels`
  and the bootstrap uncertainty in `extract_hydrological_signatures` computed `q5` as the
  95th percentile of daily flow and `q95` as the 5th, the opposite of CAMELS (Addor et al.
  2017, Table 3: `q5` = 5% flow quantile, low flow; `q95` = 95% flow quantile, high flow).
  This was a computation error under correct-looking labels, not only a rename: the values
  for each key changed. On the proof-1 gauge 01013500 the platform gave q5 = 6.357,
  q95 = 0.240 against CAMELS q5 = 0.241, q95 = 6.373. After the fix q5 = 0.2405 and
  q95 = 6.3566 (mm/day). `result.data` gains the non-underscore key `flow_quantile_convention =
  "camels_nonexceedance_v1"` (non-underscore so the tools layer keeps it in the sealed evidence; the package version moves to 0.1.1 so the run record's `env_digest` also changes); results sealed before this change lack the key (and were produced by watershed < 0.1.1) and carry the
  swapped values, so readers must treat a missing marker on a `q5`/`q95` as swapped (the
  historical records are not rewritten). The hydrologic exceedance flows of
  `flow_duration_curve` (`Q5` high, `Q95` low) are a separate, documented convention and
  are unchanged. Tests: `tests/test_q5_q95_convention.py`. Audit:
  `docs/vision-2040/findings/defect-q5-q95.md`.
