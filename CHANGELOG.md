# Changelog — aihydro-watershed

## [0.2.0] - 2026-10-22 (Science Wave 3)

Evidence: `docs/vision-2040/evidence/signature-definitions-1` (scripts, CAMELS R sources, sha256, results).

### BREAKING-VALUE

- **`slope_fdc` changes meaning.** It is now `(Q66 - Q33) / (mean(q) * 0.33)`: non-exceedance percentiles (R type 7),
  zero-flow days kept, defined only when mean(q) > 0, no logarithm. Before 0.2.0 it was the log-space Sawicz et al.
  (2011) slope (e.g. 2.697 on 01013500; now 1.528). The new value reproduces the CAMELS-US v1.2 file at 8.7e-9
  (01013500) and 2.3e-7 (01031500) relative. The old value is kept as **`slope_fdc_sawicz2011`**. The same change is
  made in `flow_duration.flow_duration_curve`. Provenance (carried as `signature_definitions["slope_fdc"]`):
  "CAMELS-US v1.2 distributed definition. Reproduces the file at <=2.3e-7 relative on 01013500 and 01031500, and
  9.7e-5 on 01516500. Its zero-structure prediction holds on all 671 basins. Source: naddor/camels@88e7a0d2d2
  hydro/hydro_signatures.R:97. It contradicts the log-space definition in Addor et al. 2017 (text and Table 3). No
  erratum or release note found. Not proven to be the script that produced the file." This is not "the CAMELS
  definition" in general: later naddor/camels code (used for CAMELS-GB) emits the log form under the same name.
  Consumers that compare `slope_fdc` across datasets or versions must read `signature_definitions`.
- **Baseflow index kernel replaced; `separate_baseflow` BFI changes (e.g. 0.36 -> 0.445 on 01031500).** One canonical kernel,
  `baseflow.lyne_hollick_ladson2013` (alpha 0.925, 3 passes forward-backward-forward, 30-day reflection, quick flow
  subtracted only where positive, final clip at zero; the Ladson et al. 2013 recipe CAMELS used), now serves
  `extract_hydrological_signatures`, `baseflow.lyne_hollick` (what the `separate_baseflow` tool calls) and the BFI
  bootstrap. It reproduces CAMELS v1.2 at 1.1e-7 (01013500: 0.585226) and 1.9e-6 (01031500: 0.445091) relative.
  Replaced: `baseflow.lyne_hollick` clamped the baseflow at every step and fed the clamped value back into the recursion
  (5-19% low, matches no reference implementation; kept, deprecated, as `lyne_hollick_legacy_clamp`), and the signatures
  filter lacked the 30-day reflection (+3e-4 relative). The two were the same filter in different algebraic forms; the
  difference was constraint handling. `BASEFLOW_SEPARATION_METHOD` is now `lyne_hollick_ladson2013`, so the
  `baseflow_method` param and the `baseflow_method=...` quality flag change with it.
- **`runoff_ratio` and `stream_elas` use complete water years only.** A water year counts when at least 95% of its days
  have both streamflow and precipitation (engineering choice, source to verify; `WATER_YEAR_MIN_PAIRED_FRACTION`).
  Incomplete years no longer enter either signature, so values can change where precipitation had holes.
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
