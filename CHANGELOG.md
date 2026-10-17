# Changelog — aihydro-watershed

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

### Precision note (not a defect fix)

- Discharge is converted once, with the exact `0.028316846592` (1 ft = 0.3048 m). Values
  from the former watershed constant `0.0283168` differ by up to 1.6e-6 relative; values
  that came through aihydro-data's `0.028316847` differ by 1.44e-8. Scale-free signatures
  (baseflow index, event frequencies and durations, timing) are unchanged; scale-bearing
  ones (q_mean, q5, q95, ...) move by that relative amount. Proof-1 gauge 01013500
  WY1990-2009: `baseflow_index` 0.5853970180143581 and `high_q_freq` 6.1 reproduce exactly.

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
