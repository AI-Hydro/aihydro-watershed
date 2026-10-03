# Changelog — aihydro-watershed

## [Unreleased]

### Fixed

- **`q5` and `q95` signatures follow CAMELS (defect P2-D0).** `compute_flow_stats_camels`
  and the bootstrap uncertainty in `extract_hydrological_signatures` computed `q5` as the
  95th percentile of daily flow and `q95` as the 5th, the opposite of CAMELS (Addor et al.
  2017, Table 3: `q5` = 5% flow quantile, low flow; `q95` = 95% flow quantile, high flow).
  This was a computation error under correct-looking labels, not only a rename: the values
  for each key changed. On the proof-1 gauge 01013500 the platform gave q5 = 6.357,
  q95 = 0.240 against CAMELS q5 = 0.241, q95 = 6.373. After the fix q5 = 0.2405 and
  q95 = 6.3566 (mm/day). `result.data` gains `_flow_quantile_convention =
  "camels_nonexceedance_v1"`; results sealed before this change lack the key and carry the
  swapped values, so readers must treat a missing marker on a `q5`/`q95` as swapped (the
  historical records are not rewritten). The hydrologic exceedance flows of
  `flow_duration_curve` (`Q5` high, `Q95` low) are a separate, documented convention and
  are unchanged. Tests: `tests/test_q5_q95_convention.py`. Audit:
  `docs/vision-2040/findings/defect-q5-q95.md`.
