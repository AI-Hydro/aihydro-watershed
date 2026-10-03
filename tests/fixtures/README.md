# Test fixtures

- `served_streamflow_01013500.csv` — daily streamflow (m3/s) for USGS 01013500 served in
  e2e proof 1 (1989-10-01 to 2009-09-30, 7305 days, area 2258.5163954900077 km2).
  Byte-identical copy of
  `docs/vision-2040/evidence/e2e-proof-1/capsule/data/served_streamflow_01013500.csv`
  (sha256 `683ccb1619efe82e3c124f8a540bdc9af93269af12efe5c3681d88d9438d359a`).
  Used by `tests/test_q5_q95_convention.py`; CAMELS values for this gauge are
  q5 = 0.2411, q95 = 6.3730 mm/day (`camels_hydro.txt`).
