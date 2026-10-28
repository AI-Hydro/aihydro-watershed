# Test fixtures

- `served_streamflow_01013500.csv` — daily streamflow (m3/s) for USGS 01013500 served in
  e2e proof 1 (1989-10-01 to 2009-09-30, 7305 days, area 2258.5163954900077 km2).
  Byte-identical copy of
  `docs/vision-2040/evidence/e2e-proof-1/capsule/data/served_streamflow_01013500.csv`
  (sha256 `683ccb1619efe82e3c124f8a540bdc9af93269af12efe5c3681d88d9438d359a`).
  Used by `tests/test_q5_q95_convention.py`; CAMELS values for this gauge are
  q5 = 0.2411, q95 = 6.3730 mm/day (`camels_hydro.txt`).
- `served_streamflow_01031500.csv` — USGS NWIS daily mean discharge (m3/s), site 01031500,
  1989-10-01 to 2009-09-30, 7305 days, no gaps. Public domain (USGS). Written from the aihydro-data
  cache entry `b9d1de8bf04e2bf6f8b463ac` (NWIS_STREAMFLOW, fetched 2026-06-17); byte-identical copy of
  `docs/vision-2040/evidence/signature-definitions-1/data/q_01031500_wy1990_2009.csv`
  (sha256 `e01921cdd60e1750e3d27323aea4f8dc32632976c68fd0e961a3b64a444cbccc`, provenance JSON beside it).
  Used by `tests/test_signature_definitions_020.py`; CAMELS attributes v2.0 values for this gauge are
  slope_fdc = 1.49401920331253 and baseflow_index = 0.445090526012048.
