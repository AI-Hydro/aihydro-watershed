# Live evidence: MERIT Hydro native-grid Earth Engine request (aihydro-watershed PR #12)

Run date 2026-11-05, Earth Engine with the owner's existing credentials (scratch `HOME`/`AIHYDRO_HOME`, credentials read-only).
Gauges: the 13 failing + 3 successful gauges of `docs/vision-2040/evidence/camels-delineation-1` (pilot). Inputs are the pilot's gauge coordinates;
reference areas come from that folder's `reference/reference_671.csv` (read only).

## Files
- `per_gauge_live_run.csv`: one row per gauge. `pilot_*` is the pilot outcome (before). `final_window_*` is the last window requested
  for a successful run, taken from the recorded bbox (cells = columns x rows; half-side is derived from the bbox, not logged separately).
  Errors are % of the MERIT polygon area against the CAMELS polygon, `area_gages2` and `area_geospa_fabric`. IoU is against the CAMELS polygon in EPSG:5070
  (computed from the polygons returned by this run).
- `raw_live_run.json`: the run's own per-gauge record (area, snap, bbox, flags, seconds, or the error), polygons removed.
- `scripts/run_live_native_grid.py`: the runner (calls `delineate_merit_gee`). `scripts/probe_native_getdownloadurl.py`: the 32/51/64/82 km native-grid
  request probe on gauge 01547700. `scripts/analyze_snap_bias_prototype.py`: snap/UPA analysis, see "Provenance gaps".

## Result
15 of 16 gauges return a polygon; 05495000 raises `MERIT_GEE_WINDOW_LIMIT` (needs a window above the cap). Before: 3 of 16 returned a polygon (the other 13 failed).
The 3 previously successful gauges differ by under 0.1% in area from the pilot.
Largest window fetched in `per_gauge_live_run.csv`: 3.75M cells (02196000, 82 km derived half-side).

Computed from `per_gauge_live_run.csv` (this run):
- vs `area_gages2`, 14 gauges (14158500 excluded: CAMELS fabric 122.9 km2 vs 237.1 km2 gages2, MERIT 233.8): mean +0.32%, median -0.06%, range -2.77% to +4.86%.
  With 14158500 included (15): mean +0.21%, median -0.13%.
- vs the CAMELS polygon, the same 14: mean -1.78%, median -0.55%. Correlation of that error with `area_geospa_fabric / area_gages2 - 1`: -0.85.
  So the "too small" pilot bias is mostly against the fabric polygon, which exceeds `area_gages2` where the two differ. n is small and CONUS-only.
- Polygon area is 0.084% to 0.185% above MERIT's own UPA at the snapped cell (15 gauges). Pixel-sum and geodesic area agree with the polygon to under 0.1% (unit test).

## Not run live / not recorded
- **The 4.5M-cell cap (`MERIT_GEE_MAX_WINDOW_CELLS`) was not run live.** The largest window verified live is 4.1M cells (see below); 4.5M is untested.
  "About 85 km half-side" is a derived size, not a measured limit.
- `requests_count` is empty: this run did not record Earth Engine request counts or bytes. Do not read the per-gauge `secs` as a cost measure (network latency and cache state varied).
- Gauge 05495000 was run twice. With a 4.0M cap the window of 2320 x 1767 cells (4.1M) was rejected. With the cap at 4.5M that window was fetched without an
  Earth Engine error (the basin still touched its edge), and the next window (3712 x 2827 = 10.5M cells) was rejected with `MERIT_GEE_WINDOW_LIMIT`.
  `raw_live_run.json` holds only the second run's record; the first run's output was console-only. So 4.1M cells is verified from that error text, not from a saved response.
- The native-grid probe on 01547700 (`scripts/probe_native_getdownloadurl.py`) was read from the console, not saved: 32 km 691x915 -> HTTP 200 1,355,505 B in 24.4 s;
  51.2 km 1105x1465 -> 200, 3,452,069 B, 20.8 s; 64 km 1380x1831 -> 200, 5,437,377 B, 19.1 s; 82 km 1769x2345 -> 200, 8,958,503 B, 41.1 s.
  The old-request failure on that gauge at the second window (HTTP 400 "User memory limit exceeded") was also read from the console.
- Code under test: `origin/main` f46c667 plus the working tree before the review fixes of PR #12 (snap normalisation, init lock, cache key, grid checks). Those
  were verified offline only (no live calls after them); the areas/snap of this run were produced by the pre-review snap code (window-maximum normalisation),
  so snapped cells could differ slightly from the final code on some gauges.

## Provenance gaps
`scripts/analyze_snap_bias_prototype.py` read tiles cached by an earlier, tiled prototype of this fix (not in the PR). The prototype's areas equal this run's to 1e-13 (15 gauges),
but its outputs (best-matching-cell analysis, mean absolute UPA error 2.2% for the best cell within 1 km vs 2.7% at the snapped cell, n = 15 excluding 14158500)
are not reproduced from this run's files and are not cited in the CHANGELOG.
