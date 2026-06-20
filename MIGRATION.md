# Migration Guide — from `aihydro-tools` to `aihydro-watershed`

Wave A3 of the ecosystem roadmap moves the scientific implementations out of
`aihydro-tools` and into `aihydro-watershed`. The old `ai_hydro.analysis.*`
and `ai_hydro.data.*` paths are now **compatibility shims** — they re-export
everything from the new locations for one release cycle.

## Who needs to migrate

- Code that directly imports from `ai_hydro.analysis.*` or `ai_hydro.data.*`
  (the shimmed modules listed below).
- Library authors building on top of the watershed/signature primitives.

Code that uses `aihydro-tools` **only as an MCP server** (the 144 MCP tools)
does not need to change.

## Import map

Replace the old import with the new one. Both work during the shim period, but
the old paths will be removed in a future release.

### Delineation

| Old | New |
|---|---|
| `ai_hydro.analysis.delineation.router` | `aihydro_watershed.delineation.router` |
| `ai_hydro.analysis.delineation.nldi_point` | `aihydro_watershed.delineation.nldi_point` |
| `ai_hydro.analysis.delineation.merit_flowdir_pipeline` | `aihydro_watershed.delineation.merit_flowdir_pipeline` |
| `ai_hydro.analysis.delineation.merit_snap` | `aihydro_watershed.delineation.merit_snap` |
| `ai_hydro.analysis.delineation.pysheds_pipeline` | `aihydro_watershed.delineation.pysheds_pipeline` |
| `ai_hydro.analysis.delineation.dem_fetch` | `aihydro_watershed.delineation.dem_fetch` |
| `ai_hydro.analysis.delineation.dem_conditioning` | `aihydro_watershed.delineation.dem_conditioning` |
| `ai_hydro.analysis.delineation.types` | `aihydro_watershed.delineation.types` |
| `ai_hydro.analysis.delineation.utils` | `aihydro_watershed.delineation.utils` |

### Characterization

| Old | New |
|---|---|
| `ai_hydro.analysis.watershed` | `aihydro_watershed.characterize.watershed` |
| `ai_hydro.analysis.geomorphic` | `aihydro_watershed.characterize.geomorphic` |
| `ai_hydro.analysis.twi` | `aihydro_watershed.characterize.twi` |
| `ai_hydro.analysis._dem` | `aihydro_watershed.characterize._dem` |

### Terrain / runoff

| Old | New |
|---|---|
| `ai_hydro.analysis.curve_number` | `aihydro_watershed.terrain.curve_number` |
| `ai_hydro.analysis.event_runoff` | `aihydro_watershed.terrain.event_runoff` |
| `ai_hydro.analysis.erosion` | `aihydro_watershed.terrain.erosion` |

### Signatures

| Old | New |
|---|---|
| `ai_hydro.analysis.signatures` | `aihydro_watershed.signatures.signatures` |
| `ai_hydro.analysis.baseflow` | `aihydro_watershed.signatures.baseflow` |
| `ai_hydro.analysis.flow_duration` | `aihydro_watershed.signatures.flow_duration` |
| `ai_hydro.analysis.flood_frequency` | `aihydro_watershed.signatures.flood_frequency` |
| `ai_hydro.analysis.drought_indices` | `aihydro_watershed.signatures.drought_indices` |

### MERIT data

| Old | New |
|---|---|
| `ai_hydro.data.merit_download` | `aihydro_watershed.merit.merit_download` |
| `ai_hydro.data.merit_manager` | `aihydro_watershed.merit.merit_manager` |
| `ai_hydro.data.merit_map_layers` | `aihydro_watershed.merit.merit_map_layers` |
| `ai_hydro.data.region_presets` | `aihydro_watershed.merit.region_presets` |
| `ai_hydro.data.wbd_layers` | `aihydro_watershed.merit.wbd_layers` |

## unittest.mock patch targets

If your tests use `unittest.mock.patch` to intercept calls in the delineation
or characterization code, update the patch target to the **new module path**
because `@patch` must match where the code actually executes.

```python
# Before
@patch("ai_hydro.analysis.delineation.nldi_point.delineate_nldi_at_point")

# After
@patch("aihydro_watershed.delineation.nldi_point.delineate_nldi_at_point")
```

Exception: if an **MCP tool** (in `ai_hydro.mcp.tools_analysis`) does a lazy
import from the shim (e.g. `from ai_hydro.data.merit_manager import MeritDataManager`),
keep the patch target at the shim path so the lazy import picks it up:

```python
# MCP-tool test — keep shim path because tools_analysis imports from the shim
with patch("ai_hydro.data.merit_manager.MeritDataManager") as MockMgr: ...
```

## Install

```bash
pip install "aihydro-watershed[delineation,analysis]"
# or for local editable dev:
pip install -e /path/to/aihydro-watershed --no-deps
```
