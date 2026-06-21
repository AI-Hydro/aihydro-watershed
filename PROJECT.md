# aihydro-watershed

## What it is
Standalone watershed delineation + characterization package, carved out of
`aihydro-tools` so hydrologists can `pip install aihydro-watershed` for just the
delineation/characterization primitives. Global pour-point delineation (NLDI +
MERIT-Hydro/pyflwdir) + geomorphics, TWI, terrain/runoff, and watershed-scale
signatures.

## Status
**v0.1.0 — Wave A extraction complete (2026-06-21).**

## Where to read next
| Goal | Go to |
|---|---|
| Architecture diagram | `ARCHITECTURE.md` |
| Migration from aihydro-tools | `MIGRATION.md` |
| The extraction plan | `../aihydro-tools/local-docs/WATERSHED_EXTRACTION_PLAN.md` |
| The decision record | `../aihydro-tools/local-docs/ADR-002-watershed-extraction.md` |
| The ecosystem roadmap | `../docs/ECOSYSTEM_ROADMAP.md` |
| The result contract | `aihydro_core.contracts` (HydroResult) |

## Wave A — complete

- **A0 ✅** `HydroResult` contract promoted to `aihydro-core`; re-export shim in tools.
- **A1 ✅** Package scaffolded: `pyproject.toml`, five-subpackage tree, layering guard.
- **A2 ✅** 5 module units moved: delineation / merit / characterize / terrain / signatures.
  27 compatibility shims left in `ai_hydro/analysis/` and `ai_hydro/data/` for one release.
- **A3 ✅** `aihydro-tools` wired to consume this package; 929 offline tests pass; `TOOL_TIERS`
  updated to include `lsh_dynamic_attributes` + `lsh_events`.
- **A4 ✅** Parity tests: NLDI delineation (Potomac 01638500) within 30 % of published area;
  fast DEM delineation (Rhine/Cologne) returns non-empty polygon. AST layering guard green.
- **A5 ✅** Docs: README, ARCHITECTURE, MIGRATION, PROJECT updated.

## Non-goals
- The flood-inundation suite (separate subsystem; not here).
- No upward import of the `ai_hydro` tools package (layering guard enforces this).

## How to test
```bash
pip install -e ".[dev]"
pytest -m "not live" -q    # offline: layering guard (always green)
pytest -m live -v          # live: NLDI + fast-DEM parity (requires network)
lint-imports               # layering contract (import-linter)
```
