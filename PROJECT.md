# aihydro-watershed

## What it is
Standalone watershed delineation + characterization package, carved out of
`aihydro-tools` so hydrologists can `pip install aihydro-watershed` for just the
delineation/characterization primitives. Global pour-point delineation (NLDI +
MERIT-Hydro/pyflwdir) + geomorphics, TWI, terrain/runoff, and watershed-scale
signatures.

## Status
Alpha — **Wave A extraction in progress.** Last updated: 2026-06-19.

## Where to read next
| Goal | Go to |
|---|---|
| The extraction plan | `../aihydro-tools/local-docs/WATERSHED_EXTRACTION_PLAN.md` |
| The decision record | `../aihydro-tools/local-docs/ADR-002-watershed-extraction.md` |
| The ecosystem roadmap | `../docs/ECOSYSTEM_ROADMAP.md` |
| The result contract | `aihydro_core.contracts` (HydroResult) |

## Current state
- **A0 ✅** Contract promoted to aihydro-core (this package depends on `aihydro-core[contracts]`).
- **A1 ⏳** Package scaffolded: pyproject, subpackage tree, layering guard.
- **A2** Move 5 module units (delineation / merit / characterize / terrain / signatures).
- **A3** Rewire aihydro-tools to consume this package.
- **A4** Tests + delineation parity (CONUS gauge + global pour point).
- **A5** Docs + migration guide.
- **Next step:** Wave A2 — migrate the delineation engine first.

## Non-goals
- The flood-inundation suite (separate subsystem; not here).
- No upward import of the `ai_hydro` tools package (layering guard enforces).

## How to test
```bash
pip install -e ".[dev]"
pytest -q                 # layering guard + (post-A2) science tests
lint-imports              # layering contract
```
