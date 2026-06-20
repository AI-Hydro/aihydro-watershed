"""
Wave A2c — deterministic module migration script.

Copies source files from aihydro-tools into aihydro-watershed subpackages,
rewrites all package-prefix imports, and reports any unmapped imports for
manual review (A2d seam work).

Run from: MCP/aihydro-watershed/
    python scripts/migrate_a2c.py [--dry-run]

Flags:
  --dry-run   print planned operations without writing anything
  --verify    after writing, import each destination module to check for
              hard SyntaxErrors (does not execute any code)
"""
from __future__ import annotations

import argparse
import ast
import re
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Repo roots
# ---------------------------------------------------------------------------
TOOLS_ROOT = Path(__file__).resolve().parent.parent.parent / "aihydro-tools"
WATERSHED_ROOT = Path(__file__).resolve().parent.parent
WATERSHED_PKG = WATERSHED_ROOT / "aihydro_watershed"

# ---------------------------------------------------------------------------
# Module move plan: (src_path_relative_to_tools, dest_path_relative_to_watershed_pkg)
# ---------------------------------------------------------------------------
MOVES: list[tuple[str, str]] = [
    # delineation/*
    ("ai_hydro/analysis/delineation/router.py",              "delineation/router.py"),
    ("ai_hydro/analysis/delineation/merit_flowdir_pipeline.py", "delineation/merit_flowdir_pipeline.py"),
    ("ai_hydro/analysis/delineation/nldi_point.py",          "delineation/nldi_point.py"),
    ("ai_hydro/analysis/delineation/merit_snap.py",          "delineation/merit_snap.py"),
    ("ai_hydro/analysis/delineation/pysheds_pipeline.py",    "delineation/pysheds_pipeline.py"),
    ("ai_hydro/analysis/delineation/dem_fetch.py",           "delineation/dem_fetch.py"),
    ("ai_hydro/analysis/delineation/dem_conditioning.py",    "delineation/dem_conditioning.py"),
    ("ai_hydro/analysis/delineation/types.py",               "delineation/types.py"),
    ("ai_hydro/analysis/delineation/utils.py",               "delineation/utils.py"),
    # characterize/*
    ("ai_hydro/analysis/watershed.py",  "characterize/watershed.py"),
    ("ai_hydro/analysis/geomorphic.py", "characterize/geomorphic.py"),
    ("ai_hydro/analysis/twi.py",        "characterize/twi.py"),
    ("ai_hydro/analysis/_dem.py",       "characterize/_dem.py"),
    # terrain/*
    ("ai_hydro/analysis/curve_number.py",  "terrain/curve_number.py"),
    ("ai_hydro/analysis/event_runoff.py",  "terrain/event_runoff.py"),
    ("ai_hydro/analysis/erosion.py",       "terrain/erosion.py"),
    # signatures/*
    ("ai_hydro/analysis/signatures.py",       "signatures/signatures.py"),
    ("ai_hydro/analysis/baseflow.py",         "signatures/baseflow.py"),
    ("ai_hydro/analysis/flow_duration.py",    "signatures/flow_duration.py"),
    ("ai_hydro/analysis/flood_frequency.py",  "signatures/flood_frequency.py"),
    ("ai_hydro/analysis/drought_indices.py",  "signatures/drought_indices.py"),
    # merit/*
    ("ai_hydro/data/merit_download.py",    "merit/merit_download.py"),
    ("ai_hydro/data/merit_manager.py",     "merit/merit_manager.py"),
    ("ai_hydro/data/merit_map_layers.py",  "merit/merit_map_layers.py"),
    ("ai_hydro/data/region_presets.py",    "merit/region_presets.py"),
    ("ai_hydro/data/wbd_layers.py",        "merit/wbd_layers.py"),
    ("ai_hydro/data/merit_manifest.yaml",  "merit/merit_manifest.yaml"),
]

# ---------------------------------------------------------------------------
# Import rewrite rules (applied in order; FIRST MATCH wins per import line)
# Each rule: (pattern_re, replacement_string)
# Replacements may use \1 etc for captured groups.
# ---------------------------------------------------------------------------
REWRITE_RULES: list[tuple[re.Pattern, str]] = [
    # 1. Internal delineation cross-imports — must come before the generic analysis rule
    (re.compile(r'\bfrom ai_hydro\.analysis\.delineation(\.[a-zA-Z_][a-zA-Z0-9_]*)? import'),
     r'from aihydro_watershed.delineation\1 import'),

    # 2. characterize modules
    (re.compile(r'\bfrom ai_hydro\.analysis\.(watershed|geomorphic|twi|_dem) import'),
     r'from aihydro_watershed.characterize.\1 import'),
    (re.compile(r'\bimport ai_hydro\.analysis\.(watershed|geomorphic|twi|_dem)\b'),
     r'import aihydro_watershed.characterize.\1'),

    # 3. terrain modules
    (re.compile(r'\bfrom ai_hydro\.analysis\.(curve_number|event_runoff|erosion) import'),
     r'from aihydro_watershed.terrain.\1 import'),
    (re.compile(r'\bimport ai_hydro\.analysis\.(curve_number|event_runoff|erosion)\b'),
     r'import aihydro_watershed.terrain.\1'),

    # 4. signatures modules
    (re.compile(r'\bfrom ai_hydro\.analysis\.(signatures|baseflow|flow_duration|flood_frequency|drought_indices) import'),
     r'from aihydro_watershed.signatures.\1 import'),
    (re.compile(r'\bimport ai_hydro\.analysis\.(signatures|baseflow|flow_duration|flood_frequency|drought_indices)\b'),
     r'import aihydro_watershed.signatures.\1'),

    # 5. merit / data modules
    (re.compile(r'\bfrom ai_hydro\.data\.(merit_download|merit_manager|merit_map_layers|region_presets|wbd_layers) import'),
     r'from aihydro_watershed.merit.\1 import'),
    (re.compile(r'\bimport ai_hydro\.data\.(merit_download|merit_manager|merit_map_layers|region_presets|wbd_layers)\b'),
     r'import aihydro_watershed.merit.\1'),

    # 6. core contract + errors → aihydro_core
    (re.compile(r'\bfrom ai_hydro\.core import'),
     r'from aihydro_core import'),
    (re.compile(r'\bimport ai_hydro\.core\b'),
     r'import aihydro_core'),

    # 7. bootstrap uncertainty → aihydro_core.science.uncertainty
    (re.compile(r'\bfrom ai_hydro\.analysis\.uncertainty import'),
     r'from aihydro_core.science.uncertainty import'),

    # 8. bare `import ai_hydro` (version stamps) → aihydro_watershed
    (re.compile(r'^\s*import ai_hydro\s*$', re.MULTILINE),
     r'import aihydro_watershed'),
    (re.compile(r'\bai_hydro\.__version__\b'),
     r'aihydro_watershed.__version__'),
]

# Imports that are NOT rewritten by the above rules — collected during run
# for manual A2d seam review.
UNMAPPED_PATTERN = re.compile(r'(?:^|\b)(?:from|import)\s+ai_hydro\b', re.MULTILINE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rewrite_source(src: str, rel_dest: str) -> tuple[str, list[str]]:
    """
    Apply all import rewrite rules to `src`.  Return (rewritten_source, warnings).
    Warnings list unmapped ai_hydro imports that need A2d attention.
    """
    result = src
    for pattern, replacement in REWRITE_RULES:
        result = pattern.sub(replacement, result)

    # Collect remaining unmapped ai_hydro imports
    warnings: list[str] = []
    for m in UNMAPPED_PATTERN.finditer(result):
        line = result[result.rfind('\n', 0, m.start()) + 1:result.find('\n', m.end())]
        warnings.append(f"  UNMAPPED in {rel_dest}: {line.strip()}")

    return result, warnings


def parse_check(source: str, path: str) -> str | None:
    """Return error message if source has a SyntaxError, else None."""
    try:
        ast.parse(source, filename=path)
        return None
    except SyntaxError as e:
        return f"SyntaxError in {path}: {e}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify", action="store_true",
                        help="syntax-check each output file after writing")
    args = parser.parse_args()

    all_warnings: list[str] = []
    errors: list[str] = []

    for src_rel, dest_rel in MOVES:
        src_path = TOOLS_ROOT / src_rel
        dest_path = WATERSHED_PKG / dest_rel

        if not src_path.exists():
            errors.append(f"MISSING source: {src_path}")
            continue

        # Non-Python assets (yaml, etc.) — copy verbatim
        if src_path.suffix != ".py":
            if args.dry_run:
                print(f"[copy] {src_rel} → {dest_rel}")
            else:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dest_path)
                print(f"copied  {dest_rel}")
            continue

        source = src_path.read_text(encoding="utf-8")
        rewritten, warns = rewrite_source(source, dest_rel)
        all_warnings.extend(warns)

        if args.dry_run:
            changed = source != rewritten
            print(f"[{'REWRITE' if changed else 'copy  '}] {src_rel} → {dest_rel}")
        else:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_path.write_text(rewritten, encoding="utf-8")
            status = "rewritten" if source != rewritten else "copied   "
            print(f"{status}  {dest_rel}")

            if args.verify:
                err = parse_check(rewritten, dest_rel)
                if err:
                    errors.append(err)

    print()
    if all_warnings:
        print("=== UNMAPPED IMPORTS (need A2d seam work) ===")
        for w in all_warnings:
            print(w)
        print()

    if errors:
        print("=== ERRORS ===")
        for e in errors:
            print(e)
        sys.exit(1)
    else:
        print("=== Done — no errors ===")
        if not args.dry_run:
            print("Next: run `pytest tests/test_layering.py` in aihydro-watershed to verify no ai_hydro imports.")


if __name__ == "__main__":
    main()
