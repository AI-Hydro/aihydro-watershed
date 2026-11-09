"""``python -m aihydro_watershed.tiles 72`` — build river tiles for staged region(s); prints JSON."""
from __future__ import annotations

import argparse
import json

from aihydro_watershed.tiles import build_merit_river_tiles


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m aihydro_watershed.tiles", description=__doc__)
    ap.add_argument("regions", nargs="+", help="Pfafstetter level-2 codes, e.g. 72 74")
    ap.add_argument("--out-dir", help="default: $AIHYDRO_HOME/merit/tiles")
    ap.add_argument("--max-zoom", type=int, default=12)
    ap.add_argument("--force", action="store_true", help="rebuild even if up to date")
    a = ap.parse_args(argv)
    for r in a.regions:
        rec = build_merit_river_tiles(r, a.out_dir, max_zoom=a.max_zoom, force=a.force)
        print(json.dumps({k: rec[k] for k in ("region", "status", "output", "reaches_per_layer", "build_seconds")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
