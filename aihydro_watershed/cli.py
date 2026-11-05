"""
``aihydro-watershed`` console entry point.

One subcommand, ``signatures``: compute the CAMELS-style hydrological
signatures from a daily streamflow CSV you already hold, **with no network
access**. It is deliberately the offline slice of the package. Delineation
needs DEM/NLDI/MERIT services and stays in the Python API
(``aihydro_watershed.delineation``); this command takes the polygon that
delineation produced as a file.

    aihydro-watershed signatures --flow-csv q.csv --area-km2 250 --geojson basin.geojson

The CSV needs a date column and a discharge column in m3/s (``--date-col`` /
``--flow-col``, default ``date`` / ``q_cms``). Precipitation is never
requested (``precipitation="skip"``), so ``runoff_ratio`` and ``stream_elas``
are null and ``_precipitation.status`` records ``not_attempted``. The JSON on
stdout is the ``HydroResult`` (``data`` + ``meta``) the Python API returns.
"""
from __future__ import annotations

import argparse
import json
import sys

from aihydro_watershed import __version__


def _read_polygon(path: str) -> dict:
    """Return a GeoJSON geometry dict from a Geometry, Feature or FeatureCollection file."""
    with open(path, encoding="utf-8") as fh:
        obj = json.load(fh)
    if obj.get("type") == "FeatureCollection":
        feats = obj.get("features") or []
        if len(feats) != 1:
            raise ValueError(
                f"{path}: expected exactly one feature (the basin), found {len(feats)}"
            )
        obj = feats[0]
    if obj.get("type") == "Feature":
        obj = obj.get("geometry")
    if not isinstance(obj, dict) or obj.get("type") not in ("Polygon", "MultiPolygon"):
        raise ValueError(f"{path}: no Polygon/MultiPolygon geometry found")
    return obj


def _cmd_signatures(args: argparse.Namespace) -> int:
    import pandas as pd

    from aihydro_watershed.signatures.signatures import extract_hydrological_signatures

    df = pd.read_csv(args.flow_csv)
    for col in (args.date_col, args.flow_col):
        if col not in df.columns:
            raise ValueError(f"{args.flow_csv}: no column {col!r} (have {list(df.columns)})")
    idx = pd.to_datetime(df[args.date_col], errors="raise")
    q = pd.Series(pd.to_numeric(df[args.flow_col], errors="coerce").to_numpy(), index=idx)
    q = q[~q.index.duplicated(keep="first")].sort_index()

    result = extract_hydrological_signatures(
        gauge_id=None,
        watershed_geojson=_read_polygon(args.geojson),
        area_km2=args.area_km2,
        start_date=str(q.index.min().date()),
        end_date=str(q.index.max().date()),
        q_cms_series=q,
        precipitation="skip",
    )
    text = json.dumps(result.to_dict(), indent=2, default=str)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    else:
        print(text)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aihydro-watershed",
        description="Watershed signatures from local files (offline). "
                    "Delineation is available through the Python API.",
    )
    parser.add_argument("--version", action="version", version=f"aihydro-watershed {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    p = sub.add_parser(
        "signatures",
        help="CAMELS-style hydrological signatures from a daily streamflow CSV (no network).",
    )
    p.add_argument("--flow-csv", required=True, metavar="FILE",
                   help="daily streamflow CSV (date column + discharge in m3/s)")
    p.add_argument("--area-km2", required=True, type=float, metavar="KM2",
                   help="drainage area of the basin, km2")
    p.add_argument("--geojson", required=True, metavar="FILE",
                   help="basin polygon (GeoJSON Geometry, Feature or one-feature FeatureCollection)")
    p.add_argument("--date-col", default="date", metavar="NAME")
    p.add_argument("--flow-col", default="q_cms", metavar="NAME")
    p.add_argument("-o", "--output", default="", metavar="FILE",
                   help="write the JSON result here instead of stdout")
    p.set_defaults(func=_cmd_signatures)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:  # surface a one-line error, like the sibling CLIs
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
