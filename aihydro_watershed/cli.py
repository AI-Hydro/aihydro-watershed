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
stdout is the ``HydroResult`` (``data`` + ``meta``) the Python API returns, plus a ``cli`` block
recording how the file was read: missing/duplicate rows, absent calendar days and warnings.
Negative discharge is refused; missing-data codes go in ``--na-values`` (default -999,-9999).
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


def _load_flow(args: argparse.Namespace):
    """Read the CSV into a daily Series and report what was done to it, never silently."""
    import pandas as pd

    na_values = [v.strip() for v in args.na_values.split(",") if v.strip()]
    df = pd.read_csv(args.flow_csv, na_values=na_values)
    for col in (args.date_col, args.flow_col):
        if col not in df.columns:
            raise ValueError(f"{args.flow_csv}: no column {col!r} (have {list(df.columns)})")
    idx = pd.to_datetime(df[args.date_col], errors="raise")
    vals = pd.to_numeric(df[args.flow_col], errors="coerce")
    notes: dict = {"n_rows": int(len(df)), "na_values": na_values, "warnings": []}

    n_nan = int(vals.isna().sum())
    notes["n_missing"] = n_nan
    if n_nan:
        notes["warnings"].append(
            f"{n_nan} row(s) are missing or non-numeric (including --na-values {na_values}); "
            "they are treated as missing days, not as zero flow")

    neg = vals < 0
    if neg.any():
        first = idx[neg].iloc[0].date()
        raise ValueError(
            f"{args.flow_csv}: {int(neg.sum())} negative discharge value(s) (first on {first}). "
            "Discharge cannot be negative; if these are a missing-data code, pass it in --na-values")

    q = pd.Series(vals.to_numpy(), index=idx)
    dup = q.index.duplicated(keep="first")
    notes["n_duplicate_dates"] = int(dup.sum())
    notes["duplicate_dates"] = sorted({str(d.date()) for d in q.index[dup]})[:20]
    if dup.any():
        notes["warnings"].append(
            f"{int(dup.sum())} duplicate date row(s): the first row per date was kept")
        q = q[~dup]
    q = q.sort_index()

    span = (q.index.max() - q.index.min()).days + 1
    absent = int(span - len(q))
    notes["n_calendar_days_absent"] = absent
    if absent:
        notes["warnings"].append(
            f"{absent} calendar day(s) between the first and last date are absent from the file")
    notes["period"] = [str(q.index.min().date()), str(q.index.max().date())]
    return q, notes


def _cmd_signatures(args: argparse.Namespace) -> int:
    from aihydro_watershed.signatures.signatures import extract_hydrological_signatures

    q, notes = _load_flow(args)
    result = extract_hydrological_signatures(
        gauge_id=None,
        watershed_geojson=_read_polygon(args.geojson),
        area_km2=args.area_km2,
        start_date=notes["period"][0],
        end_date=notes["period"][1],
        q_cms_series=q,
        precipitation="skip",
    )
    out = result.to_dict()
    out["cli"] = notes  # input handling and warnings travel with the result, not only on stderr
    for w in notes["warnings"]:
        print(f"warning: {w}", file=sys.stderr)
    text = json.dumps(out, indent=2, default=str)
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
    p.add_argument("--na-values", default="-999,-9999", metavar="V[,V…]",
                   help="comma-separated flow codes read as missing (default: -999,-9999); "
                        "negative flows that remain are refused")
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
