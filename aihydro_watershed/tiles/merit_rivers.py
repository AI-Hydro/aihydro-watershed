"""
Build local MERIT-Basins river PMTiles from a staged ``rivers##.db`` (basemap slice B2, tier T2).

The output is one PMTiles archive per Pfafstetter region, ``<AIHYDRO_HOME>/merit/tiles/rivers##.pmtiles``,
with a provenance sidecar next to it. Reaches are split into one MVT layer per **upstream-area band**
(rule Z1 in ``docs/vision-2040/findings/hydrology-basemap-2026-11-08.md``): a reach is in the tiles from
the zoom where its band starts, not by Strahler order, because order depends on source density while
upstream area is physical.

=========  ==========  ===============================
layer      min zoom    upstream area (km2)
=========  ==========  ===============================
rivers_z0  0           >= 10,000
rivers_z5  5           2,500 to < 10,000
rivers_z7  7           500 to < 2,500
rivers_z9  9           100 to < 500
rivers_z11 11          < 100 (all remaining reaches)
=========  ==========  ===============================

A client draws every layer; a layer simply has no tiles below its min zoom. Reaches with no ``uparea`` go
in the last band (shown only at the finest zoom) and are counted in the sidecar. Attributes kept:
``comid`` (the source primary key), ``uparea`` (km2), ``sorder`` (Strahler order) and ``lengthkm``.

Licence: MERIT-Basins (Lin et al. 2019) is redistributed here as derived data under CC BY-NC-SA 4.0. Keep
the tiles local, attribute them, and do not ship them in a commercially licensed bundle.

Tiles are written with ``pyogrio`` (GDAL MVT/PMTiles driver); no extra dependency is needed.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from aihydro_watershed._paths import aihydro_home
from aihydro_watershed.merit import verified_fetch as vf
from aihydro_watershed.tiles.pmtiles_merge import merge_pmtiles

#: bump when the tile layout (bands, attributes, zooms) changes so old archives are rebuilt
TILE_SCHEMA_VERSION = 1

#: (layer name, min zoom, lower area bound km2 inclusive, upper bound exclusive) — rule Z1
AREA_BANDS: tuple[tuple[str, int, float, float], ...] = (
    ("rivers_z0", 0, 10_000.0, float("inf")),
    ("rivers_z5", 5, 2_500.0, 10_000.0),
    ("rivers_z7", 7, 500.0, 2_500.0),
    ("rivers_z9", 9, 100.0, 500.0),
    ("rivers_z11", 11, float("-inf"), 100.0),
)
DEFAULT_MAX_ZOOM = 12  # 90 m source: finer zooms add size, not information; clients overzoom
KEEP_FIELDS = ("comid", "uparea", "sorder", "lengthkm")

LICENCE = "CC BY-NC-SA 4.0"
CITATION = "Lin et al. 2019"
ATTRIBUTION = "Rivers: MERIT-Basins, Lin et al. 2019, CC BY-NC-SA 4.0"
CITATION_DETAIL = (
    "Lin, P., Pan, M., Beck, H. E., et al. (2019). Global reconstruction of naturalized river flows at "
    "2.94 million reaches. Water Resources Research, 55, 6499-6516. https://doi.org/10.1029/2019WR025287"
)


def merit_root() -> Path:
    """The MERIT data root, resolved as ``MeritManager`` does."""
    env = os.environ.get("AIHYDRO_MERIT_DIR") or os.environ.get("MERIT_DATA_DIR")
    return Path(env) if env else aihydro_home() / "merit"


def tile_paths(region: str | int, out_dir: str | Path | None = None) -> tuple[Path, Path]:
    """``(pmtiles path, sidecar path)`` for a region."""
    name = f"rivers{str(int(region)).zfill(2)}.pmtiles"
    out = Path(out_dir) if out_dir else merit_root() / "tiles"
    return out / name, out / (name + ".provenance.json")


def _source_sha256(db: Path, region: str, *, verify: bool) -> tuple[str, str]:
    """The source db's sha256 and where it came from.

    ``verify=True`` goes through the existing staging verifier (pinned hash + sidecar fast path), which
    raises when the file does not match the pin. ``verify=False`` (synthetic data in tests, files outside
    the pinned set) trusts a present sidecar's ``sha256`` and hashes the file only when there is none.
    """
    if verify:
        vf.verified_path(db, vf.region_asset_spec("rivers", region))
        rec = vf.read_provenance(db) or {}
        if not rec.get("sha256"):
            raise RuntimeError(f"{db}: verification wrote no sha256 sidecar")
        return rec["sha256"], "staging_provenance_sidecar"
    rec = vf.read_provenance(db) or {}
    if rec.get("sha256") and rec.get("size_bytes") == db.stat().st_size:
        return rec["sha256"], "staging_provenance_sidecar"
    return vf.sha256_file(db), "computed_no_sidecar"


def _versions() -> dict[str, str]:
    import pyogrio

    return {
        "pyogrio": pyogrio.__version__,
        "gdal": pyogrio.__gdal_version_string__,
        "geopandas": __import__("geopandas").__version__,
    }


def _read_matches(sidecar: Path, source_sha: str, options: dict[str, Any], pm: Path) -> bool:
    try:
        rec = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    out = rec.get("output", {})
    return (
        rec.get("tile_schema_version") == TILE_SCHEMA_VERSION
        and rec.get("source", {}).get("sha256") == source_sha
        and rec.get("build_options") == options
        and pm.exists()
        and out.get("size_bytes") == pm.stat().st_size
        and out.get("sha256") == vf.sha256_file(pm)
    )


def build_merit_river_tiles(
    region: str | int,
    out_dir: str | Path | None = None,
    *,
    source_db: str | Path | None = None,
    max_zoom: int = DEFAULT_MAX_ZOOM,
    force: bool = False,
    verify_source: bool = True,
) -> dict[str, Any]:
    """Build ``rivers##.pmtiles`` for a staged Pfafstetter region and write its provenance sidecar.

    Args:
        region: Pfafstetter level-2 code, e.g. ``72`` (as staged by ``MeritManager``).
        out_dir: Output directory; default ``<AIHYDRO_HOME>/merit/tiles``.
        source_db: Override for ``rivers##.db`` (default ``<merit root>/vector/rivers##.db``).
        max_zoom: Highest tile zoom written (default 12); at least 11, where the last band starts.
        force: Rebuild even when the existing sidecar matches the source.
        verify_source: Verify the source db against the pinned hash (the default). Turn off only for files
            outside the pinned MERIT set (synthetic test data).

    Returns:
        The provenance record, with ``"status"`` ``"built"`` or ``"skipped_up_to_date"``.

    Raises:
        FileNotFoundError: the region's ``rivers##.db`` is not staged (nothing is downloaded here).
        ValueError: the db lacks a required column or has no reaches, or ``max_zoom`` is below 11.
    """
    import pyogrio

    finest = max(z for _n, z, _lo, _hi in AREA_BANDS)
    if int(max_zoom) < finest:
        raise ValueError(f"max_zoom must be >= {finest} so every reach appears at some zoom (got {max_zoom})")
    code = str(int(region)).zfill(2)
    db = Path(source_db) if source_db else merit_root() / "vector" / f"rivers{code}.db"
    if not db.exists():
        raise FileNotFoundError(f"{db} is not staged; stage Pfaf {code} rivers first (no download is done here)")
    pm, sidecar = tile_paths(code, out_dir)
    source_sha, sha_origin = _source_sha256(db, code, verify=verify_source)
    options = {
        "bands": [{"layer": n, "minzoom": z, "uparea_min_km2": None if lo == float("-inf") else lo,
                   "uparea_max_km2": None if hi == float("inf") else hi} for n, z, lo, hi in AREA_BANDS],
        "maxzoom": int(max_zoom), "driver": "PMTiles", "fields": list(KEEP_FIELDS),
    }
    if not force and _read_matches(sidecar, source_sha, options, pm):
        return {**json.loads(sidecar.read_text(encoding="utf-8")), "status": "skipped_up_to_date"}

    t0 = time.perf_counter()
    df = pyogrio.read_dataframe(db, fid_as_index=True)  # the db's INTEGER PRIMARY KEY (comid) is the fid
    if len(df) == 0:
        raise ValueError(f"{db}: no reaches")
    missing = [c for c in ("uparea", "sorder") if c not in df.columns]
    if missing:
        raise ValueError(f"{db}: missing column(s) {missing}; have {list(df.columns)}")
    df = df.rename_axis("comid").reset_index()
    for c in KEEP_FIELDS:
        if c not in df.columns:
            df[c] = None
    df = df[[*KEEP_FIELDS, "geometry"]]
    n_no_uparea = int(df["uparea"].isna().sum())

    pm.parent.mkdir(parents=True, exist_ok=True)
    band_counts: dict[str, int] = {}
    with tempfile.TemporaryDirectory(prefix="rivers-tiles-", dir=pm.parent) as tmp:
        parts: list[Path] = []
        for name, minzoom, lo, hi in AREA_BANDS:
            ua = df["uparea"]
            sel = ((ua >= lo) & (ua < hi)) | (ua.isna() if hi == 100.0 else False)  # unknown area: finest band
            sub = df[sel]
            band_counts[name] = int(len(sub))
            if not len(sub):
                continue
            part = Path(tmp) / f"{name}.pmtiles"
            pyogrio.write_dataframe(sub, part, layer=name, driver="PMTiles",
                                    MINZOOM=minzoom, MAXZOOM=int(max_zoom))
            parts.append(part)
        merged = merge_pmtiles(
            parts, pm.with_name(pm.name + ".building"),
            metadata_extra={
                "name": f"merit-basins-rivers-{code}", "attribution": ATTRIBUTION,
                "description": "MERIT-Basins river reaches, split into upstream-area band layers (rule Z1)",
            },
        )
    building = pm.with_name(pm.name + ".building")
    building.replace(pm)

    record = {
        "tile_schema_version": TILE_SCHEMA_VERSION,
        "region": code,
        "source": {
            "path": str(db), "name": db.name, "sha256": source_sha, "sha256_origin": sha_origin,
            "n_reaches": int(len(df)),
            "synthetic": bool((vf.read_provenance(db) or {}).get("synthetic", False)),
        },
        "output": {
            "path": str(pm), "name": pm.name, "size_bytes": pm.stat().st_size, "sha256": vf.sha256_file(pm),
            "tiles": merged["tiles"], "layers": merged["layers"],
        },
        "build_options": options,
        "reaches_per_layer": band_counts,
        "reaches_without_uparea_placed_in_finest_band": n_no_uparea,
        "tooling": _versions(),
        "licence": LICENCE,
        "citation": CITATION,
        "citation_detail": CITATION_DETAIL,
        "attribution": ATTRIBUTION,
        "build_seconds": round(time.perf_counter() - t0, 1),
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    sidecar.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return {**record, "status": "built"}
