"""
Lazy MERIT-Hydro / MERIT-Basins data layout under ``$AIHYDRO_HOME/merit/``
(default ``~/.aihydro/merit/``).

The default global delineation tier (MERIT-Basins vector assembly) reads the
per-region ``basins##.db`` / ``rivers##.db`` files and optional ``flowdir##.tif`` /
``accum##.tif`` rasters published for the ``delineator`` package. Every file is
pinned by SHA-256 in ``delineator_data_manifest.json`` and verified before use
(see :mod:`aihydro_watershed.merit.verified_fetch`).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import geopandas as gpd
import yaml
from shapely.geometry import Point

from aihydro_watershed._paths import aihydro_home
from aihydro_watershed.merit import verified_fetch as vf

log = logging.getLogger(__name__)

_MANIFEST_PATH = Path(__file__).parent / "merit_manifest.yaml"

# ---------------------------------------------------------------------------
# Auto-staging policy (Design B)
# ---------------------------------------------------------------------------
# Read from $AIHYDRO_HOME/config.json (default ~/.aihydro).  Keys:
#   merit_auto_stage: "tiered" | "always" | "never"   (default "tiered")
#   merit_auto_stage_max_gb: float                     (default 2.0)
# ---------------------------------------------------------------------------

def _merit_auto_stage_policy() -> tuple[str, float]:
    """Return (policy, max_gb) from ``AIHYDRO_MERIT_AUTO_STAGE`` or $AIHYDRO_HOME/config.json."""
    import json as _json
    cfg_path = aihydro_home() / "config.json"
    env_policy = os.environ.get("AIHYDRO_MERIT_AUTO_STAGE", "").strip().lower()
    try:
        if cfg_path.exists():
            cfg = _json.loads(cfg_path.read_text())
            policy = str(cfg.get("merit_auto_stage", "tiered"))
            max_gb = float(cfg.get("merit_auto_stage_max_gb", 2.0))
            return (env_policy or policy), max_gb
    except Exception:
        pass
    return (env_policy or "tiered"), 2.0


@dataclass
class BasinEnsureStatus:
    pfaf_code: str
    level2_ready: bool
    rivers_ready: bool
    catchments_ready: bool
    flowdir_ready: bool
    message: str = ""
    downloaded: list[str] | None = None


@dataclass
class RoutingRegionStatus:
    pfaf_region: str
    flowdir_ready: bool
    flowdir_path: str | None
    accum_ready: bool
    accum_path: str | None
    local_upstream_area_ready: bool
    local_upstream_area_path: str | None
    acquisition_required: bool
    acquisition_policy: str
    required_assets: tuple[str, ...]
    estimated_download_size_bytes: int | None
    metadata_path: str | None
    message: str
    downloaded: list[str] | None = None


@dataclass
class BasinsRegionStatus:
    pfaf_region: str
    catchments_ready: bool
    catchments_path: str | None
    rivers_ready: bool
    rivers_path: str | None
    acquisition_required: bool
    acquisition_policy: str
    estimated_download_size_bytes: int | None
    metadata_path: str | None
    source: str | None
    license: str
    citation: str
    message: str
    downloaded: list[str] | None = None
    integrity_error: str | None = None
    rivers_db_path: str | None = None
    megabasins_ready: bool = False


class MeritDataManager:
    """Resolve Pfafstetter basin and local MERIT file paths."""

    def __init__(self, root: Path | str | None = None, *, opener: "vf.Opener | None" = None) -> None:
        self._opener = opener
        if root is not None:
            self.root = Path(root)
        else:
            env = os.environ.get("AIHYDRO_MERIT_DIR") or os.environ.get("MERIT_DATA_DIR")
            self.root = Path(env) if env else aihydro_home() / "merit"
        self.root.mkdir(parents=True, exist_ok=True)
        self._manifest = self._load_manifest()
        self._base_url = vf.mirror_base_url()

    def _load_manifest(self) -> dict[str, Any]:
        if _MANIFEST_PATH.exists():
            with open(_MANIFEST_PATH) as f:
                return yaml.safe_load(f) or {}
        return {}

    def level2_shapefile_path(self) -> Path:
        rel = self._manifest.get("level2", {}).get("relative_dir", "shp/basins_level2")
        return self.root / rel / "merit_hydro_vect_level2.shp"

    def river_shapefile_path(self, pfaf_code: str) -> Path:
        pfaf = pfaf_code.zfill(2)
        tmpl = (
            self._manifest.get("basin_template", {})
            .get("rivers", {})
            .get("filename", "riv_pfaf_{pfaf}_MERIT_Hydro_v07_Basins_v01.shp")
        )
        rel = self._manifest.get("basin_template", {}).get("rivers", {}).get(
            "relative_dir", "shp/merit_rivers"
        )
        return self.root / rel / tmpl.format(pfaf=pfaf)

    def catchment_shapefile_path(self, pfaf_code: str) -> Path:
        pfaf = pfaf_code.zfill(2)
        tmpl = (
            self._manifest.get("basin_template", {})
            .get("catchments", {})
            .get("filename", "cat_pfaf_{pfaf}_MERIT_Hydro_v07_Basins_v01.shp")
        )
        rel = self._manifest.get("basin_template", {}).get("catchments", {}).get(
            "relative_dir", "shp/merit_catchments"
        )
        return self.root / rel / tmpl.format(pfaf=pfaf)

    def _glob_flowdir(self, pfaf_code: str) -> Path | None:
        pfaf = pfaf_code.zfill(2)
        rel = self._manifest.get("basin_template", {}).get("flowdir", {}).get(
            "relative_dir", "raster/flowdir_basins"
        )
        folder = self.root / rel
        if not folder.exists():
            return None
        matches = list(folder.glob(f"*pfaf_{pfaf}*.tif")) or list(folder.glob(f"flowdir{pfaf}.tif"))
        return matches[0] if matches else None

    def _glob_accum(self, pfaf_code: str) -> Path | None:
        pfaf = pfaf_code.zfill(2)
        rel = self._manifest.get("basin_template", {}).get("accumulation", {}).get(
            "relative_dir", "raster/accum_basins"
        )
        folder = self.root / rel
        if not folder.exists():
            return None
        matches = list(folder.glob(f"*pfaf_{pfaf}*.tif")) or list(folder.glob(f"accum{pfaf}.tif"))
        return matches[0] if matches else None

    def _local_upstream_area_path(self, pfaf_code: str) -> Path:
        pfaf = pfaf_code.zfill(2)
        return self.root / "raster" / "upstream_area_local" / f"upa_local_{pfaf}.tif"

    def _flowdir_metadata_path(self, pfaf_code: str) -> Path:
        pfaf = pfaf_code.zfill(2)
        return self.root / "metadata" / f"flowdir_{pfaf}.json"

    def _basins_metadata_path(self, pfaf_code: str) -> Path:
        pfaf = pfaf_code.zfill(2)
        return self.root / "metadata" / f"basins_{pfaf}.json"

    def _merit_license(self) -> str:
        return vf.licence_summary()

    def _merit_basins_citation(self) -> str:
        return (
            "Lin et al. 2019 (MERIT-Basins), https://www.reachhydro.org/home/params/merit-basins; "
            "Yamazaki et al. 2019 (MERIT Hydro), https://doi.org/10.1029/2019WR024873; "
            "Heberger 2025 (hybrid delineation), https://mghydro.com/pages/Heberger_delineation_2025.pdf"
        )

    # -- pinned delineator-layout data (basins##.db, rivers##.db, rasters, megabasins.db) --------

    def megabasins_path(self) -> Path:
        return self.root / "vector" / "megabasins.db"

    def basins_db_path(self, pfaf_code: str) -> Path:
        return self.root / "vector" / vf.region_asset_name("basins", pfaf_code)

    def rivers_db_path(self, pfaf_code: str) -> Path:
        return self.root / "vector" / vf.region_asset_name("rivers", pfaf_code)

    def raster_path(self, kind: str, pfaf_code: str) -> Path:
        rel = "raster/flowdir_basins" if kind == "flowdir" else "raster/accum_basins"
        return self.root / rel / vf.region_asset_name(kind, pfaf_code)

    def _asset_path(self, kind: str, pfaf_code: str) -> Path:
        if kind == "basins":
            return self.basins_db_path(pfaf_code)
        if kind == "rivers":
            return self.rivers_db_path(pfaf_code)
        return self.raster_path(kind, pfaf_code)

    def verified_asset(self, kind: str, pfaf_code: str) -> Path | None:
        """Path of a verified regional file (``basins``/``rivers``/``flowdir``/``accum``) or ``None``.

        Raises :class:`AssetIntegrityError` if a file is present but does not match its pin.
        """
        return vf.verified_path(
            self._asset_path(kind, pfaf_code), vf.region_asset_spec(kind, pfaf_code)
        )

    def verified_megabasins(self) -> Path | None:
        return vf.verified_path(self.megabasins_path(), vf.asset_spec("megabasins.db"))

    def _stage(self, spec: "vf.AssetSpec", dest: Path, progress_cb=None) -> Path:
        return vf.fetch_verified(
            spec,
            dest,
            opener=self._opener,
            progress_cb=(lambda done, tot: progress_cb(f"Downloading {spec.name}", done, tot))
            if progress_cb
            else None,
        )

    def ensure_level2_index(self, *, download: bool = False) -> Path | None:
        """Verified megabasin index (``megabasins.db``); downloaded (52.9 MB) only when asked."""
        got = self.verified_megabasins()
        if got is None and download:
            got = self._stage(vf.asset_spec("megabasins.db"), self.megabasins_path())
        return got

    def resolve_pfaf_region(self, lat: float, lon: float) -> str:
        """Resolve the MERIT Pfafstetter level-2 region for a point."""
        return self.resolve_pfaf_code(lat, lon)

    def flowdir_path(self, pfaf_code: str) -> Path | None:
        """Return the cached regional MERIT flow-direction raster path, if present."""
        return self._glob_flowdir(pfaf_code)

    def routing_region_cache(self, pfaf_region: str) -> RoutingRegionStatus:
        """Report flowdir-first regional routing cache readiness."""
        pfaf = pfaf_region.zfill(2)
        flowdir = self._glob_flowdir(pfaf)
        accum = self._glob_accum(pfaf)
        local_upa = self._local_upstream_area_path(pfaf)
        meta = self._flowdir_metadata_path(pfaf)
        return RoutingRegionStatus(
            pfaf_region=pfaf,
            flowdir_ready=flowdir is not None,
            flowdir_path=str(flowdir) if flowdir else None,
            accum_ready=accum is not None,
            accum_path=str(accum) if accum else None,
            local_upstream_area_ready=local_upa.exists(),
            local_upstream_area_path=str(local_upa) if local_upa.exists() else None,
            acquisition_required=flowdir is None,
            acquisition_policy="check_only",
            required_assets=("flowdir",),
            estimated_download_size_bytes=flowdir.stat().st_size if flowdir and flowdir.exists() else None,
            metadata_path=str(meta) if meta.exists() else None,
            message=(
                f"Regional MERIT flowdir cached for Pfaf {pfaf}."
                if flowdir
                else f"Regional MERIT flowdir missing for Pfaf {pfaf}; staging required."
            ),
        )

    def basins_region_cache(self, pfaf_region: str) -> BasinsRegionStatus:
        """Report regional MERIT-Basins vector readiness (hash-verified, never trusts an unpinned file)."""
        pfaf = pfaf_region.zfill(2)
        meta = self._basins_metadata_path(pfaf)
        err: str | None = None
        found: dict[str, Path | None] = {}
        for kind in ("basins", "rivers"):
            try:
                found[kind] = self.verified_asset(kind, pfaf)
            except vf.AssetIntegrityError as exc:
                found[kind] = None
                err = err or str(exc)
        try:
            mega = self.verified_megabasins()
        except vf.AssetIntegrityError as exc:
            mega = None
            err = err or str(exc)
        need = sum(
            vf.region_asset_spec(k, pfaf).size_bytes or 0 for k in ("basins", "rivers") if found[k] is None
        )
        if mega is None:
            need += vf.asset_spec("megabasins.db").size_bytes or 0
        basins_ok, rivers_ok = found["basins"] is not None, found["rivers"] is not None
        if err:
            message = f"{err} The file was not used."
        elif basins_ok and rivers_ok:
            message = f"MERIT-Basins vectors verified for Pfaf {pfaf}."
        else:
            message = f"MERIT-Basins vectors missing for Pfaf {pfaf}; staging required."
        return BasinsRegionStatus(
            pfaf_region=pfaf,
            catchments_ready=basins_ok,
            catchments_path=str(found["basins"]) if basins_ok else None,
            rivers_ready=rivers_ok,
            rivers_path=str(found["rivers"]) if rivers_ok else None,
            acquisition_required=not (basins_ok and rivers_ok),
            acquisition_policy="check_only",
            estimated_download_size_bytes=need or None,
            metadata_path=str(meta) if meta.exists() else None,
            source=self._base_url or vf.load_manifest()["source"]["base_url"],
            license=self._merit_license(),
            citation=self._merit_basins_citation(),
            message=message,
            integrity_error=err,
            megabasins_ready=mega is not None,
        )

    def ensure_basins_region(
        self,
        pfaf_region: str,
        *,
        acquisition_policy: str = "check_only",
        progress_cb: "Callable[[str, int, int], None] | None" = None,
    ) -> BasinsRegionStatus:
        """Ensure/report regional MERIT-Basins vectors (``basins##.db``, ``rivers##.db``, ``megabasins.db``).

        ``check_only`` never downloads. ``download`` fetches only what is missing, installing a file
        only if its SHA-256 and size match the pinned manifest (a mismatch is reported, never used).
        """
        pfaf = pfaf_region.zfill(2)
        status = self.basins_region_cache(pfaf)
        downloaded: list[str] = []
        failure: str | None = None
        if acquisition_policy in ("download", "download_if_missing"):
            todo = []
            if not status.megabasins_ready:
                todo.append((vf.asset_spec("megabasins.db"), self.megabasins_path()))
            if not status.catchments_ready:
                todo.append((vf.region_asset_spec("basins", pfaf), self.basins_db_path(pfaf)))
            if not status.rivers_ready:
                todo.append((vf.region_asset_spec("rivers", pfaf), self.rivers_db_path(pfaf)))
            for spec, dest in todo:
                try:
                    log.info("Downloading %s (%.0f MB) from %s", spec.name, (spec.size_bytes or 0) / 1e6, spec.url)
                    self._stage(spec, dest, progress_cb)
                    downloaded.append(spec.name)
                except Exception as exc:  # integrity failures and network errors both leave nothing installed
                    failure = f"{spec.name}: {exc}"
                    log.warning("MERIT vector staging failed: %s", failure)
                    break
            status = self.basins_region_cache(pfaf)
            if not status.acquisition_required:
                self._write_basins_metadata(pfaf, source=self._base_url or "mghydro_pinned_manifest")
                status = self.basins_region_cache(pfaf)
        status.acquisition_policy = acquisition_policy
        status.downloaded = downloaded or None
        if failure:
            status.message = f"Staging failed ({failure}); nothing unverified was installed."
        elif status.acquisition_required and acquisition_policy == "check_only":
            status.message = (
                f"MERIT-Basins vectors missing for Pfaf {pfaf} (~{(status.estimated_download_size_bytes or 0) / 1e6:.0f} MB "
                "to download, SHA-256 verified); call merit_ensure_basins_region(..., "
                "acquisition_policy='download') after reviewing the size."
            )
        return status

    def auto_stage_vectors(
        self,
        pfaf: str,
        progress_cb: "Callable[[str, int, int], None] | None" = None,
    ) -> BasinsRegionStatus:
        """Stage the vectors for *pfaf* under the same tiered policy as :meth:`auto_stage_flowdir`.

        ``AIHYDRO_MERIT_AUTO_STAGE=never|tiered|always`` (or ``merit_auto_stage`` in config) and
        ``merit_auto_stage_max_gb`` (default 2 GB) apply; sizes come from the pinned manifest.
        """
        pfaf = pfaf.zfill(2)
        status = self.basins_region_cache(pfaf)
        if not status.acquisition_required or status.integrity_error:
            return status
        policy, max_gb = _merit_auto_stage_policy()
        need = status.estimated_download_size_bytes or 0
        if policy == "never":
            status.message = (
                f"MERIT-Basins vectors for Pfaf {pfaf} are not staged and auto-staging is disabled "
                f"(~{need / 1e6:.0f} MB, SHA-256 verified, on demand)."
            )
            return status
        if policy == "tiered" and need > max_gb * 1e9:
            status.message = (
                f"MERIT-Basins vectors for Pfaf {pfaf} are ~{need / 1e9:.1f} GB (> {max_gb:g} GB auto-stage "
                "limit); stage explicitly with merit_ensure_basins_region(acquisition_policy='download')."
            )
            setattr(status, "action_required", "confirm_large_download")
            return status
        return self.ensure_basins_region(pfaf, acquisition_policy="download", progress_cb=progress_cb)

    def _write_basins_metadata(self, pfaf: str, *, source: str) -> None:
        meta = self._basins_metadata_path(pfaf)
        meta.parent.mkdir(parents=True, exist_ok=True)
        assets: dict[str, dict[str, Any]] = {}
        for name, path in (
            ("catchments", self.basins_db_path(pfaf)),
            ("rivers", self.rivers_db_path(pfaf)),
            ("megabasins", self.megabasins_path()),
        ):
            rec = vf.read_provenance(path) if path.exists() else None
            if rec:
                assets[name] = {
                    "path": str(path),
                    "size_bytes": rec["size_bytes"],
                    "checksum_sha256": rec["sha256"],
                    "url": rec["url"],
                }
        payload = {
            "pfaf_region": pfaf,
            "asset": "merit_basins_vectors",
            "assets": assets,
            "source": source,
            "dataset_version": vf.load_manifest()["dataset_versions"]["vectors"],
            "data_version": vf.load_manifest()["source"]["data_version"],
            "acquisition_date": datetime.now(timezone.utc).isoformat(),
            "license": self._merit_license(),
            "citation": self._merit_basins_citation(),
        }
        meta.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    def _write_flowdir_metadata(self, pfaf: str, flowdir: Path, *, source: str) -> None:
        meta = self._flowdir_metadata_path(pfaf)
        meta.parent.mkdir(parents=True, exist_ok=True)
        rec = vf.read_provenance(flowdir) or {}
        payload = {
            "pfaf_region": pfaf,
            "asset": "flowdir",
            "path": str(flowdir),
            "source": source,
            "checksum_sha256": rec.get("sha256"),
            "size_bytes": flowdir.stat().st_size,
            "acquisition_date": datetime.now(timezone.utc).isoformat(),
            "license": self._merit_license(),
            "citation": "Yamazaki et al. 2019, MERIT Hydro, https://doi.org/10.1029/2019WR024873",
        }
        meta.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    def resolve_pfaf_code(self, lat: float, lon: float) -> str:
        shp = self.level2_shapefile_path()
        pt = Point(lon, lat)
        if shp.exists():
            gdf = gpd.read_file(shp)
            hit = gdf[gdf.geometry.contains(pt)]
            id_cols = ("pfaf_code", "PFAF_ID", "BASIN", "basin")
        else:
            mega = self.verified_megabasins()
            if mega is None:
                raise FileNotFoundError(
                    f"MERIT level-2 index missing at {self.megabasins_path()}. Stage it (52.9 MB, "
                    "SHA-256 pinned) with merit_ensure_basins_region(acquisition_policy='download')."
                )
            gdf = gpd.read_file(mega, layer="megabasins")
            hit = gdf[gdf.geometry.contains(pt)]
            if hit.empty:  # coastline slivers: nearest megabasin within ~5 km (0.05 deg)
                near = gdf[gdf.geometry.distance(pt) <= 0.05]
                if not near.empty:
                    hit = near.loc[[near.geometry.distance(pt).idxmin()]]
            id_cols = ("basin", "BASIN", "pfaf_code", "PFAF_ID")
        if hit.empty:
            raise ValueError(f"({lat}, {lon}) is outside MERIT level-2 basins.")
        row = hit.iloc[0]
        code = ""
        for col in id_cols:
            if col in row.index and row[col] not in (None, ""):
                code = str(row[col])
                break
        if not code:
            raise ValueError(f"Could not read Pfaf/BASIN code from level-2 index at ({lat}, {lon}).")
        return str(int(code)).zfill(2)

    def ensure_routing_region(
        self,
        lat: float | None = None,
        lon: float | None = None,
        *,
        pfaf_region: str | None = None,
        required_assets: tuple[str, ...] = ("flowdir",),
        acquisition_policy: str = "check_only",
    ) -> RoutingRegionStatus:
        """
        Ensure/report flowdir-first regional MERIT routing assets.

        ``check_only`` never downloads. ``download`` fetches the pinned, SHA-256-verified
        ``flowdir##.tif`` and still never requires accumulation.
        """
        if pfaf_region is None:
            if lat is None or lon is None:
                raise ValueError("lat/lon or pfaf_region is required.")
            pfaf_region = self.resolve_pfaf_region(lat, lon)
        pfaf = pfaf_region.zfill(2)
        required = tuple(required_assets or ("flowdir",))
        if any(asset != "flowdir" for asset in required):
            raise ValueError("Default regional routing currently supports required_assets=('flowdir',) only.")

        status = self.routing_region_cache(pfaf)
        downloaded: list[str] = []
        failure: str | None = None
        if (
            acquisition_policy in ("download", "download_if_missing")
            and not status.flowdir_ready
        ):
            spec = vf.region_asset_spec("flowdir", pfaf)
            dest = self.raster_path("flowdir", pfaf)
            try:
                self._stage(spec, dest)
                downloaded.append(f"flowdir_{pfaf}")
                self._write_flowdir_metadata(pfaf, dest, source=spec.url)
                status = self.routing_region_cache(pfaf)
            except Exception as exc:
                failure = f"{spec.name}: {exc}"
                log.warning("MERIT flowdir staging failed: %s", failure)

        status.acquisition_policy = acquisition_policy
        status.required_assets = required
        status.downloaded = downloaded or None
        if failure:
            status.message = f"Staging failed ({failure}); nothing unverified was installed."
        elif not status.flowdir_ready and acquisition_policy == "check_only":
            status.message = (
                f"Regional MERIT flowdir missing for Pfaf {pfaf}; call "
                "merit_ensure_routing_region(..., acquisition_policy='download') "
                "after reviewing the expected regional download."
            )
        return status

    def auto_stage_flowdir(
        self,
        pfaf: str,
        progress_cb: "Callable[[str, int, int], None] | None" = None,
    ) -> "RoutingRegionStatus":
        """
        Auto-stage MERIT flowdir for *pfaf* using the tiered policy.

        Tiered policy (from ~/.aihydro/config.json):
          ≤500 MB   — silent (progress_cb still called so callers can log)
          ≤max_gb   — stage with progress messages
          >max_gb   — return status with action_required = "confirm_large_download"
          policy="never" — return status unchanged, no download attempted

        *progress_cb(message, bytes_done, bytes_total)* — caller can stream to
        ctx.report_progress or log.  Pass None to suppress.

        Returns RoutingRegionStatus; check ``.flowdir_ready`` for success.
        """
        import shutil

        policy, max_gb = _merit_auto_stage_policy()
        pfaf = pfaf.zfill(2)
        status = self.routing_region_cache(pfaf)

        if status.flowdir_ready:
            return status  # already staged — nothing to do

        if policy == "never":
            status.message = (
                "MERIT auto-staging is disabled (merit_auto_stage='never' in config). "
                "Set to 'tiered' or 'always' to enable on-demand staging."
            )
            return status

        spec = vf.region_asset_spec("flowdir", pfaf)
        url = spec.url
        size_bytes = spec.size_bytes  # pinned size from the manifest (no HEAD request needed)
        size_mb = (size_bytes or 0) / (1024 ** 2)
        size_gb = size_mb / 1024

        # Disk-free check
        free_bytes = shutil.disk_usage(self.root.parent).free
        free_gb = free_bytes / (1024 ** 3)
        if size_bytes and free_bytes < size_bytes * 2:
            status.message = (
                f"Not enough disk space to stage Pfaf {pfaf} flowdir "
                f"(needs {size_mb:.0f} MB, only {free_gb:.1f} GB free under "
                f"{self.root.parent}). Free up space or set AIHYDRO_CACHE_DIR."
            )
            if progress_cb:
                progress_cb(status.message, 0, 0)
            return status

        # Tiered gate for very large downloads
        max_bytes = max_gb * (1024 ** 3)
        if policy == "tiered" and size_bytes and size_bytes > max_bytes:
            status.message = (
                f"Pfaf {pfaf} flowdir is large (~{size_gb:.1f} GB > {max_gb:.0f} GB "
                f"threshold). To proceed: call merit_ensure_routing_region("
                f"pfaf_region='{pfaf}', acquisition_policy='download') explicitly, "
                f"or set merit_auto_stage_max_gb>{size_gb:.0f} in ~/.aihydro/config.json."
            )
            setattr(status, "action_required", "confirm_large_download")
            setattr(status, "size_gb", round(size_gb, 1))
            if progress_cb:
                progress_cb(status.message, 0, size_bytes or 0)
            return status

        # --- Proceed with download ---
        region_names = {
            "23": "Rhine / Main basin (Central Europe)",
            "24": "Danube basin (Central-Eastern Europe)",
            "51": "Ganges-Brahmaputra basin (South Asia)",
            "52": "Yangtze basin (East Asia)",
            "41": "Mississippi basin (North America)",
            "81": "Amazon basin (South America)",
        }
        region_label = region_names.get(pfaf, f"Pfaf {pfaf} region")
        size_label = f"{size_mb:.0f} MB" if size_mb < 1000 else f"{size_gb:.1f} GB"

        _start_msg = (
            f"Staging MERIT-Hydro flow-direction for the {region_label}. "
            f"This enables proper hydrological routing (MERIT-pyflwdir) instead of "
            f"raw DEM fallback. Source: Yamazaki et al. 2019; non-commercial licence, SHA-256 verified. "
            f"Download: {size_label} → {self.root / 'raster/flowdir_basins'}"
        )
        log.info(_start_msg)
        if progress_cb:
            progress_cb(_start_msg, 0, size_bytes or 0)

        dest = self.raster_path("flowdir", pfaf)
        dest.parent.mkdir(parents=True, exist_ok=True)

        success = False
        try:
            self._stage(
                spec,
                dest,
                (lambda msg, done, tot: progress_cb(
                    f"{msg} {done / 1048576:.0f} / {tot / 1048576:.0f} MB", done, tot
                ))
                if progress_cb
                else None,
            )
            success = True
        except Exception as exc:
            log.warning("MERIT flowdir staging failed: %s", exc)
            failure_msg = str(exc)
        if success:
            self._write_flowdir_metadata(pfaf, dest, source=url)
            status = self.routing_region_cache(pfaf)
            _done_msg = (
                f"MERIT flowdir for {region_label} staged successfully "
                f"({dest.stat().st_size / (1024**2):.0f} MB). "
                f"Future delineations in this region will use MERIT-pyflwdir."
            )
            log.info(_done_msg)
            if progress_cb:
                progress_cb(_done_msg, size_bytes or 0, size_bytes or 0)
        else:
            status.message = f"Download failed for Pfaf {pfaf} flowdir from {url}: {failure_msg}"
            if progress_cb:
                progress_cb(status.message, 0, size_bytes or 0)

        return status

    def ensure_basin(
        self, lat: float, lon: float, *, download: bool = True
    ) -> BasinEnsureStatus:
        """Ensure verified MERIT-Basins vectors for the basin containing (lat, lon)."""
        downloaded: list[str] = []
        level2_ready = False
        try:
            if self.level2_shapefile_path().exists():
                level2_ready = True
            else:
                level2_ready = self.ensure_level2_index(download=download) is not None
                if level2_ready and download:
                    downloaded.append("level2")
        except Exception as e:
            log.warning("Level-2 index unavailable: %s", e)

        pfaf = self.resolve_pfaf_code(lat, lon) if level2_ready else "00"
        if download and level2_ready:
            st = self.ensure_basins_region(pfaf, acquisition_policy="download")
            downloaded.extend(st.downloaded or [])
        st = self.basins_region_cache(pfaf) if level2_ready else None
        rivers_ready = bool(st and st.rivers_ready)
        catchments_ready = bool(st and st.catchments_ready)
        flowdir_ready = self._glob_flowdir(pfaf) is not None if level2_ready else False

        msg_parts = [f"Pfafstetter basin {pfaf}"]
        if not level2_ready:
            msg_parts.append("level-2 index missing")
        if st is not None and st.acquisition_required:
            msg_parts.append(st.message)

        return BasinEnsureStatus(
            pfaf_code=pfaf,
            level2_ready=level2_ready,
            rivers_ready=rivers_ready,
            catchments_ready=catchments_ready,
            flowdir_ready=flowdir_ready,
            message="; ".join(msg_parts),
            downloaded=downloaded or None,
        )

    def configure_delineator_env(self, pfaf_code: str) -> dict[str, str]:
        """Return env var dict for upstream-delineator (paths must exist)."""
        pfaf = pfaf_code.zfill(2)
        catch = self.catchment_shapefile_path(pfaf)
        river = self.river_shapefile_path(pfaf)
        flowdir = self._glob_flowdir(pfaf)
        accum_dir = self.root / "raster" / "accum_basins"
        accum_matches = list(accum_dir.glob(f"*pfaf_{pfaf}*.tif")) if accum_dir.exists() else []
        megabasins = self.level2_shapefile_path()

        env = {
            "CATCHMENT_PATH": str(catch) if catch.exists() else "",
            "RIVER_PATH": str(river) if river.exists() else "",
            "FLOW_DIR_PATH": str(flowdir) if flowdir else "",
            "ACCUM_PATH": str(accum_matches[0]) if accum_matches else "",
            "MEGABASINS_PATH": str(megabasins) if megabasins.exists() else "",
        }
        return env

    def delineator_ready(self, pfaf_code: str) -> bool:
        env = self.configure_delineator_env(pfaf_code)
        required = ("CATCHMENT_PATH", "RIVER_PATH", "FLOW_DIR_PATH")
        return all(env.get(k) for k in required)
