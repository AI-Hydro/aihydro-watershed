"""
Pinned, hash-verified acquisition of per-region MERIT-Basins / MERIT-Hydro files.

The manifest (``delineator_data_manifest.json``) pins every file served by
mghydro.com for the ``delineator`` package (v2.2.4, data version v3) by SHA-256
and size. Nothing is installed or trusted unless its digest matches; a mismatch
raises :class:`AssetIntegrityError` (fail closed) and a downloaded file that does
not match is deleted before it ever reaches its final path.

The level-2 (megabasin) index is the one file mghydro.com does not serve. It is
pinned by the git blob SHA-1 at the delineator v2.2.4 tag, and the SHA-256 of the
first verified copy is recorded in the provenance sidecar and enforced afterwards.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import IO, Any, Callable

log = logging.getLogger(__name__)

_MANIFEST_PATH = Path(__file__).parent / "delineator_data_manifest.json"
_CHUNK = 1024 * 1024
VALID_KINDS = ("basins", "rivers", "flowdir", "accum")


class AssetIntegrityError(RuntimeError):
    """A MERIT data file does not match its pinned digest or size (fail closed)."""

    code = "MERIT_ASSET_INTEGRITY_FAILED"

    def __init__(self, message: str):
        super().__init__(f"{self.code}: {message}")


@dataclass(frozen=True)
class AssetSpec:
    name: str
    url: str
    size_bytes: int | None
    sha256: str | None
    git_blob_sha1: str | None = None


@lru_cache(maxsize=1)
def load_manifest() -> dict[str, Any]:
    return json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))


def licence_summary() -> str:
    return str(load_manifest()["licence"]["summary"])


def mirror_base_url() -> str:
    """Optional lab mirror (``AIHYDRO_MERIT_BASE_URL``); files are still hash-verified."""
    return os.environ.get("AIHYDRO_MERIT_BASE_URL", "").rstrip("/")


def asset_spec(name: str) -> AssetSpec:
    man = load_manifest()
    mirror = mirror_base_url()
    lvl2 = man["level2_index"]
    if name == lvl2["name"]:
        url = f"{mirror}/{name}" if mirror else lvl2["url"]
        return AssetSpec(name, url, lvl2["size_bytes"], lvl2.get("sha256"), lvl2.get("git_blob_sha1"))
    entry = man["files"].get(name)
    if entry is None:
        raise KeyError(f"{name} is not in the pinned MERIT data manifest.")
    base = mirror or man["source"]["base_url"].rstrip("/")
    return AssetSpec(name, f"{base}/{name}", entry["size_bytes"], entry["sha256"])


def region_asset_name(kind: str, pfaf: str) -> str:
    if kind not in VALID_KINDS:
        raise ValueError(f"kind must be one of {VALID_KINDS}")
    ext = "db" if kind in ("basins", "rivers") else "tif"
    return f"{kind}{str(int(pfaf)).zfill(2)}.{ext}"


def region_asset_spec(kind: str, pfaf: str) -> AssetSpec:
    return asset_spec(region_asset_name(kind, pfaf))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1_file(path: Path) -> str:
    size = Path(path).stat().st_size
    h = hashlib.sha1(f"blob {size}\0".encode())
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def sidecar_path(path: Path) -> Path:
    return path.with_name(path.name + ".provenance.json")


def _write_sidecar(path: Path, spec: AssetSpec, sha256: str, *, source: str) -> dict[str, Any]:
    st = path.stat()
    man = load_manifest()
    rec = {
        "name": spec.name,
        "sha256": sha256,
        "size_bytes": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "url": spec.url,
        "pinned_sha256": spec.sha256,
        "pinned_git_blob_sha1": spec.git_blob_sha1,
        "source": source,
        "data_version": man["source"]["data_version"],
        "delineator_version": man["source"]["package_version"],
        "licence": man["licence"]["summary"],
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }
    sidecar_path(path).write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return rec


def _read_sidecar(path: Path) -> dict[str, Any] | None:
    sc = sidecar_path(path)
    if not sc.exists():
        return None
    try:
        return json.loads(sc.read_text(encoding="utf-8"))
    except Exception:
        return None


def read_provenance(path: Path) -> dict[str, Any] | None:
    """The verification record written when ``path`` was verified, if any."""
    return _read_sidecar(Path(path))


def verify_file(path: Path, spec: AssetSpec, *, source: str = "local_file") -> dict[str, Any]:
    """Hash ``path`` against ``spec`` and write the provenance sidecar; raise on any mismatch."""
    path = Path(path)
    size = path.stat().st_size
    if spec.size_bytes is not None and size != spec.size_bytes:
        raise AssetIntegrityError(
            f"{path.name} is {size} bytes; the pinned manifest says {spec.size_bytes}. "
            "Delete it and re-stage."
        )
    digest = sha256_file(path)
    if spec.sha256 is not None:
        if digest != spec.sha256:
            raise AssetIntegrityError(
                f"{path.name} SHA-256 {digest[:16]}... does not match the pinned {spec.sha256[:16]}.... "
                "Delete it and re-stage."
            )
    else:
        prior = _read_sidecar(path)
        if prior and prior.get("sha256") and prior["sha256"] != digest:
            raise AssetIntegrityError(
                f"{path.name} changed since it was first verified (SHA-256 differs from its provenance record)."
            )
        if spec.git_blob_sha1 is not None and git_blob_sha1_file(path) != spec.git_blob_sha1:
            raise AssetIntegrityError(
                f"{path.name} git blob SHA-1 does not match the pin for the delineator release tag."
            )
    return _write_sidecar(path, spec, digest, source=source)


def verified_path(path: Path, spec: AssetSpec) -> Path | None:
    """Return ``path`` if it exists and verifies; ``None`` if absent; raise if present but wrong.

    A matching provenance sidecar (same size and mtime) avoids re-hashing multi-hundred-MB files.
    """
    path = Path(path)
    if not path.exists():
        return None
    st = path.stat()
    rec = _read_sidecar(path)
    if (
        rec
        and rec.get("size_bytes") == st.st_size
        and rec.get("mtime_ns") == st.st_mtime_ns
        and (spec.sha256 is None or rec.get("sha256") == spec.sha256)
        and rec.get("sha256")
    ):
        return path
    verify_file(path, spec, source=(rec or {}).get("source", "local_file"))
    return path


Opener = Callable[[str], IO[bytes]]


def _default_opener(url: str) -> IO[bytes]:
    return urllib.request.urlopen(url, timeout=300)  # noqa: S310


def fetch_verified(
    spec: AssetSpec,
    dest: Path,
    *,
    opener: Opener | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
) -> Path:
    """Stream ``spec.url`` to ``dest``, installing it only if size and digest match the pin."""
    if spec.sha256 is None and spec.git_blob_sha1 is None:
        raise AssetIntegrityError(f"{spec.name} has no pinned digest; refusing to download.")
    if spec.size_bytes is None:
        raise AssetIntegrityError(f"{spec.name} has no pinned size; refusing to download.")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    sha = hashlib.sha256()
    blob = hashlib.sha1(f"blob {spec.size_bytes}\0".encode())
    got = 0
    open_ = opener or _default_opener
    try:
        with open_(spec.url) as resp, part.open("wb") as out:
            while True:
                chunk = resp.read(_CHUNK)
                if not chunk:
                    break
                out.write(chunk)
                sha.update(chunk)
                blob.update(chunk)
                got += len(chunk)
                if got > spec.size_bytes:
                    raise AssetIntegrityError(
                        f"{spec.name}: received more than the pinned {spec.size_bytes} bytes."
                    )
                if progress_cb:
                    progress_cb(got, spec.size_bytes)
        if got != spec.size_bytes:
            raise AssetIntegrityError(f"{spec.name}: got {got} bytes, pinned size is {spec.size_bytes}.")
        digest = sha.hexdigest()
        if spec.sha256 is not None and digest != spec.sha256:
            raise AssetIntegrityError(
                f"{spec.name}: SHA-256 {digest[:16]}... does not match the pinned {spec.sha256[:16]}...."
            )
        if spec.sha256 is None and blob.hexdigest() != spec.git_blob_sha1:
            raise AssetIntegrityError(f"{spec.name}: git blob SHA-1 does not match the release-tag pin.")
        os.replace(part, dest)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    _write_sidecar(dest, spec, digest, source=spec.url)
    return dest
