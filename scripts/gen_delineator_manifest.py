#!/usr/bin/env python
"""
Regenerate ``aihydro_watershed/merit/delineator_data_manifest.json`` from delineator's published hashes.

Pins are taken from ``src/delineator/constants.py`` (the ``HASHES`` table) at a tag resolved to its COMMIT SHA,
so the manifest can be audited against one immutable file. Output is deterministic (sorted keys, fixed layout).

    python scripts/gen_delineator_manifest.py                 # tag v2.2.4, keep sizes from the existing manifest
    python scripts/gen_delineator_manifest.py --refresh-sizes  # HEAD every file for Content-Length (240 small requests)
    python scripts/gen_delineator_manifest.py --constants tests/fixtures/delineator_v2.2.4/constants.py \
        --commit c56095e8625d1ee3534217b0462a8ed43027c77c      # offline: no network except none

Sizes are not derivable from ``constants.py``; without ``--refresh-sizes`` they are copied from the existing manifest
and any file missing from it makes the script stop. It downloads no data files.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path

REPO = "mheberger/delineator"
DATA_BASE = "https://mghydro.com/watersheds/data/"
OUT = Path(__file__).resolve().parents[1] / "aihydro_watershed" / "merit" / "delineator_data_manifest.json"


def _get(url: str, method: str = "GET"):
    req = urllib.request.Request(url, method=method, headers={"User-Agent": "aihydro-watershed-manifest"})
    return urllib.request.urlopen(req, timeout=30)  # noqa: S310


def resolve_commit(tag: str) -> str:
    with _get(f"https://api.github.com/repos/{REPO}/commits/{tag}") as r:
        return json.load(r)["sha"]


def blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def parse_hashes(constants_py: str) -> dict[str, str]:
    m = re.search(r"HASHES = (\{.*?\n\})", constants_py, re.S)
    if not m:
        raise SystemExit("HASHES table not found in constants.py")
    return {k: v.replace("sha256:", "") for k, v in ast.literal_eval(m.group(1)).items()}


def build(constants_bytes: bytes, commit: str, tag: str, sizes: dict[str, int], level2: dict, old: dict) -> dict:
    hashes = parse_hashes(constants_bytes.decode("utf-8"))
    missing = [n for n in hashes if n not in sizes]
    if missing:
        raise SystemExit(f"no size for {len(missing)} files (e.g. {missing[0]}); rerun with --refresh-sizes")
    man = {k: v for k, v in old.items() if k not in ("files", "source", "level2_index")}
    man["source"] = {
        "package": "delineator",
        "package_version": tag.lstrip("v"),
        "package_license": "MIT",
        "tag": tag,
        "tag_commit": commit,
        "constants_blob_sha1": blob_sha1(constants_bytes),
        "hashes_file": f"https://github.com/{REPO}/blob/{commit}/src/delineator/constants.py",
        "base_url": DATA_BASE,
        "data_version": old.get("source", {}).get("data_version", "v3"),
        "data_updated": old.get("source", {}).get("data_updated", "2026-07-12"),
        "checked": old.get("source", {}).get("checked", ""),
    }
    man["level2_index"] = level2
    man["files"] = {n: {"sha256": hashes[n], "size_bytes": sizes[n]} for n in sorted(hashes)}
    return man


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="v2.2.4")
    ap.add_argument("--commit", help="commit SHA (default: resolve --tag via the GitHub API)")
    ap.add_argument("--constants", type=Path, help="local constants.py (skips fetching it)")
    ap.add_argument("--refresh-sizes", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--check", action="store_true", help="exit 1 if --out differs from what would be written")
    a = ap.parse_args()

    old = json.loads(a.out.read_text()) if a.out.exists() else {}
    commit = a.commit or resolve_commit(a.tag)
    if a.constants:
        data = a.constants.read_bytes()
    else:
        with _get(f"https://raw.githubusercontent.com/{REPO}/{commit}/src/delineator/constants.py") as r:
            data = r.read()
    names = parse_hashes(data.decode("utf-8"))
    if a.refresh_sizes:
        sizes = {}
        for n in sorted(names):
            with _get(DATA_BASE + n, "HEAD") as r:
                sizes[n] = int(r.headers["Content-Length"])
    else:
        sizes = {n: e["size_bytes"] for n, e in old.get("files", {}).items()}
    lvl = dict(old.get("level2_index", {}))
    lvl["url"] = f"https://raw.githubusercontent.com/{REPO}/{commit}/src/delineator/data/megabasins.db"
    out = json.dumps(build(data, commit, a.tag, sizes, lvl, old), indent=1, sort_keys=True) + "\n"
    if a.check:
        sys.exit(0 if a.out.exists() and a.out.read_text() == out else 1)
    a.out.write_text(out)
    print(f"wrote {a.out} ({len(names)} files, commit {commit[:12]})")


if __name__ == "__main__":
    main()
