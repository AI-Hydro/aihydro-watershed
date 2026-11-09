"""
Merge several single-layer vector PMTiles archives into one multi-layer archive.

Why this exists: GDAL's MVT/PMTiles writer cannot do per-feature zoom limits and cannot append layers to
an existing archive (``CreateLayer() not supported by this dataset``), and ``pyogrio`` writes one layer per
call. The Z1 rule (reaches appear by upstream area) needs one layer per area band, each with its own
``MINZOOM``. So each band is written as its own archive and merged here.

A Mapbox Vector Tile is a protobuf message made only of repeated ``layers`` fields, so concatenating the
decompressed bytes of two tiles that share a tile id is itself a valid tile holding both layers. The merge
is therefore a tile-id join; no tile is decoded or re-encoded. Only PMTiles v3 with gzip/none tile
compression is accepted; anything else raises rather than guessing.

Format reference: https://github.com/protomaps/PMTiles/blob/main/spec/v3/spec.md
"""
from __future__ import annotations

import gzip
import json
import struct
from pathlib import Path
from typing import Any, Iterator

_HEADER = struct.Struct("<7sB11Q6B4iBii")
HEADER_BYTES = 127
_NONE, _GZIP = 1, 2
_ROOT_BUDGET = 16384 - HEADER_BYTES  # spec: header + root directory fit in the first 16 KiB


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    shift = result = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, pos
        shift += 7


def _decompress(data: bytes, mode: int) -> bytes:
    if mode == _GZIP:
        return gzip.decompress(data)
    if mode == _NONE:
        return data
    raise ValueError(f"unsupported PMTiles compression code {mode} (only none and gzip are handled)")


def _compress(data: bytes, mode: int) -> bytes:
    return gzip.compress(data, 6, mtime=0) if mode == _GZIP else data


def _decode_directory(raw: bytes) -> list[tuple[int, int, int, int]]:
    """Entries as ``(tile_id, run_length, length, offset)``; run_length 0 marks a leaf-directory pointer."""
    n, pos = _read_varint(raw, 0)
    ids, runs, lens, offs = [], [], [], []
    last = 0
    for _ in range(n):
        d, pos = _read_varint(raw, pos)
        last += d
        ids.append(last)
    for _ in range(n):
        v, pos = _read_varint(raw, pos)
        runs.append(v)
    for _ in range(n):
        v, pos = _read_varint(raw, pos)
        lens.append(v)
    for i in range(n):
        v, pos = _read_varint(raw, pos)
        if v == 0 and i > 0:
            offs.append(offs[i - 1] + lens[i - 1])
        else:
            offs.append(v - 1)
    return list(zip(ids, runs, lens, offs))


def _encode_directory(entries: list[tuple[int, int, int, int]]) -> bytes:
    out = bytearray(_varint(len(entries)))
    last = 0
    for tid, _r, _l, _o in entries:
        out += _varint(tid - last)
        last = tid
    for _t, run, _l, _o in entries:
        out += _varint(run)
    for _t, _r, length, _o in entries:
        out += _varint(length)
    prev_end = None
    for _t, _r, length, off in entries:
        if prev_end is not None and off == prev_end:
            out += _varint(0)
        else:
            out += _varint(off + 1)
        prev_end = off + length
    return bytes(out)


class PMTilesReader:
    """Minimal PMTiles v3 reader: header, metadata and the (tile_id -> bytes) map."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with self.path.open("rb") as fh:
            head = fh.read(HEADER_BYTES)
        if len(head) < HEADER_BYTES or head[:7] != b"PMTiles":
            raise ValueError(f"{self.path}: not a PMTiles archive")
        f = _HEADER.unpack(head)
        if f[1] != 3:
            raise ValueError(f"{self.path}: PMTiles version {f[1]} is not supported (need 3)")
        (self.root_off, self.root_len, self.meta_off, self.meta_len, self.leaf_off, self.leaf_len,
         self.data_off, self.data_len, self.n_addressed, self.n_entries, self.n_contents) = f[2:13]
        (self.clustered, self.internal_comp, self.tile_comp, self.tile_type,
         self.min_zoom, self.max_zoom) = f[13:19]
        (self.min_lon_e7, self.min_lat_e7, self.max_lon_e7, self.max_lat_e7) = f[19:23]
        self.center_zoom, self.center_lon_e7, self.center_lat_e7 = f[23:26]

    def _slice(self, fh, off: int, length: int) -> bytes:
        fh.seek(off)
        return fh.read(length)

    def metadata(self) -> dict[str, Any]:
        with self.path.open("rb") as fh:
            raw = _decompress(self._slice(fh, self.meta_off, self.meta_len), self.internal_comp)
        return json.loads(raw.decode("utf-8")) if raw else {}

    def _entries(self, fh) -> Iterator[tuple[int, int, int, int]]:
        stack = [(self.root_off, self.root_len, 0)]
        while stack:
            off, length, depth = stack.pop()
            if depth > 3:
                raise ValueError(f"{self.path}: directory nesting deeper than the spec allows")
            raw = _decompress(self._slice(fh, off, length), self.internal_comp)
            for tid, run, tlen, toff in _decode_directory(raw):
                if run == 0:
                    stack.append((self.leaf_off + toff, tlen, depth + 1))
                else:
                    yield tid, run, tlen, toff

    def tiles(self) -> Iterator[tuple[int, bytes]]:
        """Yield ``(tile_id, decompressed MVT bytes)`` for every addressed tile, in tile-id order."""
        with self.path.open("rb") as fh:
            entries = sorted(self._entries(fh))
            for tid, run, tlen, toff in entries:
                data = _decompress(self._slice(fh, self.data_off + toff, tlen), self.tile_comp)
                for k in range(run):
                    yield tid + k, data


def _tile_bounds(readers: list[PMTilesReader]) -> tuple[int, int, int, int]:
    return (
        min(r.min_lon_e7 for r in readers), min(r.min_lat_e7 for r in readers),
        max(r.max_lon_e7 for r in readers), max(r.max_lat_e7 for r in readers),
    )


def merge_pmtiles(
    sources: list[str | Path],
    dest: str | Path,
    *,
    metadata_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge ``sources`` (distinct layer names) into ``dest``; return counts for the provenance record."""
    readers = [PMTilesReader(p) for p in sources]
    if not readers:
        raise ValueError("no archives to merge")
    if any(r.tile_type != 1 for r in readers):
        raise ValueError("only vector (MVT) PMTiles can be merged")
    if len({r.tile_comp for r in readers}) != 1:
        raise ValueError("source archives use different tile compression")
    tile_comp = readers[0].tile_comp

    layers: list[dict[str, Any]] = []
    first_meta: dict[str, Any] = {}
    for r in readers:
        meta = r.metadata()
        first_meta = first_meta or meta
        vl = meta.get("vector_layers")
        if vl is None and isinstance(meta.get("json"), str):
            vl = json.loads(meta["json"]).get("vector_layers")
        layers.extend(vl or [])
    names = [lyr["id"] for lyr in layers]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate layer names across archives: {names}")

    # k-way join on tile id: concatenating MVT messages unions their layers
    iters = [r.tiles() for r in readers]
    heads: list[tuple[int, bytes] | None] = [next(it, None) for it in iters]
    dest = Path(dest)
    tmp = dest.with_name(dest.name + ".partial")
    entries: list[tuple[int, int, int, int]] = []
    offset = 0
    n_tiles = 0
    # tile data goes to a side file first: the directories precede it in the archive and need its sizes
    data_tmp = dest.with_name(dest.name + ".tiledata")
    try:
        with data_tmp.open("wb") as dfh:
            while True:
                live = [h[0] for h in heads if h is not None]
                if not live:
                    break
                tid = min(live)
                parts = []
                for i, h in enumerate(heads):
                    if h is not None and h[0] == tid:
                        parts.append(h[1])
                        heads[i] = next(iters[i], None)
                blob = _compress(b"".join(parts), tile_comp)
                dfh.write(blob)
                entries.append((tid, 1, len(blob), offset))
                offset += len(blob)
                n_tiles += 1

        # directories: one root if it fits the 16 KiB budget, else fixed-size leaves under a root
        comp = _GZIP
        root = _compress(_encode_directory(entries), comp)
        leaves = b""
        if len(root) > _ROOT_BUDGET:
            leaf_size = 4096
            while True:
                leaves = b""
                root_entries = []
                for i in range(0, len(entries), leaf_size):
                    chunk = _compress(_encode_directory(entries[i:i + leaf_size]), comp)
                    root_entries.append((entries[i][0], 0, len(chunk), len(leaves)))
                    leaves += chunk
                root = _compress(_encode_directory(root_entries), comp)
                if len(root) <= _ROOT_BUDGET:
                    break
                leaf_size *= 2

        merged = {
            "name": first_meta.get("name", dest.stem), "description": "",
            "version": first_meta.get("version", "2"), "type": "overlay", "format": "pbf",
            "minzoom": str(min(r.min_zoom for r in readers)),
            "maxzoom": str(max(r.max_zoom for r in readers)),
            "vector_layers": layers,
        }
        for k in ("center", "bounds"):  # GDAL writes these as comma strings; keep the first archive's
            if k in first_meta:
                merged[k] = first_meta[k]
        merged.update(metadata_extra or {})
        meta_blob = _compress(json.dumps(merged, indent=1).encode("utf-8"), comp)

        root_off = HEADER_BYTES
        meta_off = root_off + len(root)
        leaf_off = meta_off + len(meta_blob)
        data_off = leaf_off + len(leaves)
        lo, la, hi, ha = _tile_bounds(readers)
        r0 = readers[0]
        header = _HEADER.pack(
            b"PMTiles", 3, root_off, len(root), meta_off, len(meta_blob), leaf_off, len(leaves),
            data_off, offset, n_tiles, n_tiles, n_tiles, 1, comp, tile_comp, 1,
            min(r.min_zoom for r in readers), max(r.max_zoom for r in readers),
            lo, la, hi, ha, r0.center_zoom, r0.center_lon_e7, r0.center_lat_e7,
        )
        with tmp.open("wb") as out:
            out.write(header + root + meta_blob + leaves)
            with data_tmp.open("rb") as dfh:
                while chunk := dfh.read(1 << 20):
                    out.write(chunk)
        tmp.replace(dest)
    finally:
        data_tmp.unlink(missing_ok=True)
        tmp.unlink(missing_ok=True)
    return {"tiles": n_tiles, "layers": names, "leaf_directories": bool(leaves)}
