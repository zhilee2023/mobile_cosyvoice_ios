"""Fetch one verified member from the public ZIP via HTTP ranges.

This permits early Windows graph tests while the full resumable archive is
still downloading. The final archive must still be downloaded and verified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import urllib.request
import zipfile
import zlib

from download_model import EXPECTED_BYTES, URL

KNOWN_FLOW_GRAPH_SHA256 = "B74ECE0A1A2E165EEF53DBC8D628310A1B464323D4729959069744BAECF418D7"


def read_range(first: int, last: int) -> bytes:
    partial = Path(__file__).resolve().parents[1] / "downloads" / "cosyvoice3-mnn-mobile-fp16-complete.zip.part"
    if partial.is_file() and partial.stat().st_size > last:
        with partial.open("rb") as source:
            source.seek(first)
            data = source.read(last - first + 1)
        if len(data) == last - first + 1:
            return data
    request = urllib.request.Request(URL, headers={"Range": f"bytes={first}-{last}", "User-Agent": "CosyVoiceMobile/0.1"})
    with urllib.request.urlopen(request, timeout=90) as response:
        if response.status != 206:
            raise RuntimeError(f"Server did not honor range: HTTP {response.status}")
        expected_header = f"bytes {first}-{last}/{EXPECTED_BYTES}"
        if response.headers.get("Content-Range") != expected_header:
            raise RuntimeError(f"Unexpected Content-Range: {response.headers.get('Content-Range')}")
        data = response.read()
    if len(data) != last - first + 1:
        raise RuntimeError(f"Truncated range: {len(data)} of {last - first + 1} bytes")
    return data


def members() -> dict[str, tuple[int, int, int, int]]:
    tail_start = EXPECTED_BYTES - 131072
    tail = read_range(tail_start, EXPECTED_BYTES - 1)
    eocd_at = tail.rfind(zipfile.stringEndArchive)
    if eocd_at < 0:
        raise RuntimeError("ZIP end-of-central-directory record not found")
    eocd = struct.unpack(zipfile.structEndArchive, tail[eocd_at : eocd_at + zipfile.sizeEndCentDir])
    central_size, central_offset = eocd[5], eocd[6]
    if central_offset < tail_start or central_offset + central_size > EXPECTED_BYTES:
        raise RuntimeError("Central directory is outside fetched ZIP tail")
    directory = tail[central_offset - tail_start : central_offset - tail_start + central_size]
    result = {}
    cursor = 0
    while cursor < len(directory):
        fields = struct.unpack(zipfile.structCentralDir, directory[cursor : cursor + zipfile.sizeCentralDir])
        if fields[0] != zipfile.stringCentralDir:
            raise RuntimeError("Invalid central-directory entry")
        name_len = fields[zipfile._CD_FILENAME_LENGTH]
        extra_len = fields[zipfile._CD_EXTRA_FIELD_LENGTH]
        comment_len = fields[zipfile._CD_COMMENT_LENGTH]
        name_start = cursor + zipfile.sizeCentralDir
        name = directory[name_start : name_start + name_len].decode("utf-8")
        result[name] = (
            fields[zipfile._CD_LOCAL_HEADER_OFFSET],
            fields[zipfile._CD_COMPRESSED_SIZE],
            fields[zipfile._CD_UNCOMPRESSED_SIZE],
            fields[zipfile._CD_COMPRESS_TYPE],
        )
        cursor = name_start + name_len + extra_len + comment_len
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="Model filename from model-manifest.json")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "models" / "CosyVoice3")
    parser.add_argument("--accept-known-flow-graph-mismatch", action="store_true",
                        help="Provisional PC-only use of the observed ZIP graph; full ZIP audit still required")
    args = parser.parse_args()
    manifest = json.loads((Path(__file__).resolve().parents[1] / "model-manifest.json").read_text(encoding="utf-8"))
    entry = next((item for item in manifest["files"] if item["name"] == args.name), None)
    if entry is None:
        parser.error(f"Not a required model file: {args.name}")
    output = args.output_dir.resolve() / args.name
    if output.exists():
        digest = hashlib.sha256(output.read_bytes()).hexdigest().upper()
        existing_known_exception = (args.accept_known_flow_graph_mismatch and args.name == "flow.cfg-student-2step.batch1.fp16.mnn"
                                    and digest == KNOWN_FLOW_GRAPH_SHA256)
        if output.stat().st_size == entry["bytes"] and (digest == entry["sha256"] or existing_known_exception):
            print(f"Already {'provisionally identified' if existing_known_exception else 'verified'}: {output}")
            return 0
        raise RuntimeError(f"Refusing to overwrite different file: {output}")

    listing = members()
    matches = [name for name in listing if name == args.name or name.endswith("/" + args.name)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one ZIP member for {args.name}, found {matches}")
    offset, compressed_size, size, method = listing[matches[0]]
    local = read_range(offset, offset + zipfile.sizeFileHeader - 1)
    header = struct.unpack(zipfile.structFileHeader, local)
    if header[0] != zipfile.stringFileHeader:
        raise RuntimeError("Invalid local ZIP header")
    payload_offset = offset + zipfile.sizeFileHeader + header[zipfile._FH_FILENAME_LENGTH] + header[zipfile._FH_EXTRA_FIELD_LENGTH]
    compressed = read_range(payload_offset, payload_offset + compressed_size - 1)
    if method == zipfile.ZIP_STORED:
        data = compressed
    elif method == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(compressed, -15)
    else:
        raise RuntimeError(f"Unsupported ZIP compression: {method}")
    digest = hashlib.sha256(data).hexdigest().upper()
    known_exception = (args.accept_known_flow_graph_mismatch and args.name == "flow.cfg-student-2step.batch1.fp16.mnn"
                       and digest == KNOWN_FLOW_GRAPH_SHA256)
    if len(data) != size or size != entry["bytes"] or (digest != entry["sha256"] and not known_exception):
        raise RuntimeError(f"Member failed size/SHA-256 check: {args.name}; ZIP={size} bytes, decoded={len(data)} bytes, manifest={entry['bytes']} bytes; SHA={digest}, expected={entry['sha256']}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(data)
    if known_exception:
        print(f"PROVISIONAL: Flow graph has documented upstream manifest mismatch: {digest}")
    else:
        print(f"Verified ZIP member: {output} ({len(data)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
