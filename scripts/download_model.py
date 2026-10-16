"""Resume-download and verify the public CosyVoice3-MNN ZIP; no Git LFS needed."""

import argparse
import hashlib
from pathlib import Path
import sys
import urllib.error
import urllib.request

URL = "https://huggingface.co/VicenTrent/Cosy-Voice-MNN/resolve/main/cosyvoice3-mnn-mobile-fp16-complete.zip?download=true"
EXPECTED_BYTES = 1_399_083_563
EXPECTED_SHA256 = "B1C74DFC90972D82D8166813620A882FE37A0DC02964E19C4F33DAAFEFEB1C84"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "downloads" / "cosyvoice3-mnn-mobile-fp16-complete.zip")
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file() and output.stat().st_size == EXPECTED_BYTES and sha256(output) == EXPECTED_SHA256:
        print(f"Already verified: {output}")
        return 0
    partial = output.with_suffix(output.suffix + ".part")
    for attempt in range(5):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > EXPECTED_BYTES:
            print("Partial file is larger than expected; move it aside manually before retrying.", file=sys.stderr)
            return 1
        if offset == EXPECTED_BYTES:
            break
        next_progress = ((offset // (64 * 1024 * 1024)) + 1) * 64 * 1024 * 1024
        request = urllib.request.Request(URL, headers={"User-Agent": "CosyVoiceMobile/0.1", **({"Range": f"bytes={offset}-"} if offset else {})})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                code = response.status
                if offset and code == 206:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {offset}-"):
                        raise OSError(f"Unexpected Content-Range: {content_range}")
                    mode = "ab"
                elif code == 200:
                    mode = "wb"  # Server ignored Range: restart without duplicating bytes.
                else:
                    raise OSError(f"Unexpected HTTP {code}")
                with partial.open(mode) as destination:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        destination.write(block)
                        if destination.tell() >= next_progress:
                            print(f"Downloaded {destination.tell():,}/{EXPECTED_BYTES:,} bytes", flush=True)
                            next_progress += 64 * 1024 * 1024
            if partial.stat().st_size == EXPECTED_BYTES:
                break
            print(f"Incomplete transfer: {partial.stat().st_size}/{EXPECTED_BYTES} bytes; retrying")
        except (OSError, urllib.error.URLError) as error:
            print(f"Attempt {attempt + 1}/5 interrupted at {partial.stat().st_size if partial.exists() else 0} bytes: {error}")
    if not partial.exists() or partial.stat().st_size != EXPECTED_BYTES:
        print("Download incomplete. Rerun the same command to resume.", file=sys.stderr)
        return 1
    digest = sha256(partial)
    if digest != EXPECTED_SHA256:
        print(f"SHA-256 mismatch: {digest}. Partial file kept for inspection.", file=sys.stderr)
        return 1
    partial.replace(output)
    print(f"Verified {EXPECTED_BYTES} bytes: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
