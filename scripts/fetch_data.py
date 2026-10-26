"""Download the version-pinned open dataset. Uses only the Python standard library."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

RECORD = "https://zenodo.org/records/21549559"
FILES = [
    "README.md", "DATA_DICTIONARY.md", "MANIFEST.csv", "SHA256SUMS",
    "neuron_annotations_flywire_v2.1.0.tsv.gz",
    "right_visual_positions_all_neurons.csv", "connections_biological.csv.gz",
]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(out: Path, metadata_only: bool = False) -> None:
    out.mkdir(parents=True, exist_ok=True)
    files = FILES[:-1] if metadata_only else FILES
    for name in files:
        target = out / name
        if target.exists():
            print(f"Exists: {name}", flush=True)
            continue
        url = f"{RECORD}/files/{name}?download=1"
        partial = target.with_suffix(target.suffix + ".part")
        for attempt in range(3):
            try:
                request = Request(url, headers={"User-Agent": "frozen-fly-numerosity/0.1 research"})
                with urlopen(request, timeout=60) as response, partial.open("wb") as dst:
                    total = 0
                    last_print = time.monotonic()
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        dst.write(chunk)
                        total += len(chunk)
                        if time.monotonic() - last_print > 10:
                            print(f"{name}: {total / 1e6:.1f} MB", flush=True)
                            last_print = time.monotonic()
                partial.replace(target)
                print(f"Downloaded {name} ({total:,} bytes)", flush=True)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))
    expected = {}
    for line in (out / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2:
            expected[parts[-1].lstrip("*")] = parts[0]
    checked = []
    for name in files:
        actual = digest(out / name)
        if name in expected and actual != expected[name]:
            raise ValueError(f"Checksum mismatch: {name}; remove and download again")
        checked.append({"file": name, "sha256": actual, "verified_against_deposit": name in expected})
    (out / "download_manifest.json").write_text(json.dumps({
        "source": RECORD, "version": "1.0.0", "license": "CC-BY-4.0",
        "files": checked,
    }, indent=2), encoding="utf-8")
    print("Checksums verified; manifest written.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/raw/zenodo_21549559"))
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    fetch(args.out, args.metadata_only)
