"""Fetch the canonical FlyWire connection table for independent provenance checks."""
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time
from urllib.request import Request, urlopen


def main():
    out = Path("data/raw/flywire_783/proofread_connections_783.feather")
    out.parent.mkdir(parents=True, exist_ok=True)
    url = "https://zenodo.org/records/10676866/files/proofread_connections_783.feather?download=1"
    expected_md5 = "f48f972d262323a102aed49af1396b8a"
    if not out.exists():
        partial = out.with_suffix(".feather.part")
        size, block = 852022274, 16 * 1024 * 1024
        parts = out.with_suffix(".parts")
        parts.mkdir(exist_ok=True)
        # Reuse complete blocks from an interrupted sequential download.
        if partial.exists():
            with partial.open("rb") as previous:
                for i in range(partial.stat().st_size // block):
                    data = previous.read(block)
                    piece = parts / f"{i:04}.bin"
                    if not piece.exists():
                        piece.write_bytes(data)

        def get_piece(i):
            start, end = i * block, min((i + 1) * block, size) - 1
            piece = parts / f"{i:04}.bin"
            if piece.exists() and piece.stat().st_size == end - start + 1:
                return piece.stat().st_size
            for attempt in range(4):
                try:
                    request = Request(url, headers={"User-Agent": "frozen-fly-numerosity/0.1 research",
                                                    "Range": f"bytes={start}-{end}"})
                    with urlopen(request, timeout=90) as src:
                        if src.status != 206 or src.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                            raise ValueError("Unexpected HTTP byte-range response")
                        with piece.open("wb") as dst:
                            for chunk in iter(lambda: src.read(1024 * 1024), b""):
                                dst.write(chunk)
                    if piece.stat().st_size != end - start + 1:
                        raise ValueError("Incomplete HTTP range")
                    return piece.stat().st_size
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(2 * (attempt + 1))
        completed = 0
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(get_piece, i) for i in range((size + block - 1) // block)]
            for future in as_completed(futures):
                completed += future.result()
                print(f"Canonical table: {completed / 1e6:.1f} / 852.0 MB", flush=True)
        with partial.open("wb") as dst:
            for i in range((size + block - 1) // block):
                with (parts / f"{i:04}.bin").open("rb") as src:
                    for chunk in iter(lambda: src.read(1024 * 1024), b""):
                        dst.write(chunk)
        partial.replace(out)
    md5, sha = hashlib.md5(), hashlib.sha256()
    with out.open("rb") as f:
        for chunk in iter(lambda: f.read(2 * 1024 * 1024), b""):
            md5.update(chunk); sha.update(chunk)
    if md5.hexdigest() != expected_md5:
        raise ValueError("Canonical FlyWire MD5 mismatch")
    out.with_suffix(".sha256").write_text(sha.hexdigest() + "\n", encoding="utf-8")
    print(f"Verified canonical FlyWire table: {out.stat().st_size:,} bytes, SHA256 {sha.hexdigest()}", flush=True)
    parts = out.with_suffix(".parts")
    if parts.exists():
        if not parts.resolve().is_relative_to(Path.cwd().resolve()):
            raise ValueError("Refusing to clean download pieces outside the workspace")
        for piece in parts.glob("*.bin"):
            if piece.resolve().parent == parts.resolve():
                piece.unlink()
        if not any(parts.iterdir()):
            parts.rmdir()


if __name__ == "__main__":
    main()
