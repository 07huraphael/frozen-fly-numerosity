"""Create a portable project snapshot, excluding environments and raw public data."""
from pathlib import Path
import zipfile


def package():
    root = Path(__file__).resolve().parents[1]
    out = root / "deliverables" / "frozen-fly-project.zip"
    out.parent.mkdir(exist_ok=True)
    selected = [root / n for n in ["README.md", "PROJECT_STATUS.md", "pyproject.toml", "requirements-lock.txt", ".gitignore"]]
    for folder in ["src", "tests", "scripts", "configs", "docs", "data/processed", "results"]:
        selected.extend(p for p in (root / folder).rglob("*") if p.is_file()
                        and "__pycache__" not in p.parts and not p.suffix == ".pyc"
                        and not any(part.endswith(".egg-info") for part in p.parts))
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in selected:
            z.write(path, path.relative_to(root))
    with zipfile.ZipFile(out) as z:
        bad = z.testzip()
        if bad:
            raise RuntimeError(f"Corrupt archive member: {bad}")
    print(f"Portable snapshot: {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    package()
