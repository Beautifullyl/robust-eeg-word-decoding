from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()

    root = args.root.resolve()
    output = root / "audit" / "FILE_MANIFEST.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path != output
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    )

    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "bytes", "sha256"))
        writer.writeheader()
        for path in paths:
            writer.writerow(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": digest(path),
                }
            )

    print(f"Wrote {len(paths)} entries to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
