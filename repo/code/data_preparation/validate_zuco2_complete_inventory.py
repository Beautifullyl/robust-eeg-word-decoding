from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ZUCO2_ROOT = Path.home() / "scratch/eeg_project/datasets/zuco2_zip/zuco2"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "validation" / "zuco2"

SUBJECT_RE = re.compile(r"\bY[A-Z]{2}\b")
TASK_BLOCK_RE = re.compile(r"_(NR|TSR)(\d+)_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate non-MAT ZuCo2 inventory, materials, answers, and scripts.")
    parser.add_argument("--zuco2-root", type=Path, default=DEFAULT_ZUCO2_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def file_kind(path: Path) -> str:
    name = path.name
    suffix = path.suffix.lower()
    if name.endswith("_EEG.mat"):
        return "eeg_mat"
    if name.endswith("_ET.mat"):
        return "et_mat"
    if name.startswith("results") and suffix == ".mat":
        return "results_mat"
    if suffix == ".csv":
        return "csv"
    if suffix == ".py":
        return "python"
    if suffix == ".mat":
        return "other_mat"
    return suffix.lstrip(".") or "no_extension"


def infer_section(path: Path, root: Path) -> tuple[str, str]:
    parts = path.relative_to(root).parts
    top = parts[0] if parts else ""
    second = parts[1] if len(parts) > 1 else ""
    return top, second


def infer_subject(path: Path) -> str:
    match = SUBJECT_RE.search(path.name)
    if match:
        return match.group(0)
    for part in path.parts:
        if SUBJECT_RE.fullmatch(part):
            return part
    return ""


def infer_block(path: Path) -> str:
    match = TASK_BLOCK_RE.search(path.name)
    if match:
        return f"{match.group(1)}{match.group(2)}"
    return ""


def inventory(root: Path, out_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        section, layer = infer_section(path, root)
        row = {
            "relative_path": str(path.relative_to(root)),
            "section": section,
            "layer": layer,
            "name": path.name,
            "extension": path.suffix.lower(),
            "kind": file_kind(path),
            "subject": infer_subject(path),
            "block": infer_block(path),
            "size_mb": round(path.stat().st_size / 1024 / 1024, 6),
        }
        rows.append(row)

    out_path = out_dir / "zuco2_complete_file_inventory.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    return rows


def validate_csv_file(path: Path, root: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "relative_path": str(path.relative_to(root)),
        "ok": False,
        "n_rows": "",
        "n_columns": "",
        "columns": "",
        "error": "",
    }
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            n_rows = sum(1 for _ in reader)
        row.update(
            {
                "ok": True,
                "n_rows": n_rows,
                "n_columns": len(header),
                "columns": "|".join(header[:50]),
            }
        )
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def validate_python_file(path: Path, root: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "relative_path": str(path.relative_to(root)),
        "ok": False,
        "error": "",
    }
    try:
        ast.parse(path.read_text(encoding="utf-8"))
        row["ok"] = True
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def grouped_counts(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, int]:
    counts = Counter("::".join(str(row[key]) for key in keys) for row in rows)
    return dict(sorted(counts.items()))


def subject_block_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        if not row["subject"]:
            continue
        key = f"{row['section']}::{row['layer']}::{row['kind']}"
        counts[key][row["subject"]] += 1
    return {key: dict(sorted(value.items())) for key, value in sorted(counts.items())}


def expected_material_names() -> set[str]:
    names = set()
    for idx in range(1, 8):
        names.add(f"nr_{idx}.csv")
        names.add(f"nr_{idx}_control_questions.csv")
        names.add(f"tsr_{idx}.csv")
    return names


def main() -> int:
    args = parse_args()
    root = args.zuco2_root
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = inventory(root, out_dir)

    csv_paths = sorted((root / "task_materials").glob("*.csv")) if (root / "task_materials").exists() else []
    csv_reports = [validate_csv_file(path, root) for path in csv_paths]
    write_rows(out_dir / "zuco2_task_materials_validation.csv", csv_reports)

    script_paths = sorted((root / "scripts").rglob("*.py")) if (root / "scripts").exists() else []
    script_reports = [validate_python_file(path, root) for path in script_paths]
    write_rows(out_dir / "zuco2_scripts_validation.csv", script_reports)

    material_names = {path.name for path in csv_paths}
    missing_materials = sorted(expected_material_names() - material_names)
    extra_materials = sorted(material_names - expected_material_names())

    summary = {
        "zuco2_root": str(root),
        "n_total_files": len(rows),
        "total_size_gb": round(sum(float(row["size_mb"]) for row in rows) / 1024, 3),
        "counts_by_section": grouped_counts(rows, ("section",)),
        "counts_by_section_layer": grouped_counts(rows, ("section", "layer")),
        "counts_by_kind": grouped_counts(rows, ("kind",)),
        "counts_by_section_layer_kind": grouped_counts(rows, ("section", "layer", "kind")),
        "subject_counts_by_section_layer_kind": subject_block_counts(rows),
        "task_materials": {
            "n_csv": len(csv_paths),
            "n_ok": sum(1 for row in csv_reports if row["ok"]),
            "missing_expected_csv": missing_materials,
            "extra_csv": extra_materials,
            "errors": [row for row in csv_reports if not row["ok"]],
        },
        "scripts": {
            "n_python": len(script_paths),
            "n_ok": sum(1 for row in script_reports if row["ok"]),
            "errors": [row for row in script_reports if not row["ok"]],
        },
        "answers": {
            "n_files": sum(1 for row in rows if row["section"] == "answers"),
            "subjects": sorted({row["subject"] for row in rows if row["section"] == "answers" and row["subject"]}),
        },
    }

    summary_path = out_dir / "zuco2_complete_inventory_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Complete inventory summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
