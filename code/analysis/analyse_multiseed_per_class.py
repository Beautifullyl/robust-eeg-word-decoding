from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = ("accuracy", "balanced_accuracy", "macro_f1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze per-class performance across multiple diagnostic result seeds.")
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", choices=("content_function", "coarse_pos"), required=True)
    parser.add_argument("--model-name", default="eegnet")
    parser.add_argument("--stage-tag", default="stage28")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-pattern",
        default="*classification_results.json",
        help="Glob used inside result-dir before filtering by target.",
    )
    return parser.parse_args()


def bootstrap_ci(values: np.ndarray, n_samples: int, seed: int) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    draws = rng.choice(values, size=(n_samples, len(values)), replace=True)
    means = draws.mean(axis=1)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
        "n_units": int(len(values)),
    }


def parse_seed(path: Path) -> int | None:
    match = re.search(r"_seed_(\d+)_classification_results\.json$", path.name)
    return int(match.group(1)) if match else None


def load_result_files(result_dir: Path, target: str, include_pattern: str) -> list[Path]:
    paths = []
    for path in sorted(result_dir.glob(include_pattern)):
        if not path.name.endswith("_classification_results.json"):
            continue
        if "statistical_summary" in path.name:
            continue
        if f"_{target}_" not in path.name:
            continue
        paths.append(path)
    if not paths:
        raise SystemExit(f"No result files found for target={target} in {result_dir}")
    return paths


def confusion_metrics(confusion: np.ndarray, label_id: int) -> tuple[float, float, float, float]:
    tp = confusion[label_id, label_id]
    fp = confusion[:, label_id].sum() - tp
    fn = confusion[label_id, :].sum() - tp
    support = confusion[label_id, :].sum()
    recall = tp / support if support else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return float(precision), float(recall), float(f1), float(support)


def subject_seed_units(files: list[Path]) -> tuple[list[str], dict[str, list[dict]]]:
    label_names: list[str] | None = None
    units_by_dropout: dict[str, list[dict]] = defaultdict(list)
    for path in files:
        seed = parse_seed(path)
        with path.open("r", encoding="utf-8") as f:
            result = json.load(f)
        current_labels = result["data"]["label_names"]
        if label_names is None:
            label_names = current_labels
        elif label_names != current_labels:
            raise SystemExit(f"Label mismatch in {path}")
        folds = result.get("folds", [])
        if any("confusion_matrix" not in row for row in folds):
            raise SystemExit(f"Missing confusion_matrix in {path}")
        by_dropout_subject: dict[tuple[str, str], list[dict]] = defaultdict(list)
        for row in folds:
            subject = str(row.get("test_subject", row.get("fold", "unknown")))
            by_dropout_subject[(str(row["dropout_rate"]), subject)].append(row)
        for (dropout, subject), rows in by_dropout_subject.items():
            avg_confusion = np.mean([np.asarray(row["confusion_matrix"], dtype=np.float64) for row in rows], axis=0)
            unit = {
                "seed": seed,
                "test_subject": subject,
                "n_repeats": len(rows),
                "confusion_matrix": avg_confusion.tolist(),
            }
            for metric in METRICS:
                unit[metric] = float(np.mean([row[metric] for row in rows]))
            units_by_dropout[dropout].append(unit)
    assert label_names is not None
    return label_names, units_by_dropout


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = load_result_files(args.result_dir, args.target, args.include_pattern)
    label_names, units_by_dropout = subject_seed_units(files)
    n_classes = len(label_names)

    analysis = {
        "stage_tag": args.stage_tag,
        "model_name": args.model_name,
        "target": args.target,
        "source_result_dir": str(args.result_dir),
        "source_files": [str(path) for path in files],
        "label_names": label_names,
        "method": "Each seed-subject unit averages dropout repeats before per-class summaries and bootstrap uncertainty.",
        "by_dropout": {},
    }
    csv_rows = []
    for dropout in sorted(units_by_dropout, key=float):
        units = units_by_dropout[dropout]
        total_confusion = np.sum([np.asarray(unit["confusion_matrix"], dtype=np.float64) for unit in units], axis=0)
        row_sums = total_confusion.sum(axis=1, keepdims=True)
        normalized = np.divide(total_confusion, row_sums, out=np.zeros_like(total_confusion), where=row_sums != 0)
        per_class = {}
        for label_id, label_name in enumerate(label_names):
            precisions = []
            recalls = []
            f1s = []
            supports = []
            for unit in units:
                precision, recall, f1, support = confusion_metrics(np.asarray(unit["confusion_matrix"]), label_id)
                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)
                supports.append(support)
            per_class[label_name] = {
                "support_total": int(np.sum(supports)),
                "precision": bootstrap_ci(np.asarray(precisions), 20000, args.seed),
                "recall": bootstrap_ci(np.asarray(recalls), 20000, args.seed),
                "f1": bootstrap_ci(np.asarray(f1s), 20000, args.seed),
            }
            csv_rows.append(
                {
                    "target": args.target,
                    "model_name": args.model_name,
                    "dropout": dropout,
                    "label": label_name,
                    "support_total": int(np.sum(supports)),
                    "precision_mean": per_class[label_name]["precision"]["mean"],
                    "recall_mean": per_class[label_name]["recall"]["mean"],
                    "f1_mean": per_class[label_name]["f1"]["mean"],
                    "f1_ci95_low": per_class[label_name]["f1"]["ci95_low"],
                    "f1_ci95_high": per_class[label_name]["f1"]["ci95_high"],
                }
            )
        analysis["by_dropout"][dropout] = {
            "n_seed_subject_units": len(units),
            "confusion_matrix": total_confusion.tolist(),
            "row_normalized_confusion_matrix": normalized.tolist(),
            "overall_metrics": {
                metric: bootstrap_ci(np.asarray([unit[metric] for unit in units]), 20000, args.seed)
                for metric in METRICS
            },
            "per_class": per_class,
        }

    stem = f"{args.stage_tag}_{args.model_name}_{args.target}_multiseed_per_class"
    json_path = args.output_dir / f"{stem}.json"
    csv_path = args.output_dir / f"{stem}.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Analysis JSON: {json_path}")
    print(f"Analysis CSV: {csv_path}")
    for row in csv_rows:
        if row["dropout"] == "0.0":
            print(
                f"{row['target']} {row['label']}: "
                f"f1={row['f1_mean']:.3f} "
                f"[{row['f1_ci95_low']:.3f}, {row['f1_ci95_high']:.3f}]"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

