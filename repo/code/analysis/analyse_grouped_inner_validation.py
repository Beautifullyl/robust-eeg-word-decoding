from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


TARGETS = ("content_function", "coarse_pos")
SEEDS = (42, 43, 44, 45, 46)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage37", type=Path, required=True)
    parser.add_argument("--row-validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--bootstrap-iterations", type=int, default=50000)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def seed_from_path(path: Path) -> int:
    match = re.search(r"_seed_(\d+)_", path.name)
    if not match:
        raise ValueError(f"Cannot identify seed in {path}")
    return int(match.group(1))


def result_files(root: Path, target: str, grouped: bool) -> list[Path]:
    required_tag = "stage37_grouped_inner" if grouped else "25bcclean"
    return sorted(
        path
        for path in root.rglob("*classification_results.json")
        if "statistical_summary" not in path.name
        and target in path.name
        and required_tag in path.name
    )


def load_subject_seed_values(
    paths: list[Path], grouped: bool
) -> tuple[dict[tuple[str, int], float], dict]:
    if sorted(seed_from_path(path) for path in paths) != list(SEEDS):
        raise ValueError(f"Expected seeds {SEEDS}, found {[seed_from_path(path) for path in paths]}")

    values: dict[tuple[str, int], float] = {}
    audit = {"files": len(paths), "subjects_per_seed": {}, "data_contract": None}
    for path in paths:
        result = read_json(path)
        seed = seed_from_path(path)
        data = result["data"]
        contract = {
            "n_rows": int(data["n_rows"]),
            "timepoints": int(data["timepoints"]),
            "channels": int(data["channels"]),
        }
        if contract != {"n_rows": 75539, "timepoints": 512, "channels": 105}:
            raise ValueError(f"Unexpected data contract in {path}: {contract}")
        if audit["data_contract"] is None:
            audit["data_contract"] = contract

        rows = [
            row
            for row in result["folds"]
            if float(row["dropout_rate"]) == 0.0 and int(row["repeat"]) == 0
        ]
        subjects = [str(row["test_subject"]) for row in rows]
        if len(rows) != 18 or len(set(subjects)) != 18:
            raise ValueError(f"Expected 18 unique LOSO folds in {path}")

        for row in rows:
            if grouped:
                if row.get("inner_validation_mode") != "subject":
                    raise ValueError(f"Grouped validation is not recorded in {path}")
                if len(row.get("validation_subjects", [])) != 3:
                    raise ValueError(f"Expected three validation subjects in {path}")
            values[(str(row["test_subject"]), seed)] = float(row["balanced_accuracy"])
        audit["subjects_per_seed"][str(seed)] = len(subjects)
    return values, audit


def subject_means(values: dict[tuple[str, int], float]) -> dict[str, float]:
    by_subject: dict[str, list[float]] = defaultdict(list)
    for (subject, _seed), value in values.items():
        by_subject[subject].append(value)
    if any(len(seed_values) != len(SEEDS) for seed_values in by_subject.values()):
        raise ValueError("Every subject must have one value for each seed")
    return {subject: float(np.mean(seed_values)) for subject, seed_values in by_subject.items()}


def bootstrap(values: np.ndarray, seed: int, iterations: int) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "n_subjects": int(len(values)),
    }


def exact_sign_flip(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    observed = abs(float(values.mean()))
    bits = (np.arange(1 << len(values))[:, None] >> np.arange(len(values))) & 1
    permuted = np.abs(((bits * 2 - 1) @ values) / len(values))
    return float(np.mean(permuted >= observed - 1e-15))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_rows = []
    subject_rows = []
    audits = {}

    for target_index, target in enumerate(TARGETS):
        grouped_paths = result_files(args.stage37, target, grouped=True)
        row_paths = result_files(args.row_validation, target, grouped=False)
        grouped_values, grouped_audit = load_subject_seed_values(grouped_paths, grouped=True)
        row_values, row_audit = load_subject_seed_values(row_paths, grouped=False)
        grouped_subjects = subject_means(grouped_values)
        row_subjects = subject_means(row_values)
        subjects = sorted(set(grouped_subjects) & set(row_subjects))
        if len(subjects) != 18:
            raise ValueError(f"Expected 18 matched subjects for {target}")

        grouped_array = np.asarray([grouped_subjects[subject] for subject in subjects])
        row_array = np.asarray([row_subjects[subject] for subject in subjects])
        difference = grouped_array - row_array
        grouped_stats = bootstrap(
            grouped_array, args.bootstrap_seed + 10 * target_index, args.bootstrap_iterations
        )
        row_stats = bootstrap(
            row_array, args.bootstrap_seed + 10 * target_index + 1, args.bootstrap_iterations
        )
        difference_stats = bootstrap(
            difference, args.bootstrap_seed + 10 * target_index + 2, args.bootstrap_iterations
        )
        seed_means = {
            str(seed): float(np.mean([grouped_values[(subject, seed)] for subject in subjects]))
            for seed in SEEDS
        }
        comparison_rows.append(
            {
                "target": target,
                "grouped_mean": grouped_stats["mean"],
                "grouped_ci95_low": grouped_stats["ci95_low"],
                "grouped_ci95_high": grouped_stats["ci95_high"],
                "row_validation_mean": row_stats["mean"],
                "row_validation_ci95_low": row_stats["ci95_low"],
                "row_validation_ci95_high": row_stats["ci95_high"],
                "grouped_minus_row_mean": difference_stats["mean"],
                "difference_ci95_low": difference_stats["ci95_low"],
                "difference_ci95_high": difference_stats["ci95_high"],
                "p_exact_sign_flip": exact_sign_flip(difference),
                "n_subjects": len(subjects),
                "n_seeds": len(SEEDS),
                "grouped_seed_sd": float(np.std(list(seed_means.values()), ddof=1)),
            }
        )
        subject_rows.extend(
            {
                "target": target,
                "subject": subject,
                "grouped_inner_validation": grouped_subjects[subject],
                "row_inner_validation": row_subjects[subject],
                "grouped_minus_row": grouped_subjects[subject] - row_subjects[subject],
            }
            for subject in subjects
        )
        audits[target] = {
            "grouped": grouped_audit,
            "row_validation": row_audit,
            "grouped_seed_means": seed_means,
        }

    write_csv(args.output_dir / "workflow12_grouped_validation_comparison.csv", comparison_rows)
    write_csv(args.output_dir / "workflow12_grouped_validation_subjects.csv", subject_rows)
    payload = {
        "method": (
            "Balanced accuracy was averaged across five seeds within each held-out subject. "
            "Uncertainty used a 50,000-draw subject bootstrap, and grouped-versus-row "
            "differences used an exact two-sided subject-level sign-flip test."
        ),
        "comparison": comparison_rows,
        "audit": audits,
    }
    (args.output_dir / "stage37_grouped_validation_comparison.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload["comparison"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
