from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


TARGETS = ("content_function", "coarse_pos")
STRUCTURED_TO_RANDOM = {
    "official_frontal_left_9pairs": "random_match_n009",
    "official_frontal_right_9pairs": "random_match_n009",
    "official_frontal_bilateral_9pairs": "random_match_n018",
    "official_left_homologue_set": "random_match_n048",
    "official_right_homologue_set": "random_match_n048",
}
N_REMOVED = {
    "none": 0,
    "official_frontal_left_9pairs": 9,
    "official_frontal_right_9pairs": 9,
    "official_frontal_bilateral_9pairs": 18,
    "official_left_homologue_set": 48,
    "official_right_homologue_set": 48,
    "random_match_n009": 9,
    "random_match_n018": 18,
    "random_match_n048": 48,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage25bc", type=Path, required=True)
    parser.add_argument("--stage27", type=Path, required=True)
    parser.add_argument("--stage28", type=Path, required=True)
    parser.add_argument("--stage31", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--bootstrap-iterations", type=int, default=50000)
    return parser.parse_args()


def result_files(root: Path, required: tuple[str, ...]) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*classification_results.json")
        if "statistical_summary" not in path.name and all(token in path.name for token in required)
    )


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def seed_from_result(result: dict, path: Path) -> int:
    if "seed" in result.get("settings", {}):
        return int(result["settings"]["seed"])
    match = re.search(r"_seed_(\d+)_", path.name)
    if not match:
        raise ValueError(f"Cannot identify seed in {path}")
    return int(match.group(1))


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


def exact_sign_flip(values: np.ndarray, reference: float = 0.0) -> float:
    deltas = np.asarray(values, dtype=np.float64) - reference
    n = len(deltas)
    if n > 20:
        raise ValueError("Exact sign-flip implementation is intended for at most 20 units")
    observed = abs(float(deltas.mean()))
    bits = (np.arange(1 << n)[:, None] >> np.arange(n)) & 1
    signs = bits * 2 - 1
    permuted = np.abs(signs @ deltas / n)
    return float(np.mean(permuted >= observed - 1e-15))


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=np.float64)
    running = 0.0
    for rank, original_index in enumerate(order):
        candidate = min(1.0, (len(p_values) - rank) * p_values[original_index])
        running = max(running, candidate)
        adjusted[original_index] = running
    return adjusted.tolist()


def loso_subject_values(
    files: list[Path], dropout_rate: str = "0.0", metric: str = "balanced_accuracy"
) -> dict[str, float]:
    per_subject_seed: dict[tuple[str, int], list[float]] = defaultdict(list)
    for path in files:
        result = read_json(path)
        seed = seed_from_result(result, path)
        for row in result["folds"]:
            if str(row.get("dropout_rate")) != dropout_rate:
                continue
            subject = str(row.get("test_subject", row.get("fold")))
            per_subject_seed[(subject, seed)].append(float(row[metric]))
    per_subject: dict[str, list[float]] = defaultdict(list)
    for (subject, _seed), values in per_subject_seed.items():
        per_subject[subject].append(float(np.mean(values)))
    return {subject: float(np.mean(seed_values)) for subject, seed_values in per_subject.items()}


def descriptive_seed_mean(files: list[Path], dropout_rate: str = "0.0") -> tuple[float, float, int]:
    values = [
        float(read_json(path)["summary"][dropout_rate]["balanced_accuracy"]["mean"])
        for path in files
    ]
    return float(np.mean(values)), float(np.std(values, ddof=1)), len(values)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def subject_stats(values: dict[str, float], args: argparse.Namespace, seed_offset: int = 0) -> dict:
    ordered = np.asarray([values[key] for key in sorted(values)], dtype=np.float64)
    stats = bootstrap(ordered, args.bootstrap_seed + seed_offset, args.bootstrap_iterations)
    return {**stats, "subjects": sorted(values)}


def main_no_dropout(args: argparse.Namespace) -> list[dict]:
    rows = []
    for target in TARGETS:
        for split in ("loso", "subject_dependent", "mixed_subject"):
            eegnet_root = args.stage25bc if split == "loso" else args.stage28 / "results" / "eegnet_extra_splits"
            eegnet_tokens = (target,) if split == "loso" else (split, target)
            linear_root = args.stage28 / "results" / "linear"
            for model, root, tokens in (
                ("EEGNet", eegnet_root, eegnet_tokens),
                ("Linear", linear_root, (split, target)),
            ):
                files = result_files(root, tokens)
                mean, seed_sd, n_seeds = descriptive_seed_mean(files)
                row = {
                    "split": split,
                    "target": target,
                    "model": model,
                    "balanced_accuracy": mean,
                    "seed_sd": seed_sd,
                    "n_seeds": n_seeds,
                    "uncertainty_unit": "subject" if split != "mixed_subject" else "split-repeat",
                }
                if split != "mixed_subject":
                    subject_values = loso_subject_values(files)
                    row.update(subject_stats(subject_values, args))
                rows.append(row)
    return rows


def main_loso_supporting_metrics(args: argparse.Namespace) -> list[dict]:
    rows = []
    for target_index, target in enumerate(TARGETS):
        files = result_files(args.stage25bc, (target,))
        for metric_index, metric in enumerate(("accuracy", "balanced_accuracy", "macro_f1")):
            values = loso_subject_values(files, metric=metric)
            rows.append(
                {
                    "target": target,
                    "model": "EEGNet",
                    "metric": metric,
                    **subject_stats(values, args, seed_offset=10 * target_index + metric_index),
                }
            )
    return rows


def paired_comparisons(args: argparse.Namespace) -> list[dict]:
    rows = []
    for target in TARGETS:
        eegnet = loso_subject_values(result_files(args.stage25bc, (target,)))
        comparisons = {
            "EEGNet_minus_linear": loso_subject_values(
                result_files(args.stage28 / "results" / "linear", ("loso", target))
            ),
            "EEGNet_minus_TCN": loso_subject_values(result_files(args.stage27 / "results", (target,))),
        }
        for name, comparator in comparisons.items():
            subjects = sorted(set(eegnet) & set(comparator))
            differences = np.asarray([eegnet[subject] - comparator[subject] for subject in subjects])
            rows.append(
                {
                    "target": target,
                    "comparison": name,
                    **bootstrap(differences, args.bootstrap_seed, args.bootstrap_iterations),
                    "p_exact_sign_flip": exact_sign_flip(differences),
                }
            )
    return rows


def per_class(args: argparse.Namespace) -> list[dict]:
    rows = []
    for target in TARGETS:
        files = result_files(args.stage25bc, (target,))
        subject_seed_confusions: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)
        labels = None
        for path in files:
            result = read_json(path)
            labels = result["data"]["label_names"]
            seed = seed_from_result(result, path)
            for fold in result["folds"]:
                if float(fold["dropout_rate"]) == 0.0:
                    subject_seed_confusions[(fold["test_subject"], seed)].append(
                        np.asarray(fold["confusion_matrix"], dtype=np.float64)
                    )
        subject_confusions: dict[str, list[np.ndarray]] = defaultdict(list)
        for (subject, _seed), confusions in subject_seed_confusions.items():
            subject_confusions[subject].append(np.mean(confusions, axis=0))
        assert labels is not None
        for label_id, label in enumerate(labels):
            precision, recall, f1 = [], [], []
            for subject in sorted(subject_confusions):
                confusion = np.mean(subject_confusions[subject], axis=0)
                tp = confusion[label_id, label_id]
                fp = confusion[:, label_id].sum() - tp
                fn = confusion[label_id, :].sum() - tp
                p = tp / (tp + fp) if tp + fp else 0.0
                r = tp / (tp + fn) if tp + fn else 0.0
                score = 2 * p * r / (p + r) if p + r else 0.0
                precision.append(p)
                recall.append(r)
                f1.append(score)
            f1_stats = bootstrap(np.asarray(f1), args.bootstrap_seed, args.bootstrap_iterations)
            rows.append(
                {
                    "target": target,
                    "label": label,
                    "precision_mean": float(np.mean(precision)),
                    "recall_mean": float(np.mean(recall)),
                    "f1_mean": f1_stats["mean"],
                    "f1_ci95_low": f1_stats["ci95_low"],
                    "f1_ci95_high": f1_stats["ci95_high"],
                    "n_subjects": f1_stats["n_subjects"],
                }
            )
    return rows


def structured_dropout(args: argparse.Namespace) -> list[dict]:
    rows = []
    for target in TARGETS:
        store: dict[tuple[str, str], list[float]] = defaultdict(list)
        for path in result_files(args.stage31 / "results" / target, (target,)):
            result = read_json(path)
            for condition, subject_rows in result["subject_rows_by_condition"].items():
                for row in subject_rows:
                    store[(condition, row["test_subject"])].append(float(row["balanced_accuracy"]))
        subjects = sorted({subject for _, subject in store})
        values = {
            condition: np.asarray([np.mean(store[(condition, subject)]) for subject in subjects])
            for condition in N_REMOVED
        }
        for condition in N_REMOVED:
            stats = bootstrap(values[condition], args.bootstrap_seed, args.bootstrap_iterations)
            delta_none = values[condition] - values["none"]
            delta_none_stats = bootstrap(delta_none, args.bootstrap_seed + 1, args.bootstrap_iterations)
            matched = STRUCTURED_TO_RANDOM.get(condition)
            delta_matched = values[condition] - values[matched] if matched else None
            delta_matched_stats = (
                bootstrap(delta_matched, args.bootstrap_seed + 2, args.bootstrap_iterations)
                if delta_matched is not None
                else None
            )
            rows.append(
                {
                    "target": target,
                    "condition": condition,
                    "n_removed_channels": N_REMOVED[condition],
                    **stats,
                    "delta_vs_none": float(delta_none.mean()),
                    "delta_vs_none_ci95_low": delta_none_stats["ci95_low"],
                    "delta_vs_none_ci95_high": delta_none_stats["ci95_high"],
                    "p_vs_none_unadjusted": None if condition == "none" else exact_sign_flip(delta_none),
                    "p_vs_none_holm": None,
                    "matched_random": matched,
                    "delta_vs_matched_random": None if delta_matched is None else float(delta_matched.mean()),
                    "delta_vs_matched_ci95_low": None if delta_matched_stats is None else delta_matched_stats["ci95_low"],
                    "delta_vs_matched_ci95_high": None if delta_matched_stats is None else delta_matched_stats["ci95_high"],
                    "p_vs_matched_unadjusted": None if delta_matched is None else exact_sign_flip(delta_matched),
                    "p_vs_matched_holm": None,
                }
            )
    for source_key, output_key in (
        ("p_vs_none_unadjusted", "p_vs_none_holm"),
        ("p_vs_matched_unadjusted", "p_vs_matched_holm"),
    ):
        indices = [index for index, row in enumerate(rows) if row[source_key] is not None and row["condition"] in STRUCTURED_TO_RANDOM]
        corrected = holm_adjust([rows[index][source_key] for index in indices])
        for index, value in zip(indices, corrected):
            rows[index][output_key] = value
    return rows


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "main_no_dropout": main_no_dropout(args),
        "main_loso_supporting_metrics": main_loso_supporting_metrics(args),
        "paired_comparisons": paired_comparisons(args),
        "per_class": per_class(args),
        "structured_dropout": structured_dropout(args),
    }
    for name, rows in outputs.items():
        write_csv(args.output_dir / f"{name}.csv", rows)
    (args.output_dir / "final_subject_level_analysis.json").write_text(
        json.dumps(
            {
                "method": (
                    "For LOSO inference, repeated masks were averaged within seed and subject, then seeds "
                    "were averaged within subject. Bootstrap intervals and exact sign-flip tests therefore use "
                    "18 held-out subjects as the independent units. Holm correction is applied within each "
                    "predefined family of dropout contrasts."
                ),
                "sources": {
                    "stage25bc": str(args.stage25bc),
                    "stage27": str(args.stage27),
                    "stage28": str(args.stage28),
                    "stage31": str(args.stage31),
                },
                **outputs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    for name, rows in outputs.items():
        print(f"{name}: {len(rows)} rows")
    print(f"Output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
