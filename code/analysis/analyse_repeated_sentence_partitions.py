from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ANALYSIS_ID = "stage39_original_time_repeated_sentence_partitions_loso_v1"
TARGETS = ("content_function", "coarse_pos")
TEST_SENTENCE_FOLDS = tuple(range(5))
MODEL_SEEDS = (42, 43, 44)
METRICS = ("accuracy", "balanced_accuracy", "macro_f1")
EXPECTED_SUBJECTS = 18
EXPECTED_SENTENCES = 349


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit and aggregate Stage 39 repeated sentence-partition results."
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_822)
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows available for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_ci(
    values: np.ndarray, repeats: int, seed: int
) -> tuple[float, float]:
    if values.ndim != 1 or len(values) != EXPECTED_SUBJECTS:
        raise ValueError(f"Expected {EXPECTED_SUBJECTS} subject values, found {values.shape}")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(repeats, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def exact_sign_flip_p(values: np.ndarray, reference: float) -> float:
    differences = np.asarray(values, dtype=np.float64) - reference
    observed = abs(float(differences.mean()))
    extreme = 0
    total = 1 << len(differences)
    bit_positions = np.arange(len(differences), dtype=np.uint64)
    chunk_size = 16_384
    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        codes = np.arange(start, stop, dtype=np.uint64)[:, None]
        signs = 1.0 - 2.0 * ((codes >> bit_positions) & 1).astype(np.float64)
        permuted = np.abs((signs * differences).mean(axis=1))
        extreme += int(np.count_nonzero(permuted >= observed - 1e-15))
    return float(extreme / total)


def subject_summary(
    values: np.ndarray,
    reference: float,
    repeats: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    low, high = bootstrap_ci(values, repeats, bootstrap_seed)
    return {
        "mean": float(values.mean()),
        "standard_deviation": float(values.std(ddof=1)),
        "ci95_low": low,
        "ci95_high": high,
        "n_subjects": int(len(values)),
        "reference": reference,
        "mean_difference_from_reference": float(values.mean() - reference),
        "two_sided_exact_sign_flip_p": exact_sign_flip_p(values, reference),
    }


def descriptive_subject_summary(
    values: np.ndarray,
    repeats: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    low, high = bootstrap_ci(values, repeats, bootstrap_seed)
    return {
        "mean": float(values.mean()),
        "standard_deviation": float(values.std(ddof=1)),
        "ci95_low": low,
        "ci95_high": high,
        "n_subjects": int(len(values)),
    }


def main() -> int:
    args = parse_args()
    paths = sorted(args.results_dir.glob("stage39_formal_*_results.json"))
    expected_runs = len(TARGETS) * len(TEST_SENTENCE_FOLDS) * len(MODEL_SEEDS)
    if len(paths) != expected_runs:
        raise ValueError(f"Expected {expected_runs} formal result files, found {len(paths)}")

    records: dict[tuple[str, int, int], dict[str, object]] = {}
    sentence_sets: dict[int, set[str]] = {}
    run_rows: list[dict[str, object]] = []
    fold_metric_rows: list[dict[str, object]] = []

    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        identity = payload.get("identity", {})
        if payload.get("analysis_id") != ANALYSIS_ID:
            raise ValueError(f"Unexpected analysis_id in {path.name}")
        target = str(identity.get("target"))
        seed = int(identity.get("seed"))
        sentence_fold = int(identity.get("test_sentence_fold"))
        key = (target, sentence_fold, seed)
        if target not in TARGETS or sentence_fold not in TEST_SENTENCE_FOLDS or seed not in MODEL_SEEDS:
            raise ValueError(f"Unexpected run identity in {path.name}: {key}")
        if key in records:
            raise ValueError(f"Duplicate run identity: {key}")
        records[key] = payload

        test_sentences = {str(value) for value in payload["settings"]["test_sentence_ids"]}
        previous = sentence_sets.setdefault(sentence_fold, test_sentences)
        if previous != test_sentences:
            raise ValueError(f"Inconsistent test sentence set for fold {sentence_fold}")

        folds = payload.get("folds", [])
        subjects = [str(row["test_subject"]) for row in folds]
        if len(folds) != EXPECTED_SUBJECTS or len(set(subjects)) != EXPECTED_SUBJECTS:
            raise ValueError(f"Incomplete subject folds in {path.name}")
        if any(
            int(row.get("pairwise_subject_overlap", -1)) != 0
            or int(row.get("pairwise_sentence_overlap", -1)) != 0
            for row in folds
        ):
            raise ValueError(f"Non-zero overlap audit in {path.name}")

        run_row: dict[str, object] = {
            "target": target,
            "test_sentence_fold": sentence_fold,
            "model_seed": seed,
            "n_test_sentences": len(test_sentences),
            "n_subjects": len(folds),
            "result_file": path.name,
        }
        for metric in METRICS:
            run_row[f"mean_{metric}"] = float(
                np.mean([float(row[metric]) for row in folds])
            )
        run_rows.append(run_row)
        for row in folds:
            fold_metric_rows.append(
                {
                    "target": target,
                    "test_sentence_fold": sentence_fold,
                    "model_seed": seed,
                    "test_subject": str(row["test_subject"]),
                    **{metric: float(row[metric]) for metric in METRICS},
                }
            )

    expected_keys = {
        (target, sentence_fold, seed)
        for target in TARGETS
        for sentence_fold in TEST_SENTENCE_FOLDS
        for seed in MODEL_SEEDS
    }
    if set(records) != expected_keys:
        missing = sorted(expected_keys - set(records))
        unexpected = sorted(set(records) - expected_keys)
        raise ValueError(f"Run grid mismatch; missing={missing}, unexpected={unexpected}")

    for first in TEST_SENTENCE_FOLDS:
        for second in TEST_SENTENCE_FOLDS:
            if first < second and sentence_sets[first] & sentence_sets[second]:
                raise ValueError(f"Test sentence overlap between folds {first} and {second}")
    sentence_union = set().union(*(sentence_sets[fold] for fold in TEST_SENTENCE_FOLDS))
    if len(sentence_union) != EXPECTED_SENTENCES:
        raise ValueError(
            f"Test folds cover {len(sentence_union)} sentences, expected {EXPECTED_SENTENCES}"
        )

    values_by_target_subject_metric: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    values_by_target_fold_metric: dict[tuple[str, int, str], list[float]] = defaultdict(list)
    for row in fold_metric_rows:
        for metric in METRICS:
            values_by_target_subject_metric[
                (str(row["target"]), str(row["test_subject"]), metric)
            ].append(float(row[metric]))
            values_by_target_fold_metric[
                (str(row["target"]), int(row["test_sentence_fold"]), metric)
            ].append(float(row[metric]))

    subject_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    target_statistics: dict[str, object] = {}
    for target_index, target in enumerate(TARGETS):
        subjects = sorted(
            {
                subject
                for observed_target, subject, metric in values_by_target_subject_metric
                if observed_target == target and metric == "balanced_accuracy"
            }
        )
        if len(subjects) != EXPECTED_SUBJECTS:
            raise ValueError(f"Expected {EXPECTED_SUBJECTS} subjects for {target}, found {len(subjects)}")
        for subject in subjects:
            row: dict[str, object] = {
                "target": target,
                "test_subject": subject,
                "n_sentence_fold_seed_runs": len(TEST_SENTENCE_FOLDS) * len(MODEL_SEEDS),
            }
            for metric in METRICS:
                values = values_by_target_subject_metric[(target, subject, metric)]
                if len(values) != len(TEST_SENTENCE_FOLDS) * len(MODEL_SEEDS):
                    raise ValueError(f"Incomplete repeated measures for {target}/{subject}/{metric}")
                row[f"mean_{metric}"] = float(np.mean(values))
                row[f"standard_deviation_{metric}"] = float(np.std(values, ddof=1))
            subject_rows.append(row)

        for sentence_fold in TEST_SENTENCE_FOLDS:
            row = {
                "target": target,
                "test_sentence_fold": sentence_fold,
                "n_test_sentences": len(sentence_sets[sentence_fold]),
                "n_subject_seed_values": EXPECTED_SUBJECTS * len(MODEL_SEEDS),
            }
            for metric in METRICS:
                values = values_by_target_fold_metric[(target, sentence_fold, metric)]
                if len(values) != EXPECTED_SUBJECTS * len(MODEL_SEEDS):
                    raise ValueError(f"Incomplete fold values for {target}/fold{sentence_fold}/{metric}")
                row[f"mean_{metric}"] = float(np.mean(values))
                row[f"standard_deviation_{metric}"] = float(np.std(values, ddof=1))
            fold_rows.append(row)

        chance = 0.5 if target == "content_function" else 0.2
        metric_statistics: dict[str, object] = {}
        target_subject_rows = [row for row in subject_rows if row["target"] == target]
        for metric_index, metric in enumerate(METRICS):
            values = np.asarray(
                [float(row[f"mean_{metric}"]) for row in target_subject_rows],
                dtype=np.float64,
            )
            if metric == "balanced_accuracy":
                metric_statistics[metric] = subject_summary(
                    values,
                    chance,
                    args.bootstrap_repeats,
                    args.bootstrap_seed + target_index * 100 + metric_index,
                )
            else:
                metric_statistics[metric] = descriptive_subject_summary(
                    values,
                    args.bootstrap_repeats,
                    args.bootstrap_seed + target_index * 100 + metric_index,
                )
        target_statistics[target] = metric_statistics

    run_rows.sort(key=lambda row: (str(row["target"]), int(row["test_sentence_fold"]), int(row["model_seed"])))
    subject_rows.sort(key=lambda row: (str(row["target"]), str(row["test_subject"])))
    fold_rows.sort(key=lambda row: (str(row["target"]), int(row["test_sentence_fold"])))

    audit = {
        "expected_formal_runs": expected_runs,
        "observed_formal_runs": len(paths),
        "targets": list(TARGETS),
        "test_sentence_folds": list(TEST_SENTENCE_FOLDS),
        "model_seeds": list(MODEL_SEEDS),
        "subject_count_per_run": EXPECTED_SUBJECTS,
        "test_sentence_counts_by_fold": {
            str(fold): len(sentence_sets[fold]) for fold in TEST_SENTENCE_FOLDS
        },
        "pairwise_test_sentence_overlap": 0,
        "test_sentence_union_count": len(sentence_union),
        "all_sentences_tested_exactly_once_per_target_seed": True,
        "inference_unit": "held-out subject after averaging sentence folds and model seeds",
    }
    summary = {
        "analysis_id": ANALYSIS_ID,
        "audit": audit,
        "statistics": target_statistics,
        "interpretation_guardrail": (
            "Sentence folds and model seeds are repeated measurements, not independent inferential units. "
            "Intervals and sign-flip tests use 18 subject-level averages."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "stage39_repeated_sentence_partitions_summary.json", summary)
    write_csv(args.output_dir / "workflow11_repeated_sentence_partitions_runs.csv", run_rows)
    write_csv(args.output_dir / "workflow11_repeated_sentence_partitions_subjects.csv", subject_rows)
    write_csv(args.output_dir / "workflow11_repeated_sentence_partitions_folds.csv", fold_rows)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
