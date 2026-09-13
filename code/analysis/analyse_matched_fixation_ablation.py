from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


REPRESENTATIONS = ("first_fixation", "all_fixations")
TARGETS = ("content_function", "coarse_pos")
MODEL_SEEDS = (42, 43, 44, 45, 46)
METRICS = ("accuracy", "balanced_accuracy", "macro_f1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse the Stage 41 matched fixation ablation.")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260826)
    return parser.parse_args()


def bootstrap(values: np.ndarray, n_samples: int, seed: int) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_samples, len(values)), replace=True).mean(axis=1)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "ci95_low": float(np.percentile(draws, 2.5)),
        "ci95_high": float(np.percentile(draws, 97.5)),
        "n_subjects": int(len(values)),
    }


def exact_sign_flip_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    observed = abs(float(values.mean()))
    signs = np.array(list(np.ndindex(*(2,) * len(values))), dtype=np.int8) * 2 - 1
    null_means = np.abs((signs * values).mean(axis=1))
    return float((np.sum(null_means >= observed) + 1) / (len(null_means) + 1))


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(np.asarray(p_values, dtype=np.float64))
    adjusted = np.empty(len(p_values), dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (len(p_values) - rank) * p_values[int(index)])
        running = max(running, candidate)
        adjusted[int(index)] = running
    return adjusted.tolist()


def load_results(results_dir: Path) -> list[dict]:
    paths = sorted(results_dir.glob("*stage41_matched_fixation_formal*classification_results.json"))
    expected_count = len(REPRESENTATIONS) * len(TARGETS) * len(MODEL_SEEDS)
    if len(paths) != expected_count:
        raise SystemExit(f"Expected {expected_count} formal result files, found {len(paths)} in {results_dir}")

    outputs = []
    observed = set()
    alignment_hashes = set()
    parameter_counts: dict[str, set[int]] = defaultdict(set)
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        representation = str(payload["data"]["representation"])
        target = str(payload["data"]["target"])
        seed = int(payload["settings"]["model_seed"])
        identity = (representation, target, seed)
        if identity in observed:
            raise SystemExit(f"Duplicate result for {identity}: {path}")
        observed.add(identity)

        if payload["data"]["tensor_shape"][1:] != [512, 105]:
            raise SystemExit(f"Unmatched tensor shape in {path}: {payload['data']['tensor_shape']}")
        if payload["settings"]["inner_validation"] != "complete_subjects":
            raise SystemExit(f"Unexpected inner validation in {path}")
        if payload["settings"]["interpolation"] != "none":
            raise SystemExit(f"Unexpected interpolation setting in {path}")
        if not payload["data"]["metadata_alignment"]["same_row_order"]:
            raise SystemExit(f"Metadata alignment did not pass in {path}")
        alignment_hashes.add(payload["data"]["metadata_alignment"]["identity_sha256"])
        parameter_counts[target].add(int(payload["settings"]["model_parameter_count"]))
        outputs.append(
            {
                "path": path,
                "payload": payload,
                "representation": representation,
                "target": target,
                "seed": seed,
            }
        )

    expected = {
        (representation, target, seed)
        for representation in REPRESENTATIONS
        for target in TARGETS
        for seed in MODEL_SEEDS
    }
    if observed != expected:
        raise SystemExit(f"Missing result combinations: {sorted(expected - observed)}")
    if len(alignment_hashes) != 1:
        raise SystemExit(f"Metadata identity hashes differ: {sorted(alignment_hashes)}")
    inconsistent_targets = {
        target: sorted(counts)
        for target, counts in parameter_counts.items()
        if len(counts) != 1
    }
    if inconsistent_targets:
        raise SystemExit(
            "Model parameter counts differ between representations within a target: "
            f"{inconsistent_targets}"
        )
    return outputs


def aggregate_subjects(outputs: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    seed_coverage: dict[tuple[str, str, str], set[int]] = defaultdict(set)
    for output in outputs:
        for row in output["payload"]["folds"]:
            key = (output["representation"], output["target"], str(row["test_subject"]))
            seed_coverage[key].add(output["seed"])
            for metric in METRICS:
                grouped[key][metric].append(float(row[metric]))

    rows = []
    for (representation, target, subject), metric_values in sorted(grouped.items()):
        if seed_coverage[(representation, target, subject)] != set(MODEL_SEEDS):
            raise SystemExit(
                f"Incomplete model-seed coverage for representation={representation}, "
                f"target={target}, subject={subject}"
            )
        rows.append(
            {
                "representation": representation,
                "target": target,
                "test_subject": subject,
                "n_model_seeds": len(MODEL_SEEDS),
                **{metric: float(np.mean(metric_values[metric])) for metric in METRICS},
            }
        )
    return rows


def build_analysis(
    subject_rows: list[dict],
    n_bootstrap: int,
    seed: int,
) -> tuple[list[dict], list[dict], dict]:
    condition_rows = []
    comparison_rows = []
    detail = {"conditions": {}, "paired_comparisons": {}}
    raw_ps = []

    for target_index, target in enumerate(TARGETS):
        target_rows = [row for row in subject_rows if row["target"] == target]
        subjects = sorted(set(row["test_subject"] for row in target_rows))
        if len(subjects) != 18:
            raise SystemExit(f"Expected 18 subjects for {target}, found {len(subjects)}")

        by_representation = {}
        for representation_index, representation in enumerate(REPRESENTATIONS):
            rows = sorted(
                (row for row in target_rows if row["representation"] == representation),
                key=lambda row: row["test_subject"],
            )
            if [row["test_subject"] for row in rows] != subjects:
                raise SystemExit(f"Subject mismatch for {target}, {representation}")
            by_representation[representation] = {row["test_subject"]: row for row in rows}
            metric_stats = {
                metric: bootstrap(
                    np.asarray([row[metric] for row in rows], dtype=np.float64),
                    n_bootstrap,
                    seed + target_index * 100 + representation_index * 10,
                )
                for metric in METRICS
            }
            condition_rows.append(
                {
                    "target": target,
                    "representation": representation,
                    "n_subjects": len(subjects),
                    "n_model_seeds": len(MODEL_SEEDS),
                    **{
                        f"{metric}_{field}": values[field]
                        for metric, values in metric_stats.items()
                        for field in ("mean", "std", "ci95_low", "ci95_high")
                    },
                }
            )
            detail["conditions"][f"{target}|{representation}"] = metric_stats

        metric_deltas = {}
        for metric_index, metric in enumerate(METRICS):
            deltas = np.asarray(
                [
                    by_representation["all_fixations"][subject][metric]
                    - by_representation["first_fixation"][subject][metric]
                    for subject in subjects
                ],
                dtype=np.float64,
            )
            metric_deltas[metric] = bootstrap(
                deltas,
                n_bootstrap,
                seed + target_index * 100 + 50 + metric_index,
            )
        raw_p = exact_sign_flip_p(
            np.asarray(
                [
                    by_representation["all_fixations"][subject]["balanced_accuracy"]
                    - by_representation["first_fixation"][subject]["balanced_accuracy"]
                    for subject in subjects
                ],
                dtype=np.float64,
            )
        )
        raw_ps.append(raw_p)
        comparison_rows.append(
            {
                "target": target,
                "comparison": "all_fixations_minus_first_fixation",
                "n_subjects": len(subjects),
                "n_model_seeds": len(MODEL_SEEDS),
                **{
                    f"{metric}_delta_{field}": values[field]
                    for metric, values in metric_deltas.items()
                    for field in ("mean", "std", "ci95_low", "ci95_high")
                },
                "balanced_accuracy_sign_flip_p_raw": raw_p,
                "balanced_accuracy_sign_flip_p_holm": None,
            }
        )
        detail["paired_comparisons"][target] = {
            "direction": "all_fixations_minus_first_fixation",
            "metric_deltas": metric_deltas,
            "balanced_accuracy_sign_flip_p_raw": raw_p,
        }

    adjusted = holm_adjust(raw_ps)
    for row, adjusted_p in zip(comparison_rows, adjusted):
        row["balanced_accuracy_sign_flip_p_holm"] = adjusted_p
        detail["paired_comparisons"][row["target"]]["balanced_accuracy_sign_flip_p_holm"] = adjusted_p
    return condition_rows, comparison_rows, detail


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = load_results(args.results_dir)
    subject_rows = aggregate_subjects(outputs)
    condition_rows, comparison_rows, detail = build_analysis(
        subject_rows,
        args.bootstrap_samples,
        args.bootstrap_seed,
    )

    conditions_csv = args.output_dir / "stage41_matched_fixation_conditions.csv"
    comparisons_csv = args.output_dir / "stage41_matched_fixation_paired_comparisons.csv"
    subjects_csv = args.output_dir / "stage41_matched_fixation_subjects.csv"
    analysis_json = args.output_dir / "stage41_matched_fixation_analysis.json"
    write_csv(conditions_csv, condition_rows)
    write_csv(comparisons_csv, comparison_rows)
    write_csv(subjects_csv, subject_rows)
    analysis_json.write_text(
        json.dumps(
            {
                "method": (
                    "Both representations use matched 512 x 105 tensors, identical EEGNet capacity, identical LOSO "
                    "folds and complete-subject inner validation. Five model seeds are averaged within held-out "
                    "subject before paired inference across 18 subjects. Holm correction covers the two planned "
                    "balanced-accuracy representation comparisons."
                ),
                "representations": REPRESENTATIONS,
                "targets": TARGETS,
                "model_seeds": MODEL_SEEDS,
                "detail": detail,
                "source_files": [str(output["path"]) for output in outputs],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("STAGE 41 ANALYSIS PASSED")
    print("matched_tensor_shape=512x105")
    print("matched_model_parameter_count=true")
    print(f"Conditions CSV: {conditions_csv}")
    print(f"Comparisons CSV: {comparisons_csv}")
    print(f"Subject CSV: {subjects_csv}")
    print(f"Analysis JSON: {analysis_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
