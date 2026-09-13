from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


TARGETS = ("content_function", "coarse_pos")
MODEL_SEEDS = (42, 43, 44, 45, 46)
LOSS_RATES = (0.0, 0.1, 0.25, 0.5)
METRICS = ("accuracy", "balanced_accuracy", "macro_f1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Stage 40 persistent channel-loss results.")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260826)
    return parser.parse_args()


def exact_sign_flip_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    observed = abs(float(values.mean()))
    signs = np.array(list(np.ndindex(*(2,) * len(values))), dtype=np.int8) * 2 - 1
    null_means = np.abs((signs * values).mean(axis=1))
    return float((np.sum(null_means >= observed) + 1) / (len(null_means) + 1))


def bootstrap(values: np.ndarray, n_samples: int, seed: int) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_samples, len(values)), replace=True).mean(axis=1)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
        "n_subjects": int(len(values)),
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(np.asarray(p_values, dtype=np.float64))
    adjusted = np.empty(len(p_values), dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (len(p_values) - rank) * p_values[int(index)])
        running = max(running, candidate)
        adjusted[int(index)] = running
    return adjusted.tolist()


def load_formal_results(results_dir: Path) -> list[dict]:
    paths = sorted(results_dir.glob("*stage40_fixed_session_formal*classification_results.json"))
    if len(paths) != len(TARGETS) * len(MODEL_SEEDS):
        raise SystemExit(f"Expected 10 formal result files, found {len(paths)} in {results_dir}")

    outputs = []
    observed = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        target = str(payload["data"]["target"])
        model_seed = int(payload["settings"]["model_seed"])
        identity = (target, model_seed)
        if identity in observed:
            raise SystemExit(f"Duplicate formal result for {identity}: {path}")
        observed.add(identity)
        if payload["settings"]["inner_validation"] != "complete_subjects":
            raise SystemExit(f"Unexpected inner validation in {path}")
        if payload["settings"]["test_channel_loss"] != (
            "persistent random mask fixed across all test words for each held-out subject and repeat"
        ):
            raise SystemExit(f"Unexpected mask scope in {path}")
        if tuple(float(value) for value in payload["settings"]["loss_rates"]) != LOSS_RATES:
            raise SystemExit(f"Unexpected loss rates in {path}")
        outputs.append({"path": path, "payload": payload, "target": target, "model_seed": model_seed})

    expected = {(target, seed) for target in TARGETS for seed in MODEL_SEEDS}
    if observed != expected:
        raise SystemExit(f"Missing target/seed combinations: {sorted(expected - observed)}")
    return outputs


def audit_masks(outputs: list[dict]) -> dict:
    reference: dict[tuple[str, float, int], tuple[int, ...]] = {}
    checked = 0
    for output in outputs:
        for row in output["payload"]["folds"]:
            key = (str(row["test_subject"]), float(row["loss_rate"]), int(row["repeat"]))
            mask = tuple(int(value) for value in row["removed_channel_indices"])
            if key in reference and reference[key] != mask:
                raise SystemExit(
                    "Fixed masks differ across targets or model seeds for "
                    f"subject={key[0]}, rate={key[1]}, repeat={key[2]}"
                )
            reference[key] = mask
            if row["mask_scope"] != "fixed_across_all_test_words_for_held_out_subject":
                raise SystemExit(f"Unexpected row-level mask scope for {key}")
            checked += 1
    return {
        "status": "passed",
        "unique_subject_rate_repeat_masks": len(reference),
        "rows_checked_across_targets_and_model_seeds": checked,
        "same_masks_used_across_targets_and_model_seeds": True,
    }


def aggregate_subjects(outputs: list[dict]) -> list[dict]:
    per_seed: dict[tuple[str, str, float, int], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    removed_counts: dict[tuple[str, float], set[int]] = defaultdict(set)
    repeat_counts: dict[tuple[str, str, float, int], int] = defaultdict(int)

    for output in outputs:
        target = output["target"]
        model_seed = output["model_seed"]
        for row in output["payload"]["folds"]:
            subject = str(row["test_subject"])
            rate = float(row["loss_rate"])
            key = (target, subject, rate, model_seed)
            for metric in METRICS:
                per_seed[key][metric].append(float(row[metric]))
            repeat_counts[key] += 1
            removed_counts[(target, rate)].add(int(row["n_removed_channels"]))

    for key, count in repeat_counts.items():
        expected = 1 if key[2] == 0.0 else 20
        if count != expected:
            raise SystemExit(f"Expected {expected} masks for {key}, found {count}")
    if any(len(values) != 1 for values in removed_counts.values()):
        raise SystemExit("Removed-channel count is inconsistent within a target/rate condition.")

    per_subject: dict[tuple[str, str, float], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for (target, subject, rate, model_seed), metric_values in per_seed.items():
        for metric in METRICS:
            per_subject[(target, subject, rate)][metric].append(float(np.mean(metric_values[metric])))

    rows = []
    for (target, subject, rate), metric_values in sorted(per_subject.items()):
        if any(len(metric_values[metric]) != len(MODEL_SEEDS) for metric in METRICS):
            raise SystemExit(f"Incomplete model-seed coverage for target={target}, subject={subject}, rate={rate}")
        rows.append(
            {
                "target": target,
                "test_subject": subject,
                "loss_rate": rate,
                "n_removed_channels": next(iter(removed_counts[(target, rate)])),
                "actual_removed_fraction": next(iter(removed_counts[(target, rate)])) / 105.0,
                "n_model_seeds": len(MODEL_SEEDS),
                "masks_per_subject_per_model_seed": 1 if rate == 0.0 else 20,
                **{metric: float(np.mean(metric_values[metric])) for metric in METRICS},
            }
        )
    return rows


def build_summary(subject_rows: list[dict], n_bootstrap: int, seed: int) -> tuple[list[dict], dict]:
    table_rows = []
    detail = {}
    comparisons = []

    for target_index, target in enumerate(TARGETS):
        target_rows = [row for row in subject_rows if row["target"] == target]
        subjects = sorted(set(row["test_subject"] for row in target_rows))
        if len(subjects) != 18:
            raise SystemExit(f"Expected 18 subjects for {target}, found {len(subjects)}")
        baseline = {
            row["test_subject"]: row
            for row in target_rows
            if float(row["loss_rate"]) == 0.0
        }
        detail[target] = {}
        for rate_index, rate in enumerate(LOSS_RATES):
            rows = sorted(
                (row for row in target_rows if float(row["loss_rate"]) == rate),
                key=lambda row: row["test_subject"],
            )
            metric_stats = {
                metric: bootstrap(
                    np.asarray([row[metric] for row in rows], dtype=np.float64),
                    n_bootstrap,
                    seed + target_index * 100 + rate_index * 10,
                )
                for metric in METRICS
            }
            delta_stats = None
            raw_p = None
            if rate != 0.0:
                deltas = np.asarray(
                    [
                        row["balanced_accuracy"] - baseline[row["test_subject"]]["balanced_accuracy"]
                        for row in rows
                    ],
                    dtype=np.float64,
                )
                delta_stats = bootstrap(
                    deltas,
                    n_bootstrap,
                    seed + target_index * 100 + rate_index * 10 + 1,
                )
                raw_p = exact_sign_flip_p(deltas)
                comparisons.append(
                    {
                        "target": target,
                        "loss_rate": rate,
                        "raw_p": raw_p,
                        "delta": delta_stats,
                    }
                )
            row_out = {
                "target": target,
                "loss_rate": rate,
                "n_removed_channels": int(rows[0]["n_removed_channels"]),
                "actual_removed_fraction": float(rows[0]["actual_removed_fraction"]),
                "n_subjects": len(rows),
                "n_model_seeds": len(MODEL_SEEDS),
                "masks_per_subject_per_model_seed": int(rows[0]["masks_per_subject_per_model_seed"]),
                **{
                    f"{metric}_{field}": values[field]
                    for metric, values in metric_stats.items()
                    for field in ("mean", "std", "ci95_low", "ci95_high")
                },
                "balanced_accuracy_delta_mean": None if delta_stats is None else delta_stats["mean"],
                "balanced_accuracy_delta_ci95_low": None if delta_stats is None else delta_stats["ci95_low"],
                "balanced_accuracy_delta_ci95_high": None if delta_stats is None else delta_stats["ci95_high"],
                "delta_sign_flip_p_raw": raw_p,
                "delta_sign_flip_p_holm": None,
            }
            table_rows.append(row_out)
            detail[target][str(rate)] = {
                "metrics": metric_stats,
                "balanced_accuracy_delta_vs_no_loss": delta_stats,
                "delta_sign_flip_p_raw": raw_p,
            }

    adjusted = holm_adjust([item["raw_p"] for item in comparisons])
    adjusted_map = {
        (item["target"], item["loss_rate"]): adjusted_value
        for item, adjusted_value in zip(comparisons, adjusted)
    }
    for row in table_rows:
        key = (row["target"], row["loss_rate"])
        if key in adjusted_map:
            row["delta_sign_flip_p_holm"] = adjusted_map[key]
            detail[row["target"]][str(row["loss_rate"])]["delta_sign_flip_p_holm"] = adjusted_map[key]
    return table_rows, detail


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = load_formal_results(args.results_dir)
    mask_audit = audit_masks(outputs)
    subject_rows = aggregate_subjects(outputs)
    summary_rows, detail = build_summary(subject_rows, args.bootstrap_samples, args.bootstrap_seed)

    summary_csv = args.output_dir / "workflow15_fixed_session_channel_loss_summary.csv"
    subjects_csv = args.output_dir / "workflow15_fixed_session_channel_loss_subjects.csv"
    summary_json = args.output_dir / "stage40_fixed_session_channel_loss_analysis.json"
    write_csv(summary_csv, summary_rows)
    write_csv(subjects_csv, subject_rows)
    summary_json.write_text(
        json.dumps(
            {
                "method": (
                    "For each target, held-out subject and loss rate, mask repeats were averaged within each model "
                    "seed, then five model seeds were averaged within subject. Bootstrap intervals and exact sign-flip "
                    "tests use 18 held-out subjects as independent units. Holm correction covers six planned loss-vs-"
                    "no-loss comparisons."
                ),
                "expected_targets": TARGETS,
                "expected_model_seeds": MODEL_SEEDS,
                "expected_loss_rates": LOSS_RATES,
                "mask_audit": mask_audit,
                "summary": detail,
                "source_files": [str(output["path"]) for output in outputs],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("STAGE 40 ANALYSIS PASSED")
    print(json.dumps(mask_audit, indent=2))
    print(f"Summary CSV: {summary_csv}")
    print(f"Subject CSV: {subjects_csv}")
    print(f"Analysis JSON: {summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
