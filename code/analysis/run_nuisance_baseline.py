from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


TARGETS = {
    "content_function": ("function", "content"),
    "coarse_pos": ("function", "noun", "verb", "adjective", "adverb"),
}

FEATURE_SETS = {
    "duration_fixation": (
        "log_total_original_timepoints",
        "log_used_timepoints",
        "log_n_fixations_available",
        "log_n_valid_fixations",
        "was_truncated",
        "was_padded",
    ),
    "lexical_position": (
        "word_length",
        "relative_sentence_position",
    ),
    "combined_non_eeg": (
        "log_total_original_timepoints",
        "log_used_timepoints",
        "log_n_fixations_available",
        "log_n_valid_fixations",
        "was_truncated",
        "was_padded",
        "word_length",
        "relative_sentence_position",
    ),
}

REQUIRED_COLUMNS = {
    "subject",
    "sentence_id",
    "word_id",
    "clean_word",
    "content_function_label",
    "coarse_pos_label",
    "n_fixations_available",
    "n_valid_fixations",
    "total_original_timepoints",
    "used_timepoints",
    "was_truncated",
    "was_padded",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LOSO non-EEG control models for all-fixations word-level classification."
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--target", choices=tuple(TARGETS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--c-grid", nargs="+", type=float, default=[0.01, 0.1, 1.0, 10.0])
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--bootstrap-iterations", type=int, default=20000)
    return parser.parse_args()


def parse_bool(value: str) -> float:
    normalized = value.strip().lower()
    if normalized not in {"0", "1", "false", "true", "no", "yes"}:
        raise ValueError(f"Unrecognised Boolean value: {value!r}")
    return float(normalized in {"1", "true", "yes"})


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Metadata has no header: {path}")
        missing = sorted(REQUIRED_COLUMNS.difference(reader.fieldnames))
        if missing:
            raise ValueError(f"Metadata is missing required columns: {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Metadata contains no rows: {path}")
    return rows


def sentence_max_word_ids(rows: list[dict[str, str]]) -> dict[tuple[str, str], int]:
    maxima: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row["subject"], row["sentence_id"])
        word_id = int(row["word_id"])
        maxima[key] = max(maxima.get(key, word_id), word_id)
    return maxima


def build_feature_table(rows: list[dict[str, str]]) -> tuple[np.ndarray, list[str]]:
    maxima = sentence_max_word_ids(rows)
    feature_names = list(FEATURE_SETS["combined_non_eeg"])
    records = []
    for row in rows:
        key = (row["subject"], row["sentence_id"])
        max_word_id = maxima[key]
        relative_position = int(row["word_id"]) / max(1, max_word_id)
        records.append(
            [
                np.log1p(float(row["total_original_timepoints"])),
                np.log1p(float(row["used_timepoints"])),
                np.log1p(float(row["n_fixations_available"])),
                np.log1p(float(row["n_valid_fixations"])),
                parse_bool(row["was_truncated"]),
                parse_bool(row["was_padded"]),
                float(len(row["clean_word"])),
                float(relative_position),
            ]
        )
    features = np.asarray(records, dtype=np.float64)
    if features.shape != (len(rows), len(feature_names)):
        raise RuntimeError(f"Unexpected feature shape: {features.shape}")
    if not np.all(np.isfinite(features)):
        raise ValueError("Engineered feature table contains non-finite values")
    return features, feature_names


def bootstrap_ci(values: np.ndarray, seed: int, iterations: int) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=np.float64)
    batch_size = 2000
    for start in range(0, iterations, batch_size):
        stop = min(start + batch_size, iterations)
        samples = rng.choice(values, size=(stop - start, len(values)), replace=True)
        means[start:stop] = np.mean(samples, axis=1)
    return {
        "mean": float(np.mean(values)),
        "std_across_subjects": float(np.std(values, ddof=1)),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "n_subjects": int(len(values)),
        "bootstrap_iterations": int(iterations),
    }


def exact_sign_flip_p_value(values: np.ndarray, reference: float) -> float:
    deltas = np.asarray(values, dtype=np.float64) - reference
    observed = abs(float(np.mean(deltas)))
    n = len(deltas)
    if n > 24:
        raise ValueError("Exact sign-flip enumeration is restricted to at most 24 subjects")
    total = 1 << n
    extreme = 0
    bit_positions = np.arange(n, dtype=np.uint64)
    batch_size = 16384
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        codes = np.arange(start, stop, dtype=np.uint64)[:, None]
        signs = (((codes >> bit_positions) & 1).astype(np.float64) * 2.0) - 1.0
        permuted = np.abs(np.mean(signs * deltas, axis=1))
        extreme += int(np.sum(permuted >= observed - 1e-15))
    return float(extreme / total)


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running_max = 0.0
    count = len(ordered)
    for rank, (name, p_value) in enumerate(ordered):
        candidate = min(1.0, (count - rank) * p_value)
        running_max = max(running_max, candidate)
        adjusted[name] = running_max
    return adjusted


def fit_feature_set(
    x: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    n_classes: int,
    c_grid: list[float],
    seed: int,
    n_jobs: int,
) -> list[dict[str, object]]:
    fold_rows: list[dict[str, object]] = []
    for fold_id, test_subject in enumerate(sorted(set(subjects.tolist()))):
        train_idx = np.flatnonzero(subjects != test_subject)
        test_idx = np.flatnonzero(subjects == test_subject)
        train_groups = subjects[train_idx]
        n_splits = min(5, len(set(train_groups.tolist())))
        pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        solver="lbfgs",
                        max_iter=3000,
                        random_state=seed + fold_id,
                    ),
                ),
            ]
        )
        search = GridSearchCV(
            pipeline,
            {"model__C": c_grid},
            scoring="balanced_accuracy",
            cv=GroupKFold(n_splits=n_splits),
            n_jobs=n_jobs,
            refit=True,
            error_score="raise",
        )
        search.fit(x[train_idx], y[train_idx], groups=train_groups)
        prediction = search.predict(x[test_idx])
        fold_rows.append(
            {
                "test_subject": str(test_subject),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "train_label_counts": dict(Counter(y[train_idx].tolist())),
                "test_label_counts": dict(Counter(y[test_idx].tolist())),
                "best_c": float(search.best_params_["model__C"]),
                "inner_grouped_cv_balanced_accuracy": float(search.best_score_),
                "accuracy": float(accuracy_score(y[test_idx], prediction)),
                "balanced_accuracy": float(balanced_accuracy_score(y[test_idx], prediction)),
                "macro_f1": float(f1_score(y[test_idx], prediction, average="macro", zero_division=0)),
                "confusion_matrix": confusion_matrix(
                    y[test_idx], prediction, labels=np.arange(n_classes)
                ).tolist(),
            }
        )
    return fold_rows


def main() -> int:
    args = parse_args()
    all_rows = load_rows(args.metadata)
    label_column = "content_function_label" if args.target == "content_function" else "coarse_pos_label"
    label_names = list(TARGETS[args.target])
    label_to_id = {name: index for index, name in enumerate(label_names)}
    rows = [row for row in all_rows if row[label_column] in label_to_id]
    if not rows:
        raise ValueError(f"No eligible rows for target {args.target}")

    all_features, all_feature_names = build_feature_table(rows)
    feature_index = {name: index for index, name in enumerate(all_feature_names)}
    y = np.asarray([label_to_id[row[label_column]] for row in rows], dtype=np.int64)
    subjects = np.asarray([row["subject"] for row in rows])
    n_classes = len(label_names)
    chance = 1.0 / n_classes

    subject_count = len(set(subjects.tolist()))
    if subject_count < 3:
        raise ValueError(f"LOSO requires at least three subjects, found {subject_count}")

    feature_set_results: dict[str, object] = {}
    raw_p_values: dict[str, float] = {}
    for feature_set_id, names in FEATURE_SETS.items():
        columns = [feature_index[name] for name in names]
        x = all_features[:, columns]
        fold_rows = fit_feature_set(
            x=x,
            y=y,
            subjects=subjects,
            n_classes=n_classes,
            c_grid=args.c_grid,
            seed=args.seed,
            n_jobs=args.n_jobs,
        )
        stats = {}
        for metric_id, metric in enumerate(("accuracy", "balanced_accuracy", "macro_f1")):
            values = np.asarray([float(row[metric]) for row in fold_rows], dtype=np.float64)
            stats[metric] = bootstrap_ci(
                values,
                seed=args.seed + 1000 * len(feature_set_results) + metric_id,
                iterations=args.bootstrap_iterations,
            )
        balanced = np.asarray([float(row["balanced_accuracy"]) for row in fold_rows], dtype=np.float64)
        p_value = exact_sign_flip_p_value(balanced, chance)
        raw_p_values[feature_set_id] = p_value
        stats["balanced_accuracy_vs_chance"] = {
            "reference": chance,
            "mean_difference": float(np.mean(balanced) - chance),
            "two_sided_exact_sign_flip_p": p_value,
            "number_of_sign_patterns": int(1 << len(balanced)),
        }
        feature_set_results[feature_set_id] = {
            "features": list(names),
            "folds": fold_rows,
            "stats": stats,
        }

    adjusted = holm_adjust(raw_p_values)
    for feature_set_id, adjusted_p in adjusted.items():
        result = feature_set_results[feature_set_id]
        assert isinstance(result, dict)
        stats = result["stats"]
        assert isinstance(stats, dict)
        comparison = stats["balanced_accuracy_vs_chance"]
        assert isinstance(comparison, dict)
        comparison["holm_adjusted_p_within_target"] = adjusted_p

    output = {
        "analysis_id": "stage33_non_eeg_controls_allfix_loso_v2",
        "metadata": str(args.metadata),
        "target": args.target,
        "label_names": label_names,
        "label_counts": dict(Counter(row[label_column] for row in rows)),
        "n_rows": int(len(rows)),
        "n_subjects": int(subject_count),
        "subjects": sorted(set(subjects.tolist())),
        "split": {
            "outer": "leave-one-subject-out",
            "inner": "up to five-fold GroupKFold by training subject",
            "selection_metric": "balanced_accuracy",
        },
        "model": {
            "estimator": "class-balanced logistic regression",
            "scaling": "StandardScaler fitted within each training fold",
            "c_grid": args.c_grid,
            "seed": args.seed,
            "determinism_note": "The lbfgs model is deterministic for fixed data; repeated neural-model seeds are not required for this control.",
        },
        "feature_sets": feature_set_results,
        "multiplicity": "Holm adjustment across the three non-EEG feature sets within this target.",
        "interpretation": (
            "No EEG amplitude or waveform values are used. These controls estimate how much target "
            "information is available from segment duration, fixation count, padding/truncation, word "
            "length and sentence position. They do not prove that the EEG model uses those covariates."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    for feature_set_id, result in feature_set_results.items():
        assert isinstance(result, dict)
        stats = result["stats"]
        assert isinstance(stats, dict)
        balanced = stats["balanced_accuracy"]
        comparison = stats["balanced_accuracy_vs_chance"]
        print(
            f"{feature_set_id}: balanced_accuracy={balanced['mean']:.8f} "
            f"CI=[{balanced['ci95_low']:.8f}, {balanced['ci95_high']:.8f}] "
            f"p_exact={comparison['two_sided_exact_sign_flip_p']:.8g} "
            f"p_holm={comparison['holm_adjusted_p_within_target']:.8g}"
        )
    print(f"Results: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
