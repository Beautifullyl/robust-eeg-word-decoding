from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

from run_eegnet_diagnostic_classification import (
    TARGETS,
    bootstrap_ci,
    build_labels,
    load_metadata,
    loso_folds,
    sign_flip_p_value,
    stratified_train_val_split,
    summarize,
)
from run_eegnet_diagnostic_mixed_subject import stratified_subject_label_split
from run_eegnet_diagnostic_subject_dependent import stratified_subject_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
N_CHANNELS = 105
METRICS = ("accuracy", "balanced_accuracy", "macro_f1")
FEATURE_NAMES = ("mean", "std", "mean_abs", "max", "min")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run linear diagnostic classification on ZuCo2 word-level raw EEG.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data/processed/zuco2_nr_diagnostic_first_fixation_256",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--input-kind",
        choices=("first_fixation", "all_fixations"),
        default="first_fixation",
        help="Input representation stored in data-dir.",
    )
    parser.add_argument("--target", choices=tuple(TARGETS), required=True)
    parser.add_argument("--split", choices=("loso", "subject_dependent", "mixed_subject"), required=True)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--dropout-rates", nargs="*", type=float, default=[0.0])
    parser.add_argument("--split-repeats", type=int, default=5)
    parser.add_argument("--dropout-repeats", type=int, default=5)
    parser.add_argument("--c-grid", nargs="*", type=float, default=[0.01, 0.1, 1.0, 10.0])
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--experiment-id",
        default="",
        help="Optional provenance tag written to output filenames and settings, e.g. exp20.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def extract_summary_features(x: np.ndarray, mask: np.ndarray, keep_idx: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    n_rows = len(keep_idx)
    n_stats = len(FEATURE_NAMES)
    features = np.empty((n_rows, n_stats * x.shape[2]), dtype=np.float32)
    for start in range(0, n_rows, batch_size):
        out_slice = slice(start, min(start + batch_size, n_rows))
        idx = keep_idx[out_slice]
        xb = np.asarray(x[idx], dtype=np.float32)
        mb = np.asarray(mask[idx], dtype=bool)
        valid = mb[:, :, None]
        counts = np.maximum(valid.sum(axis=1), 1).astype(np.float32)
        xb_zero = np.where(valid, xb, 0.0)
        mean = xb_zero.sum(axis=1) / counts
        centered = np.where(valid, xb - mean[:, None, :], 0.0)
        std = np.sqrt(np.maximum((centered * centered).sum(axis=1) / counts, 0.0))
        mean_abs = np.abs(xb_zero).sum(axis=1) / counts
        xb_min = np.where(valid, xb, np.inf).min(axis=1)
        xb_max = np.where(valid, xb, -np.inf).max(axis=1)
        xb_min[~np.isfinite(xb_min)] = 0.0
        xb_max[~np.isfinite(xb_max)] = 0.0
        features[out_slice] = np.concatenate([mean, std, mean_abs, xb_max, xb_min], axis=1)
    return features


def apply_channel_dropout(features: np.ndarray, data_indices: np.ndarray, rate: float, seed: int) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32).copy()
    if rate <= 0:
        return x
    n_drop = max(1, int(round(N_CHANNELS * rate)))
    n_stats = len(FEATURE_NAMES)
    for row_id, data_idx in enumerate(data_indices):
        rng = np.random.default_rng(seed + int(data_idx))
        channels = rng.choice(N_CHANNELS, size=n_drop, replace=False)
        for stat_id in range(n_stats):
            x[row_id, stat_id * N_CHANNELS + channels] = 0.0
    return x


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(n_classes),
        zero_division=0,
    )
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=np.arange(n_classes), average="macro", zero_division=0)),
        "per_class_recall": [float(value) for value in recall],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=np.arange(n_classes)).astype(int).tolist(),
    }


def fit_linear_model(
    x_features: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    c_grid: list[float],
    max_iter: int,
    seed: int,
) -> tuple[StandardScaler, LogisticRegression, dict]:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_features[train_idx])
    x_val = scaler.transform(x_features[val_idx])
    best_model = None
    best_score = -np.inf
    tried = []
    for c_value in c_grid:
        model = LogisticRegression(
            C=float(c_value),
            class_weight="balanced",
            max_iter=max_iter,
            solver="lbfgs",
            random_state=seed,
        )
        model.fit(x_train, y[train_idx])
        pred = model.predict(x_val)
        score = balanced_accuracy_score(y[val_idx], pred)
        tried.append({"C": float(c_value), "val_balanced_accuracy": float(score)})
        if score > best_score:
            best_score = float(score)
            best_model = model
    assert best_model is not None
    return scaler, best_model, {"best_C": float(best_model.C), "best_val_balanced_accuracy": best_score, "tried": tried}


def predict_with_dropout(
    scaler: StandardScaler,
    model: LogisticRegression,
    x_features: np.ndarray,
    indices: np.ndarray,
    rate: float,
    seed: int,
) -> np.ndarray:
    x_eval = scaler.transform(x_features[indices])
    x_eval = apply_channel_dropout(x_eval, indices, rate, seed)
    return model.predict(x_eval)


def summarize_loso(rows: list[dict], dropout_rates: list[float], n_classes: int, seed: int) -> tuple[dict, dict]:
    summary = {}
    statistical_summary = {}
    for rate in dropout_rates:
        rate_rows = [row for row in rows if row["dropout_rate"] == rate]
        summary[str(rate)] = {metric: summarize([row[metric] for row in rate_rows]) for metric in METRICS}
        subject_rows = []
        for subject in sorted(set(row["test_subject"] for row in rate_rows)):
            subject_repeats = [row for row in rate_rows if row["test_subject"] == subject]
            subject_rows.append(
                {
                    "test_subject": subject,
                    "n_repeats": len(subject_repeats),
                    **{metric: float(np.mean([row[metric] for row in subject_repeats])) for metric in METRICS},
                }
            )
        statistical_summary[str(rate)] = {
            metric: bootstrap_ci(np.asarray([row[metric] for row in subject_rows]), 20000, seed)
            for metric in METRICS
        }
        balanced_values = np.asarray([row["balanced_accuracy"] for row in subject_rows], dtype=np.float64)
        statistical_summary[str(rate)]["balanced_accuracy_vs_chance"] = {
            "reference": 1.0 / n_classes,
            "mean_difference": float(balanced_values.mean() - 1.0 / n_classes),
            "two_sided_sign_flip_p": sign_flip_p_value(balanced_values, 1.0 / n_classes),
        }
    return summary, statistical_summary


def summarize_subject_dependent(rows: list[dict], dropout_rates: list[float], n_classes: int, seed: int) -> tuple[dict, dict]:
    summary = {}
    statistical_summary = {}
    for rate in dropout_rates:
        rate_rows = [row for row in rows if row["dropout_rate"] == rate]
        summary[str(rate)] = {metric: summarize([row[metric] for row in rate_rows]) for metric in METRICS}
        subject_rows = []
        for subject in sorted(set(row["test_subject"] for row in rate_rows)):
            subject_rate_rows = [row for row in rate_rows if row["test_subject"] == subject]
            subject_rows.append(
                {
                    "test_subject": subject,
                    "n_repeats": len(subject_rate_rows),
                    **{metric: float(np.mean([row[metric] for row in subject_rate_rows])) for metric in METRICS},
                }
            )
        statistical_summary[str(rate)] = {
            metric: bootstrap_ci(np.asarray([row[metric] for row in subject_rows]), 20000, seed)
            for metric in METRICS
        }
        balanced_values = np.asarray([row["balanced_accuracy"] for row in subject_rows], dtype=np.float64)
        statistical_summary[str(rate)]["balanced_accuracy_vs_chance"] = {
            "reference": 1.0 / n_classes,
            "mean_difference": float(balanced_values.mean() - 1.0 / n_classes),
            "two_sided_sign_flip_p": sign_flip_p_value(balanced_values, 1.0 / n_classes),
        }
    return summary, statistical_summary


def summarize_mixed(rows: list[dict], dropout_rates: list[float], n_classes: int, seed: int) -> tuple[dict, dict]:
    summary = {}
    statistical_summary = {}
    for rate in dropout_rates:
        rate_rows = [row for row in rows if row["dropout_rate"] == rate]
        summary[str(rate)] = {metric: summarize([row[metric] for row in rate_rows]) for metric in METRICS}
        repeat_rows = []
        for split_repeat in sorted(set(int(row["split_repeat"]) for row in rate_rows)):
            split_rows = [row for row in rate_rows if int(row["split_repeat"]) == split_repeat]
            repeat_rows.append(
                {
                    "split_repeat": int(split_repeat),
                    "n_dropout_repeats": len(split_rows),
                    **{metric: float(np.mean([row[metric] for row in split_rows])) for metric in METRICS},
                }
            )
        statistical_summary[str(rate)] = {
            metric: bootstrap_ci(np.asarray([row[metric] for row in repeat_rows]), 20000, seed)
            for metric in METRICS
        }
        balanced_values = np.asarray([row["balanced_accuracy"] for row in repeat_rows], dtype=np.float64)
        statistical_summary[str(rate)]["balanced_accuracy_vs_chance"] = {
            "reference": 1.0 / n_classes,
            "mean_difference": float(balanced_values.mean() - 1.0 / n_classes),
            "two_sided_sign_flip_p": sign_flip_p_value(balanced_values, 1.0 / n_classes),
        }
    return summary, statistical_summary


def main() -> int:
    args = parse_args()
    if args.experiment_id and not re.fullmatch(r"[A-Za-z0-9_-]+", args.experiment_id):
        raise ValueError("--experiment-id may contain only letters, numbers, underscores, and hyphens.")
    set_seed(args.seed)
    output_dir = args.output_dir if args.output_dir is not None else args.data_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(args.data_dir / "metadata.csv")
    keep_idx, y, label_names, kept_metadata = build_labels(metadata, args.target)
    if args.input_kind == "first_fixation":
        x = np.load(args.data_dir / "X_first_fixation_len256.npy", mmap_mode="r")
        mask = np.load(args.data_dir / "mask_first_fixation_len256.npy", mmap_mode="r")
        representation_tag = "first_fixation_len256"
        representation_text = "ZuCo2 NR sentenceData word rawEEG, first valid fixation only, fixed length 256, 105 channels"
        expected_shape = (256, N_CHANNELS)
    else:
        x = np.load(args.data_dir / "X_all_fixations_concat_len512.npy", mmap_mode="r")
        mask = np.load(args.data_dir / "mask_all_fixations_concat_len512.npy", mmap_mode="r")
        representation_tag = "all_fixations_concat_len512"
        representation_text = "ZuCo2 NR sentenceData word rawEEG, all valid fixations concatenated per word, fixed length 512, 105 channels"
        expected_shape = (512, N_CHANNELS)
    subjects = np.asarray([row["subject"] for row in kept_metadata])
    n_classes = len(label_names)

    print(f"target={args.target}", flush=True)
    print(f"split={args.split}", flush=True)
    print(
        f"rows={len(y)}, X_filtered_shape=({len(y)}, {x.shape[1]}, {x.shape[2]}), "
        f"labels={label_names}, subjects={len(set(subjects))}",
        flush=True,
    )
    print(f"label_counts={dict(Counter(int(value) for value in y))}", flush=True)
    print(f"balanced_chance={1.0 / n_classes:.8f}", flush=True)
    print(f"feature_stats={FEATURE_NAMES}", flush=True)
    if x.shape[0] != len(metadata):
        raise SystemExit(f"X row count does not match metadata row count: x={x.shape[0]}, metadata={len(metadata)}")
    if mask.shape[0] != len(metadata):
        raise SystemExit(f"Mask row count does not match metadata row count: mask={mask.shape[0]}, metadata={len(metadata)}")
    if x.shape[1:] != expected_shape:
        raise SystemExit(f"Unexpected X shape for {args.input_kind}: {x.shape}")
    x_features = extract_summary_features(x, mask, keep_idx)
    print(f"linear_feature_shape={x_features.shape}", flush=True)

    fold_rows = []
    train_info = {}
    split_metadata = []
    skipped_subjects = []
    skipped_strata = []

    if args.split == "loso":
        for fold_id, (subject, outer_train_idx, test_idx) in enumerate(loso_folds(subjects)):
            train_idx, val_idx = stratified_train_val_split(outer_train_idx, y, args.val_size, args.seed + fold_id)
            print(f"Fold {subject}: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}", flush=True)
            scaler, model, info = fit_linear_model(
                x_features,
                y,
                train_idx,
                val_idx,
                args.c_grid,
                args.max_iter,
                args.seed + fold_id,
            )
            train_info[str(subject)] = info
            for rate in args.dropout_rates:
                n_repeats = 1 if rate == 0 else args.dropout_repeats
                for repeat in range(n_repeats):
                    pred = predict_with_dropout(scaler, model, x_features, test_idx, rate, args.seed + fold_id * 1000 + repeat)
                    fold_rows.append(
                        {
                            "fold": subject,
                            "test_subject": subject,
                            "dropout_rate": float(rate),
                            "repeat": int(repeat),
                            "n_train": int(len(train_idx)),
                            "n_val": int(len(val_idx)),
                            "n_test": int(len(test_idx)),
                            **classification_metrics(y[test_idx], pred, n_classes),
                        }
                    )
        summary, statistical_summary = summarize_loso(fold_rows, args.dropout_rates, n_classes, args.seed)

    elif args.split == "subject_dependent":
        if args.val_size <= 0 or args.test_size <= 0 or args.val_size + args.test_size >= 1:
            raise SystemExit("--val-size and --test-size must be positive and sum to less than 1.")
        for subject_id, subject in enumerate(sorted(set(str(value) for value in subjects))):
            subject_idx = np.where(subjects == subject)[0]
            counts = Counter(int(value) for value in y[subject_idx])
            missing = [label_id for label_id in range(n_classes) if counts[label_id] < 3]
            if missing:
                skipped_subjects.append(
                    {
                        "subject": subject,
                        "reason": "fewer than 3 examples for at least one class",
                        "class_counts": {label_names[k]: int(counts[k]) for k in range(n_classes)},
                    }
                )
                print(f"Skipping subject {subject}: insufficient class counts {dict(counts)}", flush=True)
                continue
            for split_repeat in range(args.split_repeats):
                split_seed = args.seed + subject_id * 1000 + split_repeat
                train_idx, val_idx, test_idx = stratified_subject_split(
                    subject_idx,
                    y,
                    n_classes,
                    args.val_size,
                    args.test_size,
                    split_seed,
                )
                print(
                    f"Subject {subject} repeat {split_repeat}: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}",
                    flush=True,
                )
                scaler, model, info = fit_linear_model(
                    x_features,
                    y,
                    train_idx,
                    val_idx,
                    args.c_grid,
                    args.max_iter,
                    split_seed,
                )
                train_info[f"{subject}_repeat_{split_repeat}"] = info
                for rate in args.dropout_rates:
                    n_repeats = 1 if rate == 0 else args.dropout_repeats
                    for dropout_repeat in range(n_repeats):
                        pred = predict_with_dropout(
                            scaler,
                            model,
                            x_features,
                            test_idx,
                            rate,
                            split_seed * 1000 + dropout_repeat,
                        )
                        fold_rows.append(
                            {
                                "fold": subject,
                                "test_subject": subject,
                                "split_repeat": int(split_repeat),
                                "dropout_rate": float(rate),
                                "dropout_repeat": int(dropout_repeat),
                                "n_train": int(len(train_idx)),
                                "n_val": int(len(val_idx)),
                                "n_test": int(len(test_idx)),
                                **classification_metrics(y[test_idx], pred, n_classes),
                            }
                        )
        summary, statistical_summary = summarize_subject_dependent(fold_rows, args.dropout_rates, n_classes, args.seed)

    else:
        if args.val_size <= 0 or args.test_size <= 0 or args.val_size + args.test_size >= 1:
            raise SystemExit("--val-size and --test-size must be positive and sum to less than 1.")
        for split_repeat in range(args.split_repeats):
            split_seed = args.seed + split_repeat
            train_idx, val_idx, test_idx, current_skipped = stratified_subject_label_split(
                subjects,
                y,
                n_classes,
                args.val_size,
                args.test_size,
                split_seed,
            )
            skipped_strata.extend([{**row, "split_repeat": int(split_repeat)} for row in current_skipped])
            split_metadata.append(
                {
                    "split_repeat": int(split_repeat),
                    "seed": int(split_seed),
                    "n_train": int(len(train_idx)),
                    "n_val": int(len(val_idx)),
                    "n_test": int(len(test_idx)),
                }
            )
            print(f"Split repeat {split_repeat}: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}", flush=True)
            scaler, model, info = fit_linear_model(
                x_features,
                y,
                train_idx,
                val_idx,
                args.c_grid,
                args.max_iter,
                split_seed,
            )
            train_info[f"split_repeat_{split_repeat}"] = info
            for rate in args.dropout_rates:
                n_repeats = 1 if rate == 0 else args.dropout_repeats
                for dropout_repeat in range(n_repeats):
                    pred = predict_with_dropout(
                        scaler,
                        model,
                        x_features,
                        test_idx,
                        rate,
                        split_seed * 1000 + dropout_repeat,
                    )
                    fold_rows.append(
                        {
                            "split_repeat": int(split_repeat),
                            "dropout_rate": float(rate),
                            "dropout_repeat": int(dropout_repeat),
                            "n_train": int(len(train_idx)),
                            "n_val": int(len(val_idx)),
                            "n_test": int(len(test_idx)),
                            **classification_metrics(y[test_idx], pred, n_classes),
                        }
                    )
        summary, statistical_summary = summarize_mixed(fold_rows, args.dropout_rates, n_classes, args.seed)

    split_tag = args.split
    experiment_part = f"_{args.experiment_id}" if args.experiment_id else ""
    stem = (
        f"zuco2_nr_diagnostic_{representation_tag}_{split_tag}_linear_summary_stats_{args.target}"
        f"{experiment_part}_seed_{args.seed}"
    )
    output = {
        "data": {
            "data_dir": str(args.data_dir),
            "target": args.target,
            "n_rows": int(len(y)),
            "timepoints": int(x.shape[1]),
            "channels": int(x.shape[2]),
            "label_names": label_names,
            "label_counts": {label_names[k]: int(v) for k, v in Counter(int(value) for value in y).items()},
        },
        "settings": {
            "split": args.split,
            "experiment_id": args.experiment_id,
            "input_kind": args.input_kind,
            "input": representation_text,
            "feature_representation": "per-channel summary statistics over valid timepoints",
            "feature_names": list(FEATURE_NAMES),
            "normalization": "StandardScaler fitted on training rows only",
            "model": "class-balanced logistic regression",
            "c_grid": args.c_grid,
            "max_iter": args.max_iter,
            "dropout_rates": args.dropout_rates,
            "split_repeats": args.split_repeats,
            "dropout_repeats": args.dropout_repeats,
            "seed": args.seed,
        },
        "summary": summary,
        "statistical_summary": statistical_summary,
        "folds": fold_rows,
        "training": train_info,
        "split_metadata": split_metadata,
        "skipped_subjects": skipped_subjects,
        "skipped_strata": skipped_strata,
    }
    result_path = output_dir / f"{stem}_classification_results.json"
    stats_path = output_dir / f"{stem}_classification_statistical_summary.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "source_result": str(result_path),
                "method": "Bootstrap/sign-flip summary following the split-specific aggregation used for EEGNet diagnostics.",
                "stats": statistical_summary,
            },
            f,
            indent=2,
        )
    print("Summary:")
    for rate, metrics in summary.items():
        print(
            f"dropout={rate}: "
            f"accuracy={metrics['accuracy']['mean']:.3f}, "
            f"balanced_accuracy={metrics['balanced_accuracy']['mean']:.3f}, "
            f"macro_f1={metrics['macro_f1']['mean']:.3f}",
            flush=True,
        )
    print(f"Results: {result_path}")
    print(f"Statistical summary: {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
