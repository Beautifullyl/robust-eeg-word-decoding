from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from run_eegnet_diagnostic_classification import (
    N_CHANNELS,
    TARGETS,
    bootstrap_ci,
    build_labels,
    channel_stats,
    classification_metrics,
    load_metadata,
    predict,
    set_seed,
    sign_flip_p_value,
    summarize,
    train_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS = ("accuracy", "balanced_accuracy", "macro_f1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mixed-subject EEGNet diagnostic classification on ZuCo2 raw EEG.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data/processed/zuco2_nr_diagnostic_first_fixation_256",
    )
    parser.add_argument("--target", choices=tuple(TARGETS), required=True)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--dropout-rates", nargs="*", type=float, default=[0.0])
    parser.add_argument("--split-repeats", type=int, default=5)
    parser.add_argument("--dropout-repeats", type=int, default=5)
    parser.add_argument("--f1", type=int, default=8)
    parser.add_argument("--depth-multiplier", type=int, default=2)
    parser.add_argument("--f2", type=int, default=16)
    parser.add_argument("--temporal-kernel", type=int, default=31)
    parser.add_argument("--separable-kernel", type=int, default=15)
    parser.add_argument("--model-dropout", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def split_one_group(indices: np.ndarray, val_size: float, test_size: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = rng.permutation(indices)
    n_test = max(1, int(round(len(indices) * test_size)))
    n_val = max(1, int(round(len(indices) * val_size)))
    while len(indices) - n_test - n_val < 1:
        if n_val >= n_test and n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            break
    if len(indices) - n_test - n_val < 1:
        raise ValueError(f"Cannot split stratum with {len(indices)} examples.")
    test_idx = indices[:n_test]
    val_idx = indices[n_test : n_test + n_val]
    train_idx = indices[n_test + n_val :]
    return train_idx, val_idx, test_idx


def stratified_subject_label_split(
    subjects: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    val_size: float,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    rng = np.random.default_rng(seed)
    train_parts = []
    val_parts = []
    test_parts = []
    skipped_strata = []
    for subject in sorted(set(str(value) for value in subjects)):
        subject_idx = np.where(subjects == subject)[0]
        for label_id in range(n_classes):
            stratum_idx = subject_idx[y[subject_idx] == label_id]
            if len(stratum_idx) < 3:
                skipped_strata.append({"subject": subject, "label_id": int(label_id), "n": int(len(stratum_idx))})
                continue
            train_idx, val_idx, test_idx = split_one_group(stratum_idx, val_size, test_size, rng)
            train_parts.append(train_idx)
            val_parts.append(val_idx)
            test_parts.append(test_idx)
    if not train_parts or not val_parts or not test_parts:
        raise ValueError("No valid strata were available for splitting.")
    train_idx = rng.permutation(np.concatenate(train_parts))
    val_idx = rng.permutation(np.concatenate(val_parts))
    test_idx = rng.permutation(np.concatenate(test_parts))
    overlap = (set(train_idx) & set(val_idx)) | (set(train_idx) & set(test_idx)) | (set(val_idx) & set(test_idx))
    if overlap:
        raise ValueError(f"Split leakage detected: {len(overlap)} overlapping rows.")
    return train_idx, val_idx, test_idx, skipped_strata


def count_by_label(y: np.ndarray, indices: np.ndarray, label_names: list[str]) -> dict[str, int]:
    counts = Counter(int(value) for value in y[indices])
    return {label_names[label_id]: int(counts[label_id]) for label_id in range(len(label_names))}


def count_by_subject(subjects: np.ndarray, indices: np.ndarray) -> dict[str, int]:
    counts = Counter(str(value) for value in subjects[indices])
    return {subject: int(counts[subject]) for subject in sorted(counts)}


def repeat_average_rows(rows: list[dict]) -> list[dict]:
    out = []
    for split_repeat in sorted(set(int(row["split_repeat"]) for row in rows)):
        repeat_rows = [row for row in rows if int(row["split_repeat"]) == split_repeat]
        out.append(
            {
                "split_repeat": int(split_repeat),
                "n_dropout_repeats": len(repeat_rows),
                **{metric: float(np.mean([row[metric] for row in repeat_rows])) for metric in METRICS},
            }
        )
    return out


def main() -> int:
    args = parse_args()
    if args.val_size <= 0 or args.test_size <= 0 or args.val_size + args.test_size >= 1:
        raise SystemExit("--val-size and --test-size must be positive and sum to less than 1.")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = min(4, max(0, int(torch.get_num_threads()) - 1))
    print(f"Using device: {device}", flush=True)
    print(f"DataLoader workers: {num_workers}", flush=True)

    metadata = load_metadata(args.data_dir / "metadata.csv")
    keep_idx, y, label_names, kept_metadata = build_labels(metadata, args.target)
    x = np.load(args.data_dir / "X_first_fixation_len256.npy", mmap_mode="r")
    mask = np.load(args.data_dir / "mask_first_fixation_len256.npy", mmap_mode="r")
    subjects = np.asarray([row["subject"] for row in kept_metadata])
    n_classes = len(label_names)

    print(f"target={args.target}", flush=True)
    print(
        f"rows={len(y)}, X_filtered_shape=({len(y)}, {x.shape[1]}, {x.shape[2]}), "
        f"labels={label_names}, subjects={len(set(subjects))}",
        flush=True,
    )
    print(f"label_counts={dict(Counter(int(value) for value in y))}", flush=True)
    print(f"balanced_chance={1.0 / n_classes:.8f}", flush=True)
    if x.shape[1:] != (256, N_CHANNELS):
        raise SystemExit(f"Unexpected X shape: {x.shape}")

    fold_rows = []
    train_info = {}
    split_metadata = []
    all_skipped_strata = []
    for split_repeat in range(args.split_repeats):
        split_seed = args.seed + split_repeat
        train_idx, val_idx, test_idx, skipped_strata = stratified_subject_label_split(
            subjects,
            y,
            n_classes,
            args.val_size,
            args.test_size,
            split_seed,
        )
        all_skipped_strata.extend([{**row, "split_repeat": int(split_repeat)} for row in skipped_strata])
        split_metadata.append(
            {
                "split_repeat": int(split_repeat),
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "n_test": int(len(test_idx)),
                "train_label_counts": count_by_label(y, train_idx, label_names),
                "val_label_counts": count_by_label(y, val_idx, label_names),
                "test_label_counts": count_by_label(y, test_idx, label_names),
                "train_subject_counts": count_by_subject(subjects, train_idx),
                "val_subject_counts": count_by_subject(subjects, val_idx),
                "test_subject_counts": count_by_subject(subjects, test_idx),
            }
        )
        print(
            f"Split repeat {split_repeat}: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}",
            flush=True,
        )
        print(f"  test_label_counts={count_by_label(y, test_idx, label_names)}", flush=True)

        mean, std = channel_stats(x, mask, keep_idx, train_idx, args.batch_size)
        model, info = train_model(
            x,
            mask,
            y,
            keep_idx,
            train_idx,
            val_idx,
            mean,
            std,
            n_classes,
            args,
            split_seed,
            device,
            num_workers,
        )
        train_info[f"split_repeat_{split_repeat}"] = info
        for rate in args.dropout_rates:
            n_dropout_repeats = 1 if rate == 0 else args.dropout_repeats
            for dropout_repeat in range(n_dropout_repeats):
                pred = predict(
                    model,
                    x,
                    mask,
                    y,
                    keep_idx,
                    test_idx,
                    mean,
                    std,
                    args.batch_size,
                    device,
                    rate,
                    split_seed * 1000 + dropout_repeat,
                    num_workers,
                )
                fold_rows.append(
                    {
                        "fold": f"mixed_split_{split_repeat}",
                        "split_repeat": int(split_repeat),
                        "dropout_repeat": int(dropout_repeat),
                        "dropout_rate": float(rate),
                        "n_train": int(len(train_idx)),
                        "n_val": int(len(val_idx)),
                        "n_test": int(len(test_idx)),
                        **classification_metrics(y[test_idx], pred, n_classes),
                    }
                )

    summary = {}
    statistical_summary = {}
    for rate in args.dropout_rates:
        rows = [row for row in fold_rows if row["dropout_rate"] == rate]
        summary[str(rate)] = {metric: summarize([row[metric] for row in rows]) for metric in METRICS}
        repeat_rows = repeat_average_rows(rows)
        statistical_summary[str(rate)] = {
            metric: bootstrap_ci(np.asarray([row[metric] for row in repeat_rows]), 20000, args.seed)
            for metric in METRICS
        }
        balanced_values = np.asarray([row["balanced_accuracy"] for row in repeat_rows], dtype=np.float64)
        statistical_summary[str(rate)]["balanced_accuracy_vs_chance"] = {
            "reference": 1.0 / n_classes,
            "mean_difference": float(balanced_values.mean() - 1.0 / n_classes),
            "two_sided_sign_flip_p": sign_flip_p_value(balanced_values, 1.0 / n_classes),
        }

    stem = (
        f"zuco2_nr_diagnostic_first_fixation_len256_mixed_subject_eegnet_{args.target}"
        f"_f1_{args.f1}_d_{args.depth_multiplier}_f2_{args.f2}"
        f"_dropout_{str(args.model_dropout).replace('.', 'p')}"
        f"_lr_{str(args.lr).replace('.', 'p')}"
        f"_wd_{str(args.weight_decay).replace('.', 'p')}"
        f"_seed_{args.seed}"
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
            "skipped_strata": all_skipped_strata,
        },
        "settings": {
            "split": "mixed-subject random split stratified by subject and target label",
            "train_fraction_approx": float(1.0 - args.val_size - args.test_size),
            "val_fraction_approx": float(args.val_size),
            "test_fraction_approx": float(args.test_size),
            "split_repeats": int(args.split_repeats),
            "dropout_repeats": int(args.dropout_repeats),
            "input": "ZuCo2 NR sentenceData word rawEEG, first valid fixation only, fixed length 256, 105 channels",
            "normalization": "channel-wise z-score using pooled training rows and valid mask only",
            "model": "EEGNet-style compact CNN classifier",
            "loss": "class-weighted cross entropy",
            "optimizer": "AdamW",
            "f1": args.f1,
            "depth_multiplier": args.depth_multiplier,
            "f2": args.f2,
            "temporal_kernel": args.temporal_kernel,
            "separable_kernel": args.separable_kernel,
            "model_dropout": args.model_dropout,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "dropout_rates": args.dropout_rates,
            "seed": args.seed,
        },
        "split_metadata": split_metadata,
        "summary": summary,
        "statistical_summary": statistical_summary,
        "folds": fold_rows,
        "training": train_info,
    }
    result_path = args.data_dir / f"{stem}_classification_results.json"
    stats_path = args.data_dir / f"{stem}_classification_statistical_summary.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "result_file": str(result_path),
                "target": args.target,
                "summary": summary,
                "statistical_summary": statistical_summary,
                "split_metadata": split_metadata,
                "skipped_strata": all_skipped_strata,
            },
            f,
            indent=2,
        )
    print(f"Results: {result_path}", flush=True)
    print(f"Statistical summary: {stats_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
