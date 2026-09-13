from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import run_eegnet_diagnostic_all_fixations_classification as base


PROJECT_ROOT = Path(__file__).resolve().parents[2]
N_TIMEPOINTS = 512
N_CHANNELS = 105
REPRESENTATIONS = {
    "first_fixation": (
        "X_first_fixation_len512.npy",
        "mask_first_fixation_len512.npy",
    ),
    "all_fixations": (
        "X_all_fixations_concat_len512.npy",
        "mask_all_fixations_concat_len512.npy",
    ),
}
IDENTITY_COLUMNS = (
    "subject",
    "sentence_id",
    "word_id",
    "clean_word",
    "pos",
    "content_function_label",
    "coarse_pos_label",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a shape- and capacity-matched LOSO EEGNet fixation ablation. "
            "Both representations must have shape 512 x 105."
        )
    )
    parser.add_argument(
        "--first-fixation-dir",
        type=Path,
        default=PROJECT_ROOT / "data/processed/zuco2_nr_diagnostic_first_fixation_512",
    )
    parser.add_argument(
        "--all-fixations-dir",
        type=Path,
        default=PROJECT_ROOT / "data/processed/zuco2_nr_diagnostic_all_fixations_concat_512",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--representation", choices=tuple(REPRESENTATIONS), required=True)
    parser.add_argument("--target", choices=tuple(base.TARGETS), required=True)
    parser.add_argument("--output-tag", default="stage41_matched_fixation_formal")
    parser.add_argument("--val-size", type=float, default=0.15)
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
    parser.add_argument("--max-subjects", type=int, default=None, help="Smoke-test limit only.")
    return parser.parse_args()


def load_metadata(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"Metadata file not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def row_identity(row: dict[str, str]) -> tuple[str, ...]:
    missing = [column for column in IDENTITY_COLUMNS if column not in row]
    if missing:
        raise SystemExit(f"Metadata is missing identity columns: {missing}")
    return tuple(str(row[column]) for column in IDENTITY_COLUMNS)


def audit_metadata_alignment(
    first_metadata: list[dict[str, str]],
    all_metadata: list[dict[str, str]],
) -> dict[str, str | int | bool]:
    if len(first_metadata) != len(all_metadata):
        raise SystemExit(
            "First- and all-fixations metadata have different row counts: "
            f"{len(first_metadata)} versus {len(all_metadata)}"
        )
    digest = hashlib.sha256()
    for index, (first_row, all_row) in enumerate(zip(first_metadata, all_metadata)):
        first_identity = row_identity(first_row)
        all_identity = row_identity(all_row)
        if first_identity != all_identity:
            raise SystemExit(
                f"Metadata alignment failed at row {index}: "
                f"first={first_identity}, all={all_identity}"
            )
        digest.update("\x1f".join(first_identity).encode("utf-8"))
        digest.update(b"\n")
    return {
        "status": "passed",
        "same_row_order": True,
        "n_rows": len(first_metadata),
        "identity_columns": list(IDENTITY_COLUMNS),
        "identity_sha256": digest.hexdigest(),
    }


def write_subject_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "representation",
        "target",
        "model_seed",
        "test_subject",
        "n_train",
        "n_val",
        "n_test",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def main() -> int:
    args = parse_args()
    if args.max_subjects is not None and args.max_subjects < 1:
        raise SystemExit("--max-subjects must be positive.")

    first_metadata = load_metadata(args.first_fixation_dir / "metadata.csv")
    all_metadata = load_metadata(args.all_fixations_dir / "metadata.csv")
    alignment = audit_metadata_alignment(first_metadata, all_metadata)
    metadata = first_metadata if args.representation == "first_fixation" else all_metadata

    data_dir = args.first_fixation_dir if args.representation == "first_fixation" else args.all_fixations_dir
    x_name, mask_name = REPRESENTATIONS[args.representation]
    x_path = data_dir / x_name
    mask_path = data_dir / mask_name
    if not x_path.is_file() or not mask_path.is_file():
        raise SystemExit(f"Missing representation arrays: {x_path}, {mask_path}")

    x = np.load(x_path, mmap_mode="r")
    valid_mask = np.load(mask_path, mmap_mode="r")
    if x.shape != (len(metadata), N_TIMEPOINTS, N_CHANNELS):
        raise SystemExit(f"Unexpected EEG shape for {args.representation}: {x.shape}")
    if valid_mask.shape != (len(metadata), N_TIMEPOINTS):
        raise SystemExit(f"Unexpected validity-mask shape for {args.representation}: {valid_mask.shape}")

    keep_idx, y, label_names, kept_metadata = base.build_labels(metadata, args.target)
    subjects = np.asarray([row["subject"] for row in kept_metadata])
    n_classes = len(label_names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = min(4, max(0, int(torch.get_num_threads()) - 1))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base.set_seed(args.seed)

    print(
        f"device={device} representation={args.representation} target={args.target} "
        f"rows={len(y)} tensor_shape={x.shape} subjects={len(set(subjects))}",
        flush=True,
    )
    print(
        f"metadata_alignment={alignment['status']} identity_sha256={alignment['identity_sha256']}",
        flush=True,
    )

    folds = base.loso_folds(subjects)
    if args.max_subjects is not None:
        folds = folds[: args.max_subjects]

    fold_rows = []
    training = {}
    model_parameter_count = None
    for fold_id, (subject, outer_train_idx, test_idx) in enumerate(folds):
        inner_train_idx, val_idx, val_subjects = base.grouped_subject_train_val_split(
            outer_train_idx,
            subjects,
            y,
            args.val_size,
            args.seed + fold_id,
        )
        mean, std = base.channel_stats(x, valid_mask, keep_idx, inner_train_idx, args.batch_size)
        model, train_info = base.train_model(
            x,
            valid_mask,
            y,
            keep_idx,
            inner_train_idx,
            val_idx,
            mean,
            std,
            n_classes,
            args,
            args.seed + fold_id,
            device,
            num_workers,
        )
        current_parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
        if model_parameter_count is None:
            model_parameter_count = current_parameter_count
        elif model_parameter_count != current_parameter_count:
            raise RuntimeError("Model parameter count changed between LOSO folds.")

        predictions = base.predict(
            model,
            x,
            valid_mask,
            y,
            keep_idx,
            test_idx,
            mean,
            std,
            args.batch_size,
            device,
            0.0,
            args.seed,
            num_workers,
        )
        metrics = base.classification_metrics(y[test_idx], predictions, n_classes)
        row = {
            "representation": args.representation,
            "target": args.target,
            "model_seed": args.seed,
            "fold": str(subject),
            "test_subject": str(subject),
            "validation_subjects": list(val_subjects),
            "n_train": int(len(inner_train_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
            **metrics,
        }
        fold_rows.append(row)
        training[str(subject)] = {
            **train_info,
            "inner_validation_mode": "subject",
            "validation_subjects": list(val_subjects),
        }
        print(
            f"fold={subject} train={len(inner_train_idx)} val={len(val_idx)} test={len(test_idx)} "
            f"balanced_accuracy={metrics['balanced_accuracy']:.8f}",
            flush=True,
        )

    summary = {
        metric: base.summarize([row[metric] for row in fold_rows])
        for metric in ("accuracy", "balanced_accuracy", "macro_f1")
    }
    stats = {
        metric: base.bootstrap_ci(
            np.asarray([row[metric] for row in fold_rows], dtype=np.float64),
            20000,
            args.seed + offset,
        )
        for offset, metric in enumerate(("accuracy", "balanced_accuracy", "macro_f1"))
    }
    balanced_values = np.asarray([row["balanced_accuracy"] for row in fold_rows], dtype=np.float64)
    stats["balanced_accuracy_vs_chance"] = {
        "reference": 1.0 / n_classes,
        "mean_difference": float(balanced_values.mean() - 1.0 / n_classes),
        "two_sided_exact_sign_flip_p": base.sign_flip_p_value(balanced_values, 1.0 / n_classes),
    }

    tag = args.output_tag.strip()
    if tag and not tag.replace("-", "").replace("_", "").replace(".", "").isalnum():
        raise SystemExit("--output-tag may contain only letters, numbers, hyphens, underscores or dots.")
    stem = (
        f"zuco2_nr_{args.representation}_len512_loso_eegnet_{args.target}_"
        f"{tag}_modelseed_{args.seed}"
    )
    results_path = args.output_dir / f"{stem}_classification_results.json"
    stats_path = args.output_dir / f"{stem}_classification_statistical_summary.json"
    subjects_path = args.output_dir / f"{stem}_subject_summary.csv"

    output = {
        "data": {
            "representation": args.representation,
            "data_dir": str(data_dir),
            "x_file": str(x_path),
            "mask_file": str(mask_path),
            "tensor_shape": [int(value) for value in x.shape],
            "target": args.target,
            "n_labelled_rows": int(len(y)),
            "label_names": label_names,
            "label_counts": {
                label_names[label_id]: int(count)
                for label_id, count in Counter(int(value) for value in y).items()
            },
            "metadata_alignment": alignment,
        },
        "settings": {
            "comparison": "first_fixation_512_vs_all_fixations_512",
            "outer_split": "leave-one-subject-out",
            "inner_validation": "complete_subjects",
            "normalization": "channel-wise z-score fitted on inner-training valid samples only",
            "interpolation": "none",
            "tensor_timepoints": N_TIMEPOINTS,
            "channels": N_CHANNELS,
            "model": "EEGNet-style compact CNN classifier",
            "model_parameter_count": model_parameter_count,
            "model_seed": args.seed,
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
            "max_subjects": args.max_subjects,
        },
        "summary": summary,
        "statistical_summary": stats,
        "folds": fold_rows,
        "training": training,
    }
    results_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    stats_path.write_text(
        json.dumps(
            {
                "source_result": str(results_path),
                "independent_unit": "held-out subject",
                "stats": stats,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_subject_csv(subjects_path, fold_rows)

    print(f"model_parameter_count={model_parameter_count}")
    print(f"Results: {results_path}")
    print(f"Statistical summary: {stats_path}")
    print(f"Subject summary: {subjects_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
