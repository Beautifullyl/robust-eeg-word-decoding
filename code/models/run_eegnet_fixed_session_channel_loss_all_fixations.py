from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import run_eegnet_diagnostic_all_fixations_classification as base


PROJECT_ROOT = Path(__file__).resolve().parents[2]
N_CHANNELS = 105


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate all-fixations LOSO EEGNet under persistent random channel loss. "
            "Each mask is fixed across every test word of a held-out subject."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data/processed/zuco2_nr_diagnostic_all_fixations_concat_512",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--channel-labels",
        type=Path,
        default=(
            PROJECT_ROOT
            / "experiment_outputs/07_structured_channel_loss"
            / "manifest/processed_channel_labels.txt"
        ),
    )
    parser.add_argument("--target", choices=tuple(base.TARGETS), required=True)
    parser.add_argument("--loss-rates", nargs="+", type=float, default=[0.0, 0.1, 0.25, 0.5])
    parser.add_argument("--mask-repeats", type=int, default=20)
    parser.add_argument("--mask-seed", type=int, default=20260826)
    parser.add_argument("--output-tag", default="stage40_fixed_session")
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
    parser.add_argument("--seed", type=int, default=42, help="Model and split seed.")
    parser.add_argument("--max-subjects", type=int, default=None, help="Smoke-test limit only.")
    return parser.parse_args()


def load_metadata(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_channel_labels(path: Path) -> list[str]:
    if not path.is_file():
        raise SystemExit(f"Channel-label file not found: {path}")
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(labels) != N_CHANNELS:
        raise SystemExit(f"Expected {N_CHANNELS} channel labels, found {len(labels)} in {path}")
    if len(set(labels)) != N_CHANNELS:
        raise SystemExit("Channel labels must be unique.")
    return labels


def validate_rates(rates: list[float]) -> list[float]:
    values = sorted(set(float(rate) for rate in rates))
    if not values or values[0] != 0.0:
        raise SystemExit("--loss-rates must include 0.0 as the paired no-loss reference.")
    if any(rate < 0.0 or rate >= 1.0 for rate in values):
        raise SystemExit("Every loss rate must satisfy 0 <= rate < 1.")
    return values


def channels_for_rate(rate: float) -> int:
    if rate == 0.0:
        return 0
    return max(1, min(N_CHANNELS - 1, int(round(rate * N_CHANNELS))))


def deterministic_mask_seed(mask_seed: int, subject: str, rate: float, repeat: int) -> int:
    token = f"{mask_seed}|{subject}|{rate:.8f}|{repeat}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], byteorder="little", signed=False)


def make_fixed_mask(mask_seed: int, subject: str, rate: float, repeat: int) -> tuple[int, list[int]]:
    if rate == 0.0:
        return mask_seed, []
    derived_seed = deterministic_mask_seed(mask_seed, subject, rate, repeat)
    rng = np.random.default_rng(derived_seed)
    removed = sorted(rng.choice(N_CHANNELS, size=channels_for_rate(rate), replace=False).tolist())
    return derived_seed, removed


class FixedRemovedChannelDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        valid_mask: np.ndarray,
        y: np.ndarray,
        data_indices: np.ndarray,
        indices: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        removed_indices: list[int],
    ):
        self.x = x
        self.valid_mask = valid_mask
        self.y = y
        self.data_indices = data_indices.astype(np.int64)
        self.indices = indices.astype(np.int64)
        self.mean = mean
        self.std = std
        self.removed_indices = np.asarray(sorted(set(removed_indices)), dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        label_index = int(self.indices[item])
        data_index = int(self.data_indices[label_index])
        x_item = np.asarray(self.x[data_index], dtype=np.float32)
        valid = np.asarray(self.valid_mask[data_index], dtype=bool)
        x_item = (x_item - self.mean) / self.std
        x_item[~valid] = 0.0
        if self.removed_indices.size:
            x_item[:, self.removed_indices] = 0.0
        x_item = np.transpose(x_item, (1, 0))[None, :, :]
        return torch.from_numpy(x_item), torch.tensor(int(self.y[label_index]), dtype=torch.long)


def predict_fixed_mask(
    model: nn.Module,
    x: np.ndarray,
    valid_mask: np.ndarray,
    y: np.ndarray,
    data_indices: np.ndarray,
    indices: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    removed_indices: list[int],
    batch_size: int,
    device: torch.device,
    num_workers: int,
) -> np.ndarray:
    dataset = FixedRemovedChannelDataset(
        x,
        valid_mask,
        y,
        data_indices,
        indices,
        mean,
        std,
        removed_indices,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    predictions = []
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            predictions.append(torch.argmax(model(xb.to(device)), dim=1).cpu().numpy())
    return np.concatenate(predictions)


def aggregate_subject_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["test_subject"])].append(row)
    output = []
    for subject, subject_rows in sorted(grouped.items()):
        output.append(
            {
                "test_subject": subject,
                "n_masks_averaged": len(subject_rows),
                **{
                    metric: float(np.mean([row[metric] for row in subject_rows]))
                    for metric in ("accuracy", "balanced_accuracy", "macro_f1")
                },
            }
        )
    return output


def write_subject_csv(path: Path, target: str, model_seed: int, rows_by_rate: dict[str, list[dict]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "target",
            "model_seed",
            "test_subject",
            "loss_rate",
            "n_removed_channels",
            "n_masks_averaged",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rate, rows in sorted(rows_by_rate.items(), key=lambda item: float(item[0])):
            for row in rows:
                writer.writerow(
                    {
                        "target": target,
                        "model_seed": model_seed,
                        "test_subject": row["test_subject"],
                        "loss_rate": float(rate),
                        "n_removed_channels": channels_for_rate(float(rate)),
                        "n_masks_averaged": row["n_masks_averaged"],
                        "accuracy": row["accuracy"],
                        "balanced_accuracy": row["balanced_accuracy"],
                        "macro_f1": row["macro_f1"],
                    }
                )


def main() -> int:
    args = parse_args()
    args.loss_rates = validate_rates(args.loss_rates)
    if args.mask_repeats < 1:
        raise SystemExit("--mask-repeats must be positive.")
    if args.max_subjects is not None and args.max_subjects < 1:
        raise SystemExit("--max-subjects must be positive.")

    base.set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    channel_labels = load_channel_labels(args.channel_labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = min(4, max(0, int(torch.get_num_threads()) - 1))

    metadata = load_metadata(args.data_dir / "metadata.csv")
    keep_idx, y, label_names, kept_metadata = base.build_labels(metadata, args.target)
    x = np.load(args.data_dir / "X_all_fixations_concat_len512.npy", mmap_mode="r")
    valid_mask = np.load(args.data_dir / "mask_all_fixations_concat_len512.npy", mmap_mode="r")
    subjects = np.asarray([row["subject"] for row in kept_metadata])
    n_classes = len(label_names)

    if x.shape[0] != len(metadata) or valid_mask.shape[0] != len(metadata):
        raise SystemExit("X, validity mask and metadata row counts do not match.")
    if x.shape[1:] != (512, N_CHANNELS):
        raise SystemExit(f"Unexpected X shape: {x.shape}")

    print(f"device={device} workers={num_workers}", flush=True)
    print(
        f"target={args.target} rows={len(y)} subjects={len(set(subjects))} "
        f"labels={label_names} counts={dict(Counter(int(value) for value in y))}",
        flush=True,
    )
    print(
        f"loss_rates={args.loss_rates} channel_counts={[channels_for_rate(rate) for rate in args.loss_rates]} "
        f"mask_repeats={args.mask_repeats} mask_seed={args.mask_seed}",
        flush=True,
    )

    folds = base.loso_folds(subjects)
    if args.max_subjects is not None:
        folds = folds[: args.max_subjects]

    fold_rows = []
    training = {}
    mask_audit = {}
    for fold_id, (subject, outer_train_idx, test_idx) in enumerate(folds):
        inner_train_idx, val_idx, val_subjects = base.grouped_subject_train_val_split(
            outer_train_idx,
            subjects,
            y,
            args.val_size,
            args.seed + fold_id,
        )
        print(
            f"fold={subject} train={len(inner_train_idx)} val={len(val_idx)} test={len(test_idx)} "
            f"val_subjects={list(val_subjects)}",
            flush=True,
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
        training[str(subject)] = {
            **train_info,
            "inner_validation_mode": "subject",
            "validation_subjects": list(val_subjects),
        }

        for rate in args.loss_rates:
            repeats = 1 if rate == 0.0 else args.mask_repeats
            masks_seen = set()
            for repeat in range(repeats):
                derived_seed, removed_indices = make_fixed_mask(args.mask_seed, str(subject), rate, repeat)
                mask_key = tuple(removed_indices)
                if mask_key in masks_seen:
                    raise RuntimeError(f"Duplicate mask for subject={subject}, rate={rate}, repeat={repeat}")
                masks_seen.add(mask_key)
                predictions = predict_fixed_mask(
                    model,
                    x,
                    valid_mask,
                    y,
                    keep_idx,
                    test_idx,
                    mean,
                    std,
                    removed_indices,
                    args.batch_size,
                    device,
                    num_workers,
                )
                fold_rows.append(
                    {
                        "fold": str(subject),
                        "test_subject": str(subject),
                        "loss_rate": float(rate),
                        "repeat": int(repeat),
                        "mask_seed": int(args.mask_seed),
                        "derived_mask_seed": int(derived_seed),
                        "removed_channel_indices": removed_indices,
                        "removed_channel_labels": [channel_labels[index] for index in removed_indices],
                        "n_removed_channels": len(removed_indices),
                        "actual_removed_fraction": len(removed_indices) / N_CHANNELS,
                        "mask_scope": "fixed_across_all_test_words_for_held_out_subject",
                        "n_train": int(len(inner_train_idx)),
                        "n_val": int(len(val_idx)),
                        "n_test": int(len(test_idx)),
                        "validation_subjects": list(val_subjects),
                        **base.classification_metrics(y[test_idx], predictions, n_classes),
                    }
                )
            mask_audit[f"{subject}|{rate}"] = {
                "n_unique_masks": len(masks_seen),
                "expected_masks": repeats,
                "n_removed_channels": channels_for_rate(rate),
            }

    summary = {}
    statistical_summary = {}
    subject_rows_by_rate = {}
    baseline_by_subject = {}
    for rate in args.loss_rates:
        key = str(rate)
        rows = [row for row in fold_rows if row["loss_rate"] == rate]
        subject_rows = aggregate_subject_rows(rows)
        subject_rows_by_rate[key] = subject_rows
        summary[key] = {
            metric: base.summarize([row[metric] for row in subject_rows])
            for metric in ("accuracy", "balanced_accuracy", "macro_f1")
        }
        statistical_summary[key] = {
            metric: base.bootstrap_ci(
                np.asarray([row[metric] for row in subject_rows], dtype=np.float64),
                20000,
                args.mask_seed + int(round(rate * 1000)),
            )
            for metric in ("accuracy", "balanced_accuracy", "macro_f1")
        }
        balanced_values = np.asarray([row["balanced_accuracy"] for row in subject_rows], dtype=np.float64)
        statistical_summary[key]["balanced_accuracy_vs_chance"] = {
            "reference": 1.0 / n_classes,
            "mean_difference": float(balanced_values.mean() - 1.0 / n_classes),
            "two_sided_exact_sign_flip_p": base.sign_flip_p_value(balanced_values, 1.0 / n_classes),
        }
        if rate == 0.0:
            baseline_by_subject = {row["test_subject"]: row for row in subject_rows}
        else:
            deltas = np.asarray(
                [row["balanced_accuracy"] - baseline_by_subject[row["test_subject"]]["balanced_accuracy"] for row in subject_rows],
                dtype=np.float64,
            )
            statistical_summary[key]["balanced_accuracy_delta_vs_no_loss"] = {
                **base.bootstrap_ci(deltas, 20000, args.mask_seed + int(round(rate * 1000)) + 1),
                "two_sided_exact_sign_flip_p": base.sign_flip_p_value(deltas, 0.0),
            }

    tag = args.output_tag.strip()
    if tag and not tag.replace("-", "").replace("_", "").replace(".", "").isalnum():
        raise SystemExit("--output-tag may contain only letters, numbers, hyphens, underscores or dots.")
    stem = f"zuco2_nr_allfix_loso_eegnet_{args.target}_{tag}_modelseed_{args.seed}_maskseed_{args.mask_seed}"
    result_path = args.output_dir / f"{stem}_classification_results.json"
    stats_path = args.output_dir / f"{stem}_classification_statistical_summary.json"
    subject_path = args.output_dir / f"{stem}_subject_summary.csv"

    output = {
        "data": {
            "data_dir": str(args.data_dir),
            "target": args.target,
            "n_rows": int(len(y)),
            "timepoints": int(x.shape[1]),
            "channels": int(x.shape[2]),
            "label_names": label_names,
            "label_counts": {label_names[k]: int(v) for k, v in Counter(int(value) for value in y).items()},
            "channel_labels_file": str(args.channel_labels),
            "channel_labels": channel_labels,
        },
        "settings": {
            "outer_split": "leave-one-subject-out",
            "inner_validation": "complete_subjects",
            "input": "ZuCo2 Task 1 all valid fixation EEG segments concatenated per word; fixed 512 x 105 tensor",
            "normalization": "channel-wise z-score fitted on inner-training valid samples only",
            "model": "EEGNet-style compact CNN classifier",
            "loss": "class-weighted cross entropy",
            "optimizer": "AdamW",
            "test_channel_loss": "persistent random mask fixed across all test words for each held-out subject and repeat",
            "loss_rates": args.loss_rates,
            "removed_channel_counts": {str(rate): channels_for_rate(rate) for rate in args.loss_rates},
            "mask_repeats": args.mask_repeats,
            "mask_seed": args.mask_seed,
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
        "statistical_summary": statistical_summary,
        "subject_rows_by_rate": subject_rows_by_rate,
        "mask_audit": mask_audit,
        "folds": fold_rows,
        "training": training,
    }
    result_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    stats_path.write_text(
        json.dumps(
            {
                "source_result": str(result_path),
                "method": (
                    "Mask repeats are averaged within held-out subject. Subject-level bootstrap intervals and exact "
                    "sign-flip tests use the held-out subject as the independent unit."
                ),
                "stats": statistical_summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_subject_csv(subject_path, args.target, args.seed, subject_rows_by_rate)

    for rate, metrics in summary.items():
        print(
            f"loss_rate={rate} balanced_accuracy={metrics['balanced_accuracy']['mean']:.6f} "
            f"macro_f1={metrics['macro_f1']['mean']:.6f}",
            flush=True,
        )
    print(f"Results: {result_path}")
    print(f"Statistical summary: {stats_path}")
    print(f"Subject summary: {subject_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
