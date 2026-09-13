from __future__ import annotations

import argparse
import csv
import json
import random
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

OFFICIAL_ELECTRODE_PAIRS: list[tuple[str, str, str]] = [
    ("E22", "E9", "FP1/FP2"),
    ("E26", "E2", "AF7/AF8"),
    ("E23", "E3", "AF3/AF4"),
    ("E33", "E122", "F7/F8"),
    ("E27", "E123", "F5/F6"),
    ("E19", "E4", "F1/F2"),
    ("E24", "E124", "F3/F4"),
    ("E34", "E116", "FT7/FT8"),
    ("E28", "E117", "FC5/FC6"),
    ("E20", "E118", "official_pair_10"),
    ("E35", "E110", "official_pair_11"),
    ("E29", "E111", "official_pair_12"),
    ("E13", "E112", "official_pair_13"),
    ("E30", "E105", "official_pair_14"),
    ("E36", "E104", "official_pair_15"),
    ("E41", "E103", "official_pair_16"),
    ("E45", "E108", "official_pair_17"),
    ("E46", "E102", "official_pair_18"),
    ("E47", "E98", "official_pair_19"),
    ("E42", "E93", "official_pair_20"),
    ("E37", "E87", "official_pair_21"),
    ("E53", "E86", "official_pair_22"),
    ("E52", "E92", "official_pair_23"),
    ("E51", "E97", "official_pair_24"),
    ("E50", "E101", "official_pair_25"),
    ("E60", "E85", "official_pair_26"),
    ("E59", "E91", "official_pair_27"),
    ("E58", "E96", "official_pair_28"),
    ("E66", "E84", "official_pair_29"),
    ("E65", "E90", "official_pair_30"),
    ("E70", "E83", "official_pair_31"),
    ("E38", "E121", "official_pair_32"),
    ("E44", "E114", "official_pair_33"),
    ("E43", "E120", "official_pair_34"),
    ("E39", "E115", "official_pair_35"),
    ("E40", "E109", "official_pair_36"),
    ("E57", "E100", "official_pair_37"),
    ("E64", "E95", "official_pair_38"),
    ("E69", "E89", "official_pair_39"),
    ("E74", "E82", "official_pair_40"),
    ("E71", "E76", "official_pair_41"),
    ("E67", "E77", "official_pair_42"),
    ("E61", "E78", "official_pair_43"),
    ("E54", "E79", "official_pair_44"),
    ("E31", "E80", "official_pair_45"),
    ("E7", "E106", "official_pair_46"),
    ("E12", "E5", "official_pair_47"),
    ("E18", "E10", "official_pair_48"),
]


FRONTAL_PAIR_COUNT = 9
BLOCK_LABELS = {
    "none": [],
    "official_frontal_bilateral_9pairs": [
        label for pair in OFFICIAL_ELECTRODE_PAIRS[:FRONTAL_PAIR_COUNT] for label in pair[:2]
    ],
    "official_frontal_left_9pairs": [left for left, _, _ in OFFICIAL_ELECTRODE_PAIRS[:FRONTAL_PAIR_COUNT]],
    "official_frontal_right_9pairs": [right for _, right, _ in OFFICIAL_ELECTRODE_PAIRS[:FRONTAL_PAIR_COUNT]],
    "official_left_homologue_set": [left for left, _, _ in OFFICIAL_ELECTRODE_PAIRS],
    "official_right_homologue_set": [right for _, right, _ in OFFICIAL_ELECTRODE_PAIRS],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run LOSO EEGNet all-fixations diagnostic classification with official ZuCo electrode-pair "
            "structured test-time channel removal and matched random controls."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data/processed/zuco2_nr_diagnostic_all_fixations_concat_512",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--channel-labels",
        type=Path,
        default=PROJECT_ROOT
        / "experiment_outputs/09_structured_channel_loss/manifest/processed_channel_labels.txt",
    )
    parser.add_argument("--target", choices=tuple(base.TARGETS), required=True)
    parser.add_argument("--blocks", nargs="+", choices=tuple(BLOCK_LABELS), default=list(BLOCK_LABELS))
    parser.add_argument("--random-repeats", type=int, default=5)
    parser.add_argument("--output-tag", default="stage31")
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
    return parser.parse_args()


def load_metadata(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_channel_labels(path: Path) -> list[str]:
    if not path.exists():
        raise SystemExit(f"Channel-label file not found: {path}")
    labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(labels) != N_CHANNELS:
        raise SystemExit(f"Expected {N_CHANNELS} channel labels, got {len(labels)} from {path}")
    return labels


def block_indices(channel_labels: list[str], block_name: str) -> tuple[list[int], list[str]]:
    labels = BLOCK_LABELS[block_name]
    label_to_index = {label: idx for idx, label in enumerate(channel_labels)}
    present = [label_to_index[label] for label in labels if label in label_to_index]
    missing = [label for label in labels if label not in label_to_index]
    return sorted(set(present)), missing


class RemovedChannelDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        mask: np.ndarray,
        y: np.ndarray,
        data_indices: np.ndarray,
        indices: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        removed_indices: list[int],
    ):
        self.x = x
        self.mask = mask
        self.y = y
        self.data_indices = data_indices.astype(np.int64)
        self.indices = indices.astype(np.int64)
        self.mean = mean
        self.std = std
        self.removed_indices = np.asarray(sorted(set(removed_indices)), dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = int(self.indices[item])
        data_idx = int(self.data_indices[idx])
        x_item = np.asarray(self.x[data_idx], dtype=np.float32)
        mask_item = np.asarray(self.mask[data_idx], dtype=bool)
        x_item = (x_item - self.mean) / self.std
        x_item[~mask_item] = 0.0
        if self.removed_indices.size:
            x_item[:, self.removed_indices] = 0.0
        x_item = np.transpose(x_item, (1, 0))[None, :, :]
        return torch.from_numpy(x_item), torch.tensor(int(self.y[idx]), dtype=torch.long)


def predict_removed_channels(
    model: nn.Module,
    x: np.ndarray,
    mask: np.ndarray,
    y: np.ndarray,
    data_indices: np.ndarray,
    indices: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
    device: torch.device,
    removed_indices: list[int],
    num_workers: int,
) -> np.ndarray:
    dataset = RemovedChannelDataset(x, mask, y, data_indices, indices, mean, std, removed_indices)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.eval()
    predictions = []
    with torch.no_grad():
        for xb, _ in loader:
            logits = model(xb.to(device))
            predictions.append(torch.argmax(logits, dim=1).cpu().numpy())
    return np.concatenate(predictions)


def summarize_subject_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["test_subject"])].append(row)

    subject_rows = []
    for subject, subject_repeats in sorted(grouped.items()):
        merged = {
            "test_subject": subject,
            "n_rows_averaged": len(subject_repeats),
            "accuracy": float(np.mean([row["accuracy"] for row in subject_repeats])),
            "balanced_accuracy": float(np.mean([row["balanced_accuracy"] for row in subject_repeats])),
            "macro_f1": float(np.mean([row["macro_f1"] for row in subject_repeats])),
        }
        subject_rows.append(merged)
    return subject_rows


def main() -> int:
    args = parse_args()
    base.set_seed(args.seed)
    random.seed(args.seed)
    output_dir = args.output_dir if args.output_dir is not None else args.data_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    channel_labels = load_channel_labels(args.channel_labels)
    block_map = {}
    missing_by_block = {}
    for block_name in args.blocks:
        indices, missing = block_indices(channel_labels, block_name)
        block_map[block_name] = indices
        missing_by_block[block_name] = missing
        if block_name != "none" and not indices:
            raise SystemExit(f"Block {block_name} has no matching labels in {args.channel_labels}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = min(4, max(0, int(torch.get_num_threads()) - 1))
    print(f"Using device: {device}", flush=True)
    print(f"DataLoader workers: {num_workers}", flush=True)
    print("experiment=stage09_structured_channel_loss", flush=True)
    print(f"target={args.target}", flush=True)
    print(f"blocks={args.blocks}", flush=True)
    print(f"block_map={block_map}", flush=True)
    print(f"missing_by_block={missing_by_block}", flush=True)

    metadata = load_metadata(args.data_dir / "metadata.csv")
    keep_idx, y, label_names, kept_metadata = base.build_labels(metadata, args.target)
    x = np.load(args.data_dir / "X_all_fixations_concat_len512.npy", mmap_mode="r")
    mask = np.load(args.data_dir / "mask_all_fixations_concat_len512.npy", mmap_mode="r")
    subjects = np.asarray([row["subject"] for row in kept_metadata])
    n_classes = len(label_names)

    print(
        f"rows={len(y)}, X_filtered_shape=({len(y)}, {x.shape[1]}, {x.shape[2]}), "
        f"labels={label_names}, subjects={len(set(subjects))}",
        flush=True,
    )
    print(f"label_counts={dict(Counter(int(value) for value in y))}", flush=True)
    print(f"balanced_chance={1.0 / n_classes:.8f}", flush=True)
    if x.shape[0] != len(metadata):
        raise SystemExit(f"X row count does not match metadata row count: x={x.shape[0]}, metadata={len(metadata)}")
    if mask.shape[0] != len(metadata):
        raise SystemExit(f"Mask row count does not match metadata row count: mask={mask.shape[0]}, metadata={len(metadata)}")
    if x.shape[1:] != (512, N_CHANNELS):
        raise SystemExit(f"Unexpected X shape: {x.shape}")

    unique_random_counts = sorted({len(indices) for name, indices in block_map.items() if name != "none" and indices})

    fold_rows = []
    train_info = {}
    for fold_id, (subject, outer_train_idx, test_idx) in enumerate(base.loso_folds(subjects)):
        inner_train_idx, val_idx = base.stratified_train_val_split(outer_train_idx, y, args.val_size, args.seed + fold_id)
        print(f"Fold {subject}: train={len(inner_train_idx)}, val={len(val_idx)}, test={len(test_idx)}", flush=True)
        mean, std = base.channel_stats(x, mask, keep_idx, inner_train_idx, args.batch_size)
        model, info = base.train_model(
            x,
            mask,
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
        train_info[str(subject)] = info

        conditions: list[dict] = []
        for block_name, indices in block_map.items():
            conditions.append(
                {
                    "condition": block_name,
                    "condition_type": "none" if block_name == "none" else "official_pair_block",
                    "matched_to": None,
                    "random_repeat": None,
                    "removed_indices": indices,
                }
            )
        for n_removed in unique_random_counts:
            for repeat in range(args.random_repeats):
                rng = np.random.default_rng(args.seed + fold_id * 10000 + n_removed * 100 + repeat)
                random_indices = sorted(rng.choice(np.arange(N_CHANNELS), size=n_removed, replace=False).tolist())
                conditions.append(
                    {
                        "condition": f"random_match_n{n_removed:03d}",
                        "condition_type": "matched_random",
                        "matched_to": f"n_removed={n_removed}",
                        "random_repeat": repeat,
                        "removed_indices": random_indices,
                    }
                )

        for condition in conditions:
            pred = predict_removed_channels(
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
                condition["removed_indices"],
                num_workers,
            )
            removed_indices = condition["removed_indices"]
            removed_labels = [channel_labels[idx] for idx in removed_indices]
            fold_rows.append(
                {
                    "fold": subject,
                    "test_subject": subject,
                    "condition": condition["condition"],
                    "condition_type": condition["condition_type"],
                    "matched_to": condition["matched_to"],
                    "random_repeat": condition["random_repeat"],
                    "removed_channel_indices": removed_indices,
                    "removed_channel_labels": removed_labels,
                    "n_removed_channels": len(removed_indices),
                    "n_train": int(len(inner_train_idx)),
                    "n_val": int(len(val_idx)),
                    "n_test": int(len(test_idx)),
                    **base.classification_metrics(y[test_idx], pred, n_classes),
                }
            )

    summary = {}
    statistical_summary = {}
    subject_rows_by_condition = {}
    no_block_subject_balanced = {}
    conditions = sorted(set(row["condition"] for row in fold_rows))
    for condition in conditions:
        rows = [row for row in fold_rows if row["condition"] == condition]
        subject_rows = summarize_subject_rows(rows)
        subject_rows_by_condition[condition] = subject_rows
        summary[condition] = {
            metric: base.summarize([row[metric] for row in subject_rows])
            for metric in ("accuracy", "balanced_accuracy", "macro_f1")
        }
        statistical_summary[condition] = {
            metric: base.bootstrap_ci(np.asarray([row[metric] for row in subject_rows]), 20000, args.seed)
            for metric in ("accuracy", "balanced_accuracy", "macro_f1")
        }
        balanced_values = np.asarray([row["balanced_accuracy"] for row in subject_rows], dtype=np.float64)
        statistical_summary[condition]["balanced_accuracy_vs_chance"] = {
            "reference": 1.0 / n_classes,
            "mean_difference": float(balanced_values.mean() - 1.0 / n_classes),
            "two_sided_sign_flip_p": base.sign_flip_p_value(balanced_values, 1.0 / n_classes),
        }
        if condition == "none":
            no_block_subject_balanced = {row["test_subject"]: row["balanced_accuracy"] for row in subject_rows}

    if no_block_subject_balanced:
        for condition, subject_rows in subject_rows_by_condition.items():
            if condition == "none":
                continue
            paired_deltas = [
                float(row["balanced_accuracy"] - no_block_subject_balanced[row["test_subject"]])
                for row in subject_rows
            ]
            statistical_summary[condition]["balanced_accuracy_delta_vs_none"] = base.bootstrap_ci(
                np.asarray(paired_deltas, dtype=np.float64), 20000, args.seed
            )

    output_tag = args.output_tag.strip()
    if output_tag and not output_tag.replace("-", "_").replace(".", "_").isalnum():
        raise SystemExit("--output-tag may only contain letters, numbers, hyphens, underscores, or dots.")
    tag_part = f"_{output_tag}" if output_tag else ""
    stem = (
        f"zuco2_nr_diagnostic_all_fixations_concat_len512_loso_eegnet_officialpairblock_{args.target}{tag_part}"
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
            "channel_labels_file": str(args.channel_labels),
            "channel_labels": channel_labels,
            "label_names": label_names,
            "label_counts": {label_names[k]: int(v) for k, v in Counter(int(value) for value in y).items()},
        },
        "settings": {
            "split": "leave-one-subject-out",
            "input": "ZuCo2 NR sentenceData word rawEEG, all valid fixations concatenated per word, fixed length 512, 105 channels",
            "normalization": "channel-wise z-score using inner training rows and valid mask only",
            "model": "EEGNet-style compact CNN classifier",
            "loss": "class-weighted cross entropy",
            "optimizer": "AdamW",
            "structured_channel_dropout": (
                "official ZuCo electrode-pair based test-time channel removal, with matched random controls by "
                "number of removed channels"
            ),
            "official_pair_source": "ZuCo 1.0 OSF scripts/Matlab_scripts/lib/getElectrodePairs.m",
            "official_pairs": OFFICIAL_ELECTRODE_PAIRS,
            "requested_blocks": args.blocks,
            "block_label_definitions": BLOCK_LABELS,
            "block_indices": block_map,
            "missing_labels_by_block": missing_by_block,
            "random_repeats_per_removed_channel_count": args.random_repeats,
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
            "seed": args.seed,
        },
        "summary": summary,
        "statistical_summary": statistical_summary,
        "subject_rows_by_condition": subject_rows_by_condition,
        "folds": fold_rows,
        "training": train_info,
    }
    result_path = output_dir / f"{stem}_classification_results.json"
    stats_path = output_dir / f"{stem}_classification_statistical_summary.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "source_result": str(result_path),
                "method": (
                    "Subject-level bootstrap CI. Matched-random conditions average repeated masks within subject "
                    "before subject-level CI and paired delta calculation."
                ),
                "stats": statistical_summary,
            },
            f,
            indent=2,
        )

    print("Summary:")
    for condition, metrics in summary.items():
        print(
            f"condition={condition}: "
            f"accuracy={metrics['accuracy']['mean']:.3f}, "
            f"balanced_accuracy={metrics['balanced_accuracy']['mean']:.3f}, "
            f"macro_f1={metrics['macro_f1']['mean']:.3f}"
        )
    print(f"Results: {result_path}")
    print(f"Statistical summary: {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
