from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
N_CHANNELS = 105
TARGETS = {
    "content_function": ("function", "content"),
    "coarse_pos": ("function", "noun", "verb", "adjective", "adverb"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EEGNet-style diagnostic classification on ZuCo2 raw EEG.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data/processed/zuco2_nr_diagnostic_all_fixations_concat_512",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output folder for result JSON files. Defaults to data-dir for backward compatibility.",
    )
    parser.add_argument("--target", choices=tuple(TARGETS), required=True)
    parser.add_argument(
        "--output-tag",
        default="",
        help="Optional short tag inserted into output filenames so reruns do not overwrite earlier results.",
    )
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument(
        "--inner-val-mode",
        choices=("row", "subject"),
        default="row",
        help=(
            "Use the historical stratified row split or a subject-disjoint inner validation split. "
            "The outer LOSO test split is unchanged."
        ),
    )
    parser.add_argument("--dropout-rates", nargs="*", type=float, default=[0.0])
    parser.add_argument("--repeats", type=int, default=5)
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loso_folds(subjects: np.ndarray) -> list[tuple[str, np.ndarray, np.ndarray]]:
    return [
        (subject, np.where(subjects != subject)[0], np.where(subjects == subject)[0])
        for subject in sorted(set(str(value) for value in subjects))
    ]


def stratified_train_val_split(
    train_idx: np.ndarray,
    y: np.ndarray,
    val_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts = []
    val_parts = []
    for label_id in sorted(set(int(value) for value in y[train_idx])):
        label_idx = train_idx[y[train_idx] == label_id]
        label_idx = rng.permutation(label_idx)
        n_val = max(1, int(round(len(label_idx) * val_size)))
        n_val = min(n_val, len(label_idx) - 1)
        val_parts.append(label_idx[:n_val])
        train_parts.append(label_idx[n_val:])
    return rng.permutation(np.concatenate(train_parts)), rng.permutation(np.concatenate(val_parts))


def grouped_subject_train_val_split(
    train_idx: np.ndarray,
    subjects: np.ndarray,
    y: np.ndarray,
    val_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    train_subjects = sorted(set(str(value) for value in subjects[train_idx]))
    n_val_subjects = max(1, int(round(len(train_subjects) * val_size)))
    n_val_subjects = min(n_val_subjects, len(train_subjects) - 1)
    all_labels = sorted(set(int(value) for value in y[train_idx]))
    overall_distribution = np.asarray(
        [np.mean(y[train_idx] == label_id) for label_id in all_labels],
        dtype=np.float64,
    )

    candidates = list(combinations(train_subjects, n_val_subjects))
    rng = np.random.default_rng(seed)
    rng.shuffle(candidates)
    best = None
    best_score = float("inf")
    for val_subjects in candidates:
        is_val = np.isin(subjects[train_idx], np.asarray(val_subjects))
        candidate_val = train_idx[is_val]
        candidate_train = train_idx[~is_val]
        if not len(candidate_val) or not len(candidate_train):
            continue
        if set(int(value) for value in y[candidate_val]) != set(all_labels):
            continue
        if set(int(value) for value in y[candidate_train]) != set(all_labels):
            continue
        val_distribution = np.asarray(
            [np.mean(y[candidate_val] == label_id) for label_id in all_labels],
            dtype=np.float64,
        )
        score = abs(len(candidate_val) / len(train_idx) - val_size) + float(
            np.abs(val_distribution - overall_distribution).sum()
        )
        if score < best_score:
            best_score = score
            best = (candidate_train, candidate_val, tuple(val_subjects))

    if best is None:
        raise RuntimeError("Could not construct a subject-grouped validation split with all target classes.")
    inner_train_idx, val_idx, val_subjects = best
    return rng.permutation(inner_train_idx), rng.permutation(val_idx), val_subjects


def channel_stats(
    x: np.ndarray,
    mask: np.ndarray,
    data_indices: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    total = np.zeros((x.shape[2],), dtype=np.float64)
    total_sq = np.zeros((x.shape[2],), dtype=np.float64)
    count = np.zeros((x.shape[2],), dtype=np.float64)
    for start in range(0, len(indices), batch_size):
        idx = data_indices[indices[start : start + batch_size]]
        xb = np.asarray(x[idx], dtype=np.float32)
        mb = np.asarray(mask[idx], dtype=bool)
        valid = mb[:, :, None]
        total += (xb * valid).sum(axis=(0, 1))
        total_sq += ((xb * xb) * valid).sum(axis=(0, 1))
        count += valid.sum(axis=(0, 1))
    mean = total / np.maximum(count, 1.0)
    var = total_sq / np.maximum(count, 1.0) - mean * mean
    std = np.sqrt(np.maximum(var, 1e-8))
    return mean.astype(np.float32), std.astype(np.float32)


class EEGClassificationDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        mask: np.ndarray,
        y: np.ndarray,
        data_indices: np.ndarray,
        indices: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        channel_dropout: float = 0.0,
        seed: int = 42,
    ):
        self.x = x
        self.mask = mask
        self.y = y
        self.data_indices = data_indices.astype(np.int64)
        self.indices = indices.astype(np.int64)
        self.mean = mean
        self.std = std
        self.channel_dropout = channel_dropout
        self.seed = seed

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = int(self.indices[item])
        data_idx = int(self.data_indices[idx])
        x_item = np.asarray(self.x[data_idx], dtype=np.float32)
        mask_item = np.asarray(self.mask[data_idx], dtype=bool)
        x_item = (x_item - self.mean) / self.std
        x_item[~mask_item] = 0.0
        if self.channel_dropout > 0:
            rng = np.random.default_rng(self.seed + data_idx)
            n_drop = max(1, int(round(x_item.shape[1] * self.channel_dropout)))
            dropped = rng.choice(x_item.shape[1], size=n_drop, replace=False)
            x_item[:, dropped] = 0.0
        x_item = np.transpose(x_item, (1, 0))[None, :, :]
        return torch.from_numpy(x_item), torch.tensor(int(self.y[idx]), dtype=torch.long)


class EEGNetClassifier(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_timepoints: int,
        n_classes: int,
        f1: int,
        depth_multiplier: int,
        f2: int,
        temporal_kernel: int,
        separable_kernel: int,
        dropout: float,
    ):
        super().__init__()
        if temporal_kernel % 2 == 0 or separable_kernel % 2 == 0:
            raise ValueError("Use odd kernel sizes to preserve exact temporal length with symmetric padding.")
        spatial_filters = f1 * depth_multiplier
        self.features = nn.Sequential(
            nn.Conv2d(1, f1, kernel_size=(1, temporal_kernel), padding=(0, temporal_kernel // 2), bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, spatial_filters, kernel_size=(n_channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(spatial_filters),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
            nn.Conv2d(
                spatial_filters,
                spatial_filters,
                kernel_size=(1, separable_kernel),
                padding=(0, separable_kernel // 2),
                groups=spatial_filters,
                bias=False,
            ),
            nn.Conv2d(spatial_filters, f2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_timepoints)
            flat_dim = int(np.prod(self.features(dummy).shape[1:]))
        self.classifier = nn.Linear(flat_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(torch.flatten(x, start_dim=1))


def class_weights(y_train: np.ndarray, n_classes: int) -> np.ndarray:
    counts = Counter(int(value) for value in y_train)
    n_samples = len(y_train)
    return np.array([n_samples / (n_classes * counts[label_id]) for label_id in range(n_classes)], dtype=np.float32)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict:
    accuracy = float(np.mean(y_true == y_pred))
    recalls = []
    precisions = []
    f1s = []
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    for true_id, pred_id in zip(y_true, y_pred):
        confusion[int(true_id), int(pred_id)] += 1
    for label_id in range(n_classes):
        tp = int(confusion[label_id, label_id])
        fp = int(confusion[:, label_id].sum() - tp)
        fn = int(confusion[label_id, :].sum() - tp)
        support = int(confusion[label_id, :].sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(2 * precision * recall / (precision + recall) if (precision + recall) else 0.0)
    return {
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "per_class_recall": [float(value) for value in recalls],
        "confusion_matrix": confusion.tolist(),
    }


def summarize(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def bootstrap_ci(values: np.ndarray, n_samples: int, seed: int) -> dict[str, float | int]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    draws = rng.choice(values, size=(n_samples, len(values)), replace=True)
    means = draws.mean(axis=1)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
        "n_subjects": int(len(values)),
    }


def sign_flip_p_value(values: np.ndarray, reference: float) -> float:
    diffs = np.asarray(values, dtype=np.float64) - reference
    observed = abs(float(diffs.mean()))
    n = len(diffs)
    signs = np.array(list(np.ndindex(*(2,) * n)), dtype=np.int8) * 2 - 1
    null_means = np.abs((signs * diffs).mean(axis=1))
    return float((np.sum(null_means >= observed) + 1) / (len(null_means) + 1))


def evaluate_loss(
    model: nn.Module,
    dataset: Dataset,
    criterion: nn.Module,
    batch_size: int,
    device: torch.device,
    num_workers: int,
) -> float:
    model.eval()
    losses = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    with torch.no_grad():
        for xb, yb in loader:
            losses.append(float(criterion(model(xb.to(device)), yb.to(device)).cpu()))
    return float(np.mean(losses))


def train_model(
    x: np.ndarray,
    mask: np.ndarray,
    y: np.ndarray,
    data_indices: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    n_classes: int,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
    num_workers: int,
) -> tuple[nn.Module, dict]:
    set_seed(seed)
    model = EEGNetClassifier(
        n_channels=x.shape[2],
        n_timepoints=x.shape[1],
        n_classes=n_classes,
        f1=args.f1,
        depth_multiplier=args.depth_multiplier,
        f2=args.f2,
        temporal_kernel=args.temporal_kernel,
        separable_kernel=args.separable_kernel,
        dropout=args.model_dropout,
    ).to(device)
    weights = torch.from_numpy(class_weights(y[train_idx], n_classes)).float().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_dataset = EEGClassificationDataset(x, mask, y, data_indices, train_idx, mean, std)
    val_dataset = EEGClassificationDataset(x, mask, y, data_indices, val_idx, mean, std)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)

    best_state = None
    best_val = float("inf")
    best_epoch = 0
    patience_left = args.patience
    history = []
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        val_loss = evaluate_loss(model, val_dataset, criterion, args.batch_size, device, num_workers)
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_loss": val_loss})
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "best_val_loss": best_val, "epochs_run": len(history), "history": history}


def predict(
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
    channel_dropout: float,
    seed: int,
    num_workers: int,
) -> np.ndarray:
    dataset = EEGClassificationDataset(
        x,
        mask,
        y,
        data_indices,
        indices,
        mean,
        std,
        channel_dropout=channel_dropout,
        seed=seed,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.eval()
    predictions = []
    with torch.no_grad():
        for xb, _ in loader:
            logits = model(xb.to(device))
            predictions.append(torch.argmax(logits, dim=1).cpu().numpy())
    return np.concatenate(predictions)


def build_labels(metadata: list[dict[str, str]], target: str) -> tuple[np.ndarray, np.ndarray, list[str], list[dict[str, str]]]:
    label_names = list(TARGETS[target])
    label_to_id = {label: idx for idx, label in enumerate(label_names)}
    label_column = "content_function_label" if target == "content_function" else "coarse_pos_label"
    keep_indices = []
    y = []
    kept_metadata = []
    for idx, row in enumerate(metadata):
        label = row[label_column]
        if label not in label_to_id:
            continue
        keep_indices.append(idx)
        y.append(label_to_id[label])
        kept_metadata.append(row)
    return (
        np.asarray(keep_indices, dtype=np.int64),
        np.asarray(y, dtype=np.int64),
        label_names,
        kept_metadata,
    )


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    output_dir = args.output_dir if args.output_dir is not None else args.data_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = min(4, max(0, int(torch.get_num_threads()) - 1))
    print(f"Using device: {device}", flush=True)
    print(f"DataLoader workers: {num_workers}", flush=True)

    metadata = load_metadata(args.data_dir / "metadata.csv")
    keep_idx, y, label_names, kept_metadata = build_labels(metadata, args.target)
    x_full = np.load(args.data_dir / "X_all_fixations_concat_len512.npy", mmap_mode="r")
    mask_full = np.load(args.data_dir / "mask_all_fixations_concat_len512.npy", mmap_mode="r")
    x = x_full
    mask = mask_full
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
    if x.shape[0] != len(metadata):
        raise SystemExit(f"X row count does not match metadata row count: x={x.shape[0]}, metadata={len(metadata)}")
    if mask.shape[0] != len(metadata):
        raise SystemExit(f"Mask row count does not match metadata row count: mask={mask.shape[0]}, metadata={len(metadata)}")
    if x.shape[1:] != (512, N_CHANNELS):
        raise SystemExit(f"Unexpected X shape: {x.shape}")

    fold_rows = []
    train_info = {}
    for fold_id, (subject, outer_train_idx, test_idx) in enumerate(loso_folds(subjects)):
        if args.inner_val_mode == "subject":
            inner_train_idx, val_idx, val_subjects = grouped_subject_train_val_split(
                outer_train_idx,
                subjects,
                y,
                args.val_size,
                args.seed + fold_id,
            )
        else:
            inner_train_idx, val_idx = stratified_train_val_split(
                outer_train_idx,
                y,
                args.val_size,
                args.seed + fold_id,
            )
            val_subjects = tuple(sorted(set(str(value) for value in subjects[val_idx])))
        print(
            f"Fold {subject}: train={len(inner_train_idx)}, val={len(val_idx)}, test={len(test_idx)}, "
            f"inner_val_mode={args.inner_val_mode}, val_subjects={list(val_subjects)}",
            flush=True,
        )
        mean, std = channel_stats(x, mask, keep_idx, inner_train_idx, args.batch_size)
        model, info = train_model(
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
        train_info[str(subject)] = {
            **info,
            "inner_validation_mode": args.inner_val_mode,
            "validation_subjects": list(val_subjects),
        }
        for rate in args.dropout_rates:
            repeats = 1 if rate == 0 else args.repeats
            for repeat in range(repeats):
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
                    args.seed + fold_id * 1000 + repeat,
                    num_workers,
                )
                fold_rows.append(
                    {
                        "fold": subject,
                        "test_subject": subject,
                        "dropout_rate": float(rate),
                        "repeat": int(repeat),
                        "n_train": int(len(inner_train_idx)),
                        "n_val": int(len(val_idx)),
                        "n_test": int(len(test_idx)),
                        "inner_validation_mode": args.inner_val_mode,
                        "validation_subjects": list(val_subjects),
                        **classification_metrics(y[test_idx], pred, n_classes),
                    }
                )

    summary = {}
    statistical_summary = {}
    for rate in args.dropout_rates:
        rows = [row for row in fold_rows if row["dropout_rate"] == rate]
        summary[str(rate)] = {
            metric: summarize([row[metric] for row in rows])
            for metric in ("accuracy", "balanced_accuracy", "macro_f1")
        }
        subject_rows = []
        for subject in sorted(set(row["test_subject"] for row in rows)):
            subject_repeats = [row for row in rows if row["test_subject"] == subject]
            subject_rows.append(
                {
                    "test_subject": subject,
                    "n_repeats": len(subject_repeats),
                    **{
                        metric: float(np.mean([row[metric] for row in subject_repeats]))
                        for metric in ("accuracy", "balanced_accuracy", "macro_f1")
                    },
                }
            )
        statistical_summary[str(rate)] = {
            metric: bootstrap_ci(np.asarray([row[metric] for row in subject_rows]), 20000, args.seed)
            for metric in ("accuracy", "balanced_accuracy", "macro_f1")
        }
        balanced_values = np.asarray([row["balanced_accuracy"] for row in subject_rows], dtype=np.float64)
        statistical_summary[str(rate)]["balanced_accuracy_vs_chance"] = {
            "reference": 1.0 / n_classes,
            "mean_difference": float(balanced_values.mean() - 1.0 / n_classes),
            "two_sided_sign_flip_p": sign_flip_p_value(balanced_values, 1.0 / n_classes),
        }

    output_tag = args.output_tag.strip()
    if output_tag and not output_tag.replace("-", "").replace("_", "").replace(".", "").isalnum():
        raise SystemExit("--output-tag may only contain letters, numbers, hyphens, underscores, or dots.")
    tag_part = f"_{output_tag}" if output_tag else ""
    stem = (
        f"zuco2_nr_diagnostic_all_fixations_concat_len512_loso_eegnet_{args.target}{tag_part}"
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
        },
        "settings": {
            "split": "leave-one-subject-out",
            "inner_validation": args.inner_val_mode,
            "input": "ZuCo2 NR sentenceData word rawEEG, all valid fixations concatenated per word, fixed length 512, 105 channels",
            "normalization": "channel-wise z-score using inner training rows and valid mask only",
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
            "repeats": args.repeats,
            "seed": args.seed,
        },
        "summary": summary,
        "statistical_summary": statistical_summary,
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
                "method": "Subject-level bootstrap CI; dropout repeats averaged within each test subject before uncertainty estimation.",
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
            f"macro_f1={metrics['macro_f1']['mean']:.3f}"
        )
    print(f"Results: {result_path}")
    print(f"Statistical summary: {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

