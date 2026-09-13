from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
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
    parser = argparse.ArgumentParser(description="Run TCN diagnostic classification on ZuCo2 all-fixations word EEG.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data/processed/zuco2_nr_diagnostic_all_fixations_concat_512",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--target", choices=tuple(base.TARGETS), required=True)
    parser.add_argument(
        "--output-tag",
        default="stage27",
        help="Optional short tag inserted into output filenames so reruns do not overwrite earlier results.",
    )
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--dropout-rates", nargs="*", type=float, default=[0.0])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--kernel-size", type=int, default=7)
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


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("Use an odd kernel size so temporal length is preserved.")
        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TCNClassifier(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        hidden_channels: int,
        levels: int,
        kernel_size: int,
        dropout: float,
    ):
        super().__init__()
        blocks = [
            nn.Conv1d(n_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
        ]
        for level in range(levels):
            blocks.append(TemporalBlock(hidden_channels, kernel_size, dilation=2**level, dropout=dropout))
        self.encoder = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(1)
        x = self.encoder(x)
        return self.classifier(x)


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
    base.set_seed(seed)
    model = TCNClassifier(
        n_channels=x.shape[2],
        n_classes=n_classes,
        hidden_channels=args.hidden_channels,
        levels=args.levels,
        kernel_size=args.kernel_size,
        dropout=args.model_dropout,
    ).to(device)
    weights = torch.from_numpy(base.class_weights(y[train_idx], n_classes)).float().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_dataset = base.EEGClassificationDataset(x, mask, y, data_indices, train_idx, mean, std)
    val_dataset = base.EEGClassificationDataset(x, mask, y, data_indices, val_idx, mean, std)
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
        val_loss = base.evaluate_loss(model, val_dataset, criterion, args.batch_size, device, num_workers)
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


def main() -> int:
    args = parse_args()
    base.set_seed(args.seed)
    output_dir = args.output_dir if args.output_dir is not None else args.data_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = min(4, max(0, int(torch.get_num_threads()) - 1))
    print(f"Using device: {device}", flush=True)
    print(f"DataLoader workers: {num_workers}", flush=True)
    print("experiment=stage07_all_fixations_loso_tcn", flush=True)

    metadata = load_metadata(args.data_dir / "metadata.csv")
    keep_idx, y, label_names, kept_metadata = base.build_labels(metadata, args.target)
    x = np.load(args.data_dir / "X_all_fixations_concat_len512.npy", mmap_mode="r")
    mask = np.load(args.data_dir / "mask_all_fixations_concat_len512.npy", mmap_mode="r")
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
    for fold_id, (subject, outer_train_idx, test_idx) in enumerate(base.loso_folds(subjects)):
        inner_train_idx, val_idx = base.stratified_train_val_split(outer_train_idx, y, args.val_size, args.seed + fold_id)
        print(
            f"Fold {subject}: train={len(inner_train_idx)}, val={len(val_idx)}, test={len(test_idx)}",
            flush=True,
        )
        mean, std = base.channel_stats(x, mask, keep_idx, inner_train_idx, args.batch_size)
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
        train_info[str(subject)] = info
        for rate in args.dropout_rates:
            repeats = 1 if rate == 0 else args.repeats
            for repeat in range(repeats):
                pred = base.predict(
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
                        **base.classification_metrics(y[test_idx], pred, n_classes),
                    }
                )

    summary = {}
    statistical_summary = {}
    for rate in args.dropout_rates:
        rows = [row for row in fold_rows if row["dropout_rate"] == rate]
        summary[str(rate)] = {
            metric: base.summarize([row[metric] for row in rows])
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
            metric: base.bootstrap_ci(np.asarray([row[metric] for row in subject_rows]), 20000, args.seed)
            for metric in ("accuracy", "balanced_accuracy", "macro_f1")
        }
        balanced_values = np.asarray([row["balanced_accuracy"] for row in subject_rows], dtype=np.float64)
        statistical_summary[str(rate)]["balanced_accuracy_vs_chance"] = {
            "reference": 1.0 / n_classes,
            "mean_difference": float(balanced_values.mean() - 1.0 / n_classes),
            "two_sided_sign_flip_p": base.sign_flip_p_value(balanced_values, 1.0 / n_classes),
        }

    output_tag = args.output_tag.strip()
    if output_tag and not output_tag.replace("-", "").replace("_", "").replace(".", "").isalnum():
        raise SystemExit("--output-tag may only contain letters, numbers, hyphens, underscores, or dots.")
    tag_part = f"_{output_tag}" if output_tag else ""
    stem = (
        f"zuco2_nr_diagnostic_all_fixations_concat_len512_loso_tcn_{args.target}{tag_part}"
        f"_hidden_{args.hidden_channels}_levels_{args.levels}_kernel_{args.kernel_size}"
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
            "input": "ZuCo2 NR sentenceData word rawEEG, all valid fixations concatenated per word, fixed length 512, 105 channels",
            "normalization": "channel-wise z-score using inner training rows and valid mask only",
            "model": "TCN-style temporal convolutional classifier",
            "loss": "class-weighted cross entropy",
            "optimizer": "AdamW",
            "hidden_channels": args.hidden_channels,
            "levels": args.levels,
            "kernel_size": args.kernel_size,
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
