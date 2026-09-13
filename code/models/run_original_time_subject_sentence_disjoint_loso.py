from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from run_eegnet_diagnostic_all_fixations_classification import (
    TARGETS,
    bootstrap_ci,
    build_labels,
    channel_stats,
    classification_metrics,
    load_metadata,
    predict,
    set_seed,
    sign_flip_p_value,
    train_model,
)


ANALYSIS_ID = "stage38_original_time_subject_sentence_disjoint_loso_v1"
SENTENCE_SPLIT_ID = "sorted_mod5_test_fixed_rng_validation_v1"
OUTPUT_STAGE = "stage38"
EXPECTED_ROWS = 75_539
EXPECTED_SUBJECTS = 18
EXPECTED_SENTENCES = 349
EXPECTED_TIMEPOINTS = 512
EXPECTED_CHANNELS = 105


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Original-time EEGNet with jointly disjoint test subject and sentence stimuli."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", choices=tuple(TARGETS), required=True)
    parser.add_argument(
        "--run-tag",
        default="formal",
        help="Short identifier included in filenames and compatibility checks (for example smoke or formal).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Model initialisation seed.")
    parser.add_argument(
        "--split-seed",
        type=int,
        default=20_260_821,
        help="Fixed data-partition seed. Keep identical across model seeds.",
    )
    parser.add_argument(
        "--test-sentence-fold",
        type=int,
        choices=range(5),
        default=0,
        help="Modulo-5 sentence fold held out for testing (0 to 4).",
    )
    parser.add_argument("--validation-sentence-fraction", type=float, default=0.15)
    parser.add_argument("--validation-subjects", type=int, default=3)
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
    return parser.parse_args()


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def sentence_sort_key(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def sentence_partition(
    metadata: list[dict[str, str]],
    split_seed: int,
    validation_fraction: float,
    test_sentence_fold: int = 0,
) -> dict[str, object]:
    sentence_ids = sorted({row["sentence_id"] for row in metadata}, key=sentence_sort_key)
    if len(sentence_ids) != EXPECTED_SENTENCES:
        raise ValueError(f"Expected {EXPECTED_SENTENCES} sentence IDs, found {len(sentence_ids)}")
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("validation-sentence-fraction must be between 0 and 0.5")

    if test_sentence_fold not in range(5):
        raise ValueError("test-sentence-fold must be between 0 and 4")

    test_sentences = [
        sentence for index, sentence in enumerate(sentence_ids) if index % 5 == test_sentence_fold
    ]
    development_pool = [
        sentence for index, sentence in enumerate(sentence_ids) if index % 5 != test_sentence_fold
    ]
    rng = np.random.default_rng(split_seed)
    shuffled = np.asarray(development_pool, dtype=object)[rng.permutation(len(development_pool))]
    n_validation = min(
        max(1, int(round(len(development_pool) * validation_fraction))),
        len(development_pool) - 1,
    )
    validation_sentences = sorted(shuffled[:n_validation].tolist(), key=sentence_sort_key)
    training_sentences = sorted(shuffled[n_validation:].tolist(), key=sentence_sort_key)

    sentence_sets = [set(training_sentences), set(validation_sentences), set(test_sentences)]
    if any(sentence_sets[i] & sentence_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("Sentence partitions are not pairwise disjoint")
    if set(sentence_ids) != set().union(*sentence_sets):
        raise RuntimeError("Sentence partition does not cover the complete stimulus set")

    return {
        "split_id": SENTENCE_SPLIT_ID,
        "split_seed": split_seed,
        "test_sentence_fold": test_sentence_fold,
        "all_sentence_count": len(sentence_ids),
        "test_sentence_ids": test_sentences,
        "validation_sentence_ids": validation_sentences,
        "training_sentence_ids": training_sentences,
    }


def class_distribution(y: np.ndarray, indices: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(y[indices], minlength=n_classes).astype(np.float64)
    return counts / max(float(counts.sum()), 1.0)


def label_count(y: np.ndarray, indices: np.ndarray, label_names: list[str]) -> dict[str, int]:
    counts = Counter(int(value) for value in y[indices])
    return {name: int(counts.get(index, 0)) for index, name in enumerate(label_names)}


def require_class_coverage(
    y: np.ndarray,
    indices: np.ndarray,
    label_names: list[str],
    partition_name: str,
    test_subject: str,
) -> None:
    counts = label_count(y, indices, label_names)
    missing = [name for name, count in counts.items() if count == 0]
    if missing:
        raise ValueError(
            f"Missing classes in {partition_name} for test subject {test_subject}: {missing}; counts={counts}"
        )


def select_validation_subjects(
    test_subject: str,
    y: np.ndarray,
    training_rows_by_subject: dict[str, np.ndarray],
    validation_rows_by_subject: dict[str, np.ndarray],
    n_validation_subjects: int,
    selection_seed: int,
) -> tuple[tuple[str, ...], np.ndarray, np.ndarray]:
    candidate_subjects = sorted(set(training_rows_by_subject) - {test_subject})
    if not 0 < n_validation_subjects < len(candidate_subjects):
        raise ValueError("validation-subjects must leave at least one training subject")

    n_classes = len(set(int(value) for value in y))
    training_counts = {
        subject: np.bincount(y[training_rows_by_subject[subject]], minlength=n_classes).astype(np.float64)
        for subject in candidate_subjects
    }
    validation_counts = {
        subject: np.bincount(y[validation_rows_by_subject[subject]], minlength=n_classes).astype(np.float64)
        for subject in candidate_subjects
    }
    candidates = list(combinations(candidate_subjects, n_validation_subjects))
    rng = np.random.default_rng(selection_seed)
    rng.shuffle(candidates)
    best = None
    best_score = float("inf")
    for validation_subjects in candidates:
        validation_subject_set = set(validation_subjects)
        training_subject_set = [subject for subject in candidate_subjects if subject not in validation_subject_set]
        train_count = np.sum([training_counts[subject] for subject in training_subject_set], axis=0)
        validation_count = np.sum([validation_counts[subject] for subject in validation_subjects], axis=0)
        if np.any(train_count == 0) or np.any(validation_count == 0):
            continue
        score = float(np.abs(train_count / train_count.sum() - validation_count / validation_count.sum()).sum())
        if score < best_score:
            best_score = score
            best = (tuple(validation_subjects), tuple(training_subject_set))

    if best is None:
        raise RuntimeError(f"No valid grouped train/validation split for test subject {test_subject}")
    validation_subjects, training_subjects = best
    train_indices = np.concatenate([training_rows_by_subject[subject] for subject in training_subjects])
    validation_indices = np.concatenate(
        [validation_rows_by_subject[subject] for subject in validation_subjects]
    )
    return validation_subjects, train_indices, validation_indices


def assert_pairwise_disjoint(
    test_subject: str,
    subjects: np.ndarray,
    sentences: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray,
) -> None:
    partitions = {
        "train": train_indices,
        "validation": validation_indices,
        "test": test_indices,
    }
    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if set(subjects[partitions[first]]) & set(subjects[partitions[second]]):
            raise RuntimeError(f"Subject overlap between {first} and {second} for {test_subject}")
        if set(sentences[partitions[first]]) & set(sentences[partitions[second]]):
            raise RuntimeError(f"Sentence overlap between {first} and {second} for {test_subject}")


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Stage 38 model fitting requires a CUDA GPU")
    num_workers = min(4, max(0, int(torch.get_num_threads()) - 1))

    metadata = load_metadata(args.data_dir / "metadata.csv")
    data_indices, y, label_names, kept_rows = build_labels(metadata, args.target)
    subjects = np.asarray([row["subject"] for row in kept_rows], dtype=object)
    sentences = np.asarray([row["sentence_id"] for row in kept_rows], dtype=object)
    x = np.load(args.data_dir / "X_all_fixations_concat_len512.npy", mmap_mode="r")
    mask = np.load(args.data_dir / "mask_all_fixations_concat_len512.npy", mmap_mode="r")
    if x.shape != (len(metadata), EXPECTED_TIMEPOINTS, EXPECTED_CHANNELS):
        raise ValueError(f"Unexpected original-time EEG shape: {x.shape}; metadata rows={len(metadata)}")
    if mask.shape != (len(metadata), EXPECTED_TIMEPOINTS):
        raise ValueError(f"Unexpected validity-mask shape: {mask.shape}")
    if len(y) != EXPECTED_ROWS or len(set(subjects.tolist())) != EXPECTED_SUBJECTS:
        raise ValueError(f"Cohort mismatch: rows={len(y)}, subjects={len(set(subjects.tolist()))}")

    split = sentence_partition(
        metadata,
        args.split_seed,
        args.validation_sentence_fraction,
        args.test_sentence_fold,
    )
    training_sentence_set = set(split["training_sentence_ids"])
    validation_sentence_set = set(split["validation_sentence_ids"])
    test_sentence_set = set(split["test_sentence_ids"])
    all_subjects = sorted(set(subjects.tolist()))
    training_sentence_mask = np.isin(sentences, list(training_sentence_set))
    validation_sentence_mask = np.isin(sentences, list(validation_sentence_set))
    test_sentence_mask = np.isin(sentences, list(test_sentence_set))
    training_rows_by_subject = {
        subject: np.flatnonzero((subjects == subject) & training_sentence_mask)
        for subject in all_subjects
    }
    validation_rows_by_subject = {
        subject: np.flatnonzero((subjects == subject) & validation_sentence_mask)
        for subject in all_subjects
    }
    test_rows_by_subject = {
        subject: np.flatnonzero((subjects == subject) & test_sentence_mask)
        for subject in all_subjects
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_tag = args.run_tag.strip()
    if not run_tag or not run_tag.replace("-", "").replace("_", "").isalnum():
        raise ValueError("run-tag may contain only letters, numbers, hyphens and underscores")
    fold_suffix = (
        ""
        if OUTPUT_STAGE == "stage38" and args.test_sentence_fold == 0
        else f"_sentencefold_{args.test_sentence_fold}"
    )
    stem = (
        f"{OUTPUT_STAGE}_{run_tag}_{args.target}_eegnet_seed_{args.seed}{fold_suffix}"
        "_original_time_subject_sentence_disjoint"
    )
    output_path = args.output_dir / f"{stem}_results.json"
    progress_path = args.output_dir / f"{stem}_progress.json"
    identity = {
        "analysis_id": ANALYSIS_ID,
        "run_tag": run_tag,
        "target": args.target,
        "seed": args.seed,
        "split_seed": args.split_seed,
        "sentence_split_id": SENTENCE_SPLIT_ID,
        "validation_sentence_fraction": args.validation_sentence_fraction,
        "validation_subjects": args.validation_subjects,
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
    }
    if OUTPUT_STAGE != "stage38" or args.test_sentence_fold != 0:
        identity["test_sentence_fold"] = args.test_sentence_fold
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        observed = {key: existing.get("identity", {}).get(key) for key in identity}
        if observed != identity:
            raise ValueError(f"Refusing to reuse incompatible result: {output_path}")
        print(f"Complete result already exists: {output_path}", flush=True)
        return 0

    fold_rows: list[dict[str, object]] = []
    training_history: dict[str, object] = {}
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("identity") != identity:
            raise ValueError(f"Incompatible progress file: {progress_path}")
        fold_rows = list(progress.get("folds", []))
        training_history = dict(progress.get("training", {}))
        print(f"Resuming after {len(fold_rows)} completed subjects", flush=True)

    completed = {str(row["test_subject"]) for row in fold_rows}
    print(
        f"device={device} target={args.target} model_seed={args.seed} split_seed={args.split_seed} "
        f"test_sentence_fold={args.test_sentence_fold} "
        f"rows={len(y)} subjects={len(set(subjects.tolist()))} "
        f"sentences(train/val/test)={len(training_sentence_set)}/{len(validation_sentence_set)}/{len(test_sentence_set)}",
        flush=True,
    )

    for fold_id, test_subject in enumerate(all_subjects):
        if test_subject in completed:
            print(f"fold={test_subject} already complete; skipping", flush=True)
            continue
        validation_subjects, train_indices, validation_indices = select_validation_subjects(
            test_subject,
            y,
            training_rows_by_subject,
            validation_rows_by_subject,
            args.validation_subjects,
            args.split_seed + fold_id,
        )
        test_indices = test_rows_by_subject[test_subject]
        require_class_coverage(y, train_indices, label_names, "training", test_subject)
        require_class_coverage(y, validation_indices, label_names, "validation", test_subject)
        require_class_coverage(y, test_indices, label_names, "test", test_subject)
        assert_pairwise_disjoint(
            test_subject,
            subjects,
            sentences,
            train_indices,
            validation_indices,
            test_indices,
        )

        mean, std = channel_stats(x, mask, data_indices, train_indices, args.batch_size)
        print(
            f"fold={test_subject} train={len(train_indices)} val={len(validation_indices)} "
            f"test={len(test_indices)} val_subjects={list(validation_subjects)}",
            flush=True,
        )
        model, information = train_model(
            x,
            mask,
            y,
            data_indices,
            train_indices,
            validation_indices,
            mean,
            std,
            len(label_names),
            args,
            args.seed + fold_id,
            device,
            num_workers,
        )
        prediction = predict(
            model,
            x,
            mask,
            y,
            data_indices,
            test_indices,
            mean,
            std,
            args.batch_size,
            device,
            0.0,
            args.seed + fold_id,
            num_workers,
        )
        metrics = classification_metrics(y[test_indices], prediction, len(label_names))
        training_subjects = sorted(set(str(value) for value in subjects[train_indices]))
        fold_rows.append(
            {
                "test_subject": test_subject,
                "training_subjects": training_subjects,
                "validation_subjects": list(validation_subjects),
                "n_train": int(len(train_indices)),
                "n_val": int(len(validation_indices)),
                "n_test": int(len(test_indices)),
                "train_label_counts": label_count(y, train_indices, label_names),
                "validation_label_counts": label_count(y, validation_indices, label_names),
                "test_label_counts": label_count(y, test_indices, label_names),
                "pairwise_subject_overlap": 0,
                "pairwise_sentence_overlap": 0,
                **metrics,
            }
        )
        training_history[test_subject] = {
            **information,
            "training_subjects": training_subjects,
            "validation_subjects": list(validation_subjects),
        }
        write_json_atomic(
            progress_path,
            {
                "identity": identity,
                "folds": fold_rows,
                "training": training_history,
            },
        )
        print(f"fold={test_subject} BA={metrics['balanced_accuracy']:.8f}", flush=True)
        del model
        torch.cuda.empty_cache()

    if len(fold_rows) != EXPECTED_SUBJECTS:
        raise RuntimeError(f"Expected {EXPECTED_SUBJECTS} completed folds, found {len(fold_rows)}")

    statistics = {}
    for metric_index, metric in enumerate(("accuracy", "balanced_accuracy", "macro_f1")):
        values = np.asarray([float(row[metric]) for row in fold_rows], dtype=np.float64)
        statistics[metric] = bootstrap_ci(values, 20_000, args.seed + 38_100 + metric_index)
    balanced = np.asarray([float(row["balanced_accuracy"]) for row in fold_rows], dtype=np.float64)
    chance = 1.0 / len(label_names)
    statistics["balanced_accuracy_vs_chance"] = {
        "reference": chance,
        "mean_difference": float(balanced.mean() - chance),
        "two_sided_exact_sign_flip_p": sign_flip_p_value(balanced, chance),
    }

    output = {
        "analysis_id": ANALYSIS_ID,
        "identity": identity,
        "data": {
            "data_dir": str(args.data_dir),
            "eeg_file": "X_all_fixations_concat_len512.npy",
            "mask_file": "mask_all_fixations_concat_len512.npy",
            "source": "ZuCo 2.0 Task 1 sentenceData/word/rawEEG",
            "n_rows": int(len(y)),
            "n_subjects": EXPECTED_SUBJECTS,
            "n_sentences": EXPECTED_SENTENCES,
            "timepoints": EXPECTED_TIMEPOINTS,
            "channels": EXPECTED_CHANNELS,
            "label_names": label_names,
            "label_counts": {label_names[key]: int(value) for key, value in Counter(y.tolist()).items()},
        },
        "settings": {
            "split": "pairwise subject-and-sentence-disjoint train/validation/test",
            "test_sentence_rule": (
                "sentence index modulo 5 equals "
                f"{args.test_sentence_fold} after numeric/lexical sorting"
            ),
            "training_sentence_ids": split["training_sentence_ids"],
            "validation_sentence_ids": split["validation_sentence_ids"],
            "test_sentence_ids": split["test_sentence_ids"],
            "interpolation": "none",
            "physical_time_scale": "retained samples remain in original 500 Hz order",
            "fixed_window_limitation": "zero padded to 512; rows longer than 512 were truncated during extraction",
            "normalization": "channel-wise z-score fitted on training subjects and training sentences only",
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
        },
        "folds": fold_rows,
        "statistics": statistics,
        "training": training_history,
    }
    write_json_atomic(output_path, output)
    progress_path.unlink(missing_ok=True)
    print(json.dumps(statistics, indent=2), flush=True)
    print(f"Results: {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
