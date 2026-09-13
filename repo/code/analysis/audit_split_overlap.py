from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


TARGETS = {
    "content_function": ("function", "content"),
    "coarse_pos": ("function", "noun", "verb", "adjective", "adverb"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit sentence/stimulus overlap in the current split definitions.")
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--target", choices=tuple(TARGETS), required=True)
    parser.add_argument("--split", choices=("loso", "subject_dependent", "mixed_subject"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_metadata(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sentence_fingerprints(all_rows: list[dict[str, str]]) -> dict[tuple[str, str], str]:
    grouped: dict[tuple[str, str], list[tuple[int, str]]] = defaultdict(list)
    for row in all_rows:
        grouped[(row["subject"], row["sentence_id"])].append(
            (int(row["word_id"]), row.get("clean_word", "").strip().lower())
        )
    output = {}
    for key, words in grouped.items():
        sentence = " ".join(word for _, word in sorted(words) if word)
        output[key] = hashlib.sha256(sentence.encode("utf-8")).hexdigest()
    return output


def split_group(indices: np.ndarray, val_size: float, test_size: float, rng: np.random.Generator):
    indices = rng.permutation(indices)
    n = len(indices)
    n_test = max(1, int(round(n * test_size)))
    n_val = max(1, int(round(n * val_size)))
    if n_test + n_val >= n:
        n_test = 1
        n_val = 1
    return indices[n_test + n_val :], indices[n_test : n_test + n_val], indices[:n_test]


def subject_dependent_folds(subjects: np.ndarray, y: np.ndarray, args: argparse.Namespace):
    for subject_id, subject in enumerate(sorted(set(subjects.tolist()))):
        rng = np.random.default_rng(args.seed + subject_id * 1000)
        subject_idx = np.where(subjects == subject)[0]
        train_parts, val_parts, test_parts = [], [], []
        for label in sorted(set(y[subject_idx].tolist())):
            label_idx = subject_idx[y[subject_idx] == label]
            if len(label_idx) < 3:
                continue
            train, val, test = split_group(label_idx, args.val_size, args.test_size, rng)
            train_parts.append(train)
            val_parts.append(val)
            test_parts.append(test)
        yield subject, np.concatenate(train_parts), np.concatenate(val_parts), np.concatenate(test_parts)


def mixed_subject_fold(subjects: np.ndarray, y: np.ndarray, args: argparse.Namespace):
    rng = np.random.default_rng(args.seed)
    train_parts, val_parts, test_parts = [], [], []
    for subject in sorted(set(subjects.tolist())):
        subject_idx = np.where(subjects == subject)[0]
        for label in sorted(set(y[subject_idx].tolist())):
            stratum = subject_idx[y[subject_idx] == label]
            if len(stratum) < 3:
                continue
            train, val, test = split_group(stratum, args.val_size, args.test_size, rng)
            train_parts.append(train)
            val_parts.append(val)
            test_parts.append(test)
    yield "mixed_repeat_0", np.concatenate(train_parts), np.concatenate(val_parts), np.concatenate(test_parts)


def loso_folds(subjects: np.ndarray):
    for subject in sorted(set(subjects.tolist())):
        yield subject, np.where(subjects != subject)[0], np.asarray([], dtype=np.int64), np.where(subjects == subject)[0]


def audit_fold(
    fold: str,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    rows: list[dict[str, str]],
    fingerprints: dict[tuple[str, str], str],
) -> dict[str, object]:
    def local_key(index: int) -> tuple[str, str]:
        row = rows[index]
        return row["subject"], row["sentence_id"]

    def stimulus_key(index: int) -> str:
        return rows[index]["sentence_id"]

    def observed_text_key(index: int) -> str:
        row = rows[index]
        return fingerprints[(row["subject"], row["sentence_id"])]

    train_local = {local_key(int(i)) for i in train_idx}
    val_local = {local_key(int(i)) for i in val_idx}
    test_local = {local_key(int(i)) for i in test_idx}
    train_stimulus = {stimulus_key(int(i)) for i in train_idx}
    val_stimulus = {stimulus_key(int(i)) for i in val_idx}
    test_stimulus = {stimulus_key(int(i)) for i in test_idx}
    train_observed_text = {observed_text_key(int(i)) for i in train_idx}
    test_observed_text = {observed_text_key(int(i)) for i in test_idx}

    test_rows_with_local_sentence_in_train = sum(local_key(int(i)) in train_local for i in test_idx)
    test_rows_with_seen_stimulus = sum(stimulus_key(int(i)) in train_stimulus for i in test_idx)
    return {
        "fold": fold,
        "n_train_rows": int(len(train_idx)),
        "n_val_rows": int(len(val_idx)),
        "n_test_rows": int(len(test_idx)),
        "train_test_local_sentence_overlap": int(len(train_local & test_local)),
        "train_test_stimulus_overlap": int(len(train_stimulus & test_stimulus)),
        "train_test_observed_text_fingerprint_overlap": int(len(train_observed_text & test_observed_text)),
        "val_test_local_sentence_overlap": int(len(val_local & test_local)),
        "val_test_stimulus_overlap": int(len(val_stimulus & test_stimulus)),
        "test_rows_with_local_sentence_in_train": int(test_rows_with_local_sentence_in_train),
        "test_rows_with_seen_stimulus": int(test_rows_with_seen_stimulus),
        "test_fraction_local_sentence_seen": float(test_rows_with_local_sentence_in_train / max(1, len(test_idx))),
        "test_fraction_stimulus_seen": float(test_rows_with_seen_stimulus / max(1, len(test_idx))),
    }


def main() -> int:
    args = parse_args()
    all_rows = load_metadata(args.metadata)
    fingerprints = sentence_fingerprints(all_rows)
    label_column = "content_function_label" if args.target == "content_function" else "coarse_pos_label"
    label_to_id = {label: index for index, label in enumerate(TARGETS[args.target])}
    rows = [row for row in all_rows if row[label_column] in label_to_id]
    y = np.asarray([label_to_id[row[label_column]] for row in rows], dtype=np.int64)
    subjects = np.asarray([row["subject"] for row in rows])

    if args.split == "loso":
        folds = loso_folds(subjects)
    elif args.split == "subject_dependent":
        folds = subject_dependent_folds(subjects, y, args)
    else:
        folds = mixed_subject_fold(subjects, y, args)

    fold_rows = [audit_fold(str(name), train, val, test, rows, fingerprints) for name, train, val, test in folds]
    output = {
        "metadata": str(args.metadata),
        "target": args.target,
        "split": args.split,
        "seed": args.seed,
        "n_labelled_rows": len(rows),
        "n_subjects": len(set(subjects.tolist())),
        "n_unique_stimuli": len(set(fingerprints.values())),
        "interpretation": {
            "local_sentence_overlap": "Words from the same subject and sentence occur in both train and test.",
            "stimulus_overlap": "The same ZuCo task sentence_id occurs in train and test, possibly for different subjects.",
            "observed_text_fingerprint": "Auxiliary hash based only on words that retained valid EEG; it can differ across subjects because missing rows change the observed token sequence.",
            "limitation": "This metadata audit cannot prove whether fixation-locked EEG windows overlap in acquisition time; raw event timing is required for that check.",
        },
        "folds": fold_rows,
        "summary": {
            "mean_test_fraction_local_sentence_seen": float(np.mean([row["test_fraction_local_sentence_seen"] for row in fold_rows])),
            "max_test_fraction_local_sentence_seen": float(np.max([row["test_fraction_local_sentence_seen"] for row in fold_rows])),
            "mean_test_fraction_stimulus_seen": float(np.mean([row["test_fraction_stimulus_seen"] for row in fold_rows])),
            "max_test_fraction_stimulus_seen": float(np.max([row["test_fraction_stimulus_seen"] for row in fold_rows])),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], indent=2))
    print(f"Audit: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
