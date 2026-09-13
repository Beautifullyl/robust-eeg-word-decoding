from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import h5py
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_LIBS = PROJECT_ROOT / ".python-libs"
NLTK_DATA = PROJECT_ROOT / ".nltk_data"
if LOCAL_LIBS.exists():
    sys.path.insert(0, str(LOCAL_LIBS))

import nltk


if NLTK_DATA.exists():
    nltk.data.path.insert(0, str(NLTK_DATA))


DEFAULT_MAT_DIR = Path.home() / "scratch/eeg_project/datasets/zuco2_zip/zuco2/task1 - NR/Matlab files"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data/processed/zuco2_nr_diagnostic_first_fixation_256"
FUNCTION_POS = {
    "CC",
    "DT",
    "EX",
    "IN",
    "MD",
    "PDT",
    "POS",
    "PRP",
    "PRP$",
    "RP",
    "TO",
    "WDT",
    "WP",
    "WP$",
    "WRB",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract ZuCo2 NR diagnostic first-fixation RAW_EEG tensors.")
    parser.add_argument("--mat-dir", type=Path, default=DEFAULT_MAT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--channels", type=int, default=105)
    return parser.parse_args()


def clean_token(token: str) -> str:
    return re.sub(r"(^[^\w']+|[^\w']+$)", "", token).lower()


def get_subject_id(path: Path) -> str:
    match = re.search(r"results(.+?)_NR\.mat$", path.name)
    return match.group(1) if match else path.stem


def read_matlab_string(file: h5py.File, ref: h5py.Reference) -> str:
    arr = np.array(file[ref][()]).squeeze()
    return "".join(chr(int(value)) for value in arr.flatten() if int(value) != 0)


def first_reference(value: Any) -> h5py.Reference | None:
    if isinstance(value, h5py.Reference):
        return value if value else None
    arr = np.asarray(value, dtype=object)
    for item in arr.flat:
        if isinstance(item, h5py.Reference) and item:
            return item
    return None


def fixation_arrays(file: h5py.File, raw_eeg_ref: h5py.Reference) -> list[np.ndarray]:
    fixation_refs = file[raw_eeg_ref]
    arrays: list[np.ndarray] = []
    if len(fixation_refs.shape) <= 1:
        return arrays
    for fixation_idx in range(fixation_refs.shape[0]):
        ref = first_reference(fixation_refs[fixation_idx, 0])
        if ref is None:
            continue
        arrays.append(np.array(file[ref][()]).squeeze())
    return arrays


def to_time_by_channel(arr: np.ndarray, n_channels: int) -> np.ndarray | None:
    if arr.ndim != 2:
        return None
    if arr.shape[1] == n_channels:
        return arr
    if arr.shape[0] == n_channels:
        return arr.T
    return None


def fixed_length_segment(arr: np.ndarray, length: int, n_channels: int) -> tuple[np.ndarray, np.ndarray, int, bool]:
    out = np.zeros((length, n_channels), dtype=np.float32)
    mask = np.zeros((length,), dtype=np.bool_)
    original_length = int(arr.shape[0])
    used_length = min(original_length, length)
    out[:used_length] = arr[:used_length].astype(np.float32)
    mask[:used_length] = True
    return out, mask, original_length, original_length > length


def coarse_pos_label(pos: str) -> str:
    if pos in FUNCTION_POS:
        return "function"
    if pos.startswith("NN"):
        return "noun"
    if pos.startswith("VB"):
        return "verb"
    if pos.startswith("JJ"):
        return "adjective"
    if pos.startswith("RB"):
        return "adverb"
    return "ignore"


def content_function_label(pos: str) -> str:
    coarse = coarse_pos_label(pos)
    if coarse in {"noun", "verb", "adjective", "adverb"}:
        return "content"
    if coarse == "function":
        return "function"
    return "ignore"


def summarize(values: list[int | float]) -> dict[str, float | int | None]:
    clean = np.array([value for value in values if value is not None and not math.isnan(float(value))], dtype=float)
    if clean.size == 0:
        return {"n": 0, "min": None, "p05": None, "p25": None, "median": None, "p75": None, "p95": None, "max": None}
    return {
        "n": int(clean.size),
        "min": float(np.min(clean)),
        "p05": float(np.percentile(clean, 5)),
        "p25": float(np.percentile(clean, 25)),
        "median": float(np.percentile(clean, 50)),
        "p75": float(np.percentile(clean, 75)),
        "p95": float(np.percentile(clean, 95)),
        "max": float(np.max(clean)),
    }


def collect_index(mat_files: list[Path], n_channels: int) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    rows: list[dict[str, Any]] = []
    lengths: list[int] = []
    fixation_counts: list[int] = []
    counts = Counter()
    for mat_path in mat_files:
        subject = get_subject_id(mat_path)
        print(f"Indexing {mat_path.name} ...", flush=True)
        with h5py.File(mat_path, "r") as file:
            sentence_refs = file["sentenceData/word"]
            for sentence_id in range(sentence_refs.shape[0]):
                word_group = file[sentence_refs[sentence_id, 0]]
                if not isinstance(word_group, h5py.Group) or "content" not in word_group or "rawEEG" not in word_group:
                    continue
                tokens = []
                for word_id in range(word_group["content"].shape[0]):
                    content_ref = first_reference(word_group["content"][word_id, 0])
                    tokens.append(read_matlab_string(file, content_ref) if content_ref else "")
                cleaned = [clean_token(token) for token in tokens]
                tag_tokens = [token if token else original for token, original in zip(cleaned, tokens)]
                tagged = nltk.pos_tag(tag_tokens)
                for word_id, ((_, pos), clean_word) in enumerate(zip(tagged, cleaned)):
                    if not clean_word or clean_word.isdigit():
                        counts["empty_or_numeric"] += 1
                        continue
                    raw_ref = first_reference(word_group["rawEEG"][word_id, 0])
                    if raw_ref is None:
                        counts["missing_raw_ref"] += 1
                        continue
                    raw_arrays = fixation_arrays(file, raw_ref)
                    selected = None
                    used_fixation_idx = None
                    for fixation_idx, raw_arr in enumerate(raw_arrays):
                        arr = to_time_by_channel(raw_arr, n_channels)
                        if arr is None or not np.isfinite(arr).all():
                            continue
                        selected = arr
                        used_fixation_idx = fixation_idx
                        break
                    if selected is None or used_fixation_idx is None:
                        counts["no_valid_fixation"] += 1
                        continue
                    row_id = len(rows)
                    lengths.append(int(selected.shape[0]))
                    fixation_counts.append(len(raw_arrays))
                    rows.append(
                        {
                            "row_id": row_id,
                            "subject": subject,
                            "sentence_id": sentence_id,
                            "word_id": word_id,
                            "word": tokens[word_id],
                            "clean_word": clean_word,
                            "pos": pos,
                            "content_function_label": content_function_label(pos),
                            "coarse_pos_label": coarse_pos_label(pos),
                            "n_fixations_available": len(raw_arrays),
                            "used_fixation_index": used_fixation_idx,
                            "original_timepoints": int(selected.shape[0]),
                        }
                    )
                    counts["indexed"] += 1
        print(f"{subject}: indexed={sum(1 for row in rows if row['subject'] == subject)}", flush=True)
    print(f"Index counts: {dict(counts)}", flush=True)
    return rows, lengths, fixation_counts


def write_tensors(mat_files: list[Path], rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    rows_by_key = {(row["subject"], row["sentence_id"], row["word_id"]): row for row in rows}
    x_path = args.out_dir / f"X_first_fixation_len{args.length}.npy"
    mask_path = args.out_dir / f"mask_first_fixation_len{args.length}.npy"
    x_out = np.lib.format.open_memmap(
        x_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(rows), args.length, args.channels),
    )
    mask_out = np.lib.format.open_memmap(
        mask_path,
        mode="w+",
        dtype=np.bool_,
        shape=(len(rows), args.length),
    )
    counts = Counter()
    for mat_path in mat_files:
        subject = get_subject_id(mat_path)
        print(f"Writing {mat_path.name} ...", flush=True)
        with h5py.File(mat_path, "r") as file:
            sentence_refs = file["sentenceData/word"]
            for row in [r for r in rows if r["subject"] == subject]:
                word_group = file[sentence_refs[int(row["sentence_id"]), 0]]
                raw_ref = first_reference(word_group["rawEEG"][int(row["word_id"]), 0])
                if raw_ref is None:
                    raise SystemExit(f"Missing raw reference for row {row['row_id']}")
                raw_arrays = fixation_arrays(file, raw_ref)
                selected = None
                for fixation_idx, raw_arr in enumerate(raw_arrays):
                    arr = to_time_by_channel(raw_arr, args.channels)
                    if arr is None or not np.isfinite(arr).all():
                        continue
                    selected = arr
                    if fixation_idx != int(row["used_fixation_index"]):
                        raise SystemExit(f"Fixation index mismatch for row {row['row_id']}")
                    break
                if selected is None:
                    raise SystemExit(f"No valid fixation for row {row['row_id']}")
                segment, mask, original_length, truncated = fixed_length_segment(selected, args.length, args.channels)
                if original_length != int(row["original_timepoints"]):
                    raise SystemExit(f"Length mismatch for row {row['row_id']}")
                x_out[int(row["row_id"])] = segment
                mask_out[int(row["row_id"])] = mask
                row["used_timepoints"] = min(original_length, args.length)
                row["was_truncated"] = bool(truncated)
                row["was_padded"] = bool(original_length < args.length)
                counts["written"] += 1
                if truncated:
                    counts["truncated"] += 1
                if original_length < args.length:
                    counts["padded"] += 1
    x_out.flush()
    mask_out.flush()
    if counts["written"] != len(rows):
        raise SystemExit(f"Expected to write {len(rows)} rows, wrote {counts['written']}")
    print(f"Write counts: {dict(counts)}", flush=True)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    mat_files = sorted(args.mat_dir.glob("resultsY*_NR.mat"))
    if not mat_files:
        raise SystemExit(f"No ZuCo2 NR Matlab files found in {args.mat_dir}")

    rows, lengths, fixation_counts = collect_index(mat_files, args.channels)
    if not rows:
        raise SystemExit("No diagnostic rows extracted.")
    write_tensors(mat_files, rows, args)

    metadata_path = args.out_dir / "metadata.csv"
    with metadata_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "row_id",
                "subject",
                "sentence_id",
                "word_id",
                "word",
                "clean_word",
                "pos",
                "content_function_label",
                "coarse_pos_label",
                "n_fixations_available",
                "used_fixation_index",
                "original_timepoints",
                "used_timepoints",
                "was_truncated",
                "was_padded",
            ],
        )
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: int(row["row_id"])))

    summary = {
        "mat_dir": str(args.mat_dir),
        "out_dir": str(args.out_dir),
        "n_rows": len(rows),
        "x_shape": [len(rows), args.length, args.channels],
        "subjects": sorted({row["subject"] for row in rows}),
        "content_function_counts": dict(Counter(row["content_function_label"] for row in rows)),
        "coarse_pos_counts": dict(Counter(row["coarse_pos_label"] for row in rows)),
        "pos_counts": dict(Counter(row["pos"] for row in rows)),
        "original_timepoints": summarize(lengths),
        "fixation_counts": summarize(fixation_counts),
        "truncation_rate": float(np.mean([int(row["was_truncated"]) for row in rows])),
        "padding_rate": float(np.mean([int(row["was_padded"]) for row in rows])),
    }
    summary_path = args.out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Rows: {len(rows)}")
    print(f"Content/function counts: {summary['content_function_counts']}")
    print(f"Coarse POS counts: {summary['coarse_pos_counts']}")
    print(f"Metadata: {metadata_path}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
