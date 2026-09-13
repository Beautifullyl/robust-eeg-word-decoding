from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import scipy.io


OFFICIAL_PAIR_LABELS = {
    "E22",
    "E9",
    "E26",
    "E2",
    "E23",
    "E3",
    "E33",
    "E122",
    "E27",
    "E123",
    "E19",
    "E4",
    "E24",
    "E124",
    "E34",
    "E116",
    "E28",
    "E117",
}


def parse_args() -> argparse.Namespace:
    project_dir = Path(os.environ.get("PROJECT_DIR", Path.cwd()))
    parser = argparse.ArgumentParser(
        description="Export the channel labels used by ZuCo2 processed EEG files."
    )
    parser.add_argument(
        "--zuco-root",
        type=Path,
        default=Path.home() / "scratch/eeg_project/datasets/zuco2_zip/zuco2",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=project_dir
        / "experiment_outputs/09_structured_channel_loss/manifest",
    )
    parser.add_argument("--max-files", type=int, default=80)
    return parser.parse_args()


def squeeze(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.size == 1:
        return value.reshape(-1)[0]
    return value


def matlab_string(value: Any) -> str | None:
    value = squeeze(value)
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"U", "S"}:
            return "".join(str(part) for part in value.reshape(-1)).strip()
        if value.dtype == object and value.size == 1:
            return matlab_string(value.reshape(-1)[0])
    return None


def struct_field(value: Any, field: str) -> Any | None:
    value = squeeze(value)
    if isinstance(value, np.ndarray) and value.dtype.names and field in value.dtype.names:
        return value[field]
    if hasattr(value, field):
        return getattr(value, field)
    return None


def h5_string(file_handle: h5py.File, value: Any) -> str | None:
    if isinstance(value, h5py.Reference):
        if not value:
            return None
        return h5_string(file_handle, file_handle[value])

    if isinstance(value, h5py.Dataset):
        return h5_string(file_handle, value[()])

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip("\x00").strip()

    if isinstance(value, str):
        return value.strip("\x00").strip()

    if isinstance(value, np.ndarray):
        if value.dtype == object:
            parts = [h5_string(file_handle, item) for item in value.reshape(-1)]
            joined = "".join(part for part in parts if part)
            return joined.strip() or None

        if value.dtype.kind in {"S", "U"}:
            parts = []
            for item in value.reshape(-1):
                if isinstance(item, bytes):
                    parts.append(item.decode("utf-8", errors="replace"))
                else:
                    parts.append(str(item))
            return "".join(parts).strip("\x00").strip() or None

        if value.dtype.kind in {"u", "i", "f"}:
            chars = []
            for item in np.asarray(value).reshape(-1):
                code = int(item)
                if code:
                    chars.append(chr(code))
            return "".join(chars).strip("\x00").strip() or None

    if isinstance(value, np.generic):
        try:
            return h5_string(file_handle, np.asarray(value))
        except Exception:
            return None

    return None


def h5_label_list(file_handle: h5py.File, value: Any) -> list[str]:
    if isinstance(value, h5py.Group):
        if "labels" in value:
            return h5_label_list(file_handle, value["labels"])
        return []

    if isinstance(value, h5py.Dataset):
        data = value[()]
        if value.dtype == object:
            labels = []
            for ref in data.reshape(-1):
                if isinstance(ref, h5py.Reference) and ref:
                    target = file_handle[ref]
                    if isinstance(target, h5py.Group):
                        labels.extend(h5_label_list(file_handle, target))
                        continue
                label = h5_string(file_handle, ref)
                if label:
                    labels.append(label)
            return labels

        label = h5_string(file_handle, data)
        return [label] if label else []

    return []


def extract_labels_and_shape_h5(path: Path) -> tuple[list[str], tuple[int, ...] | None]:
    with h5py.File(path, "r") as file_handle:
        if "EEG" not in file_handle:
            return [], None

        eeg = file_handle["EEG"]
        data_shape = tuple(eeg["data"].shape) if isinstance(eeg, h5py.Group) and "data" in eeg else None

        labels: list[str] = []
        if isinstance(eeg, h5py.Group) and "chanlocs" in eeg:
            labels = h5_label_list(file_handle, eeg["chanlocs"])

        return labels, data_shape


def extract_labels_and_shape_scipy(path: Path) -> tuple[list[str], tuple[int, ...] | None]:
    mat = scipy.io.loadmat(path, squeeze_me=False, struct_as_record=False)
    eeg = mat.get("EEG")
    if eeg is None:
        return [], None

    data = struct_field(eeg, "data")
    data_shape = tuple(np.asarray(data).shape) if data is not None else None

    chanlocs = struct_field(eeg, "chanlocs")
    if chanlocs is None:
        return [], data_shape

    labels: list[str] = []
    for item in np.asarray(chanlocs).reshape(-1):
        label = matlab_string(struct_field(item, "labels"))
        if label:
            labels.append(label)
    return labels, data_shape


def extract_labels_and_shape(path: Path) -> tuple[list[str], tuple[int, ...] | None]:
    try:
        labels, data_shape = extract_labels_and_shape_h5(path)
        if labels:
            return labels, data_shape
    except OSError:
        pass
    return extract_labels_and_shape_scipy(path)


def candidate_files(root: Path, max_files: int) -> list[Path]:
    patterns = [
        "task1 - NR/Preprocessed/*/gip_*_EEG.mat",
        "task1 - NR/Preprocessed/**/*_EEG.mat",
        "task1 - NR/Raw data/**/*_EEG.mat",
    ]
    seen: set[Path] = set()
    files: list[Path] = []
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if path not in seen:
                seen.add(path)
                files.append(path)
            if len(files) >= max_files:
                return files
    return files


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"ZuCo root: {args.zuco_root}")
    print(f"exists: {args.zuco_root.exists()}")
    if not args.zuco_root.exists():
        return 1

    inspected = []
    for path in candidate_files(args.zuco_root, args.max_files):
        rel = str(path.relative_to(args.zuco_root))
        try:
            labels, data_shape = extract_labels_and_shape(path)
        except Exception as exc:
            inspected.append({"file": rel, "error": f"{type(exc).__name__}: {exc}"})
            continue

        entry = {
            "file": rel,
            "n_labels": len(labels),
            "data_shape": data_shape,
            "first_labels": labels[:12],
            "official_frontal_pair_labels_present": sorted(OFFICIAL_PAIR_LABELS.intersection(labels)),
        }
        inspected.append(entry)
        print(json.dumps(entry, ensure_ascii=False))

        if labels:
            n_data_channels = int(data_shape[0]) if data_shape else len(labels)
            if len(labels) >= 105:
                used_labels = labels[:105]
            else:
                used_labels = labels[:n_data_channels]

            if len(used_labels) == 105:
                labels_path = args.out_dir / "processed_channel_labels.txt"
                audit_path = args.out_dir / "processed_channel_labels_audit.json"
                labels_path.write_text("\n".join(used_labels) + "\n", encoding="utf-8")
                audit = {
                    "source_file": rel,
                    "source_data_shape": data_shape,
                    "source_n_labels": len(labels),
                    "used_n_labels": len(used_labels),
                    "used_labels": used_labels,
                    "official_frontal_pair_labels_present": sorted(OFFICIAL_PAIR_LABELS.intersection(used_labels)),
                    "note": (
                        "ZuCo first-level Matlab scripts use FullEEG.data(1:nChans, ...) with nChans=105. "
                        "This audit therefore records the first 105 labels from the selected processed EEG file."
                    ),
                }
                audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
                print(f"saved_labels={labels_path}")
                print(f"saved_audit={audit_path}")
                return 0

    audit_path = args.out_dir / "processed_channel_labels_audit_failed.json"
    audit_path.write_text(json.dumps({"inspected": inspected}, indent=2), encoding="utf-8")
    print(f"failed_to_export_labels; audit={audit_path}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
