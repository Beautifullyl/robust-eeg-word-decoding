from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage33", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = []
    for path in sorted(args.stage33.rglob("loso_*_non_eeg_controls_seed_42.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        target = payload["target"]
        for feature_set, result in payload["feature_sets"].items():
            stats = result["stats"]["balanced_accuracy"]
            rows.append(
                {
                    "target": target,
                    "feature_set": feature_set,
                    "mean": stats["mean"],
                    "ci95_low": stats["ci95_low"],
                    "ci95_high": stats["ci95_high"],
                }
            )
    if len(rows) != 6:
        raise SystemExit(f"Expected six target-feature rows, found {len(rows)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
