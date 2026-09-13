from __future__ import annotations

import ast
import csv
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
RESULTS = ROOT / "results" / "final_subject_level"

REQUIRED_RESULTS = (
    "main_loso_supporting_metrics.csv",
    "main_no_dropout.csv",
    "non_eeg_controls.csv",
    "paired_comparisons.csv",
    "per_class.csv",
    "structured_dropout.csv",
    "workflow10_grouped_validation_comparison.csv",
    "workflow10_grouped_validation_subjects.csv",
    "workflow11_repeated_sentence_partitions_folds.csv",
    "workflow11_repeated_sentence_partitions_runs.csv",
    "workflow11_repeated_sentence_partitions_subjects.csv",
    "workflow12_fixed_session_channel_loss_subjects.csv",
    "workflow12_fixed_session_channel_loss_summary.csv",
    "workflow13_matched_fixation_conditions.csv",
    "workflow13_matched_fixation_paired_comparisons.csv",
    "workflow13_matched_fixation_subjects.csv",
)

FORBIDDEN_TEXT = (
    "/mnt/iusers01/",
    "/net/scratch/",
    "C:\\Users\\",
    "$PROJECT_DIR/baseline/",
    "$PROJECT_DIR/analysis/",
    "$PROJECT_DIR/csf_jobs/",
)


def local_import_failures(python_files: list[Path]) -> list[str]:
    by_name: dict[str, list[Path]] = defaultdict(list)
    for path in python_files:
        by_name[path.stem].append(path)
    failures = []
    for path in python_files:
        tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or not node.module:
                continue
            module = node.module.split(".")[0]
            if module.startswith(("run_", "analyse_", "analyze_", "create_", "extract_", "audit_", "export_")) and module not in by_name:
                failures.append(f"{path.relative_to(ROOT)} imports missing module {module}")
    return failures


def referenced_file_failures(job_files: list[Path]) -> list[str]:
    failures = []
    project_pattern = re.compile(r"\$PROJECT_DIR/(code/[A-Za-z0-9_./-]+\.(?:py|sbatch|sh))")
    command_pattern = re.compile(
        r"\b(?:sbatch|bash)\b[^\n]*?\s+([A-Za-z0-9_./-]+\.(?:sbatch|sh))(?=\s|$)"
    )
    for path in job_files:
        text = path.read_text(encoding="utf-8-sig")
        references = project_pattern.findall(text)
        references.extend(command_pattern.findall(text))
        for relative in references:
            if not (ROOT / relative).is_file():
                failures.append(f"{path.relative_to(ROOT)} references missing file {relative}")
    return failures


def result_failures() -> list[str]:
    failures = []
    for name in REQUIRED_RESULTS:
        path = RESULTS / name
        if not path.is_file() or path.stat().st_size == 0:
            failures.append(f"Missing or empty result: {path.relative_to(ROOT)}")
            continue
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            if not next(csv.reader(handle), None):
                failures.append(f"Result has no header: {path.relative_to(ROOT)}")
    return failures


def main() -> int:
    python_files = sorted(CODE.rglob("*.py"))
    job_files = sorted((CODE / "csf_jobs").glob("*"))
    failures = local_import_failures(python_files)
    failures.extend(referenced_file_failures(job_files))
    failures.extend(result_failures())

    for path in sorted(CODE.rglob("*")):
        if not path.is_file() or path.resolve() == Path(__file__).resolve():
            continue
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        for forbidden in FORBIDDEN_TEXT:
            if forbidden in text:
                failures.append(f"{path.relative_to(ROOT)} contains forbidden path fragment {forbidden!r}")

    lines = [
        "REPRODUCIBILITY CHECK",
        f"Python files checked: {len(python_files)}",
        f"Job files checked: {len(job_files)}",
        f"Required result tables checked: {len(REQUIRED_RESULTS)}",
    ]
    if failures:
        lines.append("Status: FAILED")
        lines.extend(f"- {failure}" for failure in sorted(set(failures)))
    else:
        lines.append("Status: PASSED")
    print("\n".join(lines))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
