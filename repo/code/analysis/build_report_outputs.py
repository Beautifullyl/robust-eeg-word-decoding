from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "experiment_outputs"
RESULTS = ROOT / "results" / "final_subject_level"


def run(script: str, *arguments: str) -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "code" / "analysis" / script), *arguments],
        cwd=ROOT,
        check=True,
    )


def copy_analysis(workflow: str, names: tuple[str, ...]) -> None:
    source_dir = OUTPUTS / workflow / "analysis"
    for name in names:
        source = source_dir / name
        if not source.is_file():
            raise SystemExit(f"Missing analysis output: {source}")
        shutil.copy2(source, RESULTS / name)


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)

    run(
        "final_subject_level_analysis.py",
        "--stage25bc",
        str(OUTPUTS / "06_all_fixations_loso_eegnet"),
        "--stage27",
        str(OUTPUTS / "07_all_fixations_loso_tcn"),
        "--stage28",
        str(OUTPUTS / "08_core_baselines_and_diagnostics"),
        "--stage31",
        str(OUTPUTS / "09_structured_channel_loss"),
        "--output-dir",
        str(RESULTS),
    )
    run(
        "export_non_eeg_controls.py",
        "--stage33",
        str(OUTPUTS / "11_non_eeg_controls"),
        "--output",
        str(RESULTS / "non_eeg_controls.csv"),
    )

    copy_analysis(
        "12_grouped_inner_validation",
        (
            "workflow12_grouped_validation_comparison.csv",
            "workflow12_grouped_validation_subjects.csv",
        ),
    )
    copy_analysis(
        "14_repeated_sentence_partitions",
        (
            "workflow14_repeated_sentence_partitions_folds.csv",
            "workflow14_repeated_sentence_partitions_runs.csv",
            "workflow14_repeated_sentence_partitions_subjects.csv",
        ),
    )
    copy_analysis(
        "15_fixed_session_channel_loss",
        (
            "workflow15_fixed_session_channel_loss_subjects.csv",
            "workflow15_fixed_session_channel_loss_summary.csv",
        ),
    )
    copy_analysis(
        "16_matched_fixation_ablation",
        (
            "workflow16_matched_fixation_conditions.csv",
            "workflow16_matched_fixation_paired_comparisons.csv",
            "workflow16_matched_fixation_subjects.csv",
        ),
    )

    obsolete = RESULTS / "loso_random_dropout.csv"
    obsolete.unlink(missing_ok=True)
    run("create_final_report_figures.py")
    run("create_non_eeg_control_figure.py")
    print(RESULTS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
