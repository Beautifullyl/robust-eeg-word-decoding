from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / "code" / "csf_jobs"
OUTPUTS = ROOT / "experiment_outputs"

WORKFLOWS = (
    "01_environment_setup",
    "02_data_validation",
    "05_all_fixations_extraction",
    "06_all_fixations_loso_eegnet",
    "07_all_fixations_loso_tcn",
    "08_core_baselines_and_diagnostics",
    "09_structured_channel_loss",
    "10_split_overlap_audit",
    "11_non_eeg_controls",
    "12_grouped_inner_validation",
    "14_repeated_sentence_partitions",
    "15_fixed_session_channel_loss",
    "16_matched_fixation_ablation",
    "17_report_outputs",
)

SUBDIRECTORIES = ("logs", "results", "analysis", "scripts", "manifest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare or submit the reproducible CSF workflow.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    submit = subparsers.add_parser("submit")
    submit.add_argument(
        "workflow",
        choices=("all", "workflow12", "workflow14", "workflow15", "workflow16", "workflow17"),
    )
    submit.add_argument("--mode", choices=("extract", "smoke", "formal"), default="formal")
    return parser.parse_args()


def prepare() -> None:
    for workflow in WORKFLOWS:
        workflow_dir = OUTPUTS / workflow
        for subdirectory in SUBDIRECTORIES:
            (workflow_dir / subdirectory).mkdir(parents=True, exist_ok=True)


def submit_job(name: str, dependencies: tuple[str, ...] = ()) -> str:
    script = JOBS / name
    if not script.is_file():
        raise SystemExit(f"Missing job script: {script}")
    command = ["sbatch", "--parsable"]
    if dependencies:
        command.append(f"--dependency=afterok:{':'.join(dependencies)}")
    command.append(str(script))
    environment = os.environ.copy()
    environment["PROJECT_DIR"] = str(ROOT)
    job_id = subprocess.check_output(command, cwd=ROOT, env=environment, text=True).strip().split(";")[0]
    print(f"{name}: {job_id}")
    return job_id


def submit_all() -> None:
    environment = submit_job("01_setup_environment.sbatch")
    validation = submit_job("02_validate_zuco2_complete_inventory.sbatch", (environment,))
    all_512 = submit_job("05_extract_all_fixations_512.sbatch", (validation,))

    eegnet_cf = submit_job("06a_run_all_fixations_eegnet_content_function.sbatch", (all_512,))
    eegnet_pos = submit_job("06b_run_all_fixations_eegnet_coarse_pos.sbatch", (all_512,))
    tcn_cf = submit_job("07a_run_tcn_content_function.sbatch", (all_512,))
    tcn_pos = submit_job("07b_run_tcn_coarse_pos.sbatch", (all_512,))
    linear = submit_job("08a_run_linear_baselines.sbatch", (all_512,))
    diagnostics = submit_job("08b_run_eegnet_diagnostic_splits.sbatch", (all_512,))
    per_class = submit_job("08c_analyse_eegnet_per_class.sbatch", (eegnet_cf, eegnet_pos))

    channel_labels = submit_job("09a_export_channel_labels.sbatch", (all_512,))
    structured_cf = submit_job("09b_run_structured_channel_loss_content_function.sbatch", (all_512, channel_labels))
    structured_pos = submit_job("09c_run_structured_channel_loss_coarse_pos.sbatch", (all_512, channel_labels))
    submit_job("10_audit_split_overlap.sbatch", (all_512,))
    non_eeg = submit_job("11a_run_non_eeg_controls.sbatch", (all_512,))
    workflow12 = submit_job("12b_run_grouped_inner_validation.sbatch", (all_512,))
    workflow12_analysis = submit_job(
        "12d_analyse_grouped_inner_validation.sbatch",
        (workflow12, eegnet_cf, eegnet_pos),
    )
    workflow14 = submit_job("14b_run_repeated_sentence_partitions.sbatch", (all_512,))
    workflow14_analysis = submit_job("14c_analyse_repeated_sentence_partitions.sbatch", (workflow14,))
    workflow15 = submit_job("15b_run_fixed_session_channel_loss.sbatch", (all_512, channel_labels))
    workflow15_analysis = submit_job("15c_analyse_fixed_session_channel_loss.sbatch", (workflow15,))

    first_512 = submit_job("16a_extract_first_fixation_512.sbatch", (validation,))
    workflow16 = submit_job("16c_run_matched_fixation_ablation.sbatch", (first_512, all_512))
    workflow16_analysis = submit_job("16d_analyse_matched_fixation_ablation.sbatch", (workflow16,))

    submit_job(
        "17_build_report_outputs.sbatch",
        (
            eegnet_cf,
            eegnet_pos,
            tcn_cf,
            tcn_pos,
            linear,
            diagnostics,
            per_class,
            structured_cf,
            structured_pos,
            non_eeg,
            workflow12_analysis,
            workflow14_analysis,
            workflow15_analysis,
            workflow16_analysis,
        ),
    )


def submit_workflow(workflow: str, mode: str) -> None:
    if shutil.which("sbatch") is None:
        raise SystemExit("sbatch is unavailable. Run this command on CSF3.")
    prepare()
    if workflow == "all":
        if mode != "formal":
            raise SystemExit("The all workflow supports formal mode only.")
        submit_all()
        return
    if workflow == "workflow12":
        name = {
            "smoke": "12a_smoke_grouped_inner_validation.sbatch",
            "formal": "12b_run_grouped_inner_validation.sbatch",
        }.get(mode)
        if name is None:
            raise SystemExit(f"{workflow} does not support {mode} mode.")
        job = submit_job(name)
        if mode == "formal":
            submit_job("12d_analyse_grouped_inner_validation.sbatch", (job,))
        return
    if workflow == "workflow14":
        name = "14a_smoke_repeated_sentence_partitions.sbatch" if mode == "smoke" else "14b_run_repeated_sentence_partitions.sbatch"
        job = submit_job(name)
        if mode == "formal":
            submit_job("14c_analyse_repeated_sentence_partitions.sbatch", (job,))
        return
    if workflow == "workflow15":
        name = "15a_smoke_fixed_session_channel_loss.sbatch" if mode == "smoke" else "15b_run_fixed_session_channel_loss.sbatch"
        job = submit_job(name)
        if mode == "formal":
            submit_job("15c_analyse_fixed_session_channel_loss.sbatch", (job,))
        return
    if workflow == "workflow17":
        if mode != "formal":
            raise SystemExit("workflow17 supports formal mode only.")
        submit_job("17_build_report_outputs.sbatch")
    elif mode == "extract":
        submit_job("16a_extract_first_fixation_512.sbatch")
    elif mode == "smoke":
        submit_job("16b_smoke_matched_fixation_ablation.sbatch")
    else:
        job = submit_job("16c_run_matched_fixation_ablation.sbatch")
        submit_job("16d_analyse_matched_fixation_ablation.sbatch", (job,))


def main() -> int:
    args = parse_args()
    if args.command == "prepare":
        prepare()
        print(OUTPUTS)
    else:
        submit_workflow(args.workflow, args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
