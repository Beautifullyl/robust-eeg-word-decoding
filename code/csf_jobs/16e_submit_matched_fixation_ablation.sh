#!/bin/bash

set -euo pipefail

MODE="${1:-extract}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/scratch/eeg_project/robust-eeg-word-decoding}"
STAGE_DIR="$PROJECT_DIR/experiment_outputs/16_matched_fixation_ablation"

cd "$PROJECT_DIR"
mkdir -p "$STAGE_DIR/logs" "$STAGE_DIR/results" "$STAGE_DIR/analysis" "$STAGE_DIR/scripts" "$STAGE_DIR/manifest"

case "$MODE" in
  extract)
    JOB_ID=$(sbatch --parsable code/csf_jobs/16a_extract_first_fixation_512.sbatch)
    echo "Stage 41 extraction job: $JOB_ID"
    ;;
  smoke)
    JOB_ID=$(sbatch --parsable code/csf_jobs/16b_smoke_matched_fixation_ablation.sbatch)
    echo "Stage 41 smoke job: $JOB_ID"
    ;;
  formal)
    FORMAL_JOB_ID=$(sbatch --parsable code/csf_jobs/16c_run_matched_fixation_ablation.sbatch)
    ANALYSIS_JOB_ID=$(sbatch --parsable --dependency="afterok:${FORMAL_JOB_ID}" code/csf_jobs/16d_analyse_matched_fixation_ablation.sbatch)
    echo "Stage 41 formal array: $FORMAL_JOB_ID"
    echo "Stage 41 dependent analysis: $ANALYSIS_JOB_ID"
    ;;
  *)
    echo "Usage: bash code/csf_jobs/16e_submit_matched_fixation_ablation.sh [extract|smoke|formal]" >&2
    exit 2
    ;;
esac


