#!/bin/bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
STAGE_DIR="$PROJECT_DIR/experiment_outputs/12_grouped_inner_validation"

cd "$PROJECT_DIR"
mkdir -p "$STAGE_DIR/logs" "$STAGE_DIR/results" "$STAGE_DIR/scripts" "$STAGE_DIR/manifest"

case "${1:-smoke}" in
  smoke)
    sbatch code/csf_jobs/12a_smoke_grouped_inner_validation.sbatch
    ;;
  formal)
    sbatch code/csf_jobs/12b_run_grouped_inner_validation.sbatch
    ;;
  *)
    echo "Usage: bash code/csf_jobs/12c_submit_grouped_inner_validation.sh [smoke|formal]" >&2
    exit 2
    ;;
esac


