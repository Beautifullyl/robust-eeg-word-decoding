#!/bin/bash

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
STAGE_DIR="$PROJECT_DIR/experiment_outputs/11_non_eeg_controls"

mkdir -p "$STAGE_DIR/logs" "$STAGE_DIR/results" "$STAGE_DIR/scripts" "$STAGE_DIR/manifest"
cd "$PROJECT_DIR"

echo "Submitting Stage 33 from: $PROJECT_DIR"
echo "All outputs will be stored in: $STAGE_DIR"
sbatch code/csf_jobs/11a_run_non_eeg_controls.sbatch


