#!/bin/bash --login
#SBATCH -p serial
#SBATCH -n 1
#SBATCH -t 0-0:20
#SBATCH --mem=4G
#SBATCH -o experiment_outputs/07_structured_channel_loss/logs/export_channel_labels_%j.out

set -euo pipefail

module purge
module load apps/binapps/conda/miniforge3/25.9.1

ENV_DIR="${EEG_ENV_DIR:-$HOME/scratch/eeg_project/conda_envs/eeg-word-reproduction}"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
ZUCO_ROOT="${ZUCO2_ROOT:-$HOME/scratch/eeg_project/datasets/zuco2_zip/zuco2}"
STAGE_DIR="$PROJECT_DIR/experiment_outputs/07_structured_channel_loss"
MANIFEST_DIR="$STAGE_DIR/manifest"
SCRIPT_DIR="$STAGE_DIR/scripts"

mkdir -p "$STAGE_DIR/logs" "$MANIFEST_DIR" "$SCRIPT_DIR"
cp "$PROJECT_DIR/code/data_preparation/export_zuco2_processed_channel_labels.py" "$SCRIPT_DIR/" || true
cp "$PROJECT_DIR/code/csf_jobs/07a_export_channel_labels.sbatch" "$SCRIPT_DIR/" || true

conda activate "$ENV_DIR"
cd "$PROJECT_DIR"

python code/data_preparation/export_zuco2_processed_channel_labels.py \
  --zuco-root "$ZUCO_ROOT" \
  --out-dir "$MANIFEST_DIR"

echo "Stage 31 channel label export files:"
find "$STAGE_DIR" -maxdepth 3 -type f | sort


