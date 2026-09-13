#!/bin/bash --login
#SBATCH -p multicore
#SBATCH -n 2
#SBATCH -t 0-2:00
#SBATCH --mem=8G
#SBATCH -o experiment_outputs/02_data_validation/logs/validate_%j.out

set -euo pipefail

module purge
module load apps/binapps/conda/miniforge3/25.9.1

ENV_DIR="${EEG_ENV_DIR:-$HOME/scratch/eeg_project/conda_envs/eeg-word-reproduction}"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
ZUCO2_ROOT="${ZUCO2_ROOT:-$HOME/scratch/eeg_project/datasets/zuco2_zip/zuco2}"
STAGE_DIR="$PROJECT_DIR/experiment_outputs/02_data_validation"
OUT_DIR="$STAGE_DIR/results"

mkdir -p "$OUT_DIR" "$STAGE_DIR/logs" "$STAGE_DIR/scripts" "$STAGE_DIR/manifest"
cp "$PROJECT_DIR/code/data_preparation/validate_zuco2_complete_inventory.py" "$STAGE_DIR/scripts/" || true
cp "$PROJECT_DIR/code/csf_jobs/02_validate_zuco2_complete_inventory.sbatch" "$STAGE_DIR/scripts/" || true

conda activate "$ENV_DIR"
cd "$PROJECT_DIR"

python code/data_preparation/validate_zuco2_complete_inventory.py \
  --zuco2-root "$ZUCO2_ROOT" \
  --out-dir "$OUT_DIR"

printf 'zuco2_root=%s\noutput_dir=%s\n' "$ZUCO2_ROOT" "$OUT_DIR" > "$STAGE_DIR/manifest/paths.txt"


