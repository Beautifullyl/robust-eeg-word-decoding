#!/bin/bash --login
#SBATCH -p multicore
#SBATCH -n 4
#SBATCH -t 0-12:00
#SBATCH --mem=32G
#SBATCH -o experiment_outputs/05_all_fixations_extraction/logs/extract_%j.out

set -euo pipefail

module purge
module load apps/binapps/conda/miniforge3/25.9.1

ENV_DIR="${EEG_ENV_DIR:-$HOME/scratch/eeg_project/conda_envs/eeg-word-reproduction}"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
ZUCO2_ROOT="${ZUCO2_ROOT:-$HOME/scratch/eeg_project/datasets/zuco2_zip/zuco2}"
MAT_DIR="${ZUCO2_MAT_DIR:-$ZUCO2_ROOT/task1 - NR/Matlab files}"
OUT_DIR="$PROJECT_DIR/data/processed/zuco2_nr_diagnostic_all_fixations_concat_512"
STAGE_DIR="$PROJECT_DIR/experiment_outputs/05_all_fixations_extraction"
mkdir -p "$STAGE_DIR/logs" "$STAGE_DIR/results" "$STAGE_DIR/scripts" "$STAGE_DIR/manifest"
cp "$PROJECT_DIR/code/data_preparation/extract_zuco2_diagnostic_all_fixations_concat.py" "$STAGE_DIR/scripts/" || true
cp "$PROJECT_DIR/code/csf_jobs/05_extract_all_fixations_512.sbatch" "$STAGE_DIR/scripts/" || true
export OMP_NUM_THREADS="${SLURM_NTASKS:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_NTASKS:-1}"
export MKL_NUM_THREADS="${SLURM_NTASKS:-1}"
export NLTK_DATA="$PROJECT_DIR/.nltk_data"

conda activate "$ENV_DIR"
cd "$PROJECT_DIR"

python code/data_preparation/extract_zuco2_diagnostic_all_fixations_concat.py \
  --mat-dir "$MAT_DIR" \
  --out-dir "$OUT_DIR" \
  --length 512 \
  --channels 105

ls -lh "$OUT_DIR"
cp "$OUT_DIR/summary.json" "$STAGE_DIR/results/summary.json"
printf 'mat_dir=%s\nout_dir=%s\n' "$MAT_DIR" "$OUT_DIR" > "$STAGE_DIR/manifest/paths.txt"


