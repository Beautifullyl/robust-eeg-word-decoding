#!/bin/bash --login
#SBATCH -p serial
#SBATCH -n 1
#SBATCH -t 0-2:00
#SBATCH --mem=4G
#SBATCH --array=0-29
#SBATCH -o experiment_outputs/10_split_overlap_audit/logs/audit_%A_%a.out

set -euo pipefail

module purge
module load apps/binapps/conda/miniforge3/25.9.1

SPLITS=(loso subject_dependent mixed_subject)
TARGETS=(content_function coarse_pos)
SEEDS=(42 43 44 45 46)

SPLIT_ID=$((SLURM_ARRAY_TASK_ID / 10))
TARGET_ID=$(((SLURM_ARRAY_TASK_ID % 10) / 5))
SEED_ID=$((SLURM_ARRAY_TASK_ID % 5))
SPLIT="${SPLITS[$SPLIT_ID]}"
TARGET="${TARGETS[$TARGET_ID]}"
SEED="${SEEDS[$SEED_ID]}"

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
ENV_DIR="${EEG_ENV_DIR:-$HOME/scratch/eeg_project/conda_envs/eeg-word-reproduction}"
DATA_DIR="$PROJECT_DIR/data/processed/zuco2_nr_diagnostic_all_fixations_concat_512"
STAGE_DIR="$PROJECT_DIR/experiment_outputs/10_split_overlap_audit"
RESULT_DIR="$STAGE_DIR/results"

mkdir -p "$RESULT_DIR" "$STAGE_DIR/logs" "$STAGE_DIR/scripts" "$STAGE_DIR/manifest"
cp "$PROJECT_DIR/code/analysis/audit_split_overlap.py" "$STAGE_DIR/scripts/" || true
cp "$PROJECT_DIR/code/csf_jobs/10_audit_split_overlap.sbatch" "$STAGE_DIR/scripts/" || true

conda activate "$ENV_DIR"
cd "$PROJECT_DIR"
python code/analysis/audit_split_overlap.py \
  --metadata "$DATA_DIR/metadata.csv" \
  --target "$TARGET" \
  --split "$SPLIT" \
  --seed "$SEED" \
  --output "$RESULT_DIR/${SPLIT}_${TARGET}_seed_${SEED}_split_overlap_audit.json"

printf 'split=%s\ntarget=%s\nseed=%s\n' "$SPLIT" "$TARGET" "$SEED" > "$STAGE_DIR/manifest/${SPLIT}_${TARGET}_seed_${SEED}.txt"


