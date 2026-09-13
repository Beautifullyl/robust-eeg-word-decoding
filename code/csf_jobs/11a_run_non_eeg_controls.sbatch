#!/bin/bash --login
#SBATCH -J 33_non_eeg_control
#SBATCH -p multicore
#SBATCH -n 4
#SBATCH -t 0-6:00
#SBATCH --mem=16G
#SBATCH --array=0-1
#SBATCH -o experiment_outputs/11_non_eeg_controls/logs/non_eeg_%A_%a.out

set -euo pipefail

module purge
module load apps/binapps/conda/miniforge3/25.9.1

TARGETS=(content_function coarse_pos)
TARGET="${TARGETS[$SLURM_ARRAY_TASK_ID]}"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
ENV_DIR="${EEG_ENV_DIR:-$HOME/scratch/eeg_project/conda_envs/eeg-word-reproduction}"
DATA_DIR="$PROJECT_DIR/data/processed/zuco2_nr_diagnostic_all_fixations_concat_512"
STAGE_DIR="$PROJECT_DIR/experiment_outputs/11_non_eeg_controls"
RESULT_DIR="$STAGE_DIR/results"
SCRIPT_DIR="$STAGE_DIR/scripts"
MANIFEST_DIR="$STAGE_DIR/manifest"

mkdir -p "$RESULT_DIR" "$STAGE_DIR/logs" "$SCRIPT_DIR" "$MANIFEST_DIR"
cp "$PROJECT_DIR/code/analysis/run_nuisance_baseline.py" "$SCRIPT_DIR/run_nuisance_baseline_${TARGET}.py"
cp "$PROJECT_DIR/code/csf_jobs/11a_run_non_eeg_controls.sbatch" "$SCRIPT_DIR/33_run_nuisance_baseline_${TARGET}.sbatch"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

conda activate "$ENV_DIR"
cd "$PROJECT_DIR"

OUTPUT="$RESULT_DIR/loso_${TARGET}_non_eeg_controls_seed_42.json"
python code/analysis/run_nuisance_baseline.py \
  --metadata "$DATA_DIR/metadata.csv" \
  --target "$TARGET" \
  --seed 42 \
  --n-jobs "${SLURM_NTASKS:-1}" \
  --bootstrap-iterations 20000 \
  --output "$OUTPUT"

{
  printf 'stage=11_non_eeg_controls\n'
  printf 'job_id=%s\n' "${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-unknown}}"
  printf 'array_task_id=%s\n' "${SLURM_ARRAY_TASK_ID:-none}"
  printf 'target=%s\n' "$TARGET"
  printf 'representation=all_fixations_concat_len512\n'
  printf 'split=loso\n'
  printf 'model=class_balanced_logistic_regression\n'
  printf 'feature_sets=duration_fixation,lexical_position,combined_non_eeg\n'
  printf 'seed=42\n'
  printf 'metadata=%s\n' "$DATA_DIR/metadata.csv"
  printf 'result=%s\n' "$OUTPUT"
  printf 'completed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  sha256sum "$PROJECT_DIR/code/analysis/run_nuisance_baseline.py"
  sha256sum "$PROJECT_DIR/code/csf_jobs/11a_run_non_eeg_controls.sbatch"
} > "$MANIFEST_DIR/${TARGET}_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}_${SLURM_ARRAY_TASK_ID:-0}.txt"

echo "Stage 33 completed for target=$TARGET"
echo "Result: $OUTPUT"


