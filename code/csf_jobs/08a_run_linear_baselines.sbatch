#!/bin/bash --login
#SBATCH -p multicore
#SBATCH -n 8
#SBATCH -t 0-12:00
#SBATCH --mem=32G
#SBATCH --array=0-29
#SBATCH -o experiment_outputs/08_core_baselines_and_diagnostics/logs/linear_%A_%a.out

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

ENV_DIR="${EEG_ENV_DIR:-$HOME/scratch/eeg_project/conda_envs/eeg-word-reproduction}"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
DATA_DIR="$PROJECT_DIR/data/processed/zuco2_nr_diagnostic_all_fixations_concat_512"
STAGE_DIR="$PROJECT_DIR/experiment_outputs/08_core_baselines_and_diagnostics"
RESULT_DIR="$STAGE_DIR/results/linear"
SCRIPT_DIR="$STAGE_DIR/scripts"
MANIFEST_DIR="$STAGE_DIR/manifest"

mkdir -p "$RESULT_DIR" "$SCRIPT_DIR" "$MANIFEST_DIR" "$STAGE_DIR/logs"
cp "$PROJECT_DIR/code/models/run_linear_diagnostic_classification.py" "$SCRIPT_DIR/" || true
cp "$PROJECT_DIR/code/csf_jobs/08a_run_linear_baselines.sbatch" "$SCRIPT_DIR/" || true

export OMP_NUM_THREADS="${SLURM_NTASKS:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_NTASKS:-1}"
export MKL_NUM_THREADS="${SLURM_NTASKS:-1}"

echo "Stage 28a all-fixations linear core baseline"
echo "stage=08_core_baselines_and_diagnostics"
echo "model=linear_summary_stats"
echo "split=$SPLIT"
echo "target=$TARGET"
echo "seed=$SEED"
echo "job_id=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}} task_id=${SLURM_ARRAY_TASK_ID}"
echo "stage_dir=$STAGE_DIR"

cat > "$MANIFEST_DIR/linear_${SPLIT}_${TARGET}_seed_${SEED}.txt" <<EOF
stage=08_core_baselines_and_diagnostics
substage=28a_allfix_linear
model=linear_summary_stats
split=$SPLIT
target=$TARGET
seed=$SEED
job_id=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}
task_id=${SLURM_ARRAY_TASK_ID}
project_dir=$PROJECT_DIR
data_dir=$DATA_DIR
result_dir=$RESULT_DIR
EOF

conda activate "$ENV_DIR"
cd "$PROJECT_DIR"

python code/models/run_linear_diagnostic_classification.py \
  --data-dir "$DATA_DIR" \
  --output-dir "$RESULT_DIR" \
  --input-kind all_fixations \
  --target "$TARGET" \
  --split "$SPLIT" \
  --dropout-rates 0.0 0.1 0.25 0.5 \
  --split-repeats 5 \
  --dropout-repeats 5 \
  --c-grid 0.01 0.1 1.0 10.0 \
  --max-iter 1000 \
  --experiment-id stage28 \
  --seed "$SEED"

echo "Stage 28a result files:"
find "$STAGE_DIR" -maxdepth 4 -type f | sort


