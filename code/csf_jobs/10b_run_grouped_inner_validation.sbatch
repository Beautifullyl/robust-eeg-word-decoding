#!/bin/bash --login
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 8
#SBATCH -t 1-0:00
#SBATCH --mem=64G
#SBATCH --array=0-9%4
#SBATCH -o experiment_outputs/10_grouped_inner_validation/logs/formal_%A_%a.out

set -euo pipefail

module purge
module load libs/cuda/12.8.1
module load apps/binapps/conda/miniforge3/25.9.1

TARGETS=(content_function coarse_pos)
SEEDS=(42 43 44 45 46)
TARGET_INDEX=$((SLURM_ARRAY_TASK_ID / 5))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % 5))
TARGET="${TARGETS[$TARGET_INDEX]}"
SEED="${SEEDS[$SEED_INDEX]}"

ENV_DIR="${EEG_ENV_DIR:-$HOME/scratch/eeg_project/conda_envs/eeg-word-reproduction}"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
DATA_DIR="$PROJECT_DIR/data/processed/zuco2_nr_diagnostic_all_fixations_concat_512"
STAGE_DIR="$PROJECT_DIR/experiment_outputs/10_grouped_inner_validation"

mkdir -p "$STAGE_DIR/logs" "$STAGE_DIR/results" "$STAGE_DIR/scripts" "$STAGE_DIR/manifest"
cp "$PROJECT_DIR/code/models/run_eegnet_diagnostic_all_fixations_classification.py" "$STAGE_DIR/scripts/"
cp "$PROJECT_DIR/code/csf_jobs/10b_run_grouped_inner_validation.sbatch" "$STAGE_DIR/scripts/"

export OMP_NUM_THREADS="${SLURM_NTASKS:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_NTASKS:-1}"
export MKL_NUM_THREADS="${SLURM_NTASKS:-1}"

conda activate "$ENV_DIR"
cd "$PROJECT_DIR"

cat > "$STAGE_DIR/manifest/formal_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt" <<EOF
stage=10_grouped_inner_validation
purpose=subject_grouped_inner_validation_sensitivity
input=original_order_500Hz_samples_with_fixed_512_window_and_validity_mask
interpolation=none
outer_split=LOSO
inner_validation=three_complete_subjects
target=$TARGET
seed=$SEED
job_id=$SLURM_ARRAY_JOB_ID
task_id=$SLURM_ARRAY_TASK_ID
EOF

python code/models/run_eegnet_diagnostic_all_fixations_classification.py \
  --data-dir "$DATA_DIR" \
  --output-dir "$STAGE_DIR/results" \
  --target "$TARGET" \
  --inner-val-mode subject \
  --output-tag stage37_grouped_inner \
  --dropout-rates 0.0 \
  --repeats 1 \
  --f1 8 \
  --depth-multiplier 2 \
  --f2 16 \
  --temporal-kernel 31 \
  --separable-kernel 15 \
  --model-dropout 0.25 \
  --lr 0.001 \
  --weight-decay 0.0001 \
  --batch-size 128 \
  --max-epochs 100 \
  --patience 12 \
  --seed "$SEED"

echo "Stage 37 formal task completed: target=$TARGET seed=$SEED"
find "$STAGE_DIR/results" -maxdepth 1 -type f -name "*stage37_grouped_inner*seed_${SEED}*.json" | sort


