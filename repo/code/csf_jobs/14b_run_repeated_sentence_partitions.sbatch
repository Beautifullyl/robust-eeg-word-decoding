#!/bin/bash --login
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 8
#SBATCH -t 1-0:00
#SBATCH --mem=64G
#SBATCH --array=0-29%4
#SBATCH -o experiment_outputs/14_repeated_sentence_partitions/logs/formal_%A_%a.out

set -euo pipefail

module purge
module load libs/cuda/12.8.1
module load apps/binapps/conda/miniforge3/25.9.1

TARGETS=(content_function coarse_pos)
SEEDS=(42 43 44)
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % 3))
TARGET_INDEX=$(((SLURM_ARRAY_TASK_ID / 3) % 2))
TEST_SENTENCE_FOLD=$((SLURM_ARRAY_TASK_ID / 6))
TARGET="${TARGETS[$TARGET_INDEX]}"
SEED="${SEEDS[$SEED_INDEX]}"
SPLIT_SEED=20260822

ENV_DIR="${EEG_ENV_DIR:-$HOME/scratch/eeg_project/conda_envs/eeg-word-reproduction}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/scratch/eeg_project/robust-eeg-word-decoding}"
DATA_DIR="$PROJECT_DIR/data/processed/zuco2_nr_diagnostic_all_fixations_concat_512"
STAGE_DIR="$PROJECT_DIR/experiment_outputs/14_repeated_sentence_partitions"

mkdir -p "$STAGE_DIR/logs" "$STAGE_DIR/results" "$STAGE_DIR/scripts" "$STAGE_DIR/manifest" "$STAGE_DIR/analysis"
cp "$PROJECT_DIR/code/models/run_original_time_subject_sentence_disjoint_loso.py" "$STAGE_DIR/scripts/"
cp "$PROJECT_DIR/code/models/run_original_time_repeated_sentence_partitions_loso.py" "$STAGE_DIR/scripts/"
cp "$PROJECT_DIR/code/models/run_eegnet_diagnostic_all_fixations_classification.py" "$STAGE_DIR/scripts/"
cp "$PROJECT_DIR/code/csf_jobs/14b_run_repeated_sentence_partitions.sbatch" "$STAGE_DIR/scripts/"

export OMP_NUM_THREADS="${SLURM_NTASKS:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_NTASKS:-1}"
export MKL_NUM_THREADS="${SLURM_NTASKS:-1}"

conda activate "$ENV_DIR"
cd "$PROJECT_DIR"

cat > "$STAGE_DIR/manifest/formal_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt" <<EOF
stage=14_repeated_sentence_partitions
input=original_order_500Hz_samples_with_fixed_512_window_and_validity_mask
interpolation=none
sentence_partition=sorted_mod5_crossed_test_folds
train_validation_test_subject_overlap=zero
train_validation_test_sentence_overlap=zero
target=$TARGET
model_seed=$SEED
split_seed=$SPLIT_SEED
test_sentence_fold=$TEST_SENTENCE_FOLD
job_id=$SLURM_ARRAY_JOB_ID
task_id=$SLURM_ARRAY_TASK_ID
EOF

python code/models/run_original_time_repeated_sentence_partitions_loso.py \
  --data-dir "$DATA_DIR" \
  --output-dir "$STAGE_DIR/results" \
  --target "$TARGET" \
  --run-tag formal \
  --seed "$SEED" \
  --split-seed "$SPLIT_SEED" \
  --test-sentence-fold "$TEST_SENTENCE_FOLD" \
  --validation-sentence-fraction 0.15 \
  --validation-subjects 3 \
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
  --patience 12

echo "Stage 39 formal task completed: fold=$TEST_SENTENCE_FOLD target=$TARGET seed=$SEED"
find "$STAGE_DIR/results" -maxdepth 1 -type f \
  -name "stage39_formal_${TARGET}_eegnet_seed_${SEED}_sentencefold_${TEST_SENTENCE_FOLD}_*results.json" | sort


