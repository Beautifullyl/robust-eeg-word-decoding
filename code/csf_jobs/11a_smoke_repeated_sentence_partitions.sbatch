#!/bin/bash --login
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 8
#SBATCH -t 0-3:00
#SBATCH --mem=64G
#SBATCH -o experiment_outputs/11_repeated_sentence_partitions/logs/smoke_%j.out

set -euo pipefail

module purge
module load libs/cuda/12.8.1
module load apps/binapps/conda/miniforge3/25.9.1

ENV_DIR="${EEG_ENV_DIR:-$HOME/scratch/eeg_project/conda_envs/eeg-word-reproduction}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/scratch/eeg_project/robust-eeg-word-decoding}"
DATA_DIR="$PROJECT_DIR/data/processed/zuco2_nr_diagnostic_all_fixations_concat_512"
STAGE_DIR="$PROJECT_DIR/experiment_outputs/11_repeated_sentence_partitions"
TEST_SENTENCE_FOLD=1
SPLIT_SEED=20260822

mkdir -p "$STAGE_DIR/logs" "$STAGE_DIR/results" "$STAGE_DIR/scripts" "$STAGE_DIR/manifest" "$STAGE_DIR/analysis"
cp "$PROJECT_DIR/code/models/run_original_time_subject_sentence_disjoint_loso.py" "$STAGE_DIR/scripts/"
cp "$PROJECT_DIR/code/models/run_original_time_repeated_sentence_partitions_loso.py" "$STAGE_DIR/scripts/"
cp "$PROJECT_DIR/code/models/run_eegnet_diagnostic_all_fixations_classification.py" "$STAGE_DIR/scripts/"
cp "$PROJECT_DIR/code/csf_jobs/11a_smoke_repeated_sentence_partitions.sbatch" "$STAGE_DIR/scripts/"

export OMP_NUM_THREADS="${SLURM_NTASKS:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_NTASKS:-1}"
export MKL_NUM_THREADS="${SLURM_NTASKS:-1}"

conda activate "$ENV_DIR"
cd "$PROJECT_DIR"

cat > "$STAGE_DIR/manifest/smoke_${SLURM_JOB_ID}.txt" <<EOF
stage=11_repeated_sentence_partitions
purpose=nonzero_test_sentence_fold_smoke_test
input=original_order_500Hz_samples_with_fixed_512_window_and_validity_mask
interpolation=none
train_validation_test_subject_overlap=zero
train_validation_test_sentence_overlap=zero
target=content_function
model_seed=42
split_seed=$SPLIT_SEED
test_sentence_fold=$TEST_SENTENCE_FOLD
max_epochs=2
EOF

python code/models/run_original_time_repeated_sentence_partitions_loso.py \
  --data-dir "$DATA_DIR" \
  --output-dir "$STAGE_DIR/results" \
  --target content_function \
  --run-tag smoke \
  --seed 42 \
  --split-seed "$SPLIT_SEED" \
  --test-sentence-fold "$TEST_SENTENCE_FOLD" \
  --validation-sentence-fraction 0.15 \
  --validation-subjects 3 \
  --max-epochs 2 \
  --patience 1

echo "Stage 39 smoke test completed"
find "$STAGE_DIR" -maxdepth 3 -type f | sort


