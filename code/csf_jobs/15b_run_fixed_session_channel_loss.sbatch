#!/bin/bash --login
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 8
#SBATCH -t 1-0:00
#SBATCH --mem=64G
#SBATCH --array=0-9%4
#SBATCH -o experiment_outputs/15_fixed_session_channel_loss/logs/formal_%A_%a.out

set -euo pipefail

module purge
module load libs/cuda/12.8.1
module load apps/binapps/conda/miniforge3/25.9.1

TARGETS=(content_function coarse_pos)
MODEL_SEEDS=(42 43 44 45 46)
TARGET_INDEX=$((SLURM_ARRAY_TASK_ID / 5))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % 5))
TARGET="${TARGETS[$TARGET_INDEX]}"
MODEL_SEED="${MODEL_SEEDS[$SEED_INDEX]}"
MASK_SEED=20260826

ENV_DIR="${EEG_ENV_DIR:-$HOME/scratch/eeg_project/conda_envs/eeg-word-reproduction}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/scratch/eeg_project/robust-eeg-word-decoding}"
DATA_DIR="$PROJECT_DIR/data/processed/zuco2_nr_diagnostic_all_fixations_concat_512"
STAGE_DIR="$PROJECT_DIR/experiment_outputs/15_fixed_session_channel_loss"
CHANNEL_LABELS="$PROJECT_DIR/experiment_outputs/09_structured_channel_loss/manifest/processed_channel_labels.txt"

mkdir -p "$STAGE_DIR/logs" "$STAGE_DIR/results" "$STAGE_DIR/analysis" "$STAGE_DIR/scripts" "$STAGE_DIR/manifest"
cp "$PROJECT_DIR/code/models/run_eegnet_fixed_session_channel_loss_all_fixations.py" "$STAGE_DIR/scripts/"
cp "$PROJECT_DIR/code/models/run_eegnet_diagnostic_all_fixations_classification.py" "$STAGE_DIR/scripts/"
cp "$PROJECT_DIR/code/csf_jobs/15b_run_fixed_session_channel_loss.sbatch" "$STAGE_DIR/scripts/"
cp "$CHANNEL_LABELS" "$STAGE_DIR/manifest/processed_channel_labels.txt"

export OMP_NUM_THREADS="${SLURM_NTASKS:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_NTASKS:-1}"
export MKL_NUM_THREADS="${SLURM_NTASKS:-1}"

conda activate "$ENV_DIR"
cd "$PROJECT_DIR"

cat > "$STAGE_DIR/manifest/formal_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt" <<EOF
stage=15_fixed_session_channel_loss
run=formal
purpose=persistent_heldout_subject_channel_loss_sensitivity
input=all_fixations_original_order_500Hz_fixed_512x105_tensor
outer_split=LOSO
inner_validation=complete_subjects
mask_scope=fixed_across_all_test_words_for_held_out_subject
loss_rates=0.0,0.1,0.25,0.5
mask_repeats_per_nonzero_rate=20
target=$TARGET
model_seed=$MODEL_SEED
mask_seed=$MASK_SEED
job_id=$SLURM_ARRAY_JOB_ID
task_id=$SLURM_ARRAY_TASK_ID
EOF

python code/models/run_eegnet_fixed_session_channel_loss_all_fixations.py \
  --data-dir "$DATA_DIR" \
  --output-dir "$STAGE_DIR/results" \
  --channel-labels "$CHANNEL_LABELS" \
  --target "$TARGET" \
  --loss-rates 0.0 0.1 0.25 0.5 \
  --mask-repeats 20 \
  --mask-seed "$MASK_SEED" \
  --output-tag stage40_fixed_session_formal \
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
  --seed "$MODEL_SEED"

echo "Stage 40 formal task completed: target=$TARGET model_seed=$MODEL_SEED"
find "$STAGE_DIR/results" -maxdepth 1 -type f \
  -name "*${TARGET}*stage40_fixed_session_formal*modelseed_${MODEL_SEED}*" | sort


