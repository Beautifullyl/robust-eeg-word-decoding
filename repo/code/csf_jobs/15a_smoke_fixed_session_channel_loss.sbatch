#!/bin/bash --login
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 8
#SBATCH -t 0-2:00
#SBATCH --mem=64G
#SBATCH -o experiment_outputs/15_fixed_session_channel_loss/logs/smoke_%j.out

set -euo pipefail

module purge
module load libs/cuda/12.8.1
module load apps/binapps/conda/miniforge3/25.9.1

ENV_DIR="${EEG_ENV_DIR:-$HOME/scratch/eeg_project/conda_envs/eeg-word-reproduction}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/scratch/eeg_project/robust-eeg-word-decoding}"
DATA_DIR="$PROJECT_DIR/data/processed/zuco2_nr_diagnostic_all_fixations_concat_512"
STAGE_DIR="$PROJECT_DIR/experiment_outputs/15_fixed_session_channel_loss"
CHANNEL_LABELS="$PROJECT_DIR/experiment_outputs/09_structured_channel_loss/manifest/processed_channel_labels.txt"

mkdir -p "$STAGE_DIR/logs" "$STAGE_DIR/results" "$STAGE_DIR/analysis" "$STAGE_DIR/scripts" "$STAGE_DIR/manifest"
cp "$PROJECT_DIR/code/models/run_eegnet_fixed_session_channel_loss_all_fixations.py" "$STAGE_DIR/scripts/"
cp "$PROJECT_DIR/code/models/run_eegnet_diagnostic_all_fixations_classification.py" "$STAGE_DIR/scripts/"
cp "$PROJECT_DIR/code/csf_jobs/15a_smoke_fixed_session_channel_loss.sbatch" "$STAGE_DIR/scripts/"
cp "$CHANNEL_LABELS" "$STAGE_DIR/manifest/processed_channel_labels.txt"

export OMP_NUM_THREADS="${SLURM_NTASKS:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_NTASKS:-1}"
export MKL_NUM_THREADS="${SLURM_NTASKS:-1}"

conda activate "$ENV_DIR"
cd "$PROJECT_DIR"

cat > "$STAGE_DIR/manifest/smoke_${SLURM_JOB_ID}.txt" <<EOF
stage=15_fixed_session_channel_loss
run=smoke
purpose=verify_persistent_subject_level_test_masking_and_output_schema
target=content_function
model_seed=42
mask_seed=20260826
loss_rates=0.0,0.25
mask_repeats=2
max_subjects=1
max_epochs=2
inner_validation=complete_subjects
mask_scope=fixed_across_all_test_words_for_held_out_subject
EOF

echo "mask_scope=fixed_across_all_test_words_for_held_out_subject"

python code/models/run_eegnet_fixed_session_channel_loss_all_fixations.py \
  --data-dir "$DATA_DIR" \
  --output-dir "$STAGE_DIR/results" \
  --channel-labels "$CHANNEL_LABELS" \
  --target content_function \
  --loss-rates 0.0 0.25 \
  --mask-repeats 2 \
  --mask-seed 20260826 \
  --output-tag stage40_fixed_session_smoke \
  --f1 8 \
  --depth-multiplier 2 \
  --f2 16 \
  --temporal-kernel 31 \
  --separable-kernel 15 \
  --model-dropout 0.25 \
  --lr 0.001 \
  --weight-decay 0.0001 \
  --batch-size 128 \
  --max-epochs 2 \
  --patience 1 \
  --seed 42 \
  --max-subjects 1

echo "Stage 40 smoke test completed"
find "$STAGE_DIR/results" -maxdepth 1 -type f -name "*stage40_fixed_session_smoke*" | sort


