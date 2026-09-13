#!/bin/bash --login
#SBATCH -p serial
#SBATCH -n 1
#SBATCH -t 0-1:00
#SBATCH --mem=5G
#SBATCH -o experiment_outputs/15_fixed_session_channel_loss/logs/analyze_%j.out

set -euo pipefail

module purge
module load apps/binapps/conda/miniforge3/25.9.1

ENV_DIR="${EEG_ENV_DIR:-$HOME/scratch/eeg_project/conda_envs/eeg-word-reproduction}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/scratch/eeg_project/robust-eeg-word-decoding}"
STAGE_DIR="$PROJECT_DIR/experiment_outputs/15_fixed_session_channel_loss"

mkdir -p "$STAGE_DIR/logs" "$STAGE_DIR/results" "$STAGE_DIR/analysis" "$STAGE_DIR/scripts" "$STAGE_DIR/manifest"
cp "$PROJECT_DIR/code/analysis/analyse_fixed_session_channel_loss.py" "$STAGE_DIR/scripts/"
cp "$PROJECT_DIR/code/csf_jobs/15c_analyse_fixed_session_channel_loss.sbatch" "$STAGE_DIR/scripts/"

conda activate "$ENV_DIR"
cd "$PROJECT_DIR"

python code/analysis/analyse_fixed_session_channel_loss.py \
  --results-dir "$STAGE_DIR/results" \
  --output-dir "$STAGE_DIR/analysis" \
  --bootstrap-samples 20000 \
  --bootstrap-seed 20260826

cat > "$STAGE_DIR/manifest/analysis_${SLURM_JOB_ID}.txt" <<EOF
stage=15_fixed_session_channel_loss
purpose=aggregate_fixed_masks_and_model_seeds_at_heldout_subject_level
expected_targets=content_function,coarse_pos
expected_model_seeds=42,43,44,45,46
expected_loss_rates=0.0,0.1,0.25,0.5
holm_family=6_nonzero_loss_vs_no_loss_comparisons
EOF

echo "Stage 40 analysis completed"
find "$STAGE_DIR/analysis" -maxdepth 1 -type f | sort


