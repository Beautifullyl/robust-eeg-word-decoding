#!/bin/bash --login
#SBATCH -p serial
#SBATCH -n 1
#SBATCH -t 0-1:00
#SBATCH --mem=5G
#SBATCH -o experiment_outputs/16_matched_fixation_ablation/logs/analyze_%j.out

set -euo pipefail

module purge
module load apps/binapps/conda/miniforge3/25.9.1

ENV_DIR="${EEG_ENV_DIR:-$HOME/scratch/eeg_project/conda_envs/eeg-word-reproduction}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/scratch/eeg_project/robust-eeg-word-decoding}"
STAGE_DIR="$PROJECT_DIR/experiment_outputs/16_matched_fixation_ablation"

mkdir -p "$STAGE_DIR/logs" "$STAGE_DIR/results" "$STAGE_DIR/analysis" "$STAGE_DIR/scripts" "$STAGE_DIR/manifest"
cp "$PROJECT_DIR/code/analysis/analyse_matched_fixation_ablation.py" "$STAGE_DIR/scripts/"
cp "$PROJECT_DIR/code/csf_jobs/16d_analyse_matched_fixation_ablation.sbatch" "$STAGE_DIR/scripts/"

conda activate "$ENV_DIR"
cd "$PROJECT_DIR"

python code/analysis/analyse_matched_fixation_ablation.py \
  --results-dir "$STAGE_DIR/results" \
  --output-dir "$STAGE_DIR/analysis" \
  --bootstrap-samples 20000 \
  --bootstrap-seed 20260826

cat > "$STAGE_DIR/manifest/analysis_${SLURM_JOB_ID}.txt" <<EOF
stage=16_matched_fixation_ablation
purpose=paired_shape_and_capacity_matched_first_vs_all_fixations
independent_unit=heldout_subject
model_seeds=42,43,44,45,46
holm_family=2_balanced_accuracy_representation_comparisons
EOF

echo "Stage 41 analysis completed"
find "$STAGE_DIR/analysis" -maxdepth 1 -type f | sort


