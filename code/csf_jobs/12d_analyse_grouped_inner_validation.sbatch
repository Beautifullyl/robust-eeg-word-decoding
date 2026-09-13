#!/bin/bash --login
#SBATCH -p serial
#SBATCH -n 1
#SBATCH -t 0-1:00
#SBATCH --mem=5G
#SBATCH -o experiment_outputs/12_grouped_inner_validation/logs/analyze_%j.out

set -euo pipefail

module purge
module load apps/binapps/conda/miniforge3/25.9.1

ENV_DIR="${EEG_ENV_DIR:-$HOME/scratch/eeg_project/conda_envs/eeg-word-reproduction}"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
STAGE_DIR="$PROJECT_DIR/experiment_outputs/12_grouped_inner_validation"
ROW_VALIDATION="$PROJECT_DIR/experiment_outputs/06_all_fixations_loso_eegnet"

mkdir -p "$STAGE_DIR/logs" "$STAGE_DIR/analysis" "$STAGE_DIR/scripts" "$STAGE_DIR/manifest"
cp "$PROJECT_DIR/code/analysis/analyse_grouped_inner_validation.py" "$STAGE_DIR/scripts/"
cp "$PROJECT_DIR/code/csf_jobs/12d_analyse_grouped_inner_validation.sbatch" "$STAGE_DIR/scripts/"

conda activate "$ENV_DIR"
cd "$PROJECT_DIR"

python code/analysis/analyse_grouped_inner_validation.py \
  --stage37 "$STAGE_DIR" \
  --row-validation "$ROW_VALIDATION" \
  --output-dir "$STAGE_DIR/analysis"

cat > "$STAGE_DIR/manifest/analysis_${SLURM_JOB_ID}.txt" <<EOF
stage=12_grouped_inner_validation
purpose=compare_subject_grouped_and_row_level_inner_validation
grouped_results=$STAGE_DIR/results
row_validation_results=$ROW_VALIDATION/results
EOF

find "$STAGE_DIR/analysis" -maxdepth 1 -type f | sort
