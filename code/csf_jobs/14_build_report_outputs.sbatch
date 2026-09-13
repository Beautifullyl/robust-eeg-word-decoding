#!/bin/bash --login
#SBATCH -p serial
#SBATCH -n 1
#SBATCH -t 0-2:00
#SBATCH --mem=5G
#SBATCH -o experiment_outputs/14_report_outputs/logs/build_%j.out

set -euo pipefail

module purge
module load apps/binapps/conda/miniforge3/25.9.1

ENV_DIR="${EEG_ENV_DIR:-$HOME/scratch/eeg_project/conda_envs/eeg-word-reproduction}"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
STAGE_DIR="$PROJECT_DIR/experiment_outputs/14_report_outputs"

mkdir -p "$STAGE_DIR/logs" "$STAGE_DIR/results" "$STAGE_DIR/analysis" "$STAGE_DIR/scripts" "$STAGE_DIR/manifest"
cp "$PROJECT_DIR/code/analysis/build_report_outputs.py" "$STAGE_DIR/scripts/"
cp "$PROJECT_DIR/code/csf_jobs/14_build_report_outputs.sbatch" "$STAGE_DIR/scripts/"

conda activate "$ENV_DIR"
cd "$PROJECT_DIR"

python code/analysis/build_report_outputs.py
python code/verify_repository.py

cat > "$STAGE_DIR/manifest/build_${SLURM_JOB_ID}.txt" <<EOF
stage=14_report_outputs
purpose=assemble_report_tables_figures_and_repository_check
results=$PROJECT_DIR/results/final_subject_level
figures=$PROJECT_DIR/results/figures
EOF

find "$PROJECT_DIR/results/final_subject_level" -maxdepth 1 -type f | sort
find "$PROJECT_DIR/results/figures" -maxdepth 1 -type f | sort
