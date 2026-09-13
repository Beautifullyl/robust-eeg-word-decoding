#!/bin/bash --login
#SBATCH -p serial
#SBATCH -n 1
#SBATCH -t 0-1:00
#SBATCH --mem=5G
#SBATCH -o experiment_outputs/11_repeated_sentence_partitions/logs/analyze_%j.out

set -euo pipefail

module purge
module load apps/binapps/conda/miniforge3/25.9.1

ENV_DIR="${EEG_ENV_DIR:-$HOME/scratch/eeg_project/conda_envs/eeg-word-reproduction}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/scratch/eeg_project/robust-eeg-word-decoding}"
STAGE_DIR="$PROJECT_DIR/experiment_outputs/11_repeated_sentence_partitions"

mkdir -p "$STAGE_DIR/logs" "$STAGE_DIR/results" "$STAGE_DIR/scripts" "$STAGE_DIR/manifest" "$STAGE_DIR/analysis"
cp "$PROJECT_DIR/code/analysis/analyse_repeated_sentence_partitions.py" "$STAGE_DIR/scripts/"
cp "$PROJECT_DIR/code/csf_jobs/11c_analyse_repeated_sentence_partitions.sbatch" "$STAGE_DIR/scripts/"

conda activate "$ENV_DIR"
cd "$PROJECT_DIR"

python code/analysis/analyse_repeated_sentence_partitions.py \
  --results-dir "$STAGE_DIR/results" \
  --output-dir "$STAGE_DIR/analysis"

cat > "$STAGE_DIR/manifest/analysis_${SLURM_JOB_ID}.txt" <<EOF
stage=11_repeated_sentence_partitions
purpose=subject_level_aggregation_and_sentence_fold_coverage_audit
expected_targets=content_function,coarse_pos
expected_test_sentence_folds=0,1,2,3,4
expected_model_seeds=42,43,44
EOF

echo "Stage 39 analysis completed"
find "$STAGE_DIR/analysis" -maxdepth 1 -type f | sort


