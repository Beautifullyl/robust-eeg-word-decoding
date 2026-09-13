#!/bin/bash --login
#SBATCH -p serial
#SBATCH -n 1
#SBATCH -t 0-2:00
#SBATCH --mem=4G
#SBATCH --array=0-1
#SBATCH -o experiment_outputs/08_core_baselines_and_diagnostics/logs/per_class_%A_%a.out

set -euo pipefail

module purge
module load apps/binapps/conda/miniforge3/25.9.1

TARGETS=(content_function coarse_pos)
TARGET="${TARGETS[$SLURM_ARRAY_TASK_ID]}"

ENV_DIR="${EEG_ENV_DIR:-$HOME/scratch/eeg_project/conda_envs/eeg-word-reproduction}"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
SOURCE_RESULT_DIR="$PROJECT_DIR/experiment_outputs/06_all_fixations_loso_eegnet/results"
STAGE_DIR="$PROJECT_DIR/experiment_outputs/08_core_baselines_and_diagnostics"
RESULT_DIR="$STAGE_DIR/results/per_class"
SCRIPT_DIR="$STAGE_DIR/scripts"
MANIFEST_DIR="$STAGE_DIR/manifest"

mkdir -p "$RESULT_DIR" "$SCRIPT_DIR" "$MANIFEST_DIR" "$STAGE_DIR/logs"
cp "$PROJECT_DIR/code/analysis/analyse_multiseed_per_class.py" "$SCRIPT_DIR/" || true
cp "$PROJECT_DIR/code/csf_jobs/08c_analyse_eegnet_per_class.sbatch" "$SCRIPT_DIR/" || true

echo "Stage 28c all-fixations EEGNet per-class analysis"
echo "stage=08_core_baselines_and_diagnostics"
echo "source_stage=06_all_fixations_loso_eegnet"
echo "target=$TARGET"
echo "source_result_dir=$SOURCE_RESULT_DIR"
echo "result_dir=$RESULT_DIR"

cat > "$MANIFEST_DIR/per_class_eegnet_loso_${TARGET}.txt" <<EOF
stage=08_core_baselines_and_diagnostics
substage=28c_allfix_eegnet_loso_per_class
model=eegnet
split=loso
target=$TARGET
source_result_dir=$SOURCE_RESULT_DIR
result_dir=$RESULT_DIR
EOF

conda activate "$ENV_DIR"
cd "$PROJECT_DIR"

python code/analysis/analyse_multiseed_per_class.py \
  --result-dir "$SOURCE_RESULT_DIR" \
  --output-dir "$RESULT_DIR" \
  --target "$TARGET" \
  --model-name eegnet_allfix_loso \
  --stage-tag stage28c

echo "Stage 28c result files:"
find "$STAGE_DIR" -maxdepth 4 -type f | sort


