#!/bin/bash --login
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 8
#SBATCH -t 1-0:00
#SBATCH --mem=64G
#SBATCH --array=0-4
#SBATCH -o experiment_outputs/05_all_fixations_loso_tcn/logs/content_function_%A_%a.out

set -euo pipefail

module purge
module load libs/cuda/12.8.1
module load apps/binapps/conda/miniforge3/25.9.1

SEEDS=(42 43 44 45 46)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

ENV_DIR="${EEG_ENV_DIR:-$HOME/scratch/eeg_project/conda_envs/eeg-word-reproduction}"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
DATA_DIR="$PROJECT_DIR/data/processed/zuco2_nr_diagnostic_all_fixations_concat_512"
STAGE_DIR="$PROJECT_DIR/experiment_outputs/05_all_fixations_loso_tcn"
RESULT_DIR="$STAGE_DIR/results"
SCRIPT_DIR="$STAGE_DIR/scripts"
MANIFEST_DIR="$STAGE_DIR/manifest"

mkdir -p "$RESULT_DIR" "$SCRIPT_DIR" "$MANIFEST_DIR" "$STAGE_DIR/logs"
cp "$PROJECT_DIR/code/models/run_tcn_diagnostic_all_fixations_classification.py" "$SCRIPT_DIR/" || true
cp "$PROJECT_DIR/code/models/run_eegnet_diagnostic_all_fixations_classification.py" "$SCRIPT_DIR/" || true
cp "$PROJECT_DIR/code/csf_jobs/05a_run_tcn_content_function.sbatch" "$SCRIPT_DIR/" || true

export OMP_NUM_THREADS="${SLURM_NTASKS:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_NTASKS:-1}"
export MKL_NUM_THREADS="${SLURM_NTASKS:-1}"

echo "Formal Stage 27 all-fixations LOSO TCN"
echo "stage=05_all_fixations_loso_tcn"
echo "input=all_fixations_concat_len512"
echo "target=content_function"
echo "seed=$SEED"
echo "job_id=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}} task_id=${SLURM_ARRAY_TASK_ID}"
echo "stage_dir=$STAGE_DIR"

cat > "$MANIFEST_DIR/content_function_seed_${SEED}.txt" <<EOF
stage=05_all_fixations_loso_tcn
target=content_function
seed=$SEED
job_id=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}
task_id=${SLURM_ARRAY_TASK_ID}
project_dir=$PROJECT_DIR
data_dir=$DATA_DIR
result_dir=$RESULT_DIR
EOF

conda activate "$ENV_DIR"
cd "$PROJECT_DIR"

python code/models/run_tcn_diagnostic_all_fixations_classification.py \
  --data-dir "$DATA_DIR" \
  --output-dir "$RESULT_DIR" \
  --target content_function \
  --output-tag stage27 \
  --dropout-rates 0.0 0.1 0.25 0.5 \
  --repeats 5 \
  --hidden-channels 64 \
  --levels 4 \
  --kernel-size 7 \
  --model-dropout 0.25 \
  --lr 0.001 \
  --weight-decay 0.0001 \
  --batch-size 128 \
  --max-epochs 100 \
  --patience 12 \
  --seed "$SEED"

echo "Stage 27 content/function result files:"
find "$STAGE_DIR" -maxdepth 3 -type f | sort


