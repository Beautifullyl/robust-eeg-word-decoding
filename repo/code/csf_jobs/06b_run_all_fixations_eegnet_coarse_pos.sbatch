#!/bin/bash --login
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 8
#SBATCH -t 1-0:00
#SBATCH --mem=64G
#SBATCH --array=0-4
#SBATCH -o experiment_outputs/06_all_fixations_loso_eegnet/logs/coarse_pos_%A_%a.out

set -euo pipefail

module purge
module load libs/cuda/12.8.1
module load apps/binapps/conda/miniforge3/25.9.1

SEEDS=(42 43 44 45 46)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

ENV_DIR="${EEG_ENV_DIR:-$HOME/scratch/eeg_project/conda_envs/eeg-word-reproduction}"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
DATA_DIR="$PROJECT_DIR/data/processed/zuco2_nr_diagnostic_all_fixations_concat_512"
STAGE_DIR="$PROJECT_DIR/experiment_outputs/06_all_fixations_loso_eegnet"
RESULT_DIR="$STAGE_DIR/results"
SCRIPT_DIR="$STAGE_DIR/scripts"
MANIFEST_DIR="$STAGE_DIR/manifest"

mkdir -p "$RESULT_DIR" "$SCRIPT_DIR" "$MANIFEST_DIR"
cp "$PROJECT_DIR/code/models/run_eegnet_diagnostic_all_fixations_classification.py" "$SCRIPT_DIR/" || true
cp "$PROJECT_DIR/code/csf_jobs/06b_run_all_fixations_eegnet_coarse_pos.sbatch" "$SCRIPT_DIR/" || true

export OMP_NUM_THREADS="${SLURM_NTASKS:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_NTASKS:-1}"
export MKL_NUM_THREADS="${SLURM_NTASKS:-1}"

echo "Formal LOSO diagnostic classification"
echo "stage=06_all_fixations_loso_eegnet"
echo "input=all_fixations_concat_len512"
echo "target=coarse_pos"
echo "seed=$SEED"
echo "job_id=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}} task_id=${SLURM_ARRAY_TASK_ID}"
echo "stage_dir=$STAGE_DIR"

cat > "$MANIFEST_DIR/coarse_pos_seed_${SEED}.txt" <<EOF
stage=06_all_fixations_loso_eegnet
target=coarse_pos
seed=$SEED
job_id=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}
task_id=${SLURM_ARRAY_TASK_ID}
project_dir=$PROJECT_DIR
data_dir=$DATA_DIR
result_dir=$RESULT_DIR
EOF

conda activate "$ENV_DIR"
cd "$PROJECT_DIR"

python code/models/run_eegnet_diagnostic_all_fixations_classification.py \
  --data-dir "$DATA_DIR" \
  --output-dir "$RESULT_DIR" \
  --target coarse_pos \
  --output-tag 25bcclean \
  --dropout-rates 0.0 0.1 0.25 0.5 \
  --repeats 5 \
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
  --seed "$SEED"

echo "Result files written to $RESULT_DIR"


