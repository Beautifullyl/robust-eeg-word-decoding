#!/bin/bash --login
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 8
#SBATCH -t 1-0:00
#SBATCH --mem=64G
#SBATCH --array=0-4
#SBATCH -o experiment_outputs/07_structured_channel_loss/logs/content_function_%A_%a.out

set -euo pipefail

module purge
module load libs/cuda/12.8.1
module load apps/binapps/conda/miniforge3/25.9.1

SEEDS=(42 43 44 45 46)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

ENV_DIR="${EEG_ENV_DIR:-$HOME/scratch/eeg_project/conda_envs/eeg-word-reproduction}"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
DATA_DIR="$PROJECT_DIR/data/processed/zuco2_nr_diagnostic_all_fixations_concat_512"
STAGE_DIR="$PROJECT_DIR/experiment_outputs/07_structured_channel_loss"
RESULT_DIR="$STAGE_DIR/results/content_function"
SCRIPT_DIR="$STAGE_DIR/scripts"
MANIFEST_DIR="$STAGE_DIR/manifest"
CHANNEL_LABELS="$MANIFEST_DIR/processed_channel_labels.txt"

mkdir -p "$RESULT_DIR" "$SCRIPT_DIR" "$MANIFEST_DIR" "$STAGE_DIR/logs"
cp "$PROJECT_DIR/code/models/run_eegnet_official_pair_block_dropout_all_fixations.py" "$SCRIPT_DIR/" || true
cp "$PROJECT_DIR/code/models/run_eegnet_diagnostic_all_fixations_classification.py" "$SCRIPT_DIR/" || true
cp "$PROJECT_DIR/code/csf_jobs/07b_run_structured_channel_loss_content_function.sbatch" "$SCRIPT_DIR/" || true

if [ ! -f "$CHANNEL_LABELS" ]; then
  echo "Missing channel labels: $CHANNEL_LABELS"
  echo "Run: sbatch code/csf_jobs/07a_export_channel_labels.sbatch"
  exit 1
fi

export OMP_NUM_THREADS="${SLURM_NTASKS:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_NTASKS:-1}"
export MKL_NUM_THREADS="${SLURM_NTASKS:-1}"

echo "Formal Stage 31 official-pair block dropout"
echo "stage=07_structured_channel_loss"
echo "input=all_fixations_concat_len512"
echo "target=content_function"
echo "seed=$SEED"
echo "job_id=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}} task_id=${SLURM_ARRAY_TASK_ID}"
echo "stage_dir=$STAGE_DIR"

cat > "$MANIFEST_DIR/content_function_seed_${SEED}.txt" <<EOF
stage=07_structured_channel_loss
target=content_function
seed=$SEED
blocks=none official_frontal_bilateral_9pairs official_frontal_left_9pairs official_frontal_right_9pairs official_left_homologue_set official_right_homologue_set
random_repeats=5
job_id=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}
task_id=${SLURM_ARRAY_TASK_ID}
project_dir=$PROJECT_DIR
data_dir=$DATA_DIR
result_dir=$RESULT_DIR
channel_labels=$CHANNEL_LABELS
EOF

conda activate "$ENV_DIR"
cd "$PROJECT_DIR"

python code/models/run_eegnet_official_pair_block_dropout_all_fixations.py \
  --data-dir "$DATA_DIR" \
  --output-dir "$RESULT_DIR" \
  --channel-labels "$CHANNEL_LABELS" \
  --target content_function \
  --random-repeats 5 \
  --output-tag stage31 \
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

echo "Stage 31 content/function result files:"
find "$STAGE_DIR" -maxdepth 4 -type f | sort


