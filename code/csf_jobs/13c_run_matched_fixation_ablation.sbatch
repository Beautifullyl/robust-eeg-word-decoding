#!/bin/bash --login
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 8
#SBATCH -t 1-0:00
#SBATCH --mem=64G
#SBATCH --array=0-19%4
#SBATCH -o experiment_outputs/13_matched_fixation_ablation/logs/formal_%A_%a.out

set -euo pipefail

module purge
module load libs/cuda/12.8.1
module load apps/binapps/conda/miniforge3/25.9.1

REPRESENTATIONS=(first_fixation all_fixations)
TARGETS=(content_function coarse_pos)
MODEL_SEEDS=(42 43 44 45 46)
REPRESENTATION_INDEX=$((SLURM_ARRAY_TASK_ID / 10))
WITHIN_REPRESENTATION=$((SLURM_ARRAY_TASK_ID % 10))
TARGET_INDEX=$((WITHIN_REPRESENTATION / 5))
SEED_INDEX=$((WITHIN_REPRESENTATION % 5))
REPRESENTATION="${REPRESENTATIONS[$REPRESENTATION_INDEX]}"
TARGET="${TARGETS[$TARGET_INDEX]}"
MODEL_SEED="${MODEL_SEEDS[$SEED_INDEX]}"

ENV_DIR="${EEG_ENV_DIR:-$HOME/scratch/eeg_project/conda_envs/eeg-word-reproduction}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/scratch/eeg_project/robust-eeg-word-decoding}"
FIRST_DIR="$PROJECT_DIR/data/processed/zuco2_nr_diagnostic_first_fixation_512"
ALL_DIR="$PROJECT_DIR/data/processed/zuco2_nr_diagnostic_all_fixations_concat_512"
STAGE_DIR="$PROJECT_DIR/experiment_outputs/13_matched_fixation_ablation"

mkdir -p "$STAGE_DIR/logs" "$STAGE_DIR/results" "$STAGE_DIR/analysis" "$STAGE_DIR/scripts" "$STAGE_DIR/manifest"
cp "$PROJECT_DIR/code/models/run_eegnet_matched_fixation_ablation.py" "$STAGE_DIR/scripts/"
cp "$PROJECT_DIR/code/models/run_eegnet_diagnostic_all_fixations_classification.py" "$STAGE_DIR/scripts/"
cp "$PROJECT_DIR/code/csf_jobs/13c_run_matched_fixation_ablation.sbatch" "$STAGE_DIR/scripts/"

export OMP_NUM_THREADS="${SLURM_NTASKS:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_NTASKS:-1}"
export MKL_NUM_THREADS="${SLURM_NTASKS:-1}"

conda activate "$ENV_DIR"
cd "$PROJECT_DIR"

cat > "$STAGE_DIR/manifest/formal_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt" <<EOF
stage=13_matched_fixation_ablation
run=formal
representation=$REPRESENTATION
target=$TARGET
model_seed=$MODEL_SEED
tensor_shape=512x105
outer_split=LOSO
inner_validation=complete_subjects
interpolation=none
job_id=$SLURM_ARRAY_JOB_ID
task_id=$SLURM_ARRAY_TASK_ID
EOF

python code/models/run_eegnet_matched_fixation_ablation.py \
  --first-fixation-dir "$FIRST_DIR" \
  --all-fixations-dir "$ALL_DIR" \
  --output-dir "$STAGE_DIR/results" \
  --representation "$REPRESENTATION" \
  --target "$TARGET" \
  --output-tag stage41_matched_fixation_formal \
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
  --seed "$MODEL_SEED"

echo "Stage 41 formal task completed: representation=$REPRESENTATION target=$TARGET seed=$MODEL_SEED"


