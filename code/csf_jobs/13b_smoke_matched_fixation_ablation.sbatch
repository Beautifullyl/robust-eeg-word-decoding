#!/bin/bash --login
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 8
#SBATCH -t 0-3:00
#SBATCH --mem=64G
#SBATCH -o experiment_outputs/13_matched_fixation_ablation/logs/smoke_%j.out

set -euo pipefail

module purge
module load libs/cuda/12.8.1
module load apps/binapps/conda/miniforge3/25.9.1

ENV_DIR="${EEG_ENV_DIR:-$HOME/scratch/eeg_project/conda_envs/eeg-word-reproduction}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/scratch/eeg_project/robust-eeg-word-decoding}"
FIRST_DIR="$PROJECT_DIR/data/processed/zuco2_nr_diagnostic_first_fixation_512"
ALL_DIR="$PROJECT_DIR/data/processed/zuco2_nr_diagnostic_all_fixations_concat_512"
STAGE_DIR="$PROJECT_DIR/experiment_outputs/13_matched_fixation_ablation"

mkdir -p "$STAGE_DIR/logs" "$STAGE_DIR/results" "$STAGE_DIR/analysis" "$STAGE_DIR/scripts" "$STAGE_DIR/manifest"
cp "$PROJECT_DIR/code/models/run_eegnet_matched_fixation_ablation.py" "$STAGE_DIR/scripts/"
cp "$PROJECT_DIR/code/models/run_eegnet_diagnostic_all_fixations_classification.py" "$STAGE_DIR/scripts/"
cp "$PROJECT_DIR/code/csf_jobs/13b_smoke_matched_fixation_ablation.sbatch" "$STAGE_DIR/scripts/"

export OMP_NUM_THREADS="${SLURM_NTASKS:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_NTASKS:-1}"
export MKL_NUM_THREADS="${SLURM_NTASKS:-1}"

conda activate "$ENV_DIR"
cd "$PROJECT_DIR"

for REPRESENTATION in first_fixation all_fixations; do
  python code/models/run_eegnet_matched_fixation_ablation.py \
    --first-fixation-dir "$FIRST_DIR" \
    --all-fixations-dir "$ALL_DIR" \
    --output-dir "$STAGE_DIR/results" \
    --representation "$REPRESENTATION" \
    --target content_function \
    --output-tag stage41_matched_fixation_smoke \
    --f1 8 \
    --depth-multiplier 2 \
    --f2 16 \
    --temporal-kernel 31 \
    --separable-kernel 15 \
    --model-dropout 0.25 \
    --lr 0.001 \
    --weight-decay 0.0001 \
    --batch-size 128 \
    --max-epochs 2 \
    --patience 1 \
    --seed 42 \
    --max-subjects 1
done

cat > "$STAGE_DIR/manifest/smoke_${SLURM_JOB_ID}.txt" <<EOF
stage=13_matched_fixation_ablation
run=smoke
representations=first_fixation_512,all_fixations_512
target=content_function
model_seed=42
max_subjects=1
max_epochs=2
interpolation=none
EOF

echo "Stage 41 matched-fixation smoke test completed"
find "$STAGE_DIR/results" -maxdepth 1 -type f -name "*stage41_matched_fixation_smoke*" | sort


