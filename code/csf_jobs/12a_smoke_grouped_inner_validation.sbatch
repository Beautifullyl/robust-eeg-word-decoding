#!/bin/bash --login
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -n 8
#SBATCH -t 0-3:00
#SBATCH --mem=64G
#SBATCH -o experiment_outputs/12_grouped_inner_validation/logs/smoke_%j.out

set -euo pipefail

module purge
module load libs/cuda/12.8.1
module load apps/binapps/conda/miniforge3/25.9.1

ENV_DIR="${EEG_ENV_DIR:-$HOME/scratch/eeg_project/conda_envs/eeg-word-reproduction}"
PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
DATA_DIR="$PROJECT_DIR/data/processed/zuco2_nr_diagnostic_all_fixations_concat_512"
STAGE_DIR="$PROJECT_DIR/experiment_outputs/12_grouped_inner_validation"

mkdir -p "$STAGE_DIR/logs" "$STAGE_DIR/results" "$STAGE_DIR/scripts" "$STAGE_DIR/manifest"
cp "$PROJECT_DIR/code/models/run_eegnet_diagnostic_all_fixations_classification.py" "$STAGE_DIR/scripts/"
cp "$PROJECT_DIR/code/csf_jobs/12a_smoke_grouped_inner_validation.sbatch" "$STAGE_DIR/scripts/"

export OMP_NUM_THREADS="${SLURM_NTASKS:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_NTASKS:-1}"
export MKL_NUM_THREADS="${SLURM_NTASKS:-1}"

conda activate "$ENV_DIR"
cd "$PROJECT_DIR"

cat > "$STAGE_DIR/manifest/smoke_${SLURM_JOB_ID}.txt" <<EOF
stage=12_grouped_inner_validation
purpose=code_and_data_contract_smoke_test
input=original_order_500Hz_samples_with_fixed_512_window_and_validity_mask
interpolation=none
outer_split=LOSO
inner_validation=subject
target=content_function
seed=42
max_epochs=2
EOF

python code/models/run_eegnet_diagnostic_all_fixations_classification.py \
  --data-dir "$DATA_DIR" \
  --output-dir "$STAGE_DIR/results" \
  --target content_function \
  --inner-val-mode subject \
  --output-tag stage37_smoke_grouped_inner \
  --dropout-rates 0.0 \
  --repeats 1 \
  --max-epochs 2 \
  --patience 1 \
  --seed 42

echo "Stage 37 smoke test completed"
find "$STAGE_DIR" -maxdepth 3 -type f | sort


