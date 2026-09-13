#!/bin/bash --login
#SBATCH -p multicore
#SBATCH -n 4
#SBATCH -t 0-12:00
#SBATCH --mem=32G
#SBATCH -o experiment_outputs/13_matched_fixation_ablation/logs/extract_%j.out

set -euo pipefail

module purge
module load apps/binapps/conda/miniforge3/25.9.1

ENV_DIR="${EEG_ENV_DIR:-$HOME/scratch/eeg_project/conda_envs/eeg-word-reproduction}"
PROJECT_DIR="${PROJECT_DIR:-$HOME/scratch/eeg_project/robust-eeg-word-decoding}"
ZUCO2_ROOT="${ZUCO2_ROOT:-$HOME/scratch/eeg_project/datasets/zuco2_zip/zuco2}"
MAT_DIR="${ZUCO2_MAT_DIR:-$ZUCO2_ROOT/task1 - NR/Matlab files}"
OUT_DIR="$PROJECT_DIR/data/processed/zuco2_nr_diagnostic_first_fixation_512"
STAGE_DIR="$PROJECT_DIR/experiment_outputs/13_matched_fixation_ablation"

mkdir -p "$STAGE_DIR/logs" "$STAGE_DIR/results" "$STAGE_DIR/analysis" "$STAGE_DIR/scripts" "$STAGE_DIR/manifest"
cp "$PROJECT_DIR/code/data_preparation/extract_zuco2_diagnostic_first_fixation_timeseries.py" "$STAGE_DIR/scripts/"
cp "$PROJECT_DIR/code/csf_jobs/13a_extract_first_fixation_512.sbatch" "$STAGE_DIR/scripts/"

if [[ -e "$OUT_DIR/X_first_fixation_len512.npy" || -e "$OUT_DIR/mask_first_fixation_len512.npy" ]]; then
  echo "Refusing to overwrite existing Stage 41 extraction in $OUT_DIR" >&2
  exit 2
fi

export OMP_NUM_THREADS="${SLURM_NTASKS:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_NTASKS:-1}"
export MKL_NUM_THREADS="${SLURM_NTASKS:-1}"
export NLTK_DATA="$PROJECT_DIR/.nltk_data"

conda activate "$ENV_DIR"
cd "$PROJECT_DIR"

python code/data_preparation/extract_zuco2_diagnostic_first_fixation_timeseries.py \
  --mat-dir "$MAT_DIR" \
  --out-dir "$OUT_DIR" \
  --length 512 \
  --channels 105

python - "$OUT_DIR" <<'PY'
import csv
import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
with (root / "metadata.csv").open("r", encoding="utf-8", newline="") as handle:
    n_rows = sum(1 for _ in csv.DictReader(handle))
x = np.load(root / "X_first_fixation_len512.npy", mmap_mode="r")
mask = np.load(root / "mask_first_fixation_len512.npy", mmap_mode="r")
audit = {
    "status": "passed",
    "metadata_rows": n_rows,
    "x_shape": list(x.shape),
    "mask_shape": list(mask.shape),
    "x_dtype": str(x.dtype),
    "mask_dtype": str(mask.dtype),
    "interpolation": "none",
}
if x.shape != (n_rows, 512, 105) or mask.shape != (n_rows, 512):
    raise SystemExit(f"Stage 41 extraction shape check failed: {audit}")
(root / "stage41_extraction_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
print(json.dumps(audit, indent=2))
PY

cp "$OUT_DIR/summary.json" "$STAGE_DIR/manifest/first_fixation_512_summary.json"
cp "$OUT_DIR/stage41_extraction_audit.json" "$STAGE_DIR/manifest/first_fixation_512_audit.json"
echo "Stage 41 first-fixation 512 extraction completed"
ls -lh "$OUT_DIR"


