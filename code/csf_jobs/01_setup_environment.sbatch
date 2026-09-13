#!/bin/bash --login
#SBATCH -p serial
#SBATCH -t 0-2:00
#SBATCH --mem=4G
#SBATCH -o experiment_outputs/01_environment_setup/logs/setup_%j.out

set -euo pipefail

module purge
module load apps/binapps/conda/miniforge3/25.9.1

PROJECT_DIR="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
ENV_DIR="${EEG_ENV_DIR:-$HOME/scratch/eeg_project/conda_envs/eeg-word-reproduction}"
STAGE_DIR="$PROJECT_DIR/experiment_outputs/01_environment_setup"
MANIFEST_DIR="$STAGE_DIR/manifest"
PKG_DIR="${EEG_CONDA_PKGS_DIR:-$HOME/scratch/eeg_project/conda_pkgs}"

mkdir -p "$MANIFEST_DIR" "$STAGE_DIR/logs" "$(dirname "$ENV_DIR")" "$PKG_DIR" "$PROJECT_DIR/.nltk_data"
export CONDA_PKGS_DIRS="$PKG_DIR"

if [ ! -d "$ENV_DIR" ]; then
  conda env create -p "$ENV_DIR" -f "$PROJECT_DIR/config/environment.yml"
else
  conda env update -p "$ENV_DIR" -f "$PROJECT_DIR/config/environment.yml" --prune
fi

conda activate "$ENV_DIR"
python -m nltk.downloader -d "$PROJECT_DIR/.nltk_data" -q averaged_perceptron_tagger_eng

conda list --explicit > "$MANIFEST_DIR/conda-explicit.txt"
python -m pip freeze > "$MANIFEST_DIR/pip-freeze.txt"
python - <<'PY' > "$MANIFEST_DIR/runtime-check.txt"
import h5py
import nltk
import numpy
import pandas
import scipy
import sklearn
import torch

print("environment_ok=true")
print(f"numpy={numpy.__version__}")
print(f"scipy={scipy.__version__}")
print(f"pandas={pandas.__version__}")
print(f"sklearn={sklearn.__version__}")
print(f"h5py={h5py.__version__}")
print(f"nltk={nltk.__version__}")
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
PY

cat "$MANIFEST_DIR/runtime-check.txt"


