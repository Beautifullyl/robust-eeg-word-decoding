# Reproduction guide

## 1. Obtain the data

Download the [official ZuCo 2.0 release](https://osf.io/2urht/). The main analysis
uses the Task 1 natural reading Matlab files. Source data are not included in
this repository.

## 2. Set the paths on CSF3

Place the repository in scratch storage and work from its root directory, which
contains `code`, `config`, `documentation` and `results`. Replace the example
paths below with the locations on your account, then export them before
submitting jobs:

```bash
export PROJECT_DIR="/path/to/robust-eeg-word-decoding"
export EEG_ENV_DIR="/path/to/conda_envs/eeg-word-reproduction"
export ZUCO2_ROOT="/path/to/ZuCo2"
export ZUCO2_MAT_DIR="/path/to/ZuCo2/task1 - NR/Matlab files"
cd "$PROJECT_DIR"
```

The extraction jobs create their processed-data directories when they run.

## 3. Install the environment

Create the Python environment at the location specified by `EEG_ENV_DIR`, using
[environment.yml](../config/environment.yml). A package list is also provided in
[requirements.txt](../config/requirements.txt). The CSF jobs use the Miniforge
module; GPU workflows require a CUDA-capable PyTorch installation.

```bash
module load apps/binapps/conda/miniforge3/25.9.1
conda env create -p "$EEG_ENV_DIR" -f config/environment.yml
conda activate "$EEG_ENV_DIR"
```

When the complete workflow is submitted, Workflow 01 updates this environment,
downloads the NLTK tagger and records the installed package versions.

## 4. Create the output folders

```bash
python code/run_pipeline.py prepare
```

This creates a directory under `experiment_outputs` for each numbered workflow,
with `logs`, `results`, `analysis`, `scripts` and `manifest` subdirectories.

## 5. Run the experiments

Submit the complete workflow with:

```bash
python code/run_pipeline.py submit all
```

Jobs are linked by dependencies, so data validation and extraction finish
before model fitting begins. This command submits the formal experiments and
their analyses, not just data preparation.

For a shorter check of an individual workflow, submit a smoke test after its
input files are ready:

```bash
python code/run_pipeline.py submit workflow15 --mode smoke
python code/run_pipeline.py submit workflow16 --mode smoke
```

Workflow 15 needs the all-fixations inputs and exported channel labels.
Workflow 16 needs both the all-fixations and first-fixation inputs. To submit
first-fixation extraction separately, use:

```bash
python code/run_pipeline.py submit workflow16 --mode extract
```

Wait for extraction to finish before submitting its smoke test.

## 6. Recreate the result tables

Analysis jobs run after their corresponding experiments. Workflow 17 combines
the summaries and writes the report tables to `results/final_subject_level`.

To rebuild the tables after all formal outputs already exist, run:

```bash
python code/analysis/build_report_outputs.py
```

## 7. Recreate the figures

Workflow 17 creates the figures automatically. After the result tables are
available, they can also be rebuilt with:

```bash
python code/analysis/create_final_report_figures.py
python code/analysis/create_non_eeg_control_figure.py
```

Figures are written to `results/figures`.

## 8. Check the files

```bash
python code/verify_repository.py
python code/analysis/build_file_manifest.py
```

The verifier checks Python syntax, local code dependencies, referenced job
files, required result tables and unwanted absolute or legacy paths. It does
not rerun model training or establish that the experimental results have been
reproduced. The second command updates `audit/FILE_MANIFEST.csv` with the file
sizes and checksums.

## Experiment settings

Workflow 16 uses a `512 x 105` input for both fixation representations, with the
same EEGNet architecture and no temporal interpolation. In Workflow 15, each
sampled random mask is fixed across all test words for a held-out participant.
