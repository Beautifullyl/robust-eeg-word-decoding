# Robust EEG-based Word Decoding under Subject Shift and Channel Loss

This repository contains the code and result tables for my MSc dissertation on predicting word categories from EEG recorded during natural reading. The experiments evaluate how well the models generalise to unseen participants and how performance changes when EEG channels are missing.

Start with the [reproduction guide](documentation/REPRODUCTION_GUIDE.txt) for environment setup, data
paths and instructions for running the experiments on CSF3.

## Files and folders

- [code/data_preparation/](code/data_preparation/): source data checks and extraction of word-level EEG
  and channel labels.
- [code/models/](code/models/): linear, EEGNet and TCN models, including the channel-loss
  and fixation comparisons.
- [code/analysis/](code/analysis/): subject-level results, split checks and report figures.
- [code/csf_jobs/](code/csf_jobs/): submission scripts for CSF3.
- [config/](config/): Python environment and package requirements.
- [results/final_subject_level/](results/final_subject_level/): summary and subject-level CSV tables
  reported in the dissertation.
- [documentation/](documentation/): setup instructions, data access and the technical appendix.
- [audit/](audit/): file list, checksums and repository check results.

## Reproduction order

1. Set up the Python environment (Workflow 01).
2. Validate the complete ZuCo 2.0 release (Workflow 02).
3. Extract the 512-sample all-fixations representation (Workflow 05).
4. Run the main all-fixations EEGNet LOSO baseline (Workflow 06).
5. Run the TCN model comparison (Workflow 07).
6. Run the linear baseline, diagnostic splits and per-class analysis
   (Workflow 08).
7. Run structured channel-loss experiments (Workflow 09).
8. Audit subject and sentence overlap between splits (Workflow 10).
9. Run the non-EEG control models (Workflow 11).
10. Run LOSO with grouped inner validation (Workflow 12).
11. Repeat the sentence-disjoint evaluation across five sentence partitions
    (Workflow 14).
12. Run persistent random channel loss (Workflow 15).
13. Run the input-matched first-fixation versus all-fixations ablation
    (Workflow 16).
14. Assemble the report tables and figures, then run the repository check
    (Workflow 17).

The workflow numbers match the script filenames. Gaps in these numbers do
not indicate missing steps.

## Check the downloaded files

From the repository root, run:

```bash
python code/verify_repository.py
```

The check should finish with:

```text
Status: PASSED
```

This checks the code structure and required result files; it does not
rerun the experiments.

## Run on CSF3

After setting up the environment and data paths in the reproduction guide,
set `PROJECT_DIR` to the repository root. Create the output folders with:

```bash
python code/run_pipeline.py prepare
```

To submit one experiment, for example the persistent channel-loss test:

```bash
python code/run_pipeline.py submit workflow15 --mode formal
```

To submit all experiments and their analysis jobs:

```bash
python code/run_pipeline.py submit all
```

Workflow 17 waits for the required experiments and analyses to finish, then
writes the report tables to `results/final_subject_level`, recreates the
figures and checks the repository.

## Data

The experiments use ZuCo 2.0 Task 1 natural reading. Download the source data
from the [official release](https://osf.io/2urht/). The raw data and extracted EEG arrays are not
included here. See [data availability](documentation/DATA_AVAILABILITY.txt) for data access
details and the [reproduction guide](documentation/REPRODUCTION_GUIDE.txt) for the required paths.
