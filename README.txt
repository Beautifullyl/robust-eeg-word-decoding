ROBUST EEG-BASED WORD DECODING UNDER SUBJECT SHIFT AND CHANNEL LOSS

This repository contains the code, aggregate results and documentation used for the dissertation experiments. It does not redistribute the ZuCo 2.0 source data or extracted row-level EEG arrays.

CONTENTS

code/data_preparation
    Dataset validation, word-level EEG extraction and channel-label export.

code/models
    Linear, EEGNet and TCN models, evaluation splits, channel-loss experiments and fixation ablations.

code/analysis
    Subject-level aggregation, uncertainty estimates, split audits and figure generation.

code/csf_jobs
    CSF3 job definitions numbered in the order needed to reproduce the study.

results/final_subject_level
    Aggregate and subject-level CSV files used to check the reported results.

documentation
    Reproduction instructions, data availability and a technical appendix.

audit
    Repository integrity checks and the file manifest.

REPRODUCTION ORDER

1   Set up the Python environment (Workflow 01).
2   Validate the complete ZuCo 2.0 release (Workflow 02).
3   Extract the 512-sample all-fixations representation (Workflow 05).
4   Run the main all-fixations EEGNet LOSO baseline (Workflow 06).
5   Run the TCN model comparison (Workflow 07).
6   Run the linear baseline, diagnostic splits and per-class analysis (Workflow 08).
7   Run structured channel-loss experiments (Workflow 09).
8   Audit subject and sentence overlap between splits (Workflow 10).
9   Run the non-EEG control models (Workflow 11).
10  Run LOSO with grouped inner validation (Workflow 12).
11  Repeat the sentence-disjoint evaluation across five sentence partitions (Workflow 14).
12  Run persistent random channel loss (Workflow 15).
13  Run the input-matched first-fixation versus all-fixations ablation (Workflow 16).
14  Assemble the report tables and figures, then run the repository check (Workflow 17).

The numbered list above is the final reproduction order. Workflow identifiers retain their original experiment numbers, so the sequence is intentionally non-consecutive.

QUICK CHECK

Run:

    python code/verify_repository.py

The expected final line is:

    Status: PASSED

CSF WORKFLOW

On CSF3, set PROJECT_DIR to the repository root and run:

    python code/run_pipeline.py prepare

Individual workflows can then be submitted with, for example:

    python code/run_pipeline.py submit workflow15 --mode formal

The complete formal dependency graph can be submitted with:

    python code/run_pipeline.py submit all

The final dependency submits Workflow 17 after every required model and analysis job has completed. It assembles the aggregate tables in results/final_subject_level, recreates the figures and runs the repository check.

DATA

The source dataset is ZuCo 2.0 Task 1 natural reading. Obtain it from the official release at https://osf.io/2urht/ and set the paths described in documentation/REPRODUCTION_GUIDE.txt. Source data and extracted NPY arrays are excluded from this repository.
