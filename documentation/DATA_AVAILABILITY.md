# Data availability

The experiments use the publicly released [ZuCo 2.0 dataset](https://osf.io/2urht/), specifically Task 1 natural reading. The repository does not redistribute source Matlab files, raw EEG recordings, extracted NPY arrays or row-level derived data.

To reproduce the analysis, obtain ZuCo 2.0 from the official release above and provide the local dataset paths described in the [reproduction guide](REPRODUCTION_GUIDE.md). The data-preparation scripts validate the expected layout and generate the word-level inputs used by the model scripts.

The repository includes aggregate and subject-level result tables. These tables are sufficient to check the reported means, paired differences and uncertainty intervals without redistributing participant-level EEG waveforms.
