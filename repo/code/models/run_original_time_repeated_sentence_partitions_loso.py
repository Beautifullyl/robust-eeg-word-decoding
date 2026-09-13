from __future__ import annotations

import run_original_time_subject_sentence_disjoint_loso as runner


runner.ANALYSIS_ID = "stage39_original_time_repeated_sentence_partitions_loso_v1"
runner.SENTENCE_SPLIT_ID = "sorted_mod5_crossed_test_folds_rng_validation_v1"
runner.OUTPUT_STAGE = "stage39"


if __name__ == "__main__":
    raise SystemExit(runner.main())
