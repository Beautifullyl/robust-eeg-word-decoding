# Technical appendix

## Dataset and representation

The main dataset is ZuCo 2.0 Task 1 natural reading. The analysis includes 18 subjects and 75,539 labelled word observations. The main all-fixations representation concatenates valid fixation-aligned EEG samples and pads or truncates the sequence to 512 time points across 105 channels. No temporal interpolation is used.

## Targets

The two targets are content/function classification and five-class coarse part-of-speech classification. The latter contains function words, nouns, verbs, adjectives and adverbs.

## Evaluation

Leave-one-subject-out evaluation is the main subject-shift setting. In the grouped-validation analysis, whole subjects from the training pool are held out for inner validation. The joint subject-and-sentence analyses also exclude test sentences from model fitting. Subject-dependent and mixed-subject splits are retained as diagnostic comparisons.

## Models

The main neural baseline is EEGNet. It is compared with a class-balanced linear model and a TCN with a wider dilated temporal receptive field. Five model seeds, 42 to 46, are used for formal comparisons.

## Matched fixation ablation

Workflow 16 compares first-fixation and all-fixations inputs using the same `512 x 105` tensor shape, the same EEGNet configuration and no interpolation. All-fixations balanced accuracy is 0.588 compared with 0.536 for first-fixation on content/function, a paired increase of 0.0518 (95% CI 0.0429 to 0.0605; Holm-adjusted p = 2.29e-5). For coarse POS, the corresponding values are 0.238 and 0.215, a paired increase of 0.0231 (95% CI 0.0185 to 0.0275; Holm-adjusted p = 2.29e-5).

## Channel loss

Workflow 15 uses persistent random masks. For each held-out subject, loss rate and repeat, the selected channels remain unavailable for every test word. Workflow 09 removes predefined electrode blocks and compares them with random masks removing the same number of channels.

## Statistical analysis

The held-out subject is the independent unit for LOSO inference. Reported intervals are subject-level bootstrap intervals. Planned paired contrasts use exact sign-flip tests with Holm adjustment where multiple comparisons form one family.

## Reproducibility

Each CSF workflow keeps its logs, results, analysis outputs, copied scripts and manifests in its own output directory. Workflow 17 assembles the final report tables and figures. The repository verifier checks local code dependencies, job-script references, required result tables and unwanted absolute or legacy paths. See the [reproduction guide](REPRODUCTION_GUIDE.md) for commands and data paths.
