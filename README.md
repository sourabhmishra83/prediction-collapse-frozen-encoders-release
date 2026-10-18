# Prediction collapse in frozen encoder classifiers

Code, aggregate results, scientific figures and verification material by Sourabh Mishra for a study of checkpoint-dependent prediction collapse in frozen-encoder text and graph classifiers. MIT licensed. This is a clean public review release; manuscripts and experiment-plan files are intentionally excluded.

**Archived release DOI:** [10.5281/zenodo.23445417](https://doi.org/10.5281/zenodo.23445417)

## Review the saved evidence

Run `python review/verify_release.py` from this directory with Python 3.10 or later. It uses only the standard library, verifies every release checksum, checks expected aggregate coverage and independently recomputes hard-decision metrics from the released confusion matrices. It never trains, downloads an encoder or accesses source datasets.

For a comparison plot and summary tables, install `review/requirements.txt`, then run `python review/summarize_results.py`. Outputs go to an ignored `review_outputs/` directory. The accompanying analysis code records how private-data audits were conducted; these original audit scripts require the restricted inputs and private workspace layout and are not standalone public tests.

## Evidence map

| Location | Evidence |
|---|---|
| `experiments/existing_2026/` | Current 2026 notebook source code, with outputs removed |
| `data/historical_summaries/` | Earlier 2026 saved summary records; overlap prevents treating row counts as independent executions |
| `experiments/grouped_control/analysis/` | Separate 40-job grouped text-only robustness control |
| `full_study/` and `results/full_study/` | Executed RoBERTa architecture and update-control code; 432 job records and 3,888 endpoint/target records |
| `code/monant_stance_v1/` and `results/monant_stance_v1/` | Separate 18-job, three-class claim/article stance evaluation; 108 endpoint/population records |
| `code/second_encoder_v1/` and `results/second_encoder/` | Matched frozen BERT code and 36-job endpoint aggregates |
| `code/priority_controls_v1/` and `results/analysis/v1_8/` | 108 logistic probes, 36 mirror neural heads and 72 existing-head replays; 1,944 decision/target records |
| `code/standardization_v1/` and `results/standardization/` | Completed 72 standardized-feature fits, 648 selected decision/target records and 648 endpoint/target records |
| `results/analysis/` | Score, direction, geometry and checkpoint diagnostics from the completed 2026 evidence |
| `figures/current/` | Current scientific figure assets; filenames retain their creation names |

## Interpretation and verification limits

Keep initialization, training-ever source validation, selected and final checkpoints separate. Predictions at a fixed decision rule can collapse despite ranking signal. Training-only standardization reduced selected in-domain raw-rule collapse from 14/72 to 0/72 and increased mean macro-F1 from .603 to .714 in the matched panel. Three BERT trajectories still crossed the validation threshold. Transfer remained mixed: raw-rule cross-domain collapse decreased from 50/144 to 12/144, but mean cross-domain AUROC decreased for both encoders. Centering/scaling and changed initial predictions are not isolated mechanisms; raw baselines were not refitted contemporaneously, and software/device differences remain a limitation.

The new standardization and classifier-control checkpoint/margin checks ran in Colab. The public release supports aggregate integrity and consistency checks, not private per-example replay. Three seeds on one matched split are descriptive; targets and decision rules reuse heads. Record inventories are not independent execution counts. Historical graph provenance is distinct from source-only grouped graphs. Monant uses a separate stance task; its associated-claim article-rating derivative is excluded from the main binary benchmark.

## Reproduction and data access

Training engines and input-preparation code are included for inspection and reuse. Exact reruns also require separately acquired source data, private row allocations and run configurations, and, for checkpoint replay, the retained private weights/feature caches. Those files are intentionally absent. Do not bypass frozen-input or execution guards to substitute different assets. Source hashes and verification reports preserve the original evidence lineage. The supplied public review commands need none of these private assets. See [data access](docs/DATA_ACCESS.md) and [rights](RIGHTS.md).

## Citation and Zenodo

The reviewed code and aggregate-results release is archived in Zenodo at [10.5281/zenodo.23445417](https://doi.org/10.5281/zenodo.23445417). `CITATION.cff` supplies the DOI for GitHub’s citation feature; `.zenodo.json` retains the author and MIT license metadata. This citation identifies the archived release; later metadata-only commits do not change the deposited archive. No manuscript, journal submission or acceptance is included or implied.

Mishra S. Prediction collapse in frozen encoder classifiers: code and aggregate results. Zenodo. 2026. https://doi.org/10.5281/zenodo.23445417
