# Exposure architecture classifier

This optional module trains a three-class model from exposure and matched-control
scores. It returns probabilities for `genetics-driven`, `non-genetics-driven`, and
`correlation-driven`. Interpret these as signal patterns under the study's
assumptions, not proof of causal mechanisms. A UKBB-trained or validated classifier
is not included.

## Score definitions and inputs

The DNN callback's `featImport` files contain squared path importance scores,
written in this order: `X1,X2,...,Xp,C1,C2,...,Cp`. Its `featWeight` files contain signed
weight products; neither is a regression coefficient or odds ratio. Do not mix
these scales or train on ridge coefficients and apply that model to DNN scores.

Current DeepEXPOKE fits the statistical and PRS references in separate two-channel
networks. Retain four raw scores per exposure:

| Column | Meaning |
| --- | --- |
| `exposure_stat` | Exposure importance from the statistical-reference fit |
| `stat_control` | Matched statKO importance from that fit |
| `exposure_prs` | Exposure importance from the PRS-reference fit |
| `prs_control` | Matched prsCO importance from that fit |

The `paired` classifier adds the two magnitude differences to those four inputs.
It does not average the two exposure estimates. A `joint` schema also accepts
`exposure,prs_control,stat_control` from a separately implemented joint three-channel
model. This addition does not change the existing two-channel DNN architecture.

Every input table must include all exposures, including negative and zero scores.
An absent score is missing, never zero. Never use shuffled ranking tails as data.
Preserve cohort, outcome, feature ID, fit ID, seed, split and preprocessing
metadata alongside the scores. Training and prediction require an identical
`score_definition` and `protocol_id`; the latter names your fixed model, scaling,
feature panel, reference construction and score extraction protocol.

## Export scores and matrices

Pass names and metadata to the existing callback:

```python
callback = DNN.Job_finish_Callback(
    resultDir, pNum, feature_names=feature_ids,
    control_kind="statKO", run_id="sepsis_incidence_seed42_stat"
)
```

Each epoch now saves an additional named `*_paired_scores.csv` with all exposures,
both importance scores, their W contrast and both signed weight products. The
original headerless exports remain available. Use a fresh run directory for each
independent fit and retain the outcome and participant manifest in the controlled
analysis environment.

Recover a combined table from existing callback outputs:

```python
from DL.effects import paired_effect_table

effects = paired_effect_table(
    "stat_run/result_epoch20_featImport.csv",
    "prs_run/result_epoch20_featImport.csv",
    feature_ids,  # exact original training column order
    stat_run_index=0, prs_run_index=0,
)
effects.to_csv("all_exposure_control_effects.csv", index=False)
```

Explicit run indices are mandatory if a file contains multiple appended fits.
Choose paired fits with the same endpoint, cohort, preprocessing and split.
`read_pair_scores` defaults to the current blocked callback format; its
`layout="interleaved"` option is only for independently verified alternate formats.

For matrices, `split_aligned_matrices(stat, prs, feature_ids)` takes numeric arrays
in `[all X, all control]` order and returns X, statKO and prsCO DataFrames. It
requires the two X blocks to match exactly. Equal row counts alone are
insufficient. Preserve verified participant IDs in a separate restricted manifest;
positional row numbers are not participant identifiers. Export participant data
only to the authorized local handoff location, never into this public repository.

The command-line exporter creates a new directory with a complete table, feature
dictionary and SHA-256 source/output manifest. Use the repository's `Feature,Label`
dictionary in the original training order. For participant matrices, supply
`[X,statKO]`, `[X,prsCO]` and `[X,Y]`; all three X blocks must match exactly.
The statistical and XY headers must use dictionary IDs; both control blocks must
use `K1..Kp` in matching feature order. Descriptive PRS X headers are supported.

```bash
python -m DL.export matrices stat_matrix.txt prs_matrix.csv xy_matrix.txt \
  /private/handoff/sepsis_incidence --feature-map featuremapping.csv \
  --cohort UKBB --endpoint sepsis_incidence

python -m DL.export scores stat_run/result_epoch20_featImport.csv \
  prs_run/result_epoch20_featImport.csv /private/handoff/sepsis_effects \
  --feature-map featuremapping.csv --cohort UKBB --endpoint sepsis_incidence \
  --stat-run-index 0 --prs-run-index 0 --protocol-id paired_dnn_v1
```

The matrix command defaults to tab-separated statistical/XY inputs and comma-
separated PRS inputs; separator flags allow either format. Its compressed CSV
contains `source_row`, `X_<ID>`, `statKO_<ID>`, `prsCO_<ID>` and binary `Y`, in that
order. Matrix values are preserved without scaling or fitting. The score command
retains all raw importance scores and both W contrasts, including negative/zero
values. It does not recover scores from W summaries or verify that independently
supplied fits share a split. Existing destination directories are rejected.

## Train and predict

Install `requirements-architecture.txt` in a separate modern Python environment.
The classifier and score readers do not import the legacy TensorFlow stack.

Training needs independent labels, normally from simulations with known mechanisms
processed through the same fitting and score extraction pipeline as the target
data. UKBB exposure names or overlap-derived categories are not independent truth
labels. Mixed and null signals must be handled explicitly in the simulation design;
this three-class model only learns dominant patterns and does not validate a rule
for excluding those signals. Do not silently relabel mixed or null simulations.

```bash
python -m DL.architecture train labeled_simulation_effects.csv architecture.joblib \
  --schema paired --label-column truth_class --group-column population_id \
  --folds 5 --score-definition dnn_squared_path_importance \
  --protocol-id paired_dnn_v1

python -m DL.architecture predict architecture.joblib ukbb_effects.csv predictions.csv \
  --score-definition dnn_squared_path_importance --protocol-id paired_dnn_v1
```

The fixed multinomial logistic model standardizes inputs within each training
fold. All exposures/reference variants/outcomes from one simulated population
must share a group. Grouped cross-validation writes out-of-fold probabilities and
a validation JSON with balanced accuracy, log loss and a confusion matrix, then
fits the final model on all supplied rows. Grouping follows scikit-learn's
[StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).
Additional held-out simulation families and independent data are needed to assess
transfer. Software tests with artificial separated classes are not that validation.

An optional `--min-probability` threshold returns `unresolved` below a user-chosen
value while preserving the three class probabilities. The default reports the
largest-probability class without abstention; no biological confidence cutoff or
probability calibration is claimed. Choose any abstention cutoff on independent
validation data. Load joblib artifacts only from trusted training runs.

## Migration of historical effect summaries

The callback and FDR reader both use exposure/control blocks. This layout is
preserved and covered by a deterministic callback test. Recover complete effects
from the original `featImport` files and verify the run provenance before combining
historical summaries. A W difference alone cannot recover the two scores that
produced it. If raw scores and model weights are unavailable, rerun the original
fit from the original cohort inputs.

Filtering now keeps an infinite threshold when no candidate passes, uses the
inclusive positive boundary, and validates q (default 0.05). `controlFilter`
returns selected features; use `controlFilter_return_all` or `paired_effect_table`
to export every score. Old notebook calls with `q=0.00` should use the study's
predeclared q (the manuscript uses 0.05). PRS controls do not inherit formal
model-X FDR guarantees merely by passing through this numerical filter.

The previous feature shuffling/bootstrap flags do not implement participant
bootstrap refits. They now raise an error. `calculate_multiple_W_for_selected`
reads actual saved fit rows and rejects a request for more fits than exist.
For confidence intervals, resample participants, regenerate controls as required
by the protocol, refit both models with matched resampling, and keep each complete
score table. Recalculate intervals from those refits, including failed/null
selection outcomes under a prespecified method. Existing intervals cannot be
validated from selected tables alone.

Run the software checks with:

```bash
python -m unittest discover -s tests -v
```
