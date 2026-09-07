# Registered experimental protocol

This file freezes decisions that must not be changed after viewing headline test
results without an explicit amendment.

## Splits and horizons

- Chronological 70/10/20 raw-time split.
- Training-only per-sensor normalization; missing speed is excluded from fit.
- Input 12 steps; direct prediction of the next 12 steps.
- Headline horizons: 3, 6, and 12; all-horizon average also reported.
- Test data are never used for early stopping, calibration fitting, or threshold
  selection.

## Model selection

- Hyperparameters are selected on validation MAE plus validation calibration
  diagnostics, never on test MAE.
- Screening seed: 42. Confirmatory seeds: 1, 12, 42, 123, 2026.
- AdamW, initial learning rate 1e-3, weight decay 1e-4, maximum 100 epochs,
  early-stopping patience 15, gradient norm 5.
- A proposed variant advances only if it improves compound-shift deployment
  metrics on both datasets and does not regress clean MAE by more than 1%.

## Baseline policy

- Reproduced: historical average, persistence, ridge AR, linear, MLP, GRU,
  STGCN, DCRNN, Graph WaveNet, AGCRN, STID, STAEformer, and backbone-matched
  ablations.
- Official recent implementations are added when the data split, target,
  masking, and metric code can be made equivalent.
- Literature-reported values are a separate context table and never enter
  paired tests or superiority claims.

## Stress design

- Rates: 10%, 20%, 30%, 40%, 50%.
- Families: independent point loss, temporal bursts, contiguous graph region,
  and highest-degree hubs.
- The random stress masks are saved and shared across models for paired tests.
- Every family is stratified by free, congested, intermediate, and transition
  regimes defined before inspecting model errors.

## Statistical analysis

- Mean and standard deviation across five seeds.
- Crossed paired bootstrap over confirmatory seeds and chronological daily
  blocks, 10,000 resamples; Wilcoxon tests use seed-averaged daily differences.
- Holm correction within each hypothesis family.
- Report absolute effect, relative effect, 95% interval, adjusted p-value, and
  practical threshold; do not equate p-values with practical importance.
- Calibration: marginal and worst node/horizon/regime coverage, mean width,
  Winkler score, and calibration error.
- Selection: AURC and retained MAE/failure rate at fixed coverage; the registered
  operational failure is forecast-origin mean absolute error above 5 mph and target selective risk
  is 10% at delta 0.05. Candidate thresholds and risk certification use
  disjoint even/odd past forecast origins.

## Required ablations

1. no reliability conditioning;
2. no structured-outage augmentation;
3. independent masking only;
4. static graph only;
5. adaptive graph only;
6. no regime router;
7. fixed global conformal interval;
8. node/horizon calibration without risk-controlled abstention;
9. full ShiftGuard.

## Amendment log

- 2026-09-03: PEMS-BAY STAEformer exceeded practical memory capacity (about
  87 minutes for one epoch, 7.9 GB allocated device memory and under 1 GB free
  system RAM). Resume from the complete epoch-1 training state using microbatches
  of eight with gradient accumulation to retain optimizer batch size 32.
  Validation/test inference batches are capped at eight and test tensors are
  loaded only after training. Dropout draws and floating-point reduction order
  may differ; this is not a bitwise-equivalence claim. The change is motivated
  by resource diagnostics, not test performance. Optimizer/RNG/early-stopping
  state is saved atomically at each complete epoch for subsequent interruptions.

- 2026-08-31: protocol created before any complete model result was observed.
- 2026-09-01: before confirmatory stress runs, threshold construction and risk
  certification were split across even/odd past origins to remove outcome reuse.
  This validity correction was motivated by assumption auditing, not performance;
  all earlier screening stress values are non-confirmatory and will be replaced.
