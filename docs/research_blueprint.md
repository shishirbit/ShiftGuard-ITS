# Research blueprint (living document)

## Final title

**When Not to Trust a Traffic Forecast: Shift- and Outage-Aware Risk-Controlled
Spatiotemporal Learning for Intelligent Transportation Systems**

## Problem statement

Canonical METR-LA and PEMS-BAY studies largely optimize average point error on
a single chronological test block. Real ITS deployments must also operate when
(i) the traffic regime differs from training, (ii) sensor loss is spatially and
temporally correlated rather than independent, and (iii) the downstream system
needs to know which forecast tensors are safe enough to use. Recent work
addresses drift, missingness, uncertainty, causal invariance, or efficiency in
isolation, but the joint operational problem and its risk-coverage trade-off
remain insufficiently benchmarked.

## Tested hypothesis

A reliability-conditioned spatiotemporal forecaster trained with structured
masking and probabilistic scale estimation, paired with group-adaptive online
conformal risk control, can improve worst-regime point error and maintain useful
coverage/selective-risk behavior under joint temporal shift and graph outages,
without sacrificing canonical clean-test accuracy or real-time feasibility.
The hypothesis is partially supported: robustness, selective risk, and
efficiency improve materially, while clean PEMS-BAY MAE is 0.027 mph worse than
STAEformer and reliability conditioning itself is not supported by ablation.

## Objectives

1. Build a leakage-controlled stress protocol for temporal regimes and
   realistic structured sensor failures on METR-LA and PEMS-BAY.
2. Develop a lightweight reliability-conditioned graph-temporal point
   forecaster that explicitly consumes observation masks and local graph
   reliability.
3. Develop node/horizon/regime-adaptive uncertainty and abstention calibration
   for nonstationary, spatially correlated residuals.
4. Compare clean accuracy, robustness, calibration, selective risk, and
   efficiency against classical, deep-learning, graph, transformer, linear,
   probabilistic, and adaptation baselines.
5. Establish which component produces each gain using ablation, sensitivity,
   statistical significance, and failure analysis.

## Initial falsifiable criteria and disposition

The initial proposal was tested against the following criteria; failed criteria
trigger claim narrowing rather than being hidden:

- The systematic search finds no prior method evaluating the same joint task
  with the same outputs (point forecast, interval, and abstention decision).
- Gains are not explained only by a larger parameter count or favorable split.
- The method improves at least one deployment metric materially on both
  datasets, with no practically important regression on canonical clean MAE.
- Coverage claims are stated under assumptions appropriate to time dependence;
  no exchangeability guarantee is overclaimed under uncontrolled drift.
- Ablations isolate the value of reliability conditioning, structured masking,
  adaptive calibration, and selective decision rules.

The clean-error criterion was not met on PEMS-BAY and the architectural
necessity criterion was only partly met. Accordingly, the retained contribution
is explicitly limited to the compound-shift benchmark, structured-outage
training, and prequential origin-level risk control.

## Novelty audit (2026-09-06)

**Decision: retain the contribution with narrowed claims.** The 36-study
evidence matrix found no prior traffic-forecasting work that jointly evaluates
point forecasts, node/horizon intervals, and forecast-origin abstention across
independent, burst, contiguous-region, and hub outages on both target datasets.
Locally, 640 confirmatory stress evaluations show lower mean MAE than
STAEformer by 0.347 mph on METR-LA and 0.546 mph on PEMS-BAY, with hierarchical
95% intervals excluding zero. The registered controller retains 86%/98% of
origins at 30% severity with 8%/2% empirical failure. ShiftGuard is also much
smaller than STAEformer.

The audit rejects two broader claims. ShiftGuard does not dominate clean
PEMS-BAY accuracy, and reliability-weighted propagation plus regime routing do
not explain the robust gains. The defensible novelty is therefore the
leakage-controlled compound-shift benchmark, structured-outage training, and
prequential origin-level risk control. It is not a new clean-data SOTA or proof
that every architectural component is beneficial.

## Evaluation tiers

### Tier A: locally reproduced under one pipeline

HA, persistence, seasonal naive, linear/DLinear-style, MLP, GRU, STGCN,
DCRNN-like diffusion recurrent, Graph WaveNet, AGCRN, STID, and the proposed
model. Additional SOTA implementations are added only when their licenses and
preprocessing can be audited.

### Tier B: reported literature context

GMAN, MTGNN, GTS, STNorm, PDFormer, MegaCRN, TESTAM, STAEformer, recent
transformer/Mamba/MoE models, drift-aware methods, and probabilistic methods.
These numbers will never be presented as directly comparable when protocols
differ.

## Stress protocol

- **Clean canonical:** standard chronological test block.
- **Regime slices:** peak/off-peak, weekday/weekend, congestion transitions,
  high-volatility windows, and data-driven change-point segments.
- **Outages:** independent missingness; contiguous graph-region failure;
  high-centrality failure; and temporal burst failure at 10--50% severity.
- **Joint shift:** each outage family evaluated inside each regime stratum.
- **Calibration:** rolling prequential calibration using past residuals only.

## Success metrics

- Point: MAE, RMSE, masked MAPE; horizon 3/6/12 and all-horizon average.
- Robustness: relative degradation, worst-regime MAE, worst-outage MAE.
- UQ: empirical coverage, mean interval width, Winkler score, calibration error,
  worst-node-group coverage.
- Selective: risk-coverage curve, AURC, MAE at 90/80/70% retained coverage.
- Efficiency: parameters, FLOPs, wall-clock training, batch-1/batch-64 latency,
  and peak GPU memory.
