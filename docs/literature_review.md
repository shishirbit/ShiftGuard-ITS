# Literature review and gap analysis

Last searched: 2026-08-31. This is a living evidence synthesis; publication
status and protocol compatibility must be rechecked before submission.

## Search protocol

Databases/sources: IEEE Xplore, ACM Digital Library, ScienceDirect, Springer,
OpenReview, arXiv, and official project repositories. Queries combine traffic
forecasting, spatiotemporal graph learning, METR-LA, PEMS-BAY, uncertainty,
conformal prediction, distribution shift, test-time adaptation, missing data,
sensor outage, calibration, robustness, selective prediction, and abstention.
We include primary studies that forecast network traffic states and either use
the target datasets or introduce a directly relevant reliability method. We
exclude papers without sufficient methodological detail, non-forecasting ITS
tasks, and secondary summaries as evidence for novelty.

## Evolution of the field

### Fixed-topology graph forecasting

DCRNN introduced diffusion convolution inside an encoder-decoder recurrent
network; STGCN used graph convolution and gated temporal convolution. Graph
WaveNet added learned adaptive adjacency and dilated temporal convolutions.
These remain essential reproducible baselines, but they optimize average point
error and do not expose deployment confidence.

### Learned, dynamic, and heterogeneous spatial structure

AGCRN learns node-adaptive parameters and graph structure. GTS learns a discrete
graph; MTGNN learns directed dependencies. MegaCRN uses a memory bank to obtain
time-varying meta-graphs. PDFormer explicitly models long-range dependencies and
propagation delay. TESTAM routes inputs among heterogeneous spatial experts.
These improve contextual spatial modeling, but observed inputs and routing are
generally assumed reliable.

### Simpler and transformer-based forecasters

STID showed that identity and temporal embeddings make a simple model highly
competitive. STAEformer showed that adaptive spatiotemporal embeddings can make
a vanilla transformer SOTA. This weakens novelty claims based solely on adding
architectural complexity. Any proposed model must beat strong simple baselines
and report efficiency.

### Extreme events and temporal nonstationarity

CompFormer retrieves historical representations at test time for extreme
traffic. MegaCRN models heterogeneous/nonstationary patterns. Recent online and
drift-aware models adapt their graph or parameters to changing regimes. Recent
causal-invariance work targets OOD robustness. Therefore, "drift awareness" or
"extreme-event robustness" alone is not a defensible new problem.

Masked pretraining provides a second robustness route. STMAE uses graph-aware
spatial and patch-based temporal masking, while STD-MAE decouples spatial and
temporal reconstruction to learn longer-context representations. Their masking
objectives improve representation quality but do not by themselves calibrate
compound-shift risk. Continual spatiotemporal tuning methods further show that
sequential adaptation is an established direction.

### Missingness and sensor reliability

Imputation/forecasting and reliability-weighted graph methods study missing or
heterogeneous observations; recent cold-start work evaluates held-out sensors
and static outages. Many studies use independent random masks or a single fault
family. Joint spatially contiguous and temporally bursty failures inside
shifted regimes remain less consistently evaluated.

### Probabilistic and conformal forecasting

Deep spatiotemporal UQ work compares Bayesian, frequentist, and quantile
methods. Adaptive conformal graph and hypergraph studies provide calibrated
intervals, including recent topology-aware calibration. Thus, adding conformal
intervals to a point model is not novel. Most work reports marginal coverage
and interval width, while operational decisions require subgroup reliability
and a principled option to abstain.

A 2026 regime-dependent audit of time-series foundation models reports that
aggregate accuracy and marginal interval coverage conceal severe transition
failures on METR-LA and PEMS-BAY. This directly supports regime-stratified
coverage evaluation, but it does not address graph-structured sensor outages or
risk-controlled abstention under their interaction.

General online selective-conformal work can control risk after adaptive
selection, but current searches found no traffic-forecasting study that
evaluates selective error control under compound traffic-regime and graph-outage
shift. The proposed method must therefore cite and adapt this broader
statistical literature rather than claim abstention itself as new.

### Efficiency and mixtures

STID, STAEformer, TESTAM/TESTAM+, GC-MoE, and Weaver illustrate a trend toward
simple embeddings, selective experts, parameter-efficient routing, and
factorized attention. Proposed improvements must report parameters, latency,
memory, and accuracy--reliability trade-offs rather than accuracy alone.

## Gap matrix

| Capability | Well studied? | Remaining gap relevant here |
|---|---:|---|
| Clean canonical point accuracy | Yes | Saturated; insufficient novelty alone |
| Dynamic/learned graphs | Yes | Reliability of graph messages under outages |
| Temporal drift adaptation | Increasingly | Often evaluated without structured outages |
| Missing-data robustness | Yes | Often random/static masks, not joint regime shift |
| Conformal intervals | Increasingly | Marginal coverage can hide node/regime failures |
| Selective forecasting/abstention | Limited in traffic | Risk-coverage under graph-correlated failures |
| Joint point + interval + decision output | Limited | Unified operational evaluation is missing |
| Worst-group reliability | Limited | Node/horizon/regime conditional diagnostics |
| Efficiency under adaptation/calibration | Partial | End-to-end deployment cost rarely central |

## Defensible research gap

The defensible gap is not a new graph block. It is **risk-controlled traffic
forecasting under compound shift**: temporal regime change and graph-structured
sensor outage occur together, and the system must issue a point forecast, a
calibrated interval, and an abstention score for every node and horizon. The
study must quantify worst-group reliability and risk coverage while retaining
clean accuracy and real-time feasibility.

## Novelty threats

1. A contemporaneous paper may already combine compound shift and conformal
   abstention; repeat the search immediately before submission.
2. Improvements may come from mask augmentation rather than the proposed
   architecture; isolate this with backbone-matched ablations.
3. Rolling conformal methods under arbitrary drift do not automatically retain
   finite-sample exchangeability guarantees; state empirical and theoretical
   claims precisely.
4. METR-LA and PEMS-BAY are old, short benchmarks. The contribution must be a
   rigorous reliability protocol, not a claim of universal modern deployment.
5. Literature values use inconsistent preprocessing and MAPE masking. Keep
   reproduced and reported comparisons in separate tables.

## Core primary sources

- Li et al., *DCRNN*, ICLR 2018, arXiv:1707.01926.
- Yu et al., *STGCN*, IJCAI 2018, arXiv:1709.04875.
- Wu et al., *Graph WaveNet*, IJCAI 2019, arXiv:1906.00121.
- Bai et al., *AGCRN*, NeurIPS 2020.
- Shang et al., *GTS*, ICLR 2021.
- Shao et al., *STID*, CIKM 2022.
- Jiang et al., *MegaCRN*, AAAI 2023, arXiv:2211.14701.
- Jiang et al., *PDFormer*, AAAI 2023, arXiv:2301.07945.
- Liu et al., *STAEformer*, CIKM 2023, arXiv:2308.10425.
- Zhang et al., *CompFormer*, arXiv:2309.09074.
- Lee and Ko, *TESTAM*, ICLR 2024, OpenReview:N0nTk5BSvO.
- Wu et al., *Quantifying Uncertainty in Deep Spatiotemporal Forecasting*,
  KDD 2021, arXiv:2105.11982.
- Patil et al., conformal GNN traffic forecasting, arXiv:2407.12238.
- Cheong et al., *Weaver*, arXiv:2511.08888.
- Ghaffari et al., *GC-MoE*, arXiv:2605.30486.
- Wang and Ma, *Adaptive conformal prediction for spatio-temporal hypergraph
  neural networks*, Applied Soft Computing, 2026,
  DOI:10.1016/j.asoc.2026.116256.
- Martin et al., *A Critical Audit of Spatiotemporal Forecasting Benchmark
  Datasets and Baselines*, arXiv:2608.20980.
- Wang et al., *Do Time Series Foundation Model Benchmarks Hide
  Regime-Dependent Failures?*, ICML 2026 workshop, arXiv:2606.18367.
- Wang et al., *Cold-Start Traffic State Forecasting at Unseen Sensor
  Locations via Support-Conditioned Meta-Graph Learning*, Sensors 26(15), 2026,
  DOI:10.3390/s26154995.
- Wu et al., *Beyond Correlations: A Causal Invariance Framework for Robust
  Urban Traffic Prediction*, International Journal of Approximate Reasoning,
  2026, DOI:10.1016/j.ijar.2026.109802.
- Bao et al., *CAP: A General Algorithm for Online Selective Conformal
  Prediction with FCR Control*, JMLR 26 (2025).
- Stankeviciute et al., *Conformal Time-Series Forecasting*, NeurIPS 2021.
- Sun et al., *Revealing the Power of Masked Autoencoders in Traffic
  Forecasting (STMAE)*, arXiv:2309.15169.
- Gao et al., *Spatial-Temporal-Decoupled Masked Pre-training for
  Spatiotemporal Forecasting (STD-MAE)*, arXiv:2312.00516.
