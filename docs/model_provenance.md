# Model provenance and equivalence audit

This file prevents a compact reproduction from being mistaken for an authors'
official implementation. All models use this repository's identical normalized
inputs, chronological split, target mask, loss/evaluation code, and seed policy.

| Artifact name | Literature anchor | Local status | Material deviation |
|---|---|---|---|
| historical_average | standard seasonal baseline | native | training-only sensor/time-of-week mean |
| persistence | standard naive baseline | native | none |
| ridge_ar | classical autoregression | native | node-shared direct ridge; sampled training rows |
| linear | linear direct forecaster | native | node-shared |
| mlp | nonlinear direct forecaster | native | node-shared two-layer MLP |
| gru | GRU baseline | compact reproduction | shared node encoder; direct decoder |
| stgcn | Yu et al. (2018) | compact reproduction | one spectral spatial-temporal block |
| dcrnn | Li et al. (2018) | compact reproduction | diffusion-GRU encoder with direct rather than autoregressive decoder |
| graphwavenet | Wu et al. (2019) | compact reproduction | three dilated blocks and compact channels |
| agcrn | Bai et al. (2020) | compact reproduction | embedding-derived graph with shared recurrent weights |
| stid | Shao et al. (2022) | compact reproduction | identity/calendar/history formulation retained; compact head |
| staeformer | Liu et al. (2023) | faithful independent reproduction | published METR-LA/PEMS-BAY dimensions and temporal/spatial attention retained; PyTorch native attention |
| shiftguard | proposed | native | full registered method |

Only the faithful STAEformer configuration is treated as a locally reproduced
recent SOTA comparator. Compact canonical models are named as such in the paper;
their numbers are never presented as exact reproductions of published results.
Literature-reported values remain in a separate context table.

