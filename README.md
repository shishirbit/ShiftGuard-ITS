# ShiftGuard-ITS

Reproducible research repository for reliable traffic-speed forecasting on
METR-LA and PEMS-BAY under temporal regime shift and structured sensor outage.

The research question is not simply whether another backbone can
reduce canonical test MAE. It is whether a forecasting system can preserve
point accuracy, calibrated node/horizon-level uncertainty, and safe selective
coverage when traffic regimes change and contiguous groups of sensors fail.

## Status

- Repository and preregistered protocol: complete
- Literature evidence matrix and gap synthesis: complete; recheck before submission
- Dataset acquisition, checksum verification, preprocessing, and EDA: complete
- Models, registered ablations, stress generators, and calibration variants:
  implemented and covered by automated tests
- Clean evaluation: complete for classical controls and five-seed neural models
- Ablations: 60/60 complete
- Confirmatory compound-shift evaluation: 640/640 complete
- Efficiency benchmarks: 44/44 complete
- Statistical analysis and generated tables: complete
- LaTeX manuscript: quantitative draft compiled and visually verified

The novelty audit supports the compound-shift protocol, structured-outage
training, and forecast-origin conformal risk control. It does not support a
universal clean-accuracy SOTA claim or the necessity of every candidate
architectural component.

All numerical claims in the paper must be generated from versioned experiment
artifacts. Reported literature results will be clearly separated from locally
reproduced results.

The exact Python, package, CUDA, GPU, and driver snapshot used for confirmatory
runs is generated at `artifacts/environment.json`.

## Data availability

METR-LA and PEMS-BAY are available from the official
[DCRNN data-preparation page](https://github.com/liyaguang/DCRNN#data-preparation)
and its [linked dataset archive](https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX).
PEMS-BAY originates from the California Department of Transportation
[Performance Measurement System](https://pems.dot.ca.gov/). Raw benchmark files
are intentionally excluded from this repository.

## Reproducibility policy

- Canonical chronological 70/10/20 split, with no future-to-past leakage.
- Five-minute resolution; 12 input steps and 12 output horizons.
- At least five seeds for headline comparisons when compute permits.
- MAE, RMSE, and masked MAPE for point forecasts.
- Coverage, interval width, Winkler score, and worst-group coverage for UQ.
- Risk-coverage/AURC and selective MAE for abstaining forecasts.
- Structured outages, bursts, and regime-stratified temporal stress tests.
- Statistical tests, effect sizes, parameters, FLOPs, latency, and peak memory.

## Quick start

```powershell
python scripts/download_data.py
python scripts/preprocess.py METR-LA
python scripts/evaluate_classical.py METR-LA
python scripts/train_ridge.py METR-LA
python scripts/train.py --dataset METR-LA --model shiftguard --seed 1
python scripts/run_matrix.py --models shiftguard --datasets METR-LA PEMS-BAY --seeds 1 12 42 123 2026 --skip-stress
python scripts/run_stress_matrix.py
python scripts/aggregate_results.py
python scripts/statistical_tests.py
python scripts/stress_statistical_tests.py
python scripts/benchmark_efficiency.py --models staeformer shiftguard
```
