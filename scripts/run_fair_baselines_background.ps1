$ErrorActionPreference = "Stop"
$workspace = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $workspace

$logDirectory = Join-Path $workspace "artifacts\logs"
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$logPath = Join-Path $logDirectory "fair_baselines_background.log"
$env:PYTHONPATH = "$workspace\.runtime-deps;C:\codex-pt;$workspace\src"
$python = "C:\Users\shish\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

# Cheapest models first; STAEformer is ~80% of the compute budget.
$models = @(
    "stid_outageaug",
    "dcrnn_outageaug",
    "graphwavenet_outageaug",
    "agcrn_outageaug",
    "staeformer_outageaug",
    "stid_outageaug_laplace",
    "dcrnn_outageaug_laplace",
    "graphwavenet_outageaug_laplace",
    "agcrn_outageaug_laplace",
    "staeformer_outageaug_laplace"
)

# Seed-major order so a full single-seed comparison is available early.
foreach ($seed in 1, 12, 42, 123, 2026) {
    foreach ($model in $models) {
        $extra = @()
        if ($model -like "staeformer_*") { $extra = @("--microbatch-size", "8", "--amp") }
        & $python -u scripts\run_matrix.py `
            --models $model `
            --datasets METR-LA PEMS-BAY `
            --seeds $seed `
            --epochs 100 --patience 15 --batch-size 32 `
            --resume --skip-stress @extra *>> $logPath
        & $python -u scripts\run_stress_matrix.py `
            --models $model `
            --seeds $seed `
            --rates 0.1 0.2 0.3 0.4 0.5 `
            --primary-only *>> $logPath
    }
}
& $python -u scripts\aggregate_results.py *>> $logPath
