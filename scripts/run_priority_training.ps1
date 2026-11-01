$ErrorActionPreference = "Continue"
$workspace = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $workspace

$logPath = Join-Path $workspace "artifacts\logs\priority_training.log"
$env:PYTHONPATH = "$workspace\.runtime-deps;C:\codex-pt;$workspace\src"
$env:OMP_NUM_THREADS = "4"
$python = "C:\Users\shish\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$seeds = 1, 12, 42, 123, 2026

# Cheapest evidence first; every finished run is usable, and a rerun resumes.
$queue = @(
    "shiftguard_noreliability_noaugmentation",
    "shiftguard_noreliability_independentaugmentation",
    "stid_outageaug",
    "stid_outageaug_laplace",
    "shiftguard_noreliability_nomaskinput",
    "agcrn_outageaug",
    "graphwavenet_outageaug",
    "dcrnn_outageaug",
    "agcrn_outageaug_laplace",
    "graphwavenet_outageaug_laplace",
    "dcrnn_outageaug_laplace",
    "staeformer_outageaug",
    "staeformer_outageaug_laplace"
)
foreach ($model in $queue) {
    $extra = @()
    if ($model -like "staeformer_*") { $extra = @("--microbatch-size", "8", "--amp") }
    & $python -u scripts\run_matrix.py --models $model --datasets METR-LA PEMS-BAY --seeds $seeds `
        --epochs 100 --patience 15 --batch-size 32 --resume --skip-stress @extra *>> $logPath
}
# Evaluate the new checkpoints by rerunning scripts\run_deployment_background.ps1,
# which skips everything already scored.
