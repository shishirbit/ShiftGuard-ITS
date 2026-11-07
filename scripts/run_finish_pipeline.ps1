# Finishes the experiments unattended. Safe to relaunch after a shutdown: it waits
# for runners already alive, and each step skips what is finished.
$ErrorActionPreference = "Continue"
$workspace = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $workspace

$logPath = Join-Path $workspace "artifacts\logs\finish_pipeline.log"
$donePath = Join-Path $workspace "artifacts\logs\finish_pipeline.done"
Remove-Item -LiteralPath $donePath -ErrorAction SilentlyContinue
$env:PYTHONPATH = "$workspace\.runtime-deps;C:\codex-pt;$workspace\src;$workspace\scripts"
$python = "C:\Users\shish\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

function Note($message) {
    "$(Get-Date -Format 'yyyy-MM-dd HH:mm') $message" | Out-File -FilePath $logPath -Append -Encoding utf8
}

function Wait-Runner($pattern) {
    while (Get-CimInstance Win32_Process -Filter "Name = 'powershell.exe'" |
            Where-Object { $_.CommandLine -like "*$pattern*" }) {
        Start-Sleep -Seconds 300
    }
}

function Missing-Training {
    $expected = & $python -c "from train import OUTAGE_AUGMENTED_BASELINES as b; print(' '.join(f'{m}_outageaug{s}' for m in b if m not in ('mlp', 'gru') for s in ('', '_laplace')))"
    $count = 0
    foreach ($model in ($expected -split ' ') + "shiftguard_noreliability_nomaskinput") {
        foreach ($dataset in "METR-LA", "PEMS-BAY") {
            foreach ($seed in 1, 12, 42, 123, 2026) {
                if (-not (Test-Path "artifacts\$dataset\$model\seed_$seed\metrics.json")) { $count++ }
            }
        }
    }
    return $count
}

Note "pipeline started"
Wait-Runner "run_priority_training"

# A failed run makes the queue move on, so repeat until nothing is missing.
for ($attempt = 1; $attempt -le 3 -and (Missing-Training) -gt 0; $attempt++) {
    Note "priority training attempt $attempt, missing $(Missing-Training)"
    & (Join-Path $PSScriptRoot "run_priority_training.ps1")
}
$missing = Missing-Training
Note "training finished, missing $missing"

Wait-Runner "run_deployment_background"
Note "scoring new checkpoints"
& (Join-Path $PSScriptRoot "run_deployment_background.ps1")

# Latency is only meaningful on an idle GPU; other projects may be using it.
$quiet = 0
$waited = 0
while ($quiet -lt 10 -and $waited -lt 1440) {
    $load = [int](& nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    if ($load -lt 5) { $quiet++ } else { $quiet = 0 }
    Start-Sleep -Seconds 60
    $waited++
}
$gpuIdle = $quiet -ge 10
Note "latency benchmark, gpu idle: $gpuIdle"
& $python -u scripts\benchmark_efficiency.py --models shiftguard_noreliability staeformer `
    --batch-sizes 1 8 --warmup 20 --repeats 100 `
    --output artifacts\summary\efficiency_reference.csv *>> $logPath

& $python -u scripts\aggregate_results.py *>> $logPath
& $python -u scripts\statistical_tests.py *>> $logPath
& $python -u scripts\build_deployment_report.py *>> $logPath
Note "pipeline finished"
"missing_training=$missing gpu_idle=$gpuIdle finished=$(Get-Date -Format 'yyyy-MM-dd HH:mm')" |
    Out-File -FilePath $donePath -Encoding utf8
