$ErrorActionPreference = "Stop"
$workspace = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $workspace

$logDirectory = Join-Path $workspace "artifacts\logs"
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$logPath = Join-Path $logDirectory "reference_model_background.log"
$env:PYTHONPATH = "$workspace\.runtime-deps;C:\codex-pt;$workspace\src"
$python = "C:\Users\shish\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$reference = "shiftguard_noreliability"
$ablations = @(
    "${reference}_staticgraph",
    "${reference}_adaptivegraph",
    "${reference}_norouter",
    "${reference}_noaugmentation",
    "${reference}_independentaugmentation"
)

# 1. Evaluation only: all stress severities and calibration controls.
& $python -u scripts\run_stress_matrix.py --models $reference `
    --rates 0.1 0.2 0.3 0.4 0.5 --skip-ablations *>> $logPath

# 2. Ablations of the reference model (small; ~6-14 min per run).
& $python -u scripts\run_matrix.py --models @ablations `
    --datasets METR-LA PEMS-BAY --seeds 1 12 42 123 2026 `
    --epochs 100 --patience 15 --batch-size 32 --resume --skip-stress *>> $logPath

# 3. Stress-evaluate the new ablations (finished evaluations are skipped).
& $python -u scripts\run_stress_matrix.py --rates 0.1 0.2 0.3 0.4 0.5 *>> $logPath
& $python -u scripts\run_natural_missingness_matrix.py *>> $logPath

# 4. Tables and paired statistics.
& $python -u scripts\aggregate_results.py *>> $logPath
& $python -u scripts\statistical_tests.py *>> $logPath
& $python -u scripts\stress_statistical_tests.py *>> $logPath

# 5. Latency needs an idle GPU: wait for the fair-baseline chain to finish.
while (Get-CimInstance Win32_Process -Filter "Name = 'powershell.exe'" |
        Where-Object { $_.CommandLine -like "*run_fair_baselines_background*" }) {
    Start-Sleep -Seconds 300
}
& $python -u scripts\benchmark_efficiency.py --models $reference `
    --batch-sizes 1 8 --warmup 20 --repeats 100 `
    --output artifacts\summary\efficiency_reference.csv *>> $logPath
& $python -u scripts\aggregate_results.py *>> $logPath
