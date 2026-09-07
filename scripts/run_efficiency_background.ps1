$ErrorActionPreference = "Continue"
$workspace = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $workspace

$logDirectory = Join-Path $workspace "artifacts\logs"
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$logPath = Join-Path $logDirectory "efficiency_background.log"
$env:PYTHONPATH = "$workspace\.runtime-deps;C:\codex-pt;$workspace\src"
$python = "C:\Users\shish\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

& $python -u scripts\benchmark_efficiency.py `
    --models persistence linear mlp gru stgcn dcrnn graphwavenet agcrn stid staeformer shiftguard `
    --batch-sizes 1 8 `
    --warmup 20 `
    --repeats 100 *>> $logPath
