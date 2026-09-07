$ErrorActionPreference = "Stop"
$workspace = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $workspace

$logDirectory = Join-Path $workspace "artifacts\logs"
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$logPath = Join-Path $logDirectory "staeformer_background.log"
$env:PYTHONPATH = "$workspace\.runtime-deps;C:\codex-pt;$workspace\src"
$python = "C:\Users\shish\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

& $python -u scripts\run_matrix.py `
    --models staeformer `
    --datasets METR-LA PEMS-BAY `
    --seeds 1 12 42 123 2026 `
    --epochs 100 `
    --patience 15 `
    --batch-size 32 --microbatch-size 8 `
    --amp --resume `
    --output-root artifacts `
    --skip-stress *>> $logPath
