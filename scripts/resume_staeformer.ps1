param(
    [Parameter(Mandatory = $true)]
    [int]$WaitForPid
)

$ErrorActionPreference = "Stop"
try {
    Wait-Process -Id $WaitForPid -ErrorAction SilentlyContinue
} catch {
    # The original process may already have exited before this watchdog starts.
}

$workspace = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $workspace
$env:PYTHONPATH = "C:\codex-pt;src"
$python = "C:\Users\shish\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

& $python -u scripts\run_matrix.py `
    --models staeformer `
    --datasets METR-LA PEMS-BAY `
    --seeds 1 12 42 123 2026 `
    --epochs 100 `
    --patience 15 `
    --batch-size 32 `
    --amp `
    --output-root artifacts `
    --skip-stress
