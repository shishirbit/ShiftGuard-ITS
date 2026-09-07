$ErrorActionPreference = "Continue"
$workspace = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $workspace

$logDirectory = Join-Path $workspace "artifacts\logs"
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$logPath = Join-Path $logDirectory "stress_background.log"
$env:PYTHONPATH = "$workspace\.runtime-deps;C:\codex-pt;$workspace\src"
$python = "C:\Users\shish\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

& $python -u scripts\run_stress_matrix.py --rates 0.1 0.2 0.3 0.4 0.5 *>> $logPath
