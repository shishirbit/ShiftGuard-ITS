$ErrorActionPreference = "Continue"
$workspace = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $workspace
$env:PYTHONPATH = "$workspace\.runtime-deps;C:\codex-pt;$workspace\src"
$python = "C:\Users\shish\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$log = Join-Path $workspace "artifacts\logs\group_coverage_background.log"
& $python -u scripts\run_group_coverage_matrix.py *>> $log
