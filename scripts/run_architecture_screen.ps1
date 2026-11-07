# Screens ShiftGuard architecture candidates on one seed, then ranks them on
# validation data only. Safe to relaunch: training resumes and skips finished runs.
$ErrorActionPreference = "Continue"
$workspace = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $workspace

$logPath = Join-Path $workspace "artifacts\logs\architecture_screen.log"
$env:PYTHONPATH = "$workspace\.runtime-deps;C:\codex-pt;$workspace\src;$workspace\scripts"
$env:OMP_NUM_THREADS = "4"
$python = "C:\Users\shish\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$candidates = & $python -c "from shiftguard.registry import CANDIDATE_MODELS as c; print(' '.join(c))"
$candidates = $candidates -split ' '

foreach ($model in $candidates) {
    & $python -u scripts\run_matrix.py --models $model --datasets METR-LA PEMS-BAY --seeds 42 `
        --epochs 100 --patience 15 --batch-size 32 --resume --skip-stress *>> $logPath
}
# The recipe-equipped STAEformer forms are scored too, as the bar to clear.
& $python -u scripts\validation_model_selection.py --seeds 42 `
    --variants shiftguard_noreliability shiftguard_noreliability_norouter @candidates `
    staeformer_outageaug staeformer_outageaug_laplace `
    --output architecture_screen_validation.json *>> $logPath
"$(Get-Date -Format 'yyyy-MM-dd HH:mm') screen finished" | Out-File -FilePath $logPath -Append -Encoding utf8
