$ErrorActionPreference = "Continue"
$workspace = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $workspace

$logDirectory = Join-Path $workspace "artifacts\logs"
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$env:PYTHONPATH = "$workspace\.runtime-deps;C:\codex-pt;$workspace\src"
$env:OMP_NUM_THREADS = "6"
$env:MKL_NUM_THREADS = "6"
$python = "C:\Users\shish\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

$primary = @("native", "trace_x1", "trace_x2", "sensor_s30", "region_s30", "hub_s15")
$all = $primary + @("sensor_s10", "region_s10", "hub_s10")
$point = @("feasible")
$rich = @("oracle_feedback", "feasible", "feasible_thinned")

# Two lanes fit in 16 GB of RAM next to one training process. Each lane finishes
# METR-LA, then PEMS-BAY, for its models in order; reruns skip finished scenarios.
$lanes = @(
    @(
        @{ models = @("shiftguard_noreliability", "staeformer"); scenarios = $all; arms = $rich },
        @{ models = @("dcrnn", "graphwavenet", "agcrn"); scenarios = $primary; arms = $point }
    ),
    @(
        @{ models = @("stid", "shiftguard_noreliability_noaugmentation"); scenarios = $primary; arms = $rich },
        @{ models = @("stid_outageaug_laplace", "dcrnn_outageaug_laplace", "graphwavenet_outageaug_laplace", "agcrn_outageaug_laplace", "staeformer_outageaug_laplace"); scenarios = $primary; arms = $rich },
        @{ models = @("stid_outageaug", "dcrnn_outageaug", "graphwavenet_outageaug", "agcrn_outageaug", "staeformer_outageaug"); scenarios = $primary; arms = $point },
        @{ models = @("shiftguard_noreliability_nomaskinput", "shiftguard", "shiftguard_noreliability_norouter", "shiftguard_noreliability_staticgraph", "shiftguard_noreliability_adaptivegraph", "shiftguard_noreliability_independentaugmentation"); scenarios = $primary; arms = $point }
    )
)

$workers = for ($index = 0; $index -lt $lanes.Count; $index++) {
    Start-Job -Name "lane$index" -ArgumentList $python, $workspace, $logDirectory, $index, $lanes[$index] -ScriptBlock {
        param($python, $workspace, $logDirectory, $index, $jobs)
        Set-Location -LiteralPath $workspace
        foreach ($job in $jobs) {
            $arguments = @("-u", "scripts\evaluate_deployment.py", "--models") + $job.models + @("--scenarios") + $job.scenarios + @("--arms") + $job.arms
            & $python @arguments *>> (Join-Path $logDirectory "deployment_lane$index.log")
        }
    }
}
$workers | Wait-Job | Out-Null
