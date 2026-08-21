$ErrorActionPreference = "Stop"
$ProjectRoot = Join-Path $PSScriptRoot "..\STAnchor-BlockMemory"
Set-Location $ProjectRoot
$Python = "C:\Users\31396\.conda\envs\research\python.exe"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable does not exist: $Python"
}

$Pretrained = Join-Path $ProjectRoot "artifacts\convergence\tgge_single_view_v3_higher_order_reconstruction2\pretrain\pretrain_best_relation.pt"
$Bank = Join-Path $ProjectRoot "artifacts\convergence\tgge_single_view_v3_higher_order_reconstruction2\bank"
$StgcnBase = Join-Path $ProjectRoot "artifacts\metrla_stgcn_base_only_fulltrain_seed42_local_v3\downstream_best.pt"
$GraphWaveNetBase = Join-Path $ProjectRoot "artifacts\convergence\downstream_tgge_v3_graphwavenet_base_only_seed42\downstream_best.pt"

foreach ($Path in @($Pretrained, $Bank, $StgcnBase, $GraphWaveNetBase)) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Required experiment artifact is missing: $Path"
    }
}

function Invoke-DownstreamStep {
    param(
        [string]$Name,
        [string[]]$Arguments
    )
    Write-Host "[$(Get-Date -Format s)] START $Name"
    & $Python $Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
    Write-Host "[$(Get-Date -Format s)] DONE $Name"
}

Invoke-DownstreamStep -Name "stgcn_v3_error_aware" -Arguments @(
    "scripts/train_downstream.py",
    "--config", "configs/metrla_stgcn_tgge_v3_error_aware_posthoc_v1.yaml",
    "--pretrained-checkpoint", $Pretrained,
    "--bank", $Bank,
    "--base-checkpoint", $StgcnBase
)

Invoke-DownstreamStep -Name "graphwavenet_v3_horizon_only" -Arguments @(
    "scripts/train_downstream.py",
    "--config", "configs/metrla_graphwavenet_tgge_v3_downstream_v1.yaml",
    "--pretrained-checkpoint", $Pretrained,
    "--bank", $Bank,
    "--run-name", "convergence/downstream_tgge_v3_graphwavenet_horizon_only_seed42"
)

Invoke-DownstreamStep -Name "graphwavenet_v3_error_aware" -Arguments @(
    "scripts/train_downstream.py",
    "--config", "configs/metrla_graphwavenet_tgge_v3_error_aware_posthoc_v1.yaml",
    "--pretrained-checkpoint", $Pretrained,
    "--bank", $Bank,
    "--base-checkpoint", $GraphWaveNetBase
)
