param(
    [string]$Python = "C:\Users\31396\.conda\envs\research\python.exe"
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$QueueName = "downstream_tgge_v3_error_aware_fulltrain_queue_v4_signed_horizon"
$OutputRoot = Join-Path $ProjectRoot ("artifacts\convergence\" + $QueueName)
$LogRoot = Join-Path $ProjectRoot ("artifacts\convergence\logs\" + $QueueName)

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable does not exist: $Python"
}
foreach ($Path in @($OutputRoot, $LogRoot)) {
    if (Test-Path -LiteralPath $Path) {
        throw "Refusing to reuse or overwrite queue output: $Path"
    }
}
New-Item -ItemType Directory -Force -Path $OutputRoot, $LogRoot | Out-Null
Set-Content -Encoding UTF8 -Path (Join-Path $LogRoot "queue.started") -Value (Get-Date -Format "yyyy-MM-dd HH:mm:ss")

function Invoke-PythonStep {
    param(
        [string]$Name,
        [string[]]$Arguments
    )
    $Stdout = Join-Path $LogRoot ($Name + ".out.log")
    $Stderr = Join-Path $LogRoot ($Name + ".err.log")
    $Process = Start-Process `
        -FilePath $Python `
        -ArgumentList $Arguments `
        -WorkingDirectory $ProjectRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $Stdout `
        -RedirectStandardError $Stderr `
        -PassThru `
        -Wait
    if ($Process.ExitCode -ne 0) {
        Set-Content -Encoding UTF8 -Path (Join-Path $LogRoot ($Name + ".failed")) -Value (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
        throw "$Name failed with exit code $($Process.ExitCode); see $Stderr"
    }
    Set-Content -Encoding UTF8 -Path (Join-Path $LogRoot ($Name + ".completed")) -Value (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
}

$Pretrained = Join-Path $ProjectRoot "artifacts\convergence\tgge_single_view_v3_higher_order_reconstruction2\pretrain\pretrain_best_relation.pt"
$Bank = Join-Path $ProjectRoot "artifacts\convergence\tgge_single_view_v3_higher_order_reconstruction2\bank"
$StgcnBase = Join-Path $ProjectRoot "artifacts\quarantine\20260820_previous_downstream\artifacts\metrla_stgcn_base_only_fulltrain_seed42_local_v3\downstream_best.pt"
$GraphWaveNetBase = Join-Path $ProjectRoot "artifacts\quarantine\20260820_previous_downstream\artifacts\downstream_tgge_v3_graphwavenet_base_only_seed42\downstream_best.pt"

$Experiments = @(
    @{
        Order = "01"
        Name = "downstream_tgge_v3_stgcn_error_aware_fulltrain_seed42"
        Config = Join-Path $ProjectRoot "configs\metrla_stgcn_tgge_v3_error_aware_fulltrain_v1.yaml"
        BaseCheckpoint = $StgcnBase
    },
    @{
        Order = "02"
        Name = "downstream_tgge_v3_graphwavenet_error_aware_fulltrain_seed42"
        Config = Join-Path $ProjectRoot "configs\metrla_graphwavenet_tgge_v3_error_aware_fulltrain_v1.yaml"
        BaseCheckpoint = $GraphWaveNetBase
    }
)

foreach ($Experiment in $Experiments) {
    foreach ($InputPath in @($Experiment.Config, $Pretrained, $Experiment.BaseCheckpoint)) {
        if (-not (Test-Path -LiteralPath $InputPath -PathType Leaf)) {
            throw "Missing full-train input: $InputPath"
        }
    }
    if (-not (Test-Path -LiteralPath $Bank -PathType Container)) {
        throw "Missing full-train Bank directory: $Bank"
    }
    $RunName = "convergence/$QueueName/$($Experiment.Name)"
    $TrainArguments = @(
        (Join-Path $ProjectRoot "scripts\train_downstream.py"),
        "--config", $Experiment.Config,
        "--pretrained-checkpoint", $Pretrained,
        "--bank", $Bank,
        "--base-checkpoint", $Experiment.BaseCheckpoint,
        "--mode", "learned_topk_error_aware",
        "--candidate-protocol", "exact_calendar",
        "--level-weight", "0",
        "--seed", "42",
        "--run-name", $RunName
    )
    Invoke-PythonStep -Name ($Experiment.Order + "_train_" + $Experiment.Name) -Arguments $TrainArguments
}

Set-Content -Encoding UTF8 -Path (Join-Path $LogRoot "queue.completed") -Value (Get-Date -Format "yyyy-MM-dd HH:mm:ss")

