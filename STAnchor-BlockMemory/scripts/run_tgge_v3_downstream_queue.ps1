param(
    [string]$Python = "C:\Users\31396\.conda\envs\research\python.exe"
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$QueueName = "downstream_tgge_v3_queue"
# Error-aware entries use the current StructuredErrorCorrector implementation
# through the learned_topk_error_aware mode and its posthoc_frozen_base config.
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
    Set-Content -Encoding UTF8 -Path (Join-Path $LogRoot ($Name + ".started")) -Value (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
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
$StgcnBase = Join-Path $ProjectRoot "artifacts\metrla_stgcn_base_only_fulltrain_seed42_local_v3\downstream_best.pt"
$GraphWaveNetBase = Join-Path $ProjectRoot "artifacts\convergence\downstream_tgge_v3_graphwavenet_base_only_seed42\downstream_best.pt"

$Experiments = @(
    @{
        Order = "01"
        Name = "downstream_tgge_v3_stgcn_error_aware_seed42"
        Config = Join-Path $ProjectRoot "configs\metrla_stgcn_tgge_v3_error_aware_posthoc_v1.yaml"
        Mode = "learned_topk_error_aware"
        BaseCheckpoint = $StgcnBase
    },
    @{
        Order = "02"
        Name = "downstream_tgge_v3_graphwavenet_horizon_only_seed42"
        Config = Join-Path $ProjectRoot "configs\metrla_graphwavenet_tgge_v3_downstream_v1.yaml"
        Mode = "learned_topk_offset_decay_horizon"
        BaseCheckpoint = $null
    },
    @{
        Order = "03"
        Name = "downstream_tgge_v3_graphwavenet_error_aware_seed42"
        Config = Join-Path $ProjectRoot "configs\metrla_graphwavenet_tgge_v3_error_aware_posthoc_v1.yaml"
        Mode = "learned_topk_error_aware"
        BaseCheckpoint = $GraphWaveNetBase
    }
)

foreach ($Experiment in $Experiments) {
    foreach ($InputPath in @($Experiment.Config, $Pretrained, $Bank)) {
        if (-not (Test-Path -LiteralPath $InputPath)) {
            throw "Missing downstream input: $InputPath"
        }
    }
    if ($Experiment.BaseCheckpoint -and -not (Test-Path -LiteralPath $Experiment.BaseCheckpoint -PathType Leaf)) {
        throw "Missing matched base checkpoint: $($Experiment.BaseCheckpoint)"
    }

    $RunName = "convergence/$QueueName/$($Experiment.Name)"
    $RunRoot = Join-Path $OutputRoot $Experiment.Name
    $TrainArguments = @(
        (Join-Path $ProjectRoot "scripts\train_downstream.py"),
        "--config", $Experiment.Config,
        "--pretrained-checkpoint", $Pretrained,
        "--bank", $Bank,
        "--mode", $Experiment.Mode,
        "--candidate-protocol", "exact_calendar",
        "--level-weight", "0",
        "--seed", "42",
        "--run-name", $RunName
    )
    if ($Experiment.BaseCheckpoint) {
        $TrainArguments += @("--base-checkpoint", $Experiment.BaseCheckpoint)
    }
    Invoke-PythonStep -Name ($Experiment.Order + "_train_" + $Experiment.Name) -Arguments $TrainArguments

    $DownstreamCheckpoint = Join-Path $RunRoot "downstream_best.pt"
    if (-not (Test-Path -LiteralPath $DownstreamCheckpoint -PathType Leaf)) {
        throw "Training did not produce downstream checkpoint: $DownstreamCheckpoint"
    }
    $EvalArguments = @(
        (Join-Path $ProjectRoot "scripts\evaluate.py"),
        "--config", $Experiment.Config,
        "--pretrained-checkpoint", $Pretrained,
        "--downstream-checkpoint", $DownstreamCheckpoint,
        "--bank", $Bank,
        "--split", "val",
        "--candidate-protocol", "exact_calendar"
    )
    Invoke-PythonStep -Name ($Experiment.Order + "_evaluate_" + $Experiment.Name) -Arguments $EvalArguments

    $DiagnosticOutput = Join-Path $RunRoot "diagnostic_val.json"
    $DiagnosticArguments = @(
        (Join-Path $ProjectRoot "scripts\diagnose_downstream.py"),
        "--config", $Experiment.Config,
        "--pretrained-checkpoint", $Pretrained,
        "--downstream-checkpoint", $DownstreamCheckpoint,
        "--bank", $Bank,
        "--split", "val",
        "--candidate-protocol", "exact_calendar",
        "--output", $DiagnosticOutput
    )
    Invoke-PythonStep -Name ($Experiment.Order + "_diagnose_" + $Experiment.Name) -Arguments $DiagnosticArguments
}

Set-Content -Encoding UTF8 -Path (Join-Path $LogRoot "queue.completed") -Value (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
