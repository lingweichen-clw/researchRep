param(
    [string]$Python = "C:\Users\31396\.conda\envs\research\python.exe",
    [switch]$Smoke
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$QueueName = if ($Smoke) {
    "downstream_global288_posthoc_error_aware_smoke"
} else {
    "downstream_global288_posthoc_error_aware"
}
$OutputRoot = Join-Path $ProjectRoot ("artifacts\convergence\" + $QueueName)
$LogRoot = Join-Path $ProjectRoot ("artifacts\convergence\logs\" + $QueueName)
$PretrainedCheckpoint = Join-Path $ProjectRoot "artifacts\metrla_e5_final_latent48_global288_seed42\pretrain_best_relation.pt"
$Bank = Join-Path $ProjectRoot "artifacts\metrla_bank_e5_final_latent48_global288_local_seed42"
$BaseCheckpoint = Join-Path $ProjectRoot "artifacts\convergence\downstream_global288_controlled_init\latent48_base_only\downstream_best.pt"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable does not exist: $Python"
}
foreach ($Path in @($PretrainedCheckpoint, $BaseCheckpoint)) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Missing PostHoc input checkpoint: $Path"
    }
}
if (-not (Test-Path -LiteralPath $Bank -PathType Container)) {
    throw "Missing PostHoc Bank: $Bank"
}
foreach ($Path in @($OutputRoot, $LogRoot)) {
    if (Test-Path -LiteralPath $Path) {
        throw "Refusing to reuse or overwrite queue output: $Path"
    }
}

New-Item -ItemType Directory -Force -Path $OutputRoot, $LogRoot | Out-Null

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
        throw "Step $Name failed with exit code $($Process.ExitCode); see $Stderr"
    }
    Set-Content -Encoding UTF8 -Path (Join-Path $LogRoot ($Name + ".completed")) -Value (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
}

$Experiments = @(
    @{
        Order = "01"
        Label = "latent48_posthoc_error_aware_basecap"
        Config = Join-Path $ProjectRoot "configs\metrla_e5_final_latent48_posthoc_error_aware_v1.yaml"
    },
    @{
        Order = "02"
        Label = "latent48_posthoc_error_aware_wide"
        Config = Join-Path $ProjectRoot "configs\metrla_e5_final_latent48_posthoc_error_aware_wide_v1.yaml"
    }
)

foreach ($Experiment in $Experiments) {
    if (-not (Test-Path -LiteralPath $Experiment.Config -PathType Leaf)) {
        throw "Missing PostHoc config: $($Experiment.Config)"
    }
    $RunName = "convergence/$QueueName/$($Experiment.Label)"
    $RunRoot = Join-Path $OutputRoot $Experiment.Label
    $TrainArguments = @(
        (Join-Path $ProjectRoot "scripts\train_downstream.py"),
        "--config", $Experiment.Config,
        "--pretrained-checkpoint", $PretrainedCheckpoint,
        "--base-checkpoint", $BaseCheckpoint,
        "--bank", $Bank,
        "--candidate-protocol", "exact_calendar",
        "--level-weight", "0",
        "--seed", "42",
        "--run-name", $RunName
    )
    if ($Smoke) {
        $TrainArguments += @("--epochs", "1", "--max-batches", "1")
    }
    Invoke-PythonStep -Name ($Experiment.Order + "_train_" + $Experiment.Label) -Arguments $TrainArguments

    $DownstreamCheckpoint = Join-Path $RunRoot "downstream_best.pt"
    if (-not (Test-Path -LiteralPath $DownstreamCheckpoint -PathType Leaf)) {
        throw "Training did not produce downstream checkpoint: $DownstreamCheckpoint"
    }
    $EvalArguments = @(
        (Join-Path $ProjectRoot "scripts\evaluate.py"),
        "--config", $Experiment.Config,
        "--pretrained-checkpoint", $PretrainedCheckpoint,
        "--downstream-checkpoint", $DownstreamCheckpoint,
        "--bank", $Bank,
        "--split", "val",
        "--candidate-protocol", "exact_calendar"
    )
    if ($Smoke) {
        $EvalArguments += @("--max-batches", "1")
    }
    Invoke-PythonStep -Name ($Experiment.Order + "_evaluate_" + $Experiment.Label) -Arguments $EvalArguments

    $DiagnosticOutput = Join-Path $RunRoot "branch_diagnostics_val.json"
    $DiagnosticArguments = @(
        (Join-Path $ProjectRoot "scripts\diagnose_downstream.py"),
        "--config", $Experiment.Config,
        "--pretrained-checkpoint", $PretrainedCheckpoint,
        "--downstream-checkpoint", $DownstreamCheckpoint,
        "--bank", $Bank,
        "--split", "val",
        "--candidate-protocol", "exact_calendar",
        "--output", $DiagnosticOutput
    )
    if ($Smoke) {
        $DiagnosticArguments += @("--max-batches", "1")
    }
    Invoke-PythonStep -Name ($Experiment.Order + "_diagnose_" + $Experiment.Label) -Arguments $DiagnosticArguments
}

Set-Content -Encoding UTF8 -Path (Join-Path $LogRoot "queue.completed") -Value (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
