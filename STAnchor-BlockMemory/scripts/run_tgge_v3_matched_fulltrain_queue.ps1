param(
    [string]$Python = "C:\Users\31396\.conda\envs\research\python.exe",
    [switch]$ResumeAfterBase
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$QueueName = "downstream_tgge_v3_matched_fulltrain_queue"
$OutputRoot = Join-Path $ProjectRoot ("artifacts\convergence\" + $QueueName)
$LogRoot = Join-Path $ProjectRoot ("artifacts\convergence\logs\" + $QueueName)

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable does not exist: $Python"
}
foreach ($Path in @($OutputRoot, $LogRoot)) {
    if ((-not $ResumeAfterBase) -and (Test-Path -LiteralPath $Path)) {
        throw "Refusing to reuse or overwrite queue output: $Path"
    }
}
New-Item -ItemType Directory -Force -Path $OutputRoot, $LogRoot | Out-Null
if ($ResumeAfterBase) {
    Set-Content -Encoding UTF8 -Path (Join-Path $LogRoot "queue.resumed_after_base") -Value (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
} else {
    Set-Content -Encoding UTF8 -Path (Join-Path $LogRoot "queue.started") -Value (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
}

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
$TrainScript = Join-Path $ProjectRoot "scripts\train_downstream.py"
$EvaluateScript = Join-Path $ProjectRoot "scripts\evaluate.py"
$DiagnoseScript = Join-Path $ProjectRoot "scripts\diagnose_downstream.py"

# The aware stage uses the newest StructuredErrorCorrector implementation and the
# freshly trained matching base checkpoint for the same backbone.
$Experiments = @(
    @{
        Order = "01"
        Name = "downstream_tgge_v3_stgcn_base_only_fulltrain_seed42"
        Config = Join-Path $ProjectRoot "configs\metrla_stgcn_tgge_v3_base_only_fulltrain_v1.yaml"
        Mode = "base_only"
        BaseCheckpoint = $null
    },
    @{
        Order = "02"
        Name = "downstream_tgge_v3_stgcn_error_aware_fulltrain_seed42"
        Config = Join-Path $ProjectRoot "configs\metrla_stgcn_tgge_v3_error_aware_fulltrain_v1.yaml"
        Mode = "learned_topk_error_aware"
        BaseCheckpoint = Join-Path $OutputRoot "downstream_tgge_v3_stgcn_base_only_fulltrain_seed42\downstream_best.pt"
    },
    @{
        Order = "03"
        Name = "downstream_tgge_v3_graphwavenet_base_only_fulltrain_seed42"
        Config = Join-Path $ProjectRoot "configs\metrla_graphwavenet_tgge_v3_base_only_fulltrain_v1.yaml"
        Mode = "base_only"
        BaseCheckpoint = $null
    },
    @{
        Order = "04"
        Name = "downstream_tgge_v3_graphwavenet_error_aware_fulltrain_seed42"
        Config = Join-Path $ProjectRoot "configs\metrla_graphwavenet_tgge_v3_error_aware_fulltrain_v1.yaml"
        Mode = "learned_topk_error_aware"
        BaseCheckpoint = Join-Path $OutputRoot "downstream_tgge_v3_graphwavenet_base_only_fulltrain_seed42\downstream_best.pt"
    }
)
if ($ResumeAfterBase) {
    $Experiments = @($Experiments | Where-Object { $_.Order -ne "01" })
}

foreach ($Experiment in $Experiments) {
    foreach ($InputPath in @($Experiment.Config, $Pretrained, $Bank)) {
        if (-not (Test-Path -LiteralPath $InputPath)) {
            throw "Missing downstream input: $InputPath"
        }
    }
    if ($Experiment.BaseCheckpoint -and -not (Test-Path -LiteralPath $Experiment.BaseCheckpoint -PathType Leaf)) {
        throw "Missing freshly matched base checkpoint: $($Experiment.BaseCheckpoint)"
    }

    $RunName = "convergence/$QueueName/$($Experiment.Name)"
    $TrainArguments = @(
        $TrainScript,
        "--config", $Experiment.Config,
        "--pretrained-checkpoint", $Pretrained,
        "--bank", $Bank,
        "--mode", $Experiment.Mode,
        "--candidate-protocol", "exact_calendar",
        "--level-weight", "0",
        "--seed", "42",
        "--run-name", $RunName,
        "--disable-early-stopping"
    )
    if ($Experiment.BaseCheckpoint) {
        $TrainArguments += @("--base-checkpoint", $Experiment.BaseCheckpoint)
    }
    Invoke-PythonStep -Name ($Experiment.Order + "_train_" + $Experiment.Name) -Arguments $TrainArguments

    $RunRoot = Join-Path $OutputRoot $Experiment.Name
    $DownstreamCheckpoint = Join-Path $RunRoot "downstream_best.pt"
    if (-not (Test-Path -LiteralPath $DownstreamCheckpoint -PathType Leaf)) {
        throw "Training did not produce downstream checkpoint: $DownstreamCheckpoint"
    }

    $EvalArguments = @(
        $EvaluateScript,
        "--config", $Experiment.Config,
        "--downstream-checkpoint", $DownstreamCheckpoint,
        "--split", "val",
        "--candidate-protocol", "exact_calendar"
    )
    if ($Experiment.Mode -ne "base_only") {
        $EvalArguments += @(
            "--pretrained-checkpoint", $Pretrained,
            "--bank", $Bank
        )
    }
    Invoke-PythonStep -Name ($Experiment.Order + "_evaluate_" + $Experiment.Name) -Arguments $EvalArguments

    if ($Experiment.Mode -ne "base_only") {
        $DiagnosticOutput = Join-Path $RunRoot "diagnostic_val.json"
        $DiagnosticArguments = @(
            $DiagnoseScript,
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
}

Set-Content -Encoding UTF8 -Path (Join-Path $LogRoot "queue.completed") -Value (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
