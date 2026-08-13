param(
    [string]$Python = "C:\Users\31396\.conda\envs\research\python.exe",
    [switch]$Smoke
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Config = Join-Path $ProjectRoot "configs\metrla_e5_final_sym_profile_v1.yaml"
$PretrainedCheckpoint = Join-Path $ProjectRoot "artifacts\metrla_e5_final_sym_profile_seed42\pretrain_best_relation.pt"
$PretrainedBank = Join-Path $ProjectRoot "artifacts\metrla_bank_e5_final_sym_profile_global288_seed42"
$RandomRoot = Join-Path $ProjectRoot "artifacts\metrla_e5_final_sym_profile_global288_random_seed42"
$RandomCheckpoint = Join-Path $RandomRoot "random_checkpoint.pt"
$RandomBank = Join-Path $ProjectRoot "artifacts\metrla_bank_e5_final_sym_profile_global288_random_seed42"
$QueueName = if ($Smoke) { "downstream_global288_attribution_smoke" } else { "downstream_global288_attribution" }
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
foreach ($InputPath in @($Config, $PretrainedCheckpoint, $PretrainedBank)) {
    if (-not (Test-Path -LiteralPath $InputPath)) {
        throw "Missing downstream input: $InputPath"
    }
}

if (-not (Test-Path -LiteralPath $RandomCheckpoint -PathType Leaf)) {
    if (Test-Path -LiteralPath $RandomRoot) {
        throw "Random checkpoint directory exists but checkpoint is missing: $RandomRoot"
    }
    New-Item -ItemType Directory -Path $RandomRoot | Out-Null
    & $Python (Join-Path $ProjectRoot "scripts\init_random_checkpoint.py") `
        --config $Config `
        --output $RandomCheckpoint `
        --seed 42
    if ($LASTEXITCODE -ne 0) {
        throw "init_random_checkpoint.py failed with exit code $LASTEXITCODE"
    }
}
if (-not (Test-Path -LiteralPath $RandomBank -PathType Container)) {
    throw "Missing aligned random Bank: $RandomBank"
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
        Name = "global288_base_only_seed42"
        Mode = "base_only"
        Checkpoint = $PretrainedCheckpoint
        Bank = $PretrainedBank
    },
    @{
        Order = "02"
        Name = "global288_pretrained_offset_decay_seed42"
        Mode = "learned_topk_offset_decay_horizon"
        Checkpoint = $PretrainedCheckpoint
        Bank = $PretrainedBank
    },
    @{
        Order = "03"
        Name = "global288_random_offset_decay_seed42"
        Mode = "learned_topk_offset_decay_horizon"
        Checkpoint = $RandomCheckpoint
        Bank = $RandomBank
    }
)

foreach ($Experiment in $Experiments) {
    $RunName = $Experiment.Name
    $RelativeRun = "convergence/$QueueName/$RunName"
    $RunRoot = Join-Path $OutputRoot $RunName
    $TrainArguments = @(
        (Join-Path $ProjectRoot "scripts\train_downstream.py"),
        "--config", $Config,
        "--pretrained-checkpoint", $Experiment.Checkpoint,
        "--bank", $Experiment.Bank,
        "--mode", $Experiment.Mode,
        "--candidate-protocol", "exact_calendar",
        "--level-weight", "0",
        "--seed", "42",
        "--run-name", $RelativeRun
    )
    if ($Smoke) {
        $TrainArguments += @("--epochs", "1", "--max-batches", "1")
    }
    Invoke-PythonStep -Name ($Experiment.Order + "_train_" + $RunName) -Arguments $TrainArguments

    $DownstreamCheckpoint = Join-Path $RunRoot "downstream_best.pt"
    if (-not (Test-Path -LiteralPath $DownstreamCheckpoint -PathType Leaf)) {
        throw "Training did not produce downstream checkpoint: $DownstreamCheckpoint"
    }
    $EvalArguments = @(
        (Join-Path $ProjectRoot "scripts\evaluate.py"),
        "--config", $Config,
        "--pretrained-checkpoint", $Experiment.Checkpoint,
        "--downstream-checkpoint", $DownstreamCheckpoint,
        "--bank", $Experiment.Bank,
        "--split", "val",
        "--candidate-protocol", "exact_calendar"
    )
    if ($Smoke) {
        $EvalArguments += @("--max-batches", "1")
    }
    Invoke-PythonStep -Name ($Experiment.Order + "_evaluate_" + $RunName) -Arguments $EvalArguments

    $DiagnosticOutput = Join-Path $RunRoot "branch_diagnostics_val.json"
    $DiagnosticArguments = @(
        (Join-Path $ProjectRoot "scripts\diagnose_downstream.py"),
        "--config", $Config,
        "--pretrained-checkpoint", $Experiment.Checkpoint,
        "--downstream-checkpoint", $DownstreamCheckpoint,
        "--bank", $Experiment.Bank,
        "--split", "val",
        "--candidate-protocol", "exact_calendar",
        "--output", $DiagnosticOutput
    )
    if ($Smoke) {
        $DiagnosticArguments += @("--max-batches", "1")
    }
    Invoke-PythonStep -Name ($Experiment.Order + "_diagnose_" + $RunName) -Arguments $DiagnosticArguments
}

Set-Content -Encoding UTF8 -Path (Join-Path $LogRoot "queue.completed") -Value (Get-Date -Format "yyyy-MM-dd HH:mm:ss")

