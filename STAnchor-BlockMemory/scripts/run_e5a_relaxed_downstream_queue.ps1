param(
    [string]$Python = "C:\Users\31396\.conda\envs\research\python.exe"
)

$ErrorActionPreference = "Stop"
# Python's structured logger writes INFO records to stderr. Keep that stream in
# the per-run log files without letting PowerShell treat it as a terminating
# error; the explicit LASTEXITCODE checks below still detect failed commands.
$PSNativeCommandUseErrorActionPreference = $false
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Config = Join-Path $ProjectRoot "configs\metrla_e5_offset_decay_relation_level0_v1.yaml"
$OutputRoot = Join-Path $ProjectRoot "artifacts\convergence\downstream_candidate_protocol"
$LogRoot = Join-Path $OutputRoot "logs"
New-Item -ItemType Directory -Force -Path $OutputRoot, $LogRoot | Out-Null

function Invoke-PythonStep {
    param(
        [string[]]$Arguments,
        [string]$StandardOutput,
        [string]$StandardError,
        [string]$Label
    )
    $Process = Start-Process `
        -FilePath $Python `
        -ArgumentList $Arguments `
        -WorkingDirectory $ProjectRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput $StandardOutput `
        -RedirectStandardError $StandardError `
        -PassThru `
        -Wait
    if ($Process.ExitCode -ne 0) {
        throw "$Label failed with exit code $($Process.ExitCode)"
    }
}

$Experiments = @(
    @{
        Name = "e5a_relaxed_pretrained_seed42"
        Pretrained = Join-Path $ProjectRoot "artifacts\metrla_e5a_offset_decay_seed42\pretrain_best_relation.pt"
        Bank = Join-Path $ProjectRoot "artifacts\metrla_bank_e5a_offset_decay_relation_seed42"
    },
    @{
        Name = "e5a_relaxed_random_seed42"
        Pretrained = Join-Path $ProjectRoot "artifacts\metrla_e3_target_random_seed42\random_checkpoint.pt"
        Bank = Join-Path $ProjectRoot "artifacts\metrla_bank_e3_target_random_seed42"
    }
)

foreach ($Experiment in $Experiments) {
    $RunName = $Experiment.Name
    $RunDir = Join-Path $OutputRoot $RunName
    $RetryIndex = 1
    while ((Test-Path $RunDir) -and -not (Test-Path (Join-Path $RunDir "downstream_best.pt"))) {
        $RunName = "{0}_retry{1:00}" -f $Experiment.Name, $RetryIndex
        $RunDir = Join-Path $OutputRoot $RunName
        $RetryIndex++
    }
    $LogPath = Join-Path $LogRoot ($RunName + ".log")
    $ErrorLogPath = $LogPath + ".err"
    if (Test-Path (Join-Path $RunDir "downstream_best.pt")) {
        Write-Host "SKIP existing checkpoint: $RunName"
        continue
    }

    $Arguments = @(
        (Join-Path $ProjectRoot "scripts\train_downstream.py"),
        "--config", $Config,
        "--pretrained-checkpoint", $Experiment.Pretrained,
        "--bank", $Experiment.Bank,
        "--mode", "learned_topk_offset_decay_horizon",
        "--candidate-protocol", "relaxed_calendar",
        "--run-name", ("convergence/downstream_candidate_protocol/" + $RunName)
    )
    Write-Host "START $RunName"
    Invoke-PythonStep `
        -Arguments $Arguments `
        -StandardOutput $LogPath `
        -StandardError $ErrorLogPath `
        -Label "Downstream training for $RunName"

    $Checkpoint = Join-Path $RunDir "downstream_best.pt"
    $EvalJson = Join-Path $RunDir "val_metrics.json"
    $EvalError = Join-Path $RunDir "val_metrics.stderr.log"
    $DiagJson = Join-Path $RunDir "branch_diagnostics_val.json"
    $EvalArguments = @(
        (Join-Path $ProjectRoot "scripts\evaluate.py"),
        "--config", $Config,
        "--pretrained-checkpoint", $Experiment.Pretrained,
        "--downstream-checkpoint", $Checkpoint,
        "--bank", $Experiment.Bank,
        "--split", "val",
        "--candidate-protocol", "relaxed_calendar"
    )
    Invoke-PythonStep `
        -Arguments $EvalArguments `
        -StandardOutput $EvalJson `
        -StandardError $EvalError `
        -Label "Validation evaluation for $RunName"

    $DiagStdout = Join-Path $RunDir "diagnose.stdout.log"
    $DiagError = Join-Path $RunDir "diagnose.stderr.log"
    $DiagArguments = @(
        (Join-Path $ProjectRoot "scripts\diagnose_downstream.py"),
        "--config", $Config,
        "--pretrained-checkpoint", $Experiment.Pretrained,
        "--downstream-checkpoint", $Checkpoint,
        "--bank", $Experiment.Bank,
        "--split", "val",
        "--candidate-protocol", "relaxed_calendar",
        "--output", $DiagJson
    )
    Invoke-PythonStep `
        -Arguments $DiagArguments `
        -StandardOutput $DiagStdout `
        -StandardError $DiagError `
        -Label "Validation diagnosis for $RunName"
    Write-Host "DONE $RunName"
}

Write-Host "Queue finished: $OutputRoot"
