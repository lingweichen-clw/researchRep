param(
    [string]$Python = "C:\Users\31396\.conda\envs\research\python.exe"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$OutputRoot = Join-Path $ProjectRoot "artifacts\convergence\cfdp_probes\e5_final_pooling_attribution_seed42"
$LogRoot = Join-Path $ProjectRoot "artifacts\convergence\logs\e5_final_pooling_attribution_seed42"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable does not exist: $Python"
}
foreach ($Path in @($OutputRoot, $LogRoot)) {
    if (Test-Path -LiteralPath $Path) {
        throw "Refusing to reuse or overwrite formal output: $Path"
    }
}

$Experiments = @(
    @{
        Name = "01_local12"
        Config = Join-Path $ProjectRoot "configs\metrla_e5_final_sym_profile_local12_v1.yaml"
        Checkpoint = Join-Path $ProjectRoot "artifacts\metrla_e5_final_sym_profile_local12_seed42\pretrain_best_relation.pt"
    },
    @{
        Name = "02_global288"
        Config = Join-Path $ProjectRoot "configs\metrla_e5_final_sym_profile_v1.yaml"
        Checkpoint = Join-Path $ProjectRoot "artifacts\metrla_e5_final_sym_profile_seed42\pretrain_best_relation.pt"
    }
)
foreach ($Experiment in $Experiments) {
    foreach ($InputPath in @($Experiment.Config, $Experiment.Checkpoint)) {
        if (-not (Test-Path -LiteralPath $InputPath -PathType Leaf)) {
            throw "Missing probe input: $InputPath"
        }
    }
}

New-Item -ItemType Directory -Force -Path $OutputRoot, $LogRoot | Out-Null

function Invoke-Probe {
    param([hashtable]$Experiment)

    $Name = $Experiment.Name
    $RunRoot = Join-Path $OutputRoot $Name
    $Output = Join-Path $RunRoot "metrics.json"
    $Stdout = Join-Path $LogRoot ($Name + ".out.log")
    $Stderr = Join-Path $LogRoot ($Name + ".err.log")
    New-Item -ItemType Directory -Path $RunRoot | Out-Null
    Set-Content -Encoding UTF8 -Path (Join-Path $LogRoot ($Name + ".started")) -Value (Get-Date -Format "yyyy-MM-dd HH:mm:ss")

    $Arguments = @(
        (Join-Path $ProjectRoot "scripts\diagnose_cfdp_probes.py"),
        "--config", $Experiment.Config,
        "--checkpoint", $Experiment.Checkpoint,
        "--output", $Output,
        "--split", "val",
        "--epochs", "5"
    )
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
        throw "Probe $Name failed with exit code $($Process.ExitCode); see $Stderr"
    }
    $Result = Get-Content -LiteralPath $Output -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($null -eq $Result.validation.teacher_profile_oracle) {
        throw "Probe $Name did not produce teacher_profile_oracle metrics"
    }
    Set-Content -Encoding UTF8 -Path (Join-Path $LogRoot ($Name + ".completed")) -Value (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
}

foreach ($Experiment in $Experiments) {
    Invoke-Probe -Experiment $Experiment
}

Set-Content -Encoding UTF8 -Path (Join-Path $LogRoot "queue.completed") -Value (Get-Date -Format "yyyy-MM-dd HH:mm:ss")

