param(
    [string]$Python = "C:\Users\31396\.conda\envs\research\python.exe"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$LogRoot = Join-Path $ProjectRoot "artifacts\convergence\visualization\e5_symnorm_local12_level0\logs"
$OutputRoot = Join-Path $ProjectRoot "artifacts\convergence\visualization\e5_symnorm_local12_level0"
$Config = Join-Path $ProjectRoot "configs\metrla_e5_final_symnorm_local12_v1.yaml"
$Checkpoint = Join-Path $ProjectRoot "artifacts\metrla_e5_final_symnorm_local12_seed42\pretrain_best_relation.pt"
$PretrainedBank = Join-Path $ProjectRoot "artifacts\metrla_bank_e5_final_symnorm_local12_seed42"
$RandomCheckpoint = Join-Path $ProjectRoot "artifacts\metrla_e5_final_symnorm_local12_random_seed42\random_checkpoint.pt"
$RandomBank = Join-Path $ProjectRoot "artifacts\metrla_bank_e5_final_symnorm_local12_random_seed42"

New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

function Invoke-ExperimentStep {
    param(
        [string]$Name,
        [string[]]$Arguments
    )
    $Stdout = Join-Path $LogRoot ($Name + ".out.log")
    $Stderr = Join-Path $LogRoot ($Name + ".err.log")
    $Started = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Set-Content -Encoding UTF8 -Path (Join-Path $LogRoot ($Name + ".started")) -Value $Started
    & $Python @Arguments 1> $Stdout 2> $Stderr
    if ($LASTEXITCODE -ne 0) {
        throw "Experiment step $Name failed with exit code $LASTEXITCODE; see $Stderr"
    }
    $Finished = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Set-Content -Encoding UTF8 -Path (Join-Path $LogRoot ($Name + ".completed")) -Value $Finished
}

if (-not (Test-Path -LiteralPath $Checkpoint)) {
    throw "Missing pretrained relation checkpoint: $Checkpoint"
}
if (Test-Path -LiteralPath $PretrainedBank) {
    throw "Refusing to reuse or overwrite pretrained Bank: $PretrainedBank"
}
if (Test-Path -LiteralPath $RandomCheckpoint) {
    throw "Refusing to overwrite random checkpoint: $RandomCheckpoint"
}
if (Test-Path -LiteralPath $RandomBank) {
    throw "Refusing to reuse or overwrite random Bank: $RandomBank"
}

Invoke-ExperimentStep "01_build_pretrained_bank" @(
    (Join-Path $ProjectRoot "scripts\build_bank.py"),
    "--config", $Config,
    "--checkpoint", $Checkpoint,
    "--output-dir", $PretrainedBank,
    "--dataset-name", "METR-LA"
)

Invoke-ExperimentStep "02_init_random_checkpoint" @(
    (Join-Path $ProjectRoot "scripts\init_random_checkpoint.py"),
    "--config", $Config,
    "--output", $RandomCheckpoint,
    "--seed", "42"
)

Invoke-ExperimentStep "03_build_random_bank" @(
    (Join-Path $ProjectRoot "scripts\build_bank.py"),
    "--config", $Config,
    "--checkpoint", $RandomCheckpoint,
    "--output-dir", $RandomBank,
    "--dataset-name", "METR-LA"
)

Invoke-ExperimentStep "04_visualize_full_val_level0" @(
    (Join-Path $ProjectRoot "scripts\visualize_retrieval.py"),
    "--version", "e5a",
    "--config", $Config,
    "--checkpoint", $Checkpoint,
    "--bank", $PretrainedBank,
    "--random-checkpoint", $RandomCheckpoint,
    "--random-bank", $RandomBank,
    "--split", "val",
    "--candidate-protocol", "exact_calendar",
    "--level-weight", "0",
    "--output-dir", $OutputRoot
)

Set-Content -Encoding UTF8 -Path (Join-Path $LogRoot "queue.completed") -Value (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
