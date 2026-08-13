param(
    [string]$Python = "C:\Users\31396\.conda\envs\research\python.exe"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Config = Join-Path $ProjectRoot "configs\metrla_e5_final_sym_profile_local12_v1.yaml"
$PretrainedCheckpoint = Join-Path $ProjectRoot "artifacts\metrla_e5_final_sym_profile_local12_seed42\pretrain_best_relation.pt"
$RandomRoot = Join-Path $ProjectRoot "artifacts\metrla_e5_final_sym_profile_local12_random_seed42"
$RandomCheckpoint = Join-Path $RandomRoot "random_checkpoint.pt"
$PretrainedBank = Join-Path $ProjectRoot "artifacts\metrla_bank_e5_final_sym_profile_local12_seed42"
$RandomBank = Join-Path $ProjectRoot "artifacts\metrla_bank_e5_final_sym_profile_local12_random_seed42"
$OutputRoot = Join-Path $ProjectRoot "artifacts\convergence\visualization\e5_final_sym_profile_local12_level0"
$CfdpRoot = Join-Path $ProjectRoot "artifacts\convergence\cfdp_diagnostics\e5_final_sym_profile_local12_random_seed42"
$LogRoot = Join-Path $ProjectRoot "artifacts\convergence\logs\e5_final_sym_profile_local12_diagnostic"

New-Item -ItemType Directory -Force -Path $RandomRoot, $CfdpRoot, $LogRoot | Out-Null

function Invoke-Step {
    param(
        [string]$Name,
        [string[]]$Arguments
    )
    $Stdout = Join-Path $LogRoot ($Name + ".out.log")
    $Stderr = Join-Path $LogRoot ($Name + ".err.log")
    & $Python @Arguments 1> $Stdout 2> $Stderr
    if ($LASTEXITCODE -ne 0) {
        throw "Step $Name failed with exit code $LASTEXITCODE; see $Stderr"
    }
}

if (-not (Test-Path -LiteralPath $PretrainedCheckpoint)) {
    throw "Missing pretrained checkpoint: $PretrainedCheckpoint"
}
if (Test-Path -LiteralPath $RandomCheckpoint) {
    throw "Refusing to overwrite random checkpoint: $RandomCheckpoint"
}
if (Test-Path -LiteralPath $PretrainedBank) {
    throw "Refusing to overwrite pretrained Bank: $PretrainedBank"
}
if (Test-Path -LiteralPath $RandomBank) {
    throw "Refusing to overwrite random Bank: $RandomBank"
}
if (Test-Path -LiteralPath $OutputRoot) {
    throw "Refusing to overwrite visualization output: $OutputRoot"
}

Invoke-Step "01_init_random_checkpoint" @(
    (Join-Path $ProjectRoot "scripts\init_random_checkpoint.py"),
    "--config", $Config,
    "--output", $RandomCheckpoint,
    "--seed", "42"
)

Invoke-Step "02_diagnose_random_cfdp" @(
    (Join-Path $ProjectRoot "scripts\diagnose_cfdp.py"),
    "--config", $Config,
    "--checkpoint", $RandomCheckpoint,
    "--split", "val",
    "--output", (Join-Path $CfdpRoot "metrics.json")
)

Invoke-Step "03_build_pretrained_bank" @(
    (Join-Path $ProjectRoot "scripts\build_bank.py"),
    "--config", $Config,
    "--checkpoint", $PretrainedCheckpoint,
    "--output-dir", $PretrainedBank,
    "--dataset-name", "METR-LA"
)

Invoke-Step "04_build_random_bank" @(
    (Join-Path $ProjectRoot "scripts\build_bank.py"),
    "--config", $Config,
    "--checkpoint", $RandomCheckpoint,
    "--output-dir", $RandomBank,
    "--dataset-name", "METR-LA"
)

Invoke-Step "05_visualize_full_val_level0" @(
    (Join-Path $ProjectRoot "scripts\visualize_retrieval.py"),
    "--version", "e5a",
    "--config", $Config,
    "--checkpoint", $PretrainedCheckpoint,
    "--bank", $PretrainedBank,
    "--random-checkpoint", $RandomCheckpoint,
    "--random-bank", $RandomBank,
    "--split", "val",
    "--candidate-protocol", "exact_calendar",
    "--level-weight", "0",
    "--output-dir", $OutputRoot
)

Set-Content -Encoding UTF8 -Path (Join-Path $LogRoot "queue.completed") -Value (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
