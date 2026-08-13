param(
    [string]$Python = "C:\Users\31396\.conda\envs\research\python.exe"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Config = Join-Path $ProjectRoot "configs\metrla_e5_final_sym_profile_v1.yaml"
$PretrainedCheckpoint = Join-Path $ProjectRoot "artifacts\metrla_e5_final_sym_profile_seed42\pretrain_best_relation.pt"
$RandomRoot = Join-Path $ProjectRoot "artifacts\metrla_e5_final_sym_profile_global288_random_seed42"
$RandomCheckpoint = Join-Path $RandomRoot "random_checkpoint.pt"
$PretrainedBank = Join-Path $ProjectRoot "artifacts\metrla_bank_e5_final_sym_profile_global288_seed42"
$RandomBank = Join-Path $ProjectRoot "artifacts\metrla_bank_e5_final_sym_profile_global288_random_seed42"
$OutputRoot = Join-Path $ProjectRoot "artifacts\convergence\visualization\e5_final_sym_profile_global288_level0"
$CfdpPretrainedRoot = Join-Path $ProjectRoot "artifacts\convergence\cfdp_diagnostics\e5_final_sym_profile_global288_seed42"
$CfdpRandomRoot = Join-Path $ProjectRoot "artifacts\convergence\cfdp_diagnostics\e5_final_sym_profile_global288_random_seed42"
$LogRoot = Join-Path $ProjectRoot "artifacts\convergence\logs\e5_final_sym_profile_global288_diagnostic"

$OutputPaths = @(
    $RandomRoot,
    $PretrainedBank,
    $RandomBank,
    $OutputRoot,
    $CfdpPretrainedRoot,
    $CfdpRandomRoot,
    $LogRoot
)
foreach ($Path in $OutputPaths) {
    if (Test-Path -LiteralPath $Path) {
        throw "Refusing to reuse or overwrite formal output: $Path"
    }
}
if (-not (Test-Path -LiteralPath $PretrainedCheckpoint)) {
    throw "Missing pretrained checkpoint: $PretrainedCheckpoint"
}

New-Item -ItemType Directory -Force -Path $RandomRoot, $CfdpPretrainedRoot, $CfdpRandomRoot, $LogRoot | Out-Null

function Invoke-Step {
    param(
        [string]$Name,
        [string[]]$Arguments
    )
    $Stdout = Join-Path $LogRoot ($Name + ".out.log")
    $Stderr = Join-Path $LogRoot ($Name + ".err.log")
    Set-Content -Encoding UTF8 -Path (Join-Path $LogRoot ($Name + ".started")) -Value (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
    & $Python @Arguments 1> $Stdout 2> $Stderr
    if ($LASTEXITCODE -ne 0) {
        throw "Step $Name failed with exit code $LASTEXITCODE; see $Stderr"
    }
    Set-Content -Encoding UTF8 -Path (Join-Path $LogRoot ($Name + ".completed")) -Value (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
}

Invoke-Step "01_init_random_checkpoint" @(
    (Join-Path $ProjectRoot "scripts\init_random_checkpoint.py"),
    "--config", $Config,
    "--output", $RandomCheckpoint,
    "--seed", "42"
)

Invoke-Step "02_diagnose_pretrained_cfdp" @(
    (Join-Path $ProjectRoot "scripts\diagnose_cfdp.py"),
    "--config", $Config,
    "--checkpoint", $PretrainedCheckpoint,
    "--split", "val",
    "--output", (Join-Path $CfdpPretrainedRoot "metrics.json")
)

Invoke-Step "03_diagnose_random_cfdp" @(
    (Join-Path $ProjectRoot "scripts\diagnose_cfdp.py"),
    "--config", $Config,
    "--checkpoint", $RandomCheckpoint,
    "--split", "val",
    "--output", (Join-Path $CfdpRandomRoot "metrics.json")
)

Invoke-Step "04_build_pretrained_bank" @(
    (Join-Path $ProjectRoot "scripts\build_bank.py"),
    "--config", $Config,
    "--checkpoint", $PretrainedCheckpoint,
    "--output-dir", $PretrainedBank,
    "--dataset-name", "METR-LA"
)

Invoke-Step "05_build_random_bank" @(
    (Join-Path $ProjectRoot "scripts\build_bank.py"),
    "--config", $Config,
    "--checkpoint", $RandomCheckpoint,
    "--output-dir", $RandomBank,
    "--dataset-name", "METR-LA"
)

Invoke-Step "06_visualize_full_val_level0" @(
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
