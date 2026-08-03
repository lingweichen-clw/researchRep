param(
    [string]$Python = "C:\Users\31396\.conda\envs\research\python.exe"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

$RunRoot = Join-Path $ProjectRoot "artifacts/encoder_random_attribution_seed42"
New-Item -ItemType Directory -Path $RunRoot -Force | Out-Null
$PipelineLog = Join-Path $RunRoot "pipeline.log"

function Write-RunLog {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Write-Output $line
    Add-Content -LiteralPath $PipelineLog -Value $line -Encoding UTF8
}

function Invoke-PythonStep {
    param(
        [string]$Name,
        [string[]]$Arguments
    )
    Write-RunLog "START $Name"
    & $Python @Arguments 2>&1 | Tee-Object -FilePath (Join-Path $RunRoot "$Name.log")
    if ($LASTEXITCODE -ne 0) {
        throw "Step $Name failed with exit code $LASTEXITCODE"
    }
    Write-RunLog "DONE  $Name"
}

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python"
}

Invoke-PythonStep "01_pemsbay_source_level0" @(
    "scripts/diagnose_retrieval.py",
    "--config", "configs/pemsbay_e3_transfer_level0_v1.yaml",
    "--checkpoint", "artifacts/metrla_e3_relation_seed42/pretrain_best_relation.pt",
    "--bank", "artifacts/pemsbay_bank_from_metrla_e3_relation",
    "--split", "val",
    "--output", "artifacts/encoder_random_attribution_seed42/pemsbay_source_level0_val.json"
)

Invoke-PythonStep "02_pemsbay_random_level0" @(
    "scripts/diagnose_retrieval.py",
    "--config", "configs/pemsbay_e3_transfer_level0_v1.yaml",
    "--checkpoint", "artifacts/pemsbay_e3_target_random_seed42/random_checkpoint.pt",
    "--bank", "artifacts/pemsbay_bank_target_random_seed42",
    "--split", "val",
    "--output", "artifacts/encoder_random_attribution_seed42/pemsbay_random_level0_val.json"
)

$MetrlaRandomDir = Join-Path $ProjectRoot "artifacts/metrla_e3_target_random_seed42"
$MetrlaRandomCheckpoint = Join-Path $MetrlaRandomDir "random_checkpoint.pt"
if (Test-Path -LiteralPath $MetrlaRandomCheckpoint) {
    throw "Refusing to reuse existing METR-LA random checkpoint: $MetrlaRandomCheckpoint"
}
New-Item -ItemType Directory -Path $MetrlaRandomDir -Force | Out-Null

Invoke-PythonStep "03_metrla_random_checkpoint" @(
    "scripts/init_random_checkpoint.py",
    "--config", "configs/metrla_e3_relation_v1.yaml",
    "--output", "artifacts/metrla_e3_target_random_seed42/random_checkpoint.pt",
    "--seed", "42"
)

$MetrlaRandomBank = Join-Path $ProjectRoot "artifacts/metrla_bank_e3_target_random_seed42"
if (Test-Path -LiteralPath $MetrlaRandomBank) {
    throw "Refusing to overwrite existing METR-LA random Bank: $MetrlaRandomBank"
}

Invoke-PythonStep "04_metrla_random_bank" @(
    "scripts/build_bank.py",
    "--config", "configs/metrla_e3_relation_v1.yaml",
    "--checkpoint", "artifacts/metrla_e3_target_random_seed42/random_checkpoint.pt",
    "--output-dir", "artifacts/metrla_bank_e3_target_random_seed42",
    "--dataset-name", "METR-LA"
)

Invoke-PythonStep "05_metrla_random_level025" @(
    "scripts/diagnose_retrieval.py",
    "--config", "configs/metrla_e3_relation_v1.yaml",
    "--checkpoint", "artifacts/metrla_e3_target_random_seed42/random_checkpoint.pt",
    "--bank", "artifacts/metrla_bank_e3_target_random_seed42",
    "--split", "val",
    "--output", "artifacts/encoder_random_attribution_seed42/metrla_random_level025_val.json"
)

Invoke-PythonStep "06_metrla_source_level0" @(
    "scripts/diagnose_retrieval.py",
    "--config", "configs/metrla_e3_relation_level0_v1.yaml",
    "--checkpoint", "artifacts/metrla_e3_relation_seed42/pretrain_best_relation.pt",
    "--bank", "artifacts/metrla_bank_e3_relation_relation",
    "--split", "val",
    "--output", "artifacts/encoder_random_attribution_seed42/metrla_source_level0_val.json"
)

Invoke-PythonStep "07_metrla_random_level0" @(
    "scripts/diagnose_retrieval.py",
    "--config", "configs/metrla_e3_relation_level0_v1.yaml",
    "--checkpoint", "artifacts/metrla_e3_target_random_seed42/random_checkpoint.pt",
    "--bank", "artifacts/metrla_bank_e3_target_random_seed42",
    "--split", "val",
    "--output", "artifacts/encoder_random_attribution_seed42/metrla_random_level0_val.json"
)

Write-RunLog "ALL STEPS COMPLETED"
