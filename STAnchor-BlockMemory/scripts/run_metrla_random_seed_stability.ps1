param(
    [string]$Python = "C:\Users\31396\.conda\envs\research\python.exe"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

$RunRoot = Join-Path $ProjectRoot "artifacts/metrla_random_seed_stability"
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

function Invoke-RandomSeedExperiment {
    param([int]$Seed)

    $CheckpointDir = Join-Path $ProjectRoot "artifacts/metrla_e3_target_random_seed$Seed"
    $CheckpointPath = Join-Path $CheckpointDir "random_checkpoint.pt"
    $BankDir = Join-Path $ProjectRoot "artifacts/metrla_bank_e3_target_random_seed$Seed"

    if (Test-Path -LiteralPath $CheckpointPath) {
        throw "Refusing to reuse existing random checkpoint: $CheckpointPath"
    }
    if (Test-Path -LiteralPath $BankDir) {
        throw "Refusing to overwrite existing random Bank: $BankDir"
    }
    New-Item -ItemType Directory -Path $CheckpointDir -Force | Out-Null

    Invoke-PythonStep "seed${Seed}_01_checkpoint" @(
        "scripts/init_random_checkpoint.py",
        "--config", "configs/metrla_e3_relation_v1.yaml",
        "--output", "artifacts/metrla_e3_target_random_seed$Seed/random_checkpoint.pt",
        "--seed", "$Seed"
    )

    Invoke-PythonStep "seed${Seed}_02_bank" @(
        "scripts/build_bank.py",
        "--config", "configs/metrla_e3_relation_v1.yaml",
        "--checkpoint", "artifacts/metrla_e3_target_random_seed$Seed/random_checkpoint.pt",
        "--output-dir", "artifacts/metrla_bank_e3_target_random_seed$Seed",
        "--dataset-name", "METR-LA"
    )

    Invoke-PythonStep "seed${Seed}_03_diagnostic" @(
        "scripts/diagnose_retrieval.py",
        "--config", "configs/metrla_e3_relation_v1.yaml",
        "--checkpoint", "artifacts/metrla_e3_target_random_seed$Seed/random_checkpoint.pt",
        "--bank", "artifacts/metrla_bank_e3_target_random_seed$Seed",
        "--split", "val",
        "--output", "artifacts/metrla_random_seed_stability/metrla_random_seed${Seed}_level025_val.json"
    )
}

try {
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
        throw "Python executable not found: $Python"
    }
    Write-RunLog "PIPELINE START seeds=2024,2025"
    Invoke-RandomSeedExperiment -Seed 2024
    Invoke-RandomSeedExperiment -Seed 2025
    Write-RunLog "ALL STEPS COMPLETED"
}
catch {
    $message = $_ | Out-String
    Write-RunLog "PIPELINE FAILED"
    Set-Content -LiteralPath (Join-Path $RunRoot "pipeline_error.log") -Value $message -Encoding UTF8
    exit 1
}
