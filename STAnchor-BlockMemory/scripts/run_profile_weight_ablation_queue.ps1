param(
    [string]$Python = "C:\Users\31396\.conda\envs\research\python.exe"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Config = Join-Path $ProjectRoot "configs\metrla_e5_final_sym_profile_v1.yaml"
$PretrainedCheckpoint = Join-Path $ProjectRoot "artifacts\metrla_e5_final_sym_profile_seed42\pretrain_best_relation.pt"
$RandomCheckpoint = Join-Path $ProjectRoot "artifacts\metrla_e5_final_sym_profile_global288_random_seed42\random_checkpoint.pt"
$PretrainedBank = Join-Path $ProjectRoot "artifacts\metrla_bank_e5_final_sym_profile_global288_seed42"
$RandomBank = Join-Path $ProjectRoot "artifacts\metrla_bank_e5_final_sym_profile_global288_random_seed42"
$OutputRoot = Join-Path $ProjectRoot "artifacts\convergence\profile_weight_ablation_global288_seed42"
$LogRoot = Join-Path $ProjectRoot "artifacts\convergence\logs\profile_weight_ablation_global288_seed42"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable does not exist: $Python"
}
foreach ($InputPath in @(
    $Config,
    $PretrainedCheckpoint,
    $RandomCheckpoint,
    $PretrainedBank,
    $RandomBank
)) {
    if (-not (Test-Path -LiteralPath $InputPath)) {
        throw "Missing profile-weight ablation input: $InputPath"
    }
}
foreach ($Path in @($OutputRoot, $LogRoot)) {
    if (Test-Path -LiteralPath $Path) {
        throw "Refusing to reuse or overwrite formal output: $Path"
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
    Set-Content -LiteralPath (Join-Path $LogRoot ($Name + ".started")) `
        -Value (Get-Date -Format "yyyy-MM-dd HH:mm:ss") -Encoding UTF8
    & $Python @Arguments 1> $Stdout 2> $Stderr
    if ($LASTEXITCODE -ne 0) {
        throw "Step $Name failed with exit code $LASTEXITCODE; see $Stderr"
    }
    Set-Content -LiteralPath (Join-Path $LogRoot ($Name + ".completed")) `
        -Value (Get-Date -Format "yyyy-MM-dd HH:mm:ss") -Encoding UTF8
}

$Experiments = @(
    @{ Order = "01"; Gamma = "0"; Label = "gamma0_latent_only" },
    @{ Order = "02"; Gamma = "0.25"; Label = "gamma025_profile_latent" },
    @{ Order = "03"; Gamma = "1"; Label = "gamma1_profile_only" }
)

foreach ($Experiment in $Experiments) {
    $ExperimentOutput = Join-Path $OutputRoot $Experiment.Label
    Invoke-PythonStep -Name ($Experiment.Order + "_visualize_" + $Experiment.Label) -Arguments @(
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
        "--profile-weight-override", $Experiment.Gamma,
        "--output-dir", $ExperimentOutput
    )
}

Set-Content -LiteralPath (Join-Path $LogRoot "queue.completed") `
    -Value (Get-Date -Format "yyyy-MM-dd HH:mm:ss") -Encoding UTF8
