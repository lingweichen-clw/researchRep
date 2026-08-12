param(
    [string]$Python = "C:\Users\31396\.conda\envs\research\python.exe"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$OutputRoot = Join-Path $ProjectRoot "artifacts\convergence\candidate_protocol"
$LogRoot = Join-Path $OutputRoot "logs"
New-Item -ItemType Directory -Force -Path $OutputRoot, $LogRoot | Out-Null

$Common = @(
    (Join-Path $ProjectRoot "scripts\visualize_retrieval.py"),
    "--version", "e5a",
    "--config", (Join-Path $ProjectRoot "configs\metrla_e5_offset_decay_relation_level0_v1.yaml"),
    "--checkpoint", (Join-Path $ProjectRoot "artifacts\metrla_e5a_offset_decay_seed42\pretrain_best_relation.pt"),
    "--bank", (Join-Path $ProjectRoot "artifacts\metrla_bank_e5a_offset_decay_relation_seed42"),
    "--random-checkpoint", (Join-Path $ProjectRoot "artifacts\metrla_e3_target_random_seed42\random_checkpoint.pt"),
    "--random-bank", (Join-Path $ProjectRoot "artifacts\metrla_bank_e3_target_random_seed42"),
    "--split", "val"
)

$Experiments = @(
    @{ Protocol = "relaxed_calendar"; Directory = "e5a_relaxed_calendar" },
    @{ Protocol = "broad_causal"; Directory = "e5a_broad_causal" }
)

foreach ($Experiment in $Experiments) {
    $Protocol = $Experiment.Protocol
    $OutputDirectory = Join-Path $OutputRoot $Experiment.Directory
    $LogPath = Join-Path $LogRoot ("e5a_{0}.log" -f $Protocol)
    $ErrorLogPath = $LogPath + ".err"
    New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
    $Arguments = $Common + @(
        "--candidate-protocol", $Protocol,
        "--output-dir", $OutputDirectory
    )
    & $Python @Arguments 1> $LogPath 2> $ErrorLogPath
    if ($LASTEXITCODE -ne 0) {
        throw "Candidate protocol $Protocol failed with exit code $LASTEXITCODE"
    }
}
