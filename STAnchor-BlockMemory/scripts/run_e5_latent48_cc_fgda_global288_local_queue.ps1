param(
    [string]$Python = "C:\Users\31396\.conda\envs\research\python.exe",
    [string]$Latent48Checkpoint = "artifacts\metrla_e5_final_latent48_global288_seed42\pretrain_best_relation.pt",
    [string]$CcFgdaCheckpoint = "artifacts\metrla_e5_final_latent48_cc_fgda_global288_seed42\pretrain_best_relation.pt",
    [switch]$Smoke
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$QueueName = if ($Smoke) { "e5_latent48_cc_fgda_global288_local_smoke" } else { "e5_latent48_cc_fgda_global288_local" }
$LogRoot = Join-Path $ProjectRoot ("artifacts\convergence\logs\" + $QueueName)

function Resolve-ProjectFile {
    param([string]$Path, [string]$Description)
    $Candidate = if ([System.IO.Path]::IsPathRooted($Path)) {
        $Path
    } else {
        Join-Path $ProjectRoot $Path
    }
    if (-not (Test-Path -LiteralPath $Candidate -PathType Leaf)) {
        throw "Missing ${Description}: $Candidate"
    }
    return (Resolve-Path -LiteralPath $Candidate).Path
}

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python executable does not exist: $Python" }
if (Test-Path -LiteralPath $LogRoot) { throw "Refusing to reuse or overwrite queue log directory: $LogRoot" }
$Latent48Checkpoint = Resolve-ProjectFile $Latent48Checkpoint "Latent48 checkpoint"
$CcFgdaCheckpoint = Resolve-ProjectFile $CcFgdaCheckpoint "CC-FGDA checkpoint"
New-Item -ItemType Directory -Force -Path $LogRoot | Out-Null

function Invoke-PythonStep {
    param([string]$Name, [string[]]$Arguments)
    $Stdout = Join-Path $LogRoot ($Name + ".out.log")
    $Stderr = Join-Path $LogRoot ($Name + ".err.log")
    Set-Content -Encoding UTF8 -Path (Join-Path $LogRoot ($Name + ".started")) -Value (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
    $Process = Start-Process -FilePath $Python -ArgumentList $Arguments -WorkingDirectory $ProjectRoot -WindowStyle Hidden -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr -PassThru -Wait
    if ($Process.ExitCode -ne 0) { throw "Step $Name failed with exit code $($Process.ExitCode); see $Stderr" }
    Set-Content -Encoding UTF8 -Path (Join-Path $LogRoot ($Name + ".completed")) -Value (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
}

$Experiments = @(
    @{
        Label = "latent48"
        Config = Join-Path $ProjectRoot "configs\metrla_e5_final_latent48_global288_v1.yaml"
        Checkpoint = $Latent48Checkpoint
        BankName = "metrla_bank_e5_final_latent48_global288_local_seed42"
    },
    @{
        Label = "cc_fgda"
        Config = Join-Path $ProjectRoot "configs\metrla_e5_final_latent48_cc_fgda_global288_v1.yaml"
        Checkpoint = $CcFgdaCheckpoint
        BankName = "metrla_bank_e5_final_latent48_cc_fgda_global288_local_seed42"
    }
)

foreach ($Experiment in $Experiments) {
    $PretrainedCheckpoint = $Experiment.Checkpoint
    $RandomRoot = Join-Path $ProjectRoot ("artifacts\convergence\random_controls\" + $Experiment.Label + "_" + $QueueName)
    $RandomCheckpoint = Join-Path $RandomRoot "random_checkpoint.pt"
    $PretrainedBank = Join-Path $ProjectRoot ("artifacts\" + $Experiment.BankName)
    $RandomBank = Join-Path $ProjectRoot ("artifacts\metrla_bank_e5_final_" + $Experiment.Label + "_global288_random_local_seed42")
    $DiagnosticRoot = Join-Path $ProjectRoot ("artifacts\convergence\retrieval_diagnostics\" + $QueueName + "\" + $Experiment.Label)
    $VisualizationRoot = Join-Path $ProjectRoot ("artifacts\convergence\visualization\" + $QueueName + "\" + $Experiment.Label)
    if (-not (Test-Path -LiteralPath $PretrainedCheckpoint -PathType Leaf)) { throw "Missing pretrained relation checkpoint: $PretrainedCheckpoint" }
    foreach ($Path in @($RandomRoot, $PretrainedBank, $RandomBank, $DiagnosticRoot, $VisualizationRoot)) {
        if (Test-Path -LiteralPath $Path) { throw "Refusing to reuse local post-training output: $Path" }
    }
    New-Item -ItemType Directory -Force -Path $RandomRoot | Out-Null

    Invoke-PythonStep -Name ("01_random_checkpoint_" + $Experiment.Label) -Arguments @(
        (Join-Path $ProjectRoot "scripts\init_random_checkpoint.py"), "--config", $Experiment.Config,
        "--output", $RandomCheckpoint, "--seed", "42"
    )
    Invoke-PythonStep -Name ("02_build_pretrained_bank_" + $Experiment.Label) -Arguments @(
        (Join-Path $ProjectRoot "scripts\build_bank.py"), "--config", $Experiment.Config,
        "--checkpoint", $PretrainedCheckpoint, "--output-dir", $PretrainedBank, "--dataset-name", "METR-LA"
    )
    Invoke-PythonStep -Name ("03_build_random_bank_" + $Experiment.Label) -Arguments @(
        (Join-Path $ProjectRoot "scripts\build_bank.py"), "--config", $Experiment.Config,
        "--checkpoint", $RandomCheckpoint, "--output-dir", $RandomBank, "--dataset-name", "METR-LA"
    )

    New-Item -ItemType Directory -Force -Path $DiagnosticRoot, $VisualizationRoot | Out-Null
    $PretrainedDiagnostic = Join-Path $DiagnosticRoot "pretrained_metrics.json"
    $RandomDiagnostic = Join-Path $DiagnosticRoot "random_metrics.json"
    $DiagnosticArguments = @(
        (Join-Path $ProjectRoot "scripts\diagnose_retrieval.py"), "--config", $Experiment.Config,
        "--checkpoint", $PretrainedCheckpoint, "--bank", $PretrainedBank, "--split", "val", "--output", $PretrainedDiagnostic
    )
    if ($Smoke) { $DiagnosticArguments += @("--max-batches", "1") }
    Invoke-PythonStep -Name ("04_diagnose_pretrained_" + $Experiment.Label) -Arguments $DiagnosticArguments
    $RandomDiagnosticArguments = @(
        (Join-Path $ProjectRoot "scripts\diagnose_retrieval.py"), "--config", $Experiment.Config,
        "--checkpoint", $RandomCheckpoint, "--bank", $RandomBank, "--split", "val", "--output", $RandomDiagnostic
    )
    if ($Smoke) { $RandomDiagnosticArguments += @("--max-batches", "1") }
    Invoke-PythonStep -Name ("05_diagnose_random_" + $Experiment.Label) -Arguments $RandomDiagnosticArguments

    $VisualizationArguments = @(
        (Join-Path $ProjectRoot "scripts\visualize_retrieval.py"), "--version", "e5a",
        "--config", $Experiment.Config, "--checkpoint", $PretrainedCheckpoint, "--bank", $PretrainedBank,
        "--random-checkpoint", $RandomCheckpoint, "--random-bank", $RandomBank, "--split", "val",
        "--candidate-protocol", "exact_calendar", "--level-weight", "0", "--output-dir", $VisualizationRoot
    )
    if ($Smoke) { $VisualizationArguments += @("--max-batches", "1") }
    Invoke-PythonStep -Name ("06_visualize_" + $Experiment.Label) -Arguments $VisualizationArguments

    $Branches = @(
        @{ Order = "07"; Label = "base_only"; Mode = "base_only"; Checkpoint = $PretrainedCheckpoint; Bank = $PretrainedBank },
        @{ Order = "08"; Label = "pretrained_offset_decay"; Mode = "learned_topk_offset_decay_horizon"; Checkpoint = $PretrainedCheckpoint; Bank = $PretrainedBank },
        @{ Order = "09"; Label = "random_offset_decay"; Mode = "learned_topk_offset_decay_horizon"; Checkpoint = $RandomCheckpoint; Bank = $RandomBank }
    )
    foreach ($Branch in $Branches) {
        $RunName = "convergence/$QueueName/$($Experiment.Label)/$($Branch.Label)"
        $TrainArguments = @(
            (Join-Path $ProjectRoot "scripts\train_downstream.py"), "--config", $Experiment.Config,
            "--pretrained-checkpoint", $Branch.Checkpoint, "--bank", $Branch.Bank, "--mode", $Branch.Mode,
            "--candidate-protocol", "exact_calendar", "--level-weight", "0", "--seed", "42", "--run-name", $RunName
        )
        if ($Smoke) { $TrainArguments += @("--epochs", "1", "--max-batches", "1") }
        Invoke-PythonStep -Name ($Branch.Order + "_train_" + $Experiment.Label + "_" + $Branch.Label) -Arguments $TrainArguments
        $DownstreamRoot = Join-Path $ProjectRoot ("artifacts\" + $RunName.Replace("/", "\"))
        $DownstreamCheckpoint = Join-Path $DownstreamRoot "downstream_best.pt"
        if (-not (Test-Path -LiteralPath $DownstreamCheckpoint -PathType Leaf)) { throw "Missing downstream checkpoint: $DownstreamCheckpoint" }
        $EvalArguments = @(
            (Join-Path $ProjectRoot "scripts\evaluate.py"), "--config", $Experiment.Config,
            "--pretrained-checkpoint", $Branch.Checkpoint, "--downstream-checkpoint", $DownstreamCheckpoint,
            "--bank", $Branch.Bank, "--split", "val", "--candidate-protocol", "exact_calendar"
        )
        if ($Smoke) { $EvalArguments += @("--max-batches", "1") }
        Invoke-PythonStep -Name ($Branch.Order + "_evaluate_" + $Experiment.Label + "_" + $Branch.Label) -Arguments $EvalArguments
        $BranchDiagnostic = Join-Path $DownstreamRoot "branch_diagnostics_val.json"
        $BranchDiagnosticArguments = @(
            (Join-Path $ProjectRoot "scripts\diagnose_downstream.py"), "--config", $Experiment.Config,
            "--pretrained-checkpoint", $Branch.Checkpoint, "--downstream-checkpoint", $DownstreamCheckpoint,
            "--bank", $Branch.Bank, "--split", "val", "--candidate-protocol", "exact_calendar", "--output", $BranchDiagnostic
        )
        if ($Smoke) { $BranchDiagnosticArguments += @("--max-batches", "1") }
        Invoke-PythonStep -Name ($Branch.Order + "_diagnose_" + $Experiment.Label + "_" + $Branch.Label) -Arguments $BranchDiagnosticArguments
    }
}

Set-Content -Encoding UTF8 -Path (Join-Path $LogRoot "queue.completed") -Value (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
