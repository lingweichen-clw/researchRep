param(
    [string]$Python = "C:\Users\31396\.conda\envs\research\python.exe",
    [ValidateSet("pretrain")]
    [string]$Stage = "pretrain",
    [switch]$Smoke
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$QueueName = if ($Smoke) { "e5_latent48_cc_fgda_global288_smoke" } else { "e5_latent48_cc_fgda_global288" }
$LogRoot = Join-Path $ProjectRoot ("artifacts\convergence\logs\" + $QueueName)

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python executable does not exist: $Python" }
if (Test-Path -LiteralPath $LogRoot) { throw "Refusing to reuse or overwrite queue log directory: $LogRoot" }
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
        RunName = "metrla_e5_final_latent48_global288_seed42"
    },
    @{
        Label = "cc_fgda"
        Config = Join-Path $ProjectRoot "configs\metrla_e5_final_latent48_cc_fgda_global288_v1.yaml"
        RunName = "metrla_e5_final_latent48_cc_fgda_global288_seed42"
    }
)

foreach ($Experiment in $Experiments) {
    $RunRoot = Join-Path $ProjectRoot ("artifacts\" + $Experiment.RunName)
    if (Test-Path -LiteralPath $RunRoot) { throw "Refusing to reuse pretraining output: $RunRoot" }
    $Arguments = @((Join-Path $ProjectRoot "scripts\pretrain.py"), "--config", $Experiment.Config)
    if ($Smoke) { $Arguments += @("--epochs", "1", "--max-batches", "1") }
    Invoke-PythonStep -Name ("01_pretrain_" + $Experiment.Label) -Arguments $Arguments
}

Set-Content -Encoding UTF8 -Path (Join-Path $LogRoot "queue.completed") -Value (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
