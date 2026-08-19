param(
    [string]$Python = "C:\Users\31396\.conda\envs\research\python.exe",
    [switch]$Smoke
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$QueueName = "tgge_single_view_v3_higher_order"
$OutputRoot = Join-Path $ProjectRoot ("artifacts\convergence\" + $QueueName)
$LogRoot = Join-Path $ProjectRoot ("artifacts\convergence\logs\" + $QueueName)

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable does not exist: $Python"
}
foreach ($Path in @($OutputRoot, $LogRoot)) {
    if (Test-Path -LiteralPath $Path) {
        throw "Refusing to reuse or overwrite queue output: $Path"
    }
}
New-Item -ItemType Directory -Force -Path $OutputRoot, $LogRoot | Out-Null
Set-Content -Encoding UTF8 -Path (Join-Path $LogRoot "queue.started") -Value (Get-Date -Format "yyyy-MM-dd HH:mm:ss")

$Name = "01_pretrain_single_view_masked_relation_v3"
$Stdout = Join-Path $LogRoot ($Name + ".out.log")
$Stderr = Join-Path $LogRoot ($Name + ".err.log")
Set-Content -Encoding UTF8 -Path (Join-Path $LogRoot ($Name + ".started")) -Value (Get-Date -Format "yyyy-MM-dd HH:mm:ss")

$Arguments = @(
    (Join-Path $ProjectRoot "scripts\pretrain.py"),
    "--config", (Join-Path $ProjectRoot "configs\metrla_e5_tgge_single_view_masked_relation_v3.yaml"),
    "--run-name", "convergence/$QueueName/pretrain"
)
if ($Smoke) {
    $Arguments += @("--epochs", "1", "--max-batches", "2")
}

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
    Set-Content -Encoding UTF8 -Path (Join-Path $LogRoot "queue.failed") -Value (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
    throw "Pretraining failed with exit code $($Process.ExitCode); see $Stderr"
}

Set-Content -Encoding UTF8 -Path (Join-Path $LogRoot ($Name + ".completed")) -Value (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
Set-Content -Encoding UTF8 -Path (Join-Path $LogRoot "queue.completed") -Value (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
