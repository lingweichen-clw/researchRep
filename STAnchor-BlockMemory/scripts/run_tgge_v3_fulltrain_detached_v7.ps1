param(
    [string]$Python = "C:\Users\31396\.conda\envs\research\python.exe"
)
$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$QueueName = "downstream_tgge_v3_fulltrain_detached_v7"
$OutputRoot = Join-Path $ProjectRoot ("artifacts\convergence\" + $QueueName)
$LogRoot = Join-Path $ProjectRoot ("artifacts\convergence\logs\" + $QueueName)
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python executable does not exist: $Python" }
if (Test-Path -LiteralPath $OutputRoot) { throw "Refusing to reuse output: $OutputRoot" }
if (Test-Path -LiteralPath $LogRoot) { throw "Refusing to reuse log: $LogRoot" }
New-Item -ItemType Directory -Force -Path $OutputRoot,$LogRoot | Out-Null
Set-Content -Encoding UTF8 -Path (Join-Path $LogRoot "queue.started") -Value (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
$Pretrained = Join-Path $ProjectRoot "artifacts\convergence\tgge_single_view_v3_higher_order_reconstruction2\pretrain\pretrain_best_relation.pt"
$Bank = Join-Path $ProjectRoot "artifacts\convergence\tgge_single_view_v3_higher_order_reconstruction2\bank"
$Experiments = @(
    @{ Order="01"; Name="stgcn_seed42"; Config=(Join-Path $ProjectRoot "configs\metrla_stgcn_tgge_v3_error_aware_fulltrain_v1.yaml"); Base=(Join-Path $ProjectRoot "artifacts\quarantine\20260820_previous_downstream\artifacts\metrla_stgcn_base_only_fulltrain_seed42_local_v3\downstream_best.pt") },
    @{ Order="02"; Name="graphwavenet_seed42"; Config=(Join-Path $ProjectRoot "configs\metrla_graphwavenet_tgge_v3_error_aware_fulltrain_v1.yaml"); Base=(Join-Path $ProjectRoot "artifacts\quarantine\20260820_previous_downstream\artifacts\downstream_tgge_v3_graphwavenet_base_only_seed42\downstream_best.pt") }
)
foreach($E in $Experiments){
    foreach($Input in @($E.Config,$E.Base,$Pretrained)){ if(-not (Test-Path -LiteralPath $Input -PathType Leaf)){ throw "Missing input: $Input" } }
    if(-not (Test-Path -LiteralPath $Bank -PathType Container)){ throw "Missing Bank: $Bank" }
    $name = "$($E.Order)_$($E.Name)"
    $args = @("scripts\train_downstream.py","--config",$E.Config,"--pretrained-checkpoint",$Pretrained,"--bank",$Bank,"--base-checkpoint",$E.Base,"--mode","learned_topk_error_aware","--candidate-protocol","exact_calendar","--level-weight","0","--seed","42","--run-name","convergence/$QueueName/$($E.Name)")
    Set-Content -Encoding UTF8 -Path (Join-Path $LogRoot "$name.started") -Value (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
    & $Python @args
    if($LASTEXITCODE -ne 0){ Set-Content -Encoding UTF8 -Path (Join-Path $LogRoot "$name.failed") -Value (Get-Date -Format "yyyy-MM-dd HH:mm:ss"); throw "$name failed with exit code $LASTEXITCODE" }
    Set-Content -Encoding UTF8 -Path (Join-Path $LogRoot "$name.completed") -Value (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
}
Set-Content -Encoding UTF8 -Path (Join-Path $LogRoot "queue.completed") -Value (Get-Date -Format "yyyy-MM-dd HH:mm:ss")

