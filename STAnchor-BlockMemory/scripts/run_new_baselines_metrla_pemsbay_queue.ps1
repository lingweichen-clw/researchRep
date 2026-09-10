$ErrorActionPreference = 'Stop'

# New-baseline METR-LA + PEMS-BAY queue.
# Backbones: st_norm, dlinear, st_ssdl, dcrnn
# Per backbone: METR-LA baseonly / trained Router / random Router
#               PEMS-BAY baseonly / source Router / finetuned Router
# Total training jobs: 24
#
# Temporary Banks live only under artifacts/new_baseline_queue_banks.
# A Bank is deleted only after every job that consumes it has finished.

$repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $repo

function Assert-RequiredPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$Description
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        throw ('Missing ' + $Description + ': ' + $Path)
    }
}

function Remove-QueueBank {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $resolvedQueueRoot = (Resolve-Path -LiteralPath $queueBankRoot -ErrorAction Stop).Path
    $resolvedTarget = [System.IO.Path]::GetFullPath((Join-Path $repo $Path))
    $prefix = $resolvedQueueRoot.TrimEnd('\') + '\'
    if (-not $resolvedTarget.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw ('Refusing to remove a path outside the queue bank root: ' + $resolvedTarget)
    }
    if (Test-Path -LiteralPath $resolvedTarget) {
        Remove-Item -LiteralPath $resolvedTarget -Recurse -Force
        Write-Host ('Removed temporary Bank: ' + $resolvedTarget)
    }
}

function Invoke-Python {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Label,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    Write-Host ''
    Write-Host ('===== START ' + $Label + ' =====')
    & python -u @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw ('Command failed: ' + $Label + ' (exit code ' + $LASTEXITCODE + ')')
    }
    Write-Host ('===== DONE ' + $Label + ' =====')
}

function Invoke-BaseOnly {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Label,
        [Parameter(Mandatory = $true)]
        [string]$Config,
        [Parameter(Mandatory = $true)]
        [string]$RunName
    )

    Invoke-Python -Label $Label -Arguments @(
        'scripts\train_downstream.py',
        '--config', $Config,
        '--mode', 'base_only',
        '--seed', '42',
        '--run-name', $RunName
    )
}

function Invoke-Router {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Label,
        [Parameter(Mandatory = $true)]
        [string]$Config,
        [Parameter(Mandatory = $true)]
        [string]$EncoderCheckpoint,
        [Parameter(Mandatory = $true)]
        [string]$Bank,
        [Parameter(Mandatory = $true)]
        [string]$BaseCheckpoint,
        [Parameter(Mandatory = $true)]
        [string]$RunName
    )

    Invoke-Python -Label $Label -Arguments @(
        'scripts\train_downstream.py',
        '--config', $Config,
        '--pretrained-checkpoint', $EncoderCheckpoint,
        '--bank', $Bank,
        '--base-checkpoint', $BaseCheckpoint,
        '--mode', 'learned_topk_error_aware',
        '--candidate-protocol', 'weekday_radius1_overlap',
        '--level-weight', '0',
        '--candidate-quality-weight', '0',
        '--epochs', '50',
        '--disable-early-stopping',
        '--frozen-path-cache',
        '--seed', '42',
        '--run-name', $RunName
    )
}

function Invoke-BuildBank {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Label,
        [Parameter(Mandatory = $true)]
        [string]$Config,
        [Parameter(Mandatory = $true)]
        [string]$Checkpoint,
        [Parameter(Mandatory = $true)]
        [string]$OutputDir,
        [Parameter(Mandatory = $true)]
        [string]$DatasetName
    )

    Remove-QueueBank -Path $OutputDir
    Invoke-Python -Label $Label -Arguments @(
        'scripts\build_bank.py',
        '--config', $Config,
        '--checkpoint', $Checkpoint,
        '--output-dir', $OutputDir,
        '--dataset-name', $DatasetName
    )
    Assert-RequiredPath (Join-Path $repo (Join-Path $OutputDir 'manifest.json')) ($Label + ' manifest')
}

$sourceCheckpoint = 'artifacts\metrla_e5_tgge_hn_offset_decay_v2_transfer_hidden128_ffn2_b16_seed42\pretrain_best.pt'
$finetuneCheckpoint = 'artifacts\cross_dataset_t1\pemsbay_head_adapter_seed42\retrieval_t1_best.pt'
$metrlaEncoderConfig = 'configs\metrla_e5_tgge_hn_offset_decay_v2_transfer_hidden128_ffn2_b16.yaml'
$pemsbaySourceConfig = 'configs\cross_dataset_pemsbay_source_encoder_stage1.yaml'
$pemsbayFinetuneConfig = 'configs\cross_dataset_pemsbay_t1_head_adapter.yaml'
$queueBankRoot = 'artifacts\new_baseline_queue_banks'
$randomCheckpoint = ($queueBankRoot + '\metrla_random_checkpoint.pt')
$metrlaTrainedBank = ($queueBankRoot + '\metrla_trained')
$metrlaRandomBank = ($queueBankRoot + '\metrla_random')
$pemsbaySourceBank = ($queueBankRoot + '\pemsbay_source')
$pemsbayFinetuneBank = ($queueBankRoot + '\pemsbay_finetuned')

New-Item -ItemType Directory -Path (Join-Path $repo $queueBankRoot) -Force | Out-Null

$backbones = @('st_norm', 'dlinear', 'st_ssdl', 'dcrnn')
$metrlaBaseConfigs = [ordered]@{
    st_norm = 'configs\formal_baseonly_st_norm.yaml'
    dlinear = 'configs\formal_baseonly_dlinear.yaml'
    st_ssdl = 'configs\formal_baseonly_st_ssdl.yaml'
    dcrnn = 'configs\formal_baseonly_dcrnn.yaml'
}
$metrlaTrainedConfigs = [ordered]@{
    st_norm = 'configs\formal_base_as_candidate_st_norm.yaml'
    dlinear = 'configs\formal_base_as_candidate_dlinear.yaml'
    st_ssdl = 'configs\formal_base_as_candidate_st_ssdl.yaml'
    dcrnn = 'configs\formal_base_as_candidate_dcrnn.yaml'
}
$metrlaRandomConfigs = [ordered]@{
    st_norm = 'configs\ablation_random_bank_router_st_norm.yaml'
    dlinear = 'configs\ablation_random_bank_router_dlinear.yaml'
    st_ssdl = 'configs\ablation_random_bank_router_st_ssdl.yaml'
    dcrnn = 'configs\ablation_random_bank_router_dcrnn.yaml'
}
$pemsbayBaseConfigs = [ordered]@{
    st_norm = 'configs\cross_dataset_pemsbay_baseonly_st_norm.yaml'
    dlinear = 'configs\cross_dataset_pemsbay_baseonly_dlinear.yaml'
    st_ssdl = 'configs\cross_dataset_pemsbay_baseonly_st_ssdl.yaml'
    dcrnn = 'configs\cross_dataset_pemsbay_baseonly_dcrnn.yaml'
}
$pemsbayRouterConfigs = [ordered]@{
    st_norm = 'configs\cross_dataset_pemsbay_router_st_norm.yaml'
    dlinear = 'configs\cross_dataset_pemsbay_router_dlinear.yaml'
    st_ssdl = 'configs\cross_dataset_pemsbay_router_st_ssdl.yaml'
    dcrnn = 'configs\cross_dataset_pemsbay_router_dcrnn.yaml'
}

Assert-RequiredPath $sourceCheckpoint 'source encoder checkpoint'
Assert-RequiredPath $finetuneCheckpoint 'PEMS-BAY finetuned encoder checkpoint'
Assert-RequiredPath $metrlaEncoderConfig 'METR-LA encoder config'
Assert-RequiredPath $pemsbaySourceConfig 'PEMS-BAY source encoder config'
Assert-RequiredPath $pemsbayFinetuneConfig 'PEMS-BAY finetune encoder config'
Assert-RequiredPath (Join-Path $repo '..\data\METRLA_data\METR-LA.h5') 'METR-LA data file'
Assert-RequiredPath (Join-Path $repo '..\data\METRLA_data\adj_mx.pkl') 'METR-LA adjacency'
Assert-RequiredPath (Join-Path $repo '..\data\pemsBay_data\pems-bay.h5') 'PEMS-BAY data file'
Assert-RequiredPath (Join-Path $repo '..\data\pemsBay_data\adj_mx_bay.pkl') 'PEMS-BAY adjacency'
foreach ($table in @($metrlaBaseConfigs, $metrlaTrainedConfigs, $metrlaRandomConfigs, $pemsbayBaseConfigs, $pemsbayRouterConfigs)) {
    foreach ($config in $table.Values) {
        Assert-RequiredPath $config $config
    }
}

$completed = 0
$metrlaBaseCheckpoints = @{}
$pemsbayBaseCheckpoints = @{}

Write-Host ''
Write-Host '######## METR-LA Base-only ########'
foreach ($backbone in $backbones) {
    $runName = ('convergence/formal_20260910_baseonly_' + $backbone + '_seed42')
    Invoke-BaseOnly -Label ('metrla_baseonly_' + $backbone) -Config $metrlaBaseConfigs[$backbone] -RunName $runName
    $metrlaBaseCheckpoints[$backbone] = ('artifacts\convergence\formal_20260910_baseonly_' + $backbone + '_seed42\downstream_best.pt')
    Assert-RequiredPath $metrlaBaseCheckpoints[$backbone] ('METR-LA ' + $backbone + ' Base-only checkpoint')
    $completed++
}

Invoke-BuildBank -Label 'metrla_trained_bank' -Config $metrlaEncoderConfig -Checkpoint $sourceCheckpoint -OutputDir $metrlaTrainedBank -DatasetName 'metrla_trained_queue'
Write-Host ''
Write-Host '######## METR-LA trained Router ########'
foreach ($backbone in $backbones) {
    Invoke-Router -Label ('metrla_trained_router_' + $backbone) -Config $metrlaTrainedConfigs[$backbone] -EncoderCheckpoint $sourceCheckpoint -Bank $metrlaTrainedBank -BaseCheckpoint $metrlaBaseCheckpoints[$backbone] -RunName ('convergence/formal_20260910_trained_router_' + $backbone + '_seed42')
    $completed++
}
Remove-QueueBank -Path $metrlaTrainedBank

Write-Host ''
Write-Host '===== BUILD METR-LA random encoder ====='
& python -u scripts\build_random_checkpoint.py --config $metrlaEncoderConfig --output $randomCheckpoint --seed 42
if ($LASTEXITCODE -ne 0) {
    throw ('Random encoder build failed (exit code ' + $LASTEXITCODE + ')')
}
Assert-RequiredPath (Join-Path $repo $randomCheckpoint) 'METR-LA random encoder checkpoint'
Invoke-BuildBank -Label 'metrla_random_bank' -Config $metrlaEncoderConfig -Checkpoint $randomCheckpoint -OutputDir $metrlaRandomBank -DatasetName 'metrla_random_queue'
Write-Host ''
Write-Host '######## METR-LA random Router ########'
foreach ($backbone in $backbones) {
    Invoke-Router -Label ('metrla_random_router_' + $backbone) -Config $metrlaRandomConfigs[$backbone] -EncoderCheckpoint $randomCheckpoint -Bank $metrlaRandomBank -BaseCheckpoint $metrlaBaseCheckpoints[$backbone] -RunName ('convergence/ablation_random_bank_router_' + $backbone + '_seed42')
    $completed++
}
Remove-QueueBank -Path $metrlaRandomBank

Write-Host ''
Write-Host '######## PEMS-BAY Base-only ########'
foreach ($backbone in $backbones) {
    $runName = ('cross_dataset_pemsbay/pemsbay_baseonly_' + $backbone + '_seed42')
    Invoke-BaseOnly -Label ('pemsbay_baseonly_' + $backbone) -Config $pemsbayBaseConfigs[$backbone] -RunName $runName
    $pemsbayBaseCheckpoints[$backbone] = ('artifacts\cross_dataset_pemsbay\pemsbay_baseonly_' + $backbone + '_seed42\downstream_best.pt')
    Assert-RequiredPath $pemsbayBaseCheckpoints[$backbone] ('PEMS-BAY ' + $backbone + ' Base-only checkpoint')
    $completed++
}

Invoke-BuildBank -Label 'pemsbay_source_bank' -Config $pemsbaySourceConfig -Checkpoint $sourceCheckpoint -OutputDir $pemsbaySourceBank -DatasetName 'pemsbay_source_queue'
Write-Host ''
Write-Host '######## PEMS-BAY source Router ########'
foreach ($backbone in $backbones) {
    Invoke-Router -Label ('pemsbay_source_router_' + $backbone) -Config $pemsbayRouterConfigs[$backbone] -EncoderCheckpoint $sourceCheckpoint -Bank $pemsbaySourceBank -BaseCheckpoint $pemsbayBaseCheckpoints[$backbone] -RunName ('cross_dataset_pemsbay/pemsbay_source_router_' + $backbone + '_seed42')
    $completed++
}
Remove-QueueBank -Path $pemsbaySourceBank

Invoke-BuildBank -Label 'pemsbay_finetuned_bank' -Config $pemsbayFinetuneConfig -Checkpoint $finetuneCheckpoint -OutputDir $pemsbayFinetuneBank -DatasetName 'pemsbay_finetuned_queue'
Write-Host ''
Write-Host '######## PEMS-BAY finetuned Router ########'
foreach ($backbone in $backbones) {
    Invoke-Router -Label ('pemsbay_finetuned_router_' + $backbone) -Config $pemsbayRouterConfigs[$backbone] -EncoderCheckpoint $finetuneCheckpoint -Bank $pemsbayFinetuneBank -BaseCheckpoint $pemsbayBaseCheckpoints[$backbone] -RunName ('cross_dataset_pemsbay/pemsbay_finetuned_router_' + $backbone + '_seed42')
    $completed++
}
Remove-QueueBank -Path $pemsbayFinetuneBank

if (Test-Path -LiteralPath (Join-Path $repo $randomCheckpoint)) {
    Remove-Item -LiteralPath (Join-Path $repo $randomCheckpoint) -Force
    Write-Host ('Removed temporary random encoder: ' + $randomCheckpoint)
}

if ($completed -ne 24) {
    throw ('Expected 24 completed jobs, observed ' + $completed)
}

Write-Host ''
Write-Host '===== ALL 24 NEW-BASELINE JOBS COMPLETED ====='
Write-Host 'Backbones: st_norm, dlinear, st_ssdl, dcrnn'
Write-Host 'METR-LA: Base-only + trained Router + random Router'
Write-Host 'PEMS-BAY: Base-only + source Router + finetuned Router'
