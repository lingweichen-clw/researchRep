$ErrorActionPreference = 'Stop'

# 24-group cross-dataset downstream queue:
#   PEMS04 and PEMS08
#   each: 4 Base-only + 4 source-Bank Router + 4 finetuned-Bank Router
#
# The queue owns only the temporary banks below.  A bank is removed only after
# all four Router jobs that consume it finish successfully.  Checkpoints,
# logs, and metrics under artifacts/cross_dataset_final are retained.

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
        throw "Missing $Description`: $Path"
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
        throw "Refusing to remove a path outside the queue bank root: $resolvedTarget"
    }
    if (Test-Path -LiteralPath $resolvedTarget) {
        Remove-Item -LiteralPath $resolvedTarget -Recurse -Force
        Write-Host "Removed temporary Bank: $resolvedTarget"
    }
}

function Invoke-DownstreamJob {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Label,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    Write-Host ''
    Write-Host "===== START $Label ====="
    & python -u scripts\train_downstream.py @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Training failed: $Label (exit code $LASTEXITCODE)"
    }
    Write-Host "===== DONE $Label ====="
}

function Invoke-BaseOnly {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Dataset,
        [Parameter(Mandatory = $true)]
        [string]$Backbone,
        [Parameter(Mandatory = $true)]
        [string]$Config
    )

    Invoke-DownstreamJob -Label "${Dataset}_baseonly_${Backbone}" -Arguments @(
        '--config', $Config,
        '--mode', 'base_only',
        '--epochs', '50',
        '--disable-early-stopping',
        '--seed', '42',
        '--run-name', "cross_dataset_final/${Dataset}_baseonly_${Backbone}_seed42"
    )
}

function Invoke-Router {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Dataset,
        [Parameter(Mandatory = $true)]
        [string]$Variant,
        [Parameter(Mandatory = $true)]
        [string]$Backbone,
        [Parameter(Mandatory = $true)]
        [string]$Config,
        [Parameter(Mandatory = $true)]
        [string]$EncoderCheckpoint,
        [Parameter(Mandatory = $true)]
        [string]$Bank,
        [Parameter(Mandatory = $true)]
        [string]$BaseCheckpoint
    )

    Invoke-DownstreamJob -Label "${Dataset}_${Variant}_router_${Backbone}" -Arguments @(
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
        '--run-name', "cross_dataset_final/${Dataset}_${Variant}_router_${Backbone}_seed42"
    )
}

$sourceCheckpoint = 'artifacts\metrla_e5_tgge_hn_offset_decay_v2_transfer_hidden128_ffn2_b16_seed42\pretrain_best.pt'
$finetuneCheckpoints = @{
    pems04 = 'artifacts\cross_dataset_t1\pems04_head_adapter_seed42\retrieval_t1_best.pt'
    pems08 = 'artifacts\cross_dataset_t1\pems08_head_adapter_seed42\retrieval_t1_best.pt'
}
$queueBankRoot = 'artifacts\cross_dataset_queue_banks'
New-Item -ItemType Directory -Path (Join-Path $repo $queueBankRoot) -Force | Out-Null

$datasets = [ordered]@{
    pems04 = @{
        DataProbe = '..\data\pems04_data\pems04.npz'
        SourceConfig = 'configs\cross_dataset_pems04_source_encoder_stage1.yaml'
        FinetuneConfig = 'configs\cross_dataset_pems04_t1_head_adapter.yaml'
        BaseConfigs = [ordered]@{
            gwn = 'configs\cross_dataset_pems04_baseonly_gwn.yaml'
            stgcn = 'configs\cross_dataset_pems04_baseonly_stgcn.yaml'
            staeformer = 'configs\cross_dataset_pems04_baseonly_staeformer.yaml'
            argcn = 'configs\cross_dataset_pems04_baseonly_argcn.yaml'
        }
        RouterConfigs = [ordered]@{
            gwn = 'configs\cross_dataset_pems04_router_gwn.yaml'
            stgcn = 'configs\cross_dataset_pems04_router_stgcn.yaml'
            staeformer = 'configs\cross_dataset_pems04_router_staeformer.yaml'
            argcn = 'configs\cross_dataset_pems04_router_argcn.yaml'
        }
    }
    pems08 = @{
        DataProbe = '..\data\PEMS08\PEMS08.npz'
        SourceConfig = 'configs\cross_dataset_pems08_source_encoder_stage1.yaml'
        FinetuneConfig = 'configs\cross_dataset_pems08_t1_head_adapter.yaml'
        BaseConfigs = [ordered]@{
            gwn = 'configs\cross_dataset_pems08_baseonly_gwn.yaml'
            stgcn = 'configs\cross_dataset_pems08_baseonly_stgcn.yaml'
            staeformer = 'configs\cross_dataset_pems08_baseonly_staeformer.yaml'
            argcn = 'configs\cross_dataset_pems08_baseonly_argcn.yaml'
        }
        RouterConfigs = [ordered]@{
            gwn = 'configs\cross_dataset_pems08_router_gwn.yaml'
            stgcn = 'configs\cross_dataset_pems08_router_stgcn.yaml'
            staeformer = 'configs\cross_dataset_pems08_router_staeformer.yaml'
            argcn = 'configs\cross_dataset_pems08_router_argcn.yaml'
        }
    }
}

Assert-RequiredPath $sourceCheckpoint 'source encoder checkpoint'
foreach ($datasetName in $datasets.Keys) {
    $spec = $datasets[$datasetName]
    Assert-RequiredPath (Join-Path $repo $spec.DataProbe) "$datasetName data file"
    Assert-RequiredPath $spec.SourceConfig "$datasetName source config"
    Assert-RequiredPath $spec.FinetuneConfig "$datasetName finetune config"
    Assert-RequiredPath $finetuneCheckpoints[$datasetName] "$datasetName finetuned encoder checkpoint"
    foreach ($config in $spec.BaseConfigs.Values) {
        Assert-RequiredPath $config "$datasetName Base-only config"
    }
    foreach ($config in $spec.RouterConfigs.Values) {
        Assert-RequiredPath $config "$datasetName Router config"
    }
}

$backbones = @('gwn', 'stgcn', 'staeformer', 'argcn')
$completed = 0

foreach ($datasetName in $datasets.Keys) {
    $spec = $datasets[$datasetName]
    $baseCheckpoints = @{}

    Write-Host ''
    Write-Host "######## DATASET $datasetName : Base-only (4 jobs) ########"
    foreach ($backbone in $backbones) {
        Invoke-BaseOnly -Dataset $datasetName -Backbone $backbone -Config $spec.BaseConfigs[$backbone]
        $baseCheckpoints[$backbone] = "artifacts\cross_dataset_final\${datasetName}_baseonly_${backbone}_seed42\downstream_best.pt"
        Assert-RequiredPath $baseCheckpoints[$backbone] "$datasetName $backbone Base-only checkpoint"
        $completed++
    }

    foreach ($variant in @('source', 'finetuned')) {
        $bankRelative = "${queueBankRoot}\${datasetName}_${variant}"
        $bankAbsolute = Join-Path $repo $bankRelative
        $bankConfig = if ($variant -eq 'source') { $spec.SourceConfig } else { $spec.FinetuneConfig }
        $encoderCheckpoint = if ($variant -eq 'source') { $sourceCheckpoint } else { $finetuneCheckpoints[$datasetName] }

        # A stale queue-owned partial bank is safe to remove before rebuilding.
        Remove-QueueBank -Path $bankRelative
        Write-Host ''
        Write-Host "===== BUILD $datasetName $variant Bank ====="
        & python -u scripts\build_bank.py `
            '--config' $bankConfig `
            '--checkpoint' $encoderCheckpoint `
            '--output-dir' $bankRelative `
            '--dataset-name' "${datasetName}_${variant}_queue"
        if ($LASTEXITCODE -ne 0) {
            throw "Bank build failed: $datasetName $variant (exit code $LASTEXITCODE)"
        }
        Assert-RequiredPath (Join-Path $bankAbsolute 'manifest.json') "$datasetName $variant Bank manifest"

        Write-Host ''
        Write-Host "######## DATASET $datasetName : $variant Router (4 jobs) ########"
        foreach ($backbone in $backbones) {
            Invoke-Router `
                -Dataset $datasetName `
                -Variant $variant `
                -Backbone $backbone `
                -Config $spec.RouterConfigs[$backbone] `
                -EncoderCheckpoint $encoderCheckpoint `
                -Bank $bankRelative `
                -BaseCheckpoint $baseCheckpoints[$backbone]
            $completed++
        }

        # Do not leave the large temporary Bank on the experiment machine.
        Remove-QueueBank -Path $bankRelative
    }
}

if ($completed -ne 24) {
    throw "Expected 24 completed jobs, observed $completed"
}

Write-Host ''
Write-Host '===== ALL 24 CROSS-DATASET JOBS COMPLETED ====='
Write-Host 'Datasets: PEMS04, PEMS08'
Write-Host 'Per dataset: 4 Base-only + 4 source Router + 4 finetuned Router'
