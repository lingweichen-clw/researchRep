$ErrorActionPreference = 'Stop'

# Resolve the repository from this script location so the queue is portable
# between the local machine and the experiment machine.
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

$sourceCheckpoint = 'artifacts\metrla_e5_tgge_hn_offset_decay_v2_transfer_hidden128_ffn2_b16_seed42\pretrain_best.pt'
$sourceBank = 'artifacts\cross_dataset_stage1\pemsbay_source_bank'
$finetuneCheckpoint = 'artifacts\cross_dataset_t1\pemsbay_head_adapter_seed42\retrieval_t1_best.pt'
$finetuneBank = 'artifacts\cross_dataset_t1\pemsbay_head_adapter_bank'

$dataRoot = (Resolve-Path (Join-Path $repo '..\data')).Path
Assert-RequiredPath (Join-Path $dataRoot 'pemsBay_data\pems-bay.h5') 'PEMS-BAY data file'
Assert-RequiredPath (Join-Path $dataRoot 'pemsBay_data\adj_mx_bay.pkl') 'PEMS-BAY adjacency file'
Assert-RequiredPath $sourceCheckpoint 'source encoder checkpoint'
Assert-RequiredPath (Join-Path $sourceBank 'manifest.json') 'source Bank manifest'
Assert-RequiredPath $finetuneCheckpoint 'finetuned encoder checkpoint'
Assert-RequiredPath (Join-Path $finetuneBank 'manifest.json') 'finetuned Bank manifest'

$baseCheckpoints = [ordered]@{
    gwn        = 'artifacts\cross_dataset_stage1\pemsbay_baseonly_gwn_seed42\downstream_best.pt'
    stgcn      = 'artifacts\cross_dataset_stage1\pemsbay_baseonly_stgcn_seed42\downstream_best.pt'
    staeformer = 'artifacts\cross_dataset_stage1\pemsbay_baseonly_staeformer_seed42\downstream_best.pt'
    argcn      = 'artifacts\cross_dataset_stage1\pemsbay_baseonly_argcn_seed42\downstream_best.pt'
}

$routerConfigs = [ordered]@{
    gwn        = 'configs\cross_dataset_pemsbay_router_gwn.yaml'
    stgcn      = 'configs\cross_dataset_pemsbay_router_stgcn.yaml'
    staeformer = 'configs\cross_dataset_pemsbay_router_staeformer.yaml'
    argcn      = 'configs\cross_dataset_pemsbay_router_argcn.yaml'
}

foreach ($name in $baseCheckpoints.Keys) {
    Assert-RequiredPath $baseCheckpoints[$name] "$name Base-only checkpoint"
    Assert-RequiredPath $routerConfigs[$name] "$name Router config"
}

$jobs = @(
    [pscustomobject]@{
        Label = 'pemsbay_source_router_gwn'
        Config = $routerConfigs.gwn
        EncoderCheckpoint = $sourceCheckpoint
        Bank = $sourceBank
        BaseCheckpoint = $baseCheckpoints.gwn
        RunName = 'cross_dataset_final/pemsbay_source_router_gwn_seed42'
    },
    [pscustomobject]@{
        Label = 'pemsbay_source_router_stgcn'
        Config = $routerConfigs.stgcn
        EncoderCheckpoint = $sourceCheckpoint
        Bank = $sourceBank
        BaseCheckpoint = $baseCheckpoints.stgcn
        RunName = 'cross_dataset_final/pemsbay_source_router_stgcn_seed42'
    },
    [pscustomobject]@{
        Label = 'pemsbay_source_router_staeformer'
        Config = $routerConfigs.staeformer
        EncoderCheckpoint = $sourceCheckpoint
        Bank = $sourceBank
        BaseCheckpoint = $baseCheckpoints.staeformer
        RunName = 'cross_dataset_final/pemsbay_source_router_staeformer_seed42'
    },
    [pscustomobject]@{
        Label = 'pemsbay_source_router_argcn'
        Config = $routerConfigs.argcn
        EncoderCheckpoint = $sourceCheckpoint
        Bank = $sourceBank
        BaseCheckpoint = $baseCheckpoints.argcn
        RunName = 'cross_dataset_final/pemsbay_source_router_argcn_seed42'
    },
    [pscustomobject]@{
        Label = 'pemsbay_finetuned_router_gwn'
        Config = $routerConfigs.gwn
        EncoderCheckpoint = $finetuneCheckpoint
        Bank = $finetuneBank
        BaseCheckpoint = $baseCheckpoints.gwn
        RunName = 'cross_dataset_final/pemsbay_finetuned_router_gwn_seed42'
    },
    [pscustomobject]@{
        Label = 'pemsbay_finetuned_router_stgcn'
        Config = $routerConfigs.stgcn
        EncoderCheckpoint = $finetuneCheckpoint
        Bank = $finetuneBank
        BaseCheckpoint = $baseCheckpoints.stgcn
        RunName = 'cross_dataset_final/pemsbay_finetuned_router_stgcn_seed42'
    },
    [pscustomobject]@{
        Label = 'pemsbay_finetuned_router_staeformer'
        Config = $routerConfigs.staeformer
        EncoderCheckpoint = $finetuneCheckpoint
        Bank = $finetuneBank
        BaseCheckpoint = $baseCheckpoints.staeformer
        RunName = 'cross_dataset_final/pemsbay_finetuned_router_staeformer_seed42'
    },
    [pscustomobject]@{
        Label = 'pemsbay_finetuned_router_argcn'
        Config = $routerConfigs.argcn
        EncoderCheckpoint = $finetuneCheckpoint
        Bank = $finetuneBank
        BaseCheckpoint = $baseCheckpoints.argcn
        RunName = 'cross_dataset_final/pemsbay_finetuned_router_argcn_seed42'
    }
)

foreach ($job in $jobs) {
    Write-Host ''
    Write-Host "===== START $($job.Label) ====="

    $cliArgs = @(
        '--config', $job.Config,
        '--pretrained-checkpoint', $job.EncoderCheckpoint,
        '--bank', $job.Bank,
        '--base-checkpoint', $job.BaseCheckpoint,
        '--mode', 'learned_topk_error_aware',
        '--candidate-protocol', 'weekday_radius1_overlap',
        '--level-weight', '0',
        '--candidate-quality-weight', '0',
        '--epochs', '50',
        '--disable-early-stopping',
        '--frozen-path-cache',
        '--seed', '42',
        '--run-name', $job.RunName
    )

    # Direct invocation keeps Python's normal stdout/stderr visible without
    # routing it through PowerShell's Tee-Object error formatting.
    & python -u scripts\train_downstream.py @cliArgs

    if ($LASTEXITCODE -ne 0) {
        throw "Training failed: $($job.Label) (exit code $LASTEXITCODE)"
    }

    Write-Host "===== DONE $($job.Label) ====="
}

Write-Host ''
Write-Host '===== ALL 8 PEMS-BAY ROUTER JOBS COMPLETED ====='
