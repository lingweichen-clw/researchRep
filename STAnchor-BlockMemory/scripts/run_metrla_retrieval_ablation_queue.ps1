param(
    [switch]$RandomOnly,
    [switch]$RawL1Only
)

$ErrorActionPreference = 'Stop'

if ($RandomOnly -and $RawL1Only) {
    throw 'RandomOnly and RawL1Only are mutually exclusive.'
}
$runRandom = -not $RawL1Only
$runRawL1 = -not $RandomOnly

# METR-LA retrieval ablations after the case-study evidence is already in.
# Only three runs: random ARGCN, raw-L1 ARGCN / GWN.
# Same Router, weekday_radius1_overlap, Top-12, frozen path cache.
# Build missing banks, but do not delete the official trained or random banks.

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

$encoderConfig = 'configs\metrla_e5_tgge_hn_offset_decay_v2_transfer_hidden128_ffn2_b16.yaml'
$trainedCheckpoint = 'artifacts\metrla_e5_tgge_hn_offset_decay_v2_transfer_hidden128_ffn2_b16_seed42\pretrain_best.pt'
$trainedBank = 'artifacts\case_bank_hn_offset_decay_v2_transfer_hidden128_ffn2_b16_seed42'
$randomCheckpoint = 'artifacts\metrla_e5_tgge_hn_offset_decay_v2_transfer_hidden128_ffn2_b16_random_seed42\random_checkpoint.pt'
$randomBank = 'artifacts\case_bank_hn_offset_decay_v2_transfer_hidden128_ffn2_b16_random_seed42'

$randomJobs = @(
    @{
        Label = 'random_router_argcn'
        Config = 'configs\ablation_random_bank_router_argcn.yaml'
        Base = 'artifacts\convergence\formal_20260826_argcn_base_only_v1\downstream_best.pt'
        RunName = 'convergence/ablation_random_bank_router_argcn_seed42'
    }
)
$rawl1Jobs = @(
    @{
        Label = 'rawl1_router_gwn'
        Config = 'configs\ablation_rawl1_router_gwn.yaml'
        Base = 'artifacts\convergence\downstream_tgge_v3_matched_fulltrain_queue\downstream_tgge_v3_graphwavenet_base_only_fulltrain_seed42\downstream_best.pt'
        RunName = 'convergence/ablation_rawl1_router_gwn_seed42'
    },
    @{
        Label = 'rawl1_router_argcn'
        Config = 'configs\ablation_rawl1_router_argcn.yaml'
        Base = 'artifacts\convergence\formal_20260826_argcn_base_only_v1\downstream_best.pt'
        RunName = 'convergence/ablation_rawl1_router_argcn_seed42'
    }
)

Assert-RequiredPath $encoderConfig 'METR-LA encoder config'
Assert-RequiredPath (Join-Path $repo '..\data\METRLA_data\METR-LA.h5') 'METR-LA data file'
Assert-RequiredPath (Join-Path $repo '..\data\METRLA_data\adj_mx.pkl') 'METR-LA adjacency'
foreach ($job in @($randomJobs + $rawl1Jobs | Where-Object {
    ($runRandom -and $_.Label.StartsWith('random_')) -or
    ($runRawL1 -and $_.Label.StartsWith('rawl1_'))
})) {
    Assert-RequiredPath $job.Config $job.Label
    Assert-RequiredPath $job.Base ($job.Label + ' frozen base')
}

if ($runRandom) {
    if (-not (Test-Path -LiteralPath $randomCheckpoint)) {
        New-Item -ItemType Directory -Path (Split-Path -Parent (Join-Path $repo $randomCheckpoint)) -Force | Out-Null
        Invoke-Python -Label 'build_random_encoder' -Arguments @(
            'scripts\build_random_checkpoint.py',
            '--config', $encoderConfig,
            '--output', $randomCheckpoint,
            '--seed', '42'
        )
    }
    Assert-RequiredPath $randomCheckpoint 'random encoder checkpoint'

    if (-not (Test-Path -LiteralPath (Join-Path $randomBank 'manifest.json'))) {
        Invoke-Python -Label 'build_random_bank' -Arguments @(
            'scripts\build_bank.py',
            '--config', $encoderConfig,
            '--checkpoint', $randomCheckpoint,
            '--output-dir', $randomBank,
            '--dataset-name', 'metrla_random_official'
        )
    }
    Assert-RequiredPath (Join-Path $randomBank 'manifest.json') 'random Bank manifest'
}

if ($runRawL1) {
    Assert-RequiredPath $trainedCheckpoint 'trained encoder checkpoint'
    if (-not (Test-Path -LiteralPath (Join-Path $trainedBank 'manifest.json'))) {
        Invoke-Python -Label 'build_trained_bank' -Arguments @(
            'scripts\build_bank.py',
            '--config', $encoderConfig,
            '--checkpoint', $trainedCheckpoint,
            '--output-dir', $trainedBank,
            '--dataset-name', 'metrla_official'
        )
    }
    Assert-RequiredPath (Join-Path $trainedBank 'manifest.json') 'trained Bank manifest'
}

$completed = 0
if ($runRandom) {
    foreach ($job in $randomJobs) {
        Invoke-Python -Label $job.Label -Arguments @(
            'scripts\train_downstream.py',
            '--config', $job.Config,
            '--pretrained-checkpoint', $randomCheckpoint,
            '--bank', $randomBank,
            '--base-checkpoint', $job.Base,
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
        $completed++
    }
}
if ($runRawL1) {
    foreach ($job in $rawl1Jobs) {
        Invoke-Python -Label $job.Label -Arguments @(
            'scripts\train_downstream.py',
            '--config', $job.Config,
            '--pretrained-checkpoint', $trainedCheckpoint,
            '--bank', $trainedBank,
            '--base-checkpoint', $job.Base,
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
        $completed++
    }
}

$expected = 0
if ($runRandom) { $expected += $randomJobs.Count }
if ($runRawL1) { $expected += $rawl1Jobs.Count }
if ($completed -ne $expected) {
    throw ('Expected ' + $expected + ' completed jobs, observed ' + $completed)
}

Write-Host ''
Write-Host '===== METR-LA RETRIEVAL ABLATION QUEUE COMPLETED ====='
if ($runRandom) { Write-Host 'Finished: random ARGCN' }
if ($runRawL1) { Write-Host 'Finished: raw-L1 GWN, raw-L1 ARGCN' }
