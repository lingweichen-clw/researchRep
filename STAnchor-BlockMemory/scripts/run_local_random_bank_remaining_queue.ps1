$ErrorActionPreference = 'Stop'

# Local remaining random-Bank ablations for the original four backbones.
# STGCN is already finished. This queue only runs STAEformer and ARGCN.
# GWN random is intentionally skipped. Reuse the official local random encoder/Bank if they already exist.
# Do not delete the official random Bank.

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
$randomCheckpoint = 'artifacts\metrla_e5_tgge_hn_offset_decay_v2_transfer_hidden128_ffn2_b16_random_seed42\random_checkpoint.pt'
$randomBank = 'artifacts\case_bank_hn_offset_decay_v2_transfer_hidden128_ffn2_b16_random_seed42'
$jobs = @(
    @{
        Label = 'random_router_staeformer'
        Config = 'configs\ablation_random_bank_router_staeformer.yaml'
        Base = 'artifacts\convergence\formal_20260828_staeformer_base_only_v2\downstream_best.pt'
        RunName = 'convergence/ablation_random_bank_router_staeformer_seed42'
    },
    @{
        Label = 'random_router_argcn'
        Config = 'configs\ablation_random_bank_router_argcn.yaml'
        Base = 'artifacts\convergence\formal_20260826_argcn_base_only_v1\downstream_best.pt'
        RunName = 'convergence/ablation_random_bank_router_argcn_seed42'
    }
)

Assert-RequiredPath $encoderConfig 'METR-LA encoder config'
Assert-RequiredPath (Join-Path $repo '..\data\METRLA_data\METR-LA.h5') 'METR-LA data file'
Assert-RequiredPath (Join-Path $repo '..\data\METRLA_data\adj_mx.pkl') 'METR-LA adjacency'
foreach ($job in $jobs) {
    Assert-RequiredPath $job.Config $job.Label
    Assert-RequiredPath $job.Base ($job.Label + ' frozen base')
}

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

$completed = 0
foreach ($job in $jobs) {
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

if ($completed -ne 2) {
    throw ('Expected 2 completed jobs, observed ' + $completed)
}

Write-Host ''
Write-Host '===== LOCAL RANDOM-BANK QUEUE COMPLETED ====='
Write-Host 'Finished: STAEformer, ARGCN'
Write-Host 'Skipped: STGCN (already finished), GWN (not in the narrowed ablation)'
