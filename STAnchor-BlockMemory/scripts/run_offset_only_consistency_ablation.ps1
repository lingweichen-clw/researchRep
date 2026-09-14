param()

$ErrorActionPreference = 'Stop'

# One new encoder, one matching Bank, and two ARGCN payload controls.
# Existing completed artifacts are preserved and skipped, so the queue is resumable.

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

$encoderConfig = 'configs\metrla_e5_tgge_hn_offset_only_v1_transfer_hidden128_ffn2_b16.yaml'
$encoderCheckpoint = 'artifacts\metrla_e5_tgge_hn_offset_only_v1_transfer_hidden128_ffn2_b16_seed42\pretrain_best.pt'
$encoderBank = 'artifacts\case_bank_hn_offset_only_v1_transfer_hidden128_ffn2_b16_seed42'
$baseCheckpoint = 'artifacts\convergence\formal_20260826_argcn_base_only_v1\downstream_best.pt'
$randomCheckpoint = 'artifacts\metrla_e5_tgge_hn_offset_decay_v2_transfer_hidden128_ffn2_b16_random_seed42\random_checkpoint.pt'
$randomBank = 'artifacts\case_bank_hn_offset_decay_v2_transfer_hidden128_ffn2_b16_random_seed42'
$caseStudyDir = 'artifacts\casestudy_hn_offset_only_v1_hidden128_ffn2\visualization_weekday_radius1_overlap'
$payloadMetrics = 'artifacts\casestudy_hn_offset_only_v1_hidden128_ffn2\future_payload_comparison.json'

$jobs = @(
    @{
        Label = 'offset_only_teacher_offset_decay_router_argcn'
        Config = 'configs\ablation_offset_only_teacher_offset_decay_router_argcn.yaml'
        RunName = 'convergence/ablation_offset_only_teacher_offset_decay_router_argcn_seed42'
    },
    @{
        Label = 'offset_only_teacher_offset_only_router_argcn'
        Config = 'configs\ablation_offset_only_teacher_offset_only_router_argcn.yaml'
        RunName = 'convergence/ablation_offset_only_teacher_offset_only_router_argcn_seed42'
    }
)

Assert-RequiredPath $encoderConfig 'Offset-only encoder config'
Assert-RequiredPath $baseCheckpoint 'frozen ARGCN base checkpoint'
Assert-RequiredPath $randomCheckpoint 'matched-random encoder checkpoint'
Assert-RequiredPath (Join-Path $randomBank 'manifest.json') 'matched-random Bank manifest'
Assert-RequiredPath (Join-Path $repo '..\data\METRLA_data\METR-LA.h5') 'METR-LA data file'
Assert-RequiredPath (Join-Path $repo '..\data\METRLA_data\adj_mx.pkl') 'METR-LA adjacency'
foreach ($job in $jobs) {
    Assert-RequiredPath $job.Config ($job.Label + ' config')
}

if (-not (Test-Path -LiteralPath $encoderCheckpoint)) {
    Invoke-Python -Label 'offset_only_pretraining' -Arguments @(
        'scripts/pretrain.py',
        '--config', $encoderConfig
    )
} else {
    Write-Host ('SKIP completed pretraining: ' + $encoderCheckpoint)
}
Assert-RequiredPath $encoderCheckpoint 'Offset-only encoder checkpoint'

if (-not (Test-Path -LiteralPath (Join-Path $encoderBank 'manifest.json'))) {
    Invoke-Python -Label 'offset_only_bank' -Arguments @(
        'scripts/build_bank.py',
        '--config', $encoderConfig,
        '--checkpoint', $encoderCheckpoint,
        '--output-dir', $encoderBank,
        '--dataset-name', 'metrla_offset_only_seed42'
    )
} else {
    Write-Host ('SKIP completed Bank: ' + $encoderBank)
}
Assert-RequiredPath (Join-Path $encoderBank 'manifest.json') 'Offset-only Bank manifest'

if (-not (Test-Path -LiteralPath (Join-Path $caseStudyDir 'metrics.json'))) {
    Invoke-Python -Label 'offset_only_retrieval_case_study' -Arguments @(
        'scripts/visualize_retrieval.py',
        '--version', 'hn_offset_only_v1',
        '--config', $encoderConfig,
        '--checkpoint', $encoderCheckpoint,
        '--bank', $encoderBank,
        '--random-checkpoint', $randomCheckpoint,
        '--random-bank', $randomBank,
        '--split', 'val',
        '--output-dir', $caseStudyDir,
        '--candidate-protocol', 'weekday_radius1_overlap',
        '--level-weight', '0'
    )
} else {
    Write-Host ('SKIP completed retrieval CaseStudy: ' + $caseStudyDir)
}

if (-not (Test-Path -LiteralPath $payloadMetrics)) {
    Invoke-Python -Label 'offset_only_payload_comparison' -Arguments @(
        'scripts/compare_future_payloads.py',
        '--config', $encoderConfig,
        '--checkpoint', $encoderCheckpoint,
        '--bank', $encoderBank,
        '--split', 'val',
        '--candidate-protocol', 'weekday_radius1_overlap',
        '--output', $payloadMetrics
    )
} else {
    Write-Host ('SKIP completed payload comparison: ' + $payloadMetrics)
}

$completed = 0
foreach ($job in $jobs) {
    $bestCheckpoint = Join-Path (Join-Path 'artifacts' $job.RunName) 'downstream_best.pt'
    if (Test-Path -LiteralPath $bestCheckpoint) {
        Write-Host ('SKIP completed downstream run: ' + $job.Label)
        $completed++
        continue
    }
    Invoke-Python -Label $job.Label -Arguments @(
        'scripts/train_downstream.py',
        '--config', $job.Config,
        '--pretrained-checkpoint', $encoderCheckpoint,
        '--bank', $encoderBank,
        '--base-checkpoint', $baseCheckpoint,
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

if ($completed -ne $jobs.Count) {
    throw ('Expected ' + $jobs.Count + ' completed downstream jobs, observed ' + $completed)
}

Write-Host ''
Write-Host '===== OFFSET-ONLY CONSISTENCY ABLATION QUEUE COMPLETED ====='
Write-Host 'Finished: one Offset-only encoder/Bank and two matched ARGCN payload controls.'
