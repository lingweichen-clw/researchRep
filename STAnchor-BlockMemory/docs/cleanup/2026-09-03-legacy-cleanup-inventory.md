# 2026-09-03 Legacy Cleanup Inventory

本清单是删除前的审计记录。所有路径均相对于
`D:\projects\researchProjects\TrafficRobustST\STAnchor-BlockMemory`，执行删除时必须解析为绝对路径并确认位于该项目目录内。

## KEEP: current runtime and evidence

### Current retrieval and Bank

- `artifacts/metrla_e5_tgge_hn_offset_decay_v2_transfer_hidden128_ffn2_b16_seed42/`
- `artifacts/metrla_e5_tgge_hn_offset_decay_v2_transfer_hidden128_ffn2_b16_random_seed42/`
- `artifacts/case_bank_hn_offset_decay_v2_transfer_hidden128_ffn2_b16_seed42/`
- `artifacts/case_bank_hn_offset_decay_v2_transfer_hidden128_ffn2_b16_random_seed42/`
- `configs/formal_base_as_candidate_{argcn,gwn,staeformer,stgcn}.yaml`
- `configs/cross_dataset_pemsbay_source_encoder_stage1.yaml`

### Current downstream and cross-dataset results

- `artifacts/convergence/formal_20260901_final_router_argcn_seed42/`
- `artifacts/convergence/formal_20260901_final_router_gwn_seed42/`
- `artifacts/convergence/formal_20260901_final_router_staeformer_seed42/`
- `artifacts/convergence/formal_20260901_final_router_stgcn_seed42/`
- `artifacts/convergence/downstream_tgge_v3_matched_fulltrain_queue/`
- `artifacts/cross_dataset_stage0/`
- `artifacts/cross_dataset_stage1/`

### Current case-study evidence

- `artifacts/casestudy_hn_offset_decay_v2_hidden128_ffn2/`
- `doc/诊断报告合集/E5-TGGE-HN-OffsetDecay-v2-hidden128-最终CaseStudy报告.md`
- current weekday-radius retrieval diagnostics and UMAP/PCA source files

### Runtime interfaces that must remain

- `scripts/pretrain.py`, `scripts/build_bank.py`, `scripts/train_downstream.py`, `scripts/evaluate.py`
- `stanchor/bank/`, `stanchor/retrieval/retriever.py`, `stanchor/retrieval/strategies.py`
- `stanchor/models/encoder.py`, `stanchor/models/pretraining.py`, `stanchor/models/retrieval_head.py`
- `stanchor/models/retrieval_router.py` (`RetrievalAwareMHAResidualRouter`)
- four current backbone implementations
- `ConfidenceHead` and `SafeResidualFusion` state-bearing compatibility fields, because finalized checkpoints contain these keys even when frozen/unused

### Reproducibility dependencies discovered during final audit

- `artifacts/convergence/formal_20260826_argcn_base_only_v1/` must be kept despite
  its historical name. The finalized ArgCN Router log records this exact
  Base-only checkpoint and fingerprint as its frozen backbone source.
- `artifacts/convergence/formal_20260828_staeformer_base_only_v2/` remains the
  STAEformer Base-only source.
- The Graph WaveNet and STGCN Base-only sources remain under
  `artifacts/convergence/downstream_tgge_v3_matched_fulltrain_queue/`.

## REVIEW / ARCHIVE: do not delete in this pass

- all untracked files shown by `git status`, unless the user explicitly identifies them as disposable
- shared `stanchor/losses/pretraining.py`, `stanchor/retrieval/semantic_profile.py`, and `stanchor/retrieval/trend_residual.py` until their current import closure is removed
- `stanchor/diagnostics/retrieval_visualization.py` because the current case-study pipeline still imports it

## DELETE: explicit obsolete artifacts

The following root artifact directories are old E2/E3/early-E5 or superseded early calibrator outputs and have no dependency from the finalized runtime or current evidence:

- `artifacts/case_bank_hn_offset_decay_v1_seed42/`
- `artifacts/metrla_e2_day96_topk_horizon_seed2025/`
- `artifacts/metrla_e5a_base_only_seed42/`
- `artifacts/metrla_e5a_offset_decay_horizon_seed42/`
- `artifacts/metrla_e5a_offset_decay_seed42/`
- `artifacts/metrla_e5_final_global288_error_aware_seed42/`
- `artifacts/metrla_e5_final_latent48_global288_seed42/`
- `artifacts/metrla_e5_final_symnorm_local12_seed42/`
- `artifacts/metrla_e5_final_symnorm_seed42/`
- `artifacts/metrla_e5_tgge_latent48_relation_only_v2_seed42/`
- `artifacts/metrla_e5_tgge_latent48_v2_seed42/`
- `artifacts/metrla_e5_tgge_single_view_hn_offset_decay_v1_seed42/`
- `artifacts/metrla_e5_tgge_single_view_masked_relation_rank_top2_seed42/`
- `artifacts/metrla_learned_topk_horizon_seed2024/`
- `artifacts/metrla_learned_topk_horizon_seed2025/`
- `artifacts/metrla_learned_topk_horizon_seed42/`
- `artifacts/metrla_pretrain_seed42/`
- `artifacts/metrla_random_seed_stability/`
- `artifacts/metrla_raw_l1_topk_horizon_seed2024/`
- `artifacts/metrla_raw_l1_topk_horizon_seed2025/`
- `artifacts/metrla_raw_l1_topk_horizon_seed42/`
- `artifacts/metrla_weekly_mean_horizon_seed42/`
- `artifacts/e5_t1/`
- `artifacts/e5_t1a/`

The following convergence/diagnostic directories are old protocol, E2/E3/E5A, early HorizonMixer, old confidence, or smoke-only outputs:

- `artifacts/convergence/candidate_protocol/`
- `artifacts/convergence/downstream_candidate_protocol/`
- `artifacts/convergence/teacher_metric_diagnostic/`
- `artifacts/convergence/visualization/e5a/`
- `artifacts/convergence/visualization/tgge_relation_only_v2/`
- `artifacts/convergence/visualization/tgge_single_view_v3_reconstruction2/`
- `artifacts/convergence/retrieval_temperature/`
- `artifacts/convergence/formal_20260828_horizon_mixer_staeformer_seed42/`
- `artifacts/convergence/formal_20260826_stgcn_set_attention_forecast_50/`
- `artifacts/convergence/validation_20260827_hn_cq_w005_gwn_3epoch/`
- `artifacts/convergence/validation_20260827_hn_cq_w005_stgcn_3epoch/`
- `artifacts/convergence/validation_20260827_hn_cq_w010_gwn_3epoch/`
- `artifacts/convergence/validation_20260827_hn_cq_w010_stgcn_3epoch/`
- `artifacts/convergence/validation_20260827_hn_cq_w020_gwn_3epoch/`
- `artifacts/convergence/validation_20260827_hn_cq_w020_stgcn_3epoch/`
- `artifacts/convergence/logs/tgge_single_view_v3/`
- `artifacts/convergence/logs/tgge_single_view_v3_higher_order_reconstruction2/`
- `artifacts/convergence/logs/e5_final_pooling_attribution_seed42/`

Diagnostic candidate-pool outputs under `artifacts/diagnostics` are deleted only for `exact_calendar`, `relaxed_calendar`, and `relaxed_calendar_diverse`; the current `weekday_radius1_overlap` outputs remain.

The following loose root-level files were audited and are obsolete E2/E3/E5A,
relation-only, confidence, or superseded case-study outputs:

- `artifacts/case_norank_relation_v2_val.json`
- `artifacts/case_rank005_v3_val.json`
- `artifacts/case_rank01_v3_val.json`
- `artifacts/e3_level0_metrla_baseline.log`
- `artifacts/e3_level0_pemsbay_baseline.log`
- `artifacts/e5_t0_metrla_val.json`
- `artifacts/e5_t0_metrla_val.stdout.log`
- `artifacts/e5_t0_pemsbay_val.json`
- `artifacts/e5_t0_pemsbay_val.stdout.log`
- `artifacts/metrla_e2_day96_retrieval_diagnostics_val.json`
- `artifacts/metrla_e2_e3_horizon_multiseed_val_summary.json`
- `artifacts/metrla_e3_confidence_multiseed_test_summary.json`
- `artifacts/metrla_e3_confidence_multiseed_val_summary.json`
- `artifacts/metrla_e3_offset_decay_horizon_seed42.stdout.log`
- `artifacts/metrla_e3_relation_relation_val_diagnostics.json`
- `artifacts/metrla_e3_relation_total_val_diagnostics.json`
- `artifacts/metrla_e5_final_latent48_global288_random_seed42.pt`
- `artifacts/metrla_e5_tgge_latent48_v2_random_seed42.pt`
- `artifacts/metrla_e5a_bank_relation_seed42_build.log`
- `artifacts/metrla_e5a_bank_total_seed42_build.log`
- `artifacts/metrla_e5a_base_only_seed42.stdout.log`
- `artifacts/metrla_e5a_offset_decay_horizon_seed42.stdout.log`
- `artifacts/metrla_e5a_relation_level0_val.log`
- `artifacts/metrla_e5a_total_level0_val.log`
- `artifacts/metrla_multiseed_val_summary.json`
- `artifacts/metrla_random_offset_decay_horizon_seed42.stdout.log`
- `artifacts/metrla_retrieval_diagnostics_val.json`
- `artifacts/pemsbay_e5a_bank_relation_seed42_build.log`
- `artifacts/pemsbay_e5a_level0_val.log`
- `artifacts/convergence/diagnostics/spatial_residual_v2_gwn_val_smoke.json`

## DELETE: obsolete source and tests after runtime closure update

Candidate calibrator implementations that are not the finalized Router:

- `LegacyCandidateSetHorizonCorrector`
- `CandidateSetHorizonCorrector`
- `StructuredErrorCorrector`
- `HorizonAwareAggregationHead`
- `TrajectoryConditionedCandidateSetHorizonCorrector`
- `TransformerCandidateRouter`

The current `RetrievalAwareMHAResidualRouter`, the downstream output contract, and compatibility `ConfidenceHead`/`SafeResidualFusion` fields are not deleted. Old implementation classes are removed only together with their target branches, configs, and tests, after a strict checkpoint load smoke.

Obsolete standalone diagnostics/tests to remove after reference search confirms no current caller:

- `stanchor/diagnostics/cfdp.py`, `cfdp_probe.py`, `counterfactual.py`, `direct_gain.py`, `retrieval_temperature.py`, `teacher_metric_diagnostic.py`, `trend_residual.py`
- `scripts/diagnose_counterfactual.py`, `diagnose_direct_gain_policy.py`, `diagnose_retrieval_temperature.py`, `diagnose_teacher_metrics.py`
- matching `test_*` modules for the deleted diagnostics and calibrators

## Deletion record

No deletion is valid unless the final section below is filled with the actual command output, deleted path count, bytes, and preserved-load verification.

### Batch A: artifacts

- status: completed
- deleted paths: 49 directories total (23 root legacy experiment directories, 19 convergence/diagnostic directories, and 7 obsolete candidate-pool diagnostic directories).
- bytes before deletion: 643,372,385 bytes across 293 files (597,307,342 + 33,885,893 + 12,179,150 bytes).
- verification: every explicit allow-list path was absent after deletion; all retained v2/random Banks, v2 checkpoints, four final Router result directories, cross-dataset artifacts, current CaseStudy artifacts, and REVIEW directories remained present.

### Batch B: configs/scripts/tests

- status: completed
- deleted paths: 32 files (14 obsolete configs, 4 obsolete diagnostic scripts, 7 obsolete diagnostic modules, and 7 obsolete tests).
- verification: all 32 paths are absent; no supported code/config references remain. `compileall` and the full unittest suite pass after deletion.

### Batch C: source compatibility cleanup

- status: completed
- deleted paths: obsolete calibrator branches and target construction branches removed from `stanchor/engine/target.py`; `stanchor/models/trajectory_calibrator.py` retained as a documented compatibility import for the finalized Router. `ConfidenceHead` and `SafeResidualFusion` retained because current downstream interfaces/checkpoints and tests still require their state-bearing fields.
- verification: current `RetrievalAwareMHAResidualRouter` is the only executable error-aware calibrator; current configs construct successfully; preserved checkpoint/Bank load smoke and full tests pass.

### Batch D: missed loose legacy artifacts and obsolete archives

- status: completed and verified.
- scope: loose root-level E2/E3/E5A JSON/log/checkpoint files; obsolete v1
  CaseStudy, E5-T1 baseline, validation backup and quarantine directories; and
  obsolete convergence directories for global288, early confidence/quality,
  HorizonMixer, signed-horizon, single-view variants and superseded
  visualizations.
- safety exception: restore and retain
  `artifacts/convergence/formal_20260826_argcn_base_only_v1/`, because the final
  ArgCN Router depends on its frozen backbone checkpoint.
- historical Markdown reports are retained as research provenance even when
  they cite deleted artifacts; those citations are not runtime dependencies.

- deleted loose files: 29 root-level files and 1 current-diagnostics smoke JSON;
  all were verified as obsolete or debug-only before deletion.
- deleted archives: `casestudy_hn_offset_decay`, `e5_t1_baselines`,
  `metrla_base_only_seed42`, `quarantine`, and `validation_backup_20260825`.
- obsolete convergence directories approved for explicit deletion:
  `aggregation_comparison_v1`, the five `downstream_global288_*` families,
  the three August 26 exact-calendar cache runs, the two August 27 candidate-
  quality runs, `gradient_boundary_20260822_graphwavenet_fulltrain`, the two
  `horizon_aware_20260822_*` runs, `line_ab_validation`, the three
  `tgge_single_view_v3*` runs, `tgge_v3_signed_horizon_formal`, and the
  superseded `visualization` directory. Their five matching global288 log
  directories are also deleted; matched-queue Base-only logs remain.
- final provenance audit classified `downstream_tgge_v3_fulltrain_detached_v7`
  as an obsolete detached-calibrator run and
  `formal_20260901_retrieval_aware_mha_argcn_seed42` as the superseded
  `lr=0.001, gamma=0.95` run. Both are deleted together with detached-v7 logs.
- only the two Base-only subdirectories are retained inside
  `downstream_tgge_v3_matched_fulltrain_queue`; its Graph WaveNet/STGCN
  `error_aware_fulltrain` subdirectories and corresponding queue logs are
  deleted because the final Router checkpoints do not depend on them.
- deleted temporary workspace: project-local `tmp` contained only UMAP/t-SNE
  probes, rendered reference pages, a temporary UMAP package copy, and Python
  caches; empty `artifacts/tmp` was also removed. Formal CaseStudy outputs and
  maintained diagnostic scripts/tests remain.
- Batch D deletion total: 84 explicitly enumerated targets, 573 files, and
  92,096,345 bytes. No wildcard deletion was used.

### Batch E: obsolete Markdown reports and superseded plans

- status: completed and verified.
- scope: historical diagnostic reports, completed exploratory reports, and
  implementation plans for deleted E2/E3/E5A, RelationOnly/Joint-v3,
  Latent48, v1/pre-v2 encoders, confidence fusion, HorizonMixer,
  TransformerCandidateRouter, exact-calendar/diverse-calendar experiments,
  and superseded case-study drafts.
- preserved: the v2 hidden128 final CaseStudy report; current v2 pretraining,
  Router, frozen-cache, STAEformer protocol, weekday-radius, and
  cross-dataset plans; plus the maintained Graph WaveNet audit note and
  current diagnostic scripts.
- deleted documents: 48 Markdown files (21 obsolete diagnostic reports, 5
  old experiment-step reports, 9 old optimization plans, 7 stale root-level
  design/implementation notes, 4 old execution plans, and 2
  RelationOnly/Joint-v3 specifications), totaling 49,182 bytes from `HEAD`.
- verification: every removed document described an artifact, protocol, or
  calibrator no longer present in the finalized runtime/evidence closure. No
  executable source or current retained document references any deleted
  filename.

## Final Verification Evidence

- `python -m pytest -q`: 173 tests passed. The only warnings were Matplotlib
  dependency deprecations and a non-fatal Windows ACL warning preventing pytest
  from writing `.pytest_cache`.
- `python -m compileall -q stanchor scripts tests`: passed.
- `python scripts/pretrain.py --help`, `python scripts/build_bank.py --help`,
  and `python scripts/train_downstream.py --help`: passed.
- The v2 trained and random checkpoints loaded strictly into the current
  958,704-parameter pretraining model.
- The trained and random Banks loaded with matching manifests:
  METR-LA, 15,876 events, 207 nodes, retrieval dimension 64.
- All four formal Router YAMLs load as `retrieval_aware_mha_router` with
  `weekday_radius1_overlap` and node Top-12.
- ArgCN, Graph WaveNet, STAEformer, and STGCN finalized downstream checkpoints
  loaded with `strict=True`; each restores the current 879,693-parameter Router.
- Explicit reference scans over `stanchor`, `scripts`, `tests`, `configs`, and
  `README.md` found no executable references to deleted calibrators,
  diagnostics, loose E2/E3/E5A artifacts, or deleted configs.
