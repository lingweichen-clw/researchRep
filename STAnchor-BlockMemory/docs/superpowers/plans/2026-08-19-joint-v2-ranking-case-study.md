# Joint v2 Ranking Case Study Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add leakage-safe, pretraining-aligned local ranking diagnostics for the Joint v2 checkpoint and produce a complete case-study report with figures.

**Architecture:** Extend the existing retrieval visualization diagnostic with local ranking metrics and an explicit `pretrain_broad_causal` alias for the existing broad causal candidate sampler. Keep the existing deployment path unchanged. Generate matched pretrained/random Banks and run both protocols, then render training and retrieval figures into a self-contained report.

**Tech Stack:** Python, PyTorch, NumPy, SciPy, Matplotlib, existing STAnchor Bank and diagnostic APIs.

---

### Task 1: Add local ranking metric tests

**Files:**
- Modify: `tests/test_retrieval_visualization.py`
- Modify: `stanchor/diagnostics/retrieval_visualization.py`

- [ ] Add tests for one anchor with known orderings: perfect ranking gives Spearman/Kendall/NDCG@5/Recall@1 equal to 1; reversed ranking gives negative local rank scores and Recall@1 equal to 0.
- [ ] Add a test showing a five-candidate pool is eligible for Recall@1 and NDCG@5 while Recall@5 remains secondary.
- [ ] Add a test that ties are excluded from Kendall's denominator and that an anchor with fewer than two valid candidates is excluded from local rank aggregates.
- [ ] Run `python -m pytest tests/test_retrieval_visualization.py -q` and confirm the new tests fail before implementation.

### Task 2: Implement anchor-wise metrics and protocol metadata

**Files:**
- Modify: `stanchor/diagnostics/retrieval_visualization.py`
- Modify: `scripts/visualize_retrieval.py`

- [ ] Implement a pure NumPy helper returning per-anchor Spearman, Kendall, Recall@1, NDCG@5, candidate counts, and eligibility masks from `[B,N,R]` key/teacher distances.
- [ ] Use stable rank handling and finite masks; for Kendall count only strict candidate pairs.
- [ ] Use graded relevance `exp(-teacher_distance / temperature)` and ideal DCG for NDCG@5.
- [ ] Extend supported versions with `tgge_joint` and accept `pretrain_broad_causal` as the primary protocol while mapping it to the existing model-independent broad causal sampler.
- [ ] Add local metric aggregates and random-expectation summaries to `metrics.json` and `ranking_metrics.csv` without removing legacy Recall@5 fields.
- [ ] Update figure labels so Recall@5 is explicitly secondary and local metrics are primary.
- [ ] Run the focused tests and `python -m pytest tests/test_retrieval_visualization.py -q`.

### Task 3: Build matched Joint v2 Banks and controls

**Files:**
- Create: `scripts/create_random_pretrain_checkpoint.py` only if an existing random checkpoint utility cannot be reused.
- Create: `scripts/run_joint_v2_case_study.ps1`.

- [ ] Verify `pretrain_best.pt` loads with `configs/metrla_e5_tgge_latent48_v2.yaml` and record its checkpoint epoch and fingerprint.
- [ ] Create a same-config random initialization checkpoint with the existing checkpoint contract, without training it.
- [ ] Build pretrained and random Banks with identical event axes under `artifacts/metrla_bank_e5_tgge_latent48_v2_seed42` and `artifacts/metrla_bank_e5_tgge_latent48_v2_random_seed42`.
- [ ] Validate both Banks with `_validate_bank`; abort on fingerprint, graph, scaler, or event-axis mismatch.
- [ ] Run the diagnostic once with `--candidate-protocol pretrain_broad_causal` and once with `--candidate-protocol exact_calendar`, omitting `--max-batches`.

### Task 4: Render complete report

**Files:**
- Create: `scripts/plot_joint_v2_training.py`.
- Create: `doc/诊断报告合集/E5-TGGE-Joint-v2-Ranking-CaseStudy.md`.

- [ ] Plot epoch-wise train/validation total, reconstruction, and relation losses from `pretrain_metrics.jsonl`; mark epoch 26 and label the run Joint v2.
- [ ] Embed local ranking comparison, candidate-count/random-baseline, alignment, and deterministic Top-5 case figures.
- [ ] Define every special term at first use and state that query future is used only after ranking for teacher metrics, cases, and plots.
- [ ] State that Relation-only is unfinished and excluded from conclusions.
- [ ] Include a Keep/Remove/Next decision based on both protocols and all primary metrics.

### Task 5: Verification

**Files:**
- No source changes.

- [ ] Run all focused visualization tests and the case-study CLI without a batch cap.
- [ ] Check every PNG exists and has nonzero size; inspect image dimensions.
- [ ] Check `metrics.json` reports `complete_validation=true`, both protocols, local metrics, and future boundary metadata.
- [ ] Run `git diff --check` and record exact output paths and measured timings.
