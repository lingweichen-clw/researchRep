# Relation-only Pretraining and Vectorized Route Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add a clean-only Future-Relation pretraining mode and vectorize mixed-range route selection without changing route semantics.

**Architecture:** Add explicit pretraining objective and reconstruction weight fields. Relation-only calls the existing clean encoder/retrieval path and the existing OffsetDecay/SymNorm relation loss directly. The route branch receives precomputed static candidate partitions and uses batched partition top-k with the same quota fallback. Validation interval is configurable and defaults to the existing every-epoch behavior.

**Tech Stack:** Python 3.10, PyTorch, unittest, YAML, existing STAnchor-BlockMemory data and checkpoint contracts.

---

### Task 1: Add configuration and document contracts

**Files:**
- Modify: `stanchor/config.py`
- Create: `configs/metrla_e5_tgge_latent48_relation_only_v2.yaml`
- Modify: `doc/优化方案合集/E5-Latent48-TGGE-Structured-Error-Corrector-v2优化方案.md`
- Modify: `doc/优化方案合集/E5-Final-可泛化未来语义检索与风险感知校准统一优化方案.md`
- Test: `tests/test_pretraining_cli.py`

- [x] **Step 1: Write failing config tests**

Add tests asserting the new config loads with `objective=relation_only`, `reconstruction_weight=0`, `retrieval_weight=1`, `validation_interval=2`, and a v2-isolated run name.

- [x] **Step 2: Run the focused test and observe the expected missing-field failure**

Run `conda run -n research python -m unittest tests.test_pretraining_cli.PretrainingCliTest.test_relation_only_config_contract -v` and expect an `AttributeError` or config validation failure.

- [x] **Step 3: Add fields and validation**

Add `objective`, `reconstruction_weight`, and `validation_interval` to `PretrainConfig`. Accept `joint` and `relation_only`; require non-negative reconstruction weight and positive interval; require relation-only reconstruction weight to be zero.

- [x] **Step 4: Add the isolated relation-only YAML and update both self-contained design documents**

Copy the v2 data/model/relation protocol, set relation-only objective and independent `artifacts/metrla_e5_tgge_latent48_relation_only_v2_seed42` output names, and document future-information boundaries and keep/remove decisions.

- [x] **Step 5: Run config tests**

Run the focused test and expect PASS.

### Task 2: Add the clean-only training path

**Files:**
- Modify: `stanchor/models/pretraining.py`
- Modify: `stanchor/losses/pretraining.py`
- Modify: `stanchor/engine/pretrainer.py`
- Test: `tests/test_pretraining_flow.py`

- [x] **Step 1: Write failing relation-only tests**

Test that `forward_relation` returns `[B,N,48]` keys without a mask, relation-only loss is finite, encoder/retrieval gradients are nonzero, and reconstruction head receives no gradient.

- [x] **Step 2: Run the focused tests and observe the missing API failure**

Run `conda run -n research python -m unittest tests.test_pretraining_flow.PretrainingFlowTest.test_relation_only_flow -v` and expect a missing method or function failure.

- [x] **Step 3: Implement clean-only encoding and relation loss**

Expose `encode_clean` through a small relation-only wrapper and reuse `future_relation_retrieval_loss` with the configured teacher mode. Return a `PretrainLoss` with connected zero reconstruction and `retrieval_weight * relation_loss` total.

- [x] **Step 4: Branch the epoch loop by objective**

For `relation_only`, skip mask sampling, masked tokens and reconstruction head. Preserve the existing joint path byte-for-byte in behavior when `objective=joint`.

- [x] **Step 5: Run focused flow tests**

Run `conda run -n research python -m unittest tests.test_pretraining_flow -v` and expect all tests PASS.

### Task 3: Vectorize route selection

**Files:**
- Modify: `stanchor/data/graph.py`
- Modify: `stanchor/models/encoder.py`
- Test: `tests/test_encoder.py`

- [x] **Step 1: Write failing equivalence tests**

Compare the new batched selector with a small reference selector on random scores for normal degree and scarce local/remote degree graphs. Assert indices, validity masks and local-slot masks match.

- [x] **Step 2: Run the tests and observe the expected missing vectorized API failure**

Run `conda run -n research python -m unittest tests.test_encoder.MixedRangeRouteAttentionTest.test_vectorized_route_matches_reference -v`.

- [x] **Step 3: Precompute candidate partitions**

Add a graph helper returning padded direct/remote source indices and valid masks. Compute the helper once in encoder forward and pass it through blocks.

- [x] **Step 4: Replace target-node Python loops with batched partition top-k**

Use masked score tensors, batched top-k, per-node available counts, and vectorized quota fallback. Keep the old selector as a private reference helper only for tests until equivalence is verified.

- [x] **Step 5: Run route tests and pretraining flow tests**

Run `conda run -n research python -m unittest tests.test_encoder tests.test_pretraining_flow -v`.

### Task 4: Add validation interval and diagnostics

**Files:**
- Modify: `stanchor/engine/pretrainer.py`
- Modify: `tests/test_pretraining_flow.py`
- Modify: `tests/test_pretraining_cli.py`

- [x] **Step 1: Write a failing validation interval test**

Run a two-epoch tiny loader with `validation_interval=2`; assert training processes the same batches, validation is called only on epoch 2, and final validation is forced.

- [x] **Step 2: Implement interval-aware validation**

Use `epoch % validation_interval == 0 or epoch == epochs`; record `val_evaluated` and do not update early stopping on skipped epochs.

- [x] **Step 3: Add startup logging**

Log objective, reconstruction/retrieval weights, validation interval, route parameters, and whether the epoch was evaluated.

- [x] **Step 4: Run CLI and flow tests**

Run `conda run -n research python -m unittest tests.test_pretraining_cli tests.test_pretraining_flow -v`.

### Task 5: Verification and smoke handoff

**Files:**
- Modify: `docs/superpowers/plans/2026-08-19-relation-only-vectorized-route.md`

- [x] **Step 1: Run focused and complete tests**

Run `conda run -n research python -m unittest tests.test_encoder tests.test_graph_loading tests.test_pretraining_flow tests.test_pretraining_cli -v` and then `conda run -n research python -m unittest discover -s tests -v`.

- [x] **Step 2: Run compileall**

Run `conda run -n research python -m compileall -q stanchor scripts tests`.

- [x] **Step 3: Run one-batch CUDA relation-only smoke**

Run `conda run -n research python scripts/pretrain.py --config configs/metrla_e5_tgge_latent48_relation_only_v2.yaml --run-name relation_only_smoke --epochs 1 --max-batches 1`, verify no masked reconstruction path is logged, then remove only that smoke directory.

- [x] **Step 4: Update the plan with evidence and handoff command**

## Verification Evidence

- Focused route, pretraining-flow, and relation-only config tests: 21 passed.
- Full test suite: 161 passed; the stale test for the removed
  `metrla_e5_final_sym_profile_local12_v1.yaml` was removed instead of
  restoring that obsolete configuration.
- `conda run -n research python -m compileall -q stanchor scripts tests`: passed.
- CUDA smoke: `relation_only_smoke`, one epoch and one batch; clean-only
  relation loss ran with `val_mask=0`, checkpoints were written, and the
  smoke directory was deleted after verification.
- Handoff command:

  ```powershell
  conda run -n research python scripts/pretrain.py `
    --config configs/metrla_e5_tgge_latent48_relation_only_v2.yaml `
    --run-name metrla_e5_tgge_latent48_relation_only_v2_seed42
  ```

Record test counts, known unrelated failures, measured route speed/parameter counts, and the experiment-machine command.
