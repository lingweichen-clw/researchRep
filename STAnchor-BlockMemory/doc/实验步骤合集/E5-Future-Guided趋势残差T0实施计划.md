# E5 Future-Guided Trend-Residual T0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不重新预训练、不修改 Bank schema、不接入 confidence 的前提下，实现并运行 E5 的 offset/trend residual 检索诊断，判断趋势残差是否值得进入 T1 预训练。

**Architecture:** 新增纯张量趋势残差工具，支持 `[B,T,N,C]` query 与 `[B,R,T,N,C]` candidate；新增独立 T0 diagnostic，复用当前 E3 encoder、calendar causal pool、Bank future 和目标数据 scaler。所有 candidate future 先在 candidate-local 坐标残差化，再在 query-local 坐标重构，确保比较的是同一候选集合和同一检索权重。

**Tech Stack:** Python 3.10、PyTorch、NumPy、项目内 `unittest`、现有 STAnchor Bank/diagnostics API。

---

## 0. 实验契约

### 数据与张量

- query forecast context：`x [B,12,N,C]`；
- event candidate context：`contexts [B,R,12,N,C]`；
- event candidate future：`future [B,H,N,R,C]`；
- query future：`y [B,H,N,C]`，只用于离线指标和 oracle；
- learned node selection：`event_ids / weights [B,N,K]`；
- 默认 `trend_length=12`，只使用 query 可见历史末端。

### 因果边界

- 可部署方法只能根据 query history、历史 Bank 和 calendar metadata 排序；
- query future 只允许计算 `future_oracle` 与 rank-correlation 诊断；
- Bank candidate 必须继续满足 `candidate.future_end < query.context_start`；
- validation/test future 不写入 Bank；
- 跨域只迁移 encoder，PEMS-BAY payload 来自 PEMS-BAY training history。

### T0 方法

| 方法 | 排序 | payload | 权重 |
|---|---|---|---|
| `learned_raw_topk` | 当前 E3 node key | raw future | 当前 E3 weights |
| `learned_offset_topk` | 当前 E3 node key | offset residual | 当前 E3 weights |
| `learned_offset_decay_topk` | 当前 E3 node key | horizon-decayed offset residual | 当前 E3 weights |
| `learned_trend_topk` | 当前 E3 node key | trend residual | 当前 E3 weights |
| `fixed_offset_topk` | history offset-residual Pearson | offset residual | Top-K softmax |
| `fixed_trend_topk` | history trend-residual Pearson | trend residual | Top-K softmax |
| `future_oracle_trend_top1` | query future residual，仅诊断 | trend residual | Top-1 |

T0 不修改下游 decoder，也不运行 confidence。

## Task 1: Trend-residual pure tensor contract

**Files:**
- Create: `stanchor/retrieval/trend_residual.py`
- Create: `tests/test_trend_residual.py`

- [x] **Step 1: Write failing tests for exact linear reconstruction**

```python
def test_trend_residual_reconstructs_query_local_future(self):
    candidate_context = torch.tensor([[[[10.0]], [[12.0]], [[14.0]]]])
    candidate_future = torch.tensor([[[[17.0]], [[19.0]]]])
    query_context = torch.tensor([[[[100.0]], [[103.0]], [[106.0]]]])
    candidate_stats = estimate_local_trend(candidate_context, torch.ones_like(candidate_context, dtype=torch.bool), 3)
    query_stats = estimate_local_trend(query_context, torch.ones_like(query_context, dtype=torch.bool), 3)
    residual, valid = residualize_future(candidate_future, torch.ones_like(candidate_future, dtype=torch.bool), candidate_stats)
    reconstructed, reconstructed_valid = reconstruct_future(residual, valid, query_stats)
    self.assertTrue(torch.allclose(reconstructed.flatten(), torch.tensor([110.5, 113.5]), atol=1.0e-5))
    self.assertTrue(bool(reconstructed_valid.all()))
```

- [x] **Step 2: Run the focused test and verify RED**

Run:

```powershell
conda run -n research python -m unittest tests.test_trend_residual -v
```

Expected: import failure because `stanchor.retrieval.trend_residual` does not exist.

- [x] **Step 3: Implement mask-aware statistics and transforms**

Required API:

```python
@dataclass(frozen=True)
class LocalTrendStatistics:
    level: torch.Tensor
    slope: torch.Tensor
    scale: torch.Tensor
    valid: torch.Tensor

def estimate_local_trend(values, observed, trend_length, mode="trend", eps=1.0e-6): ...
def residualize_context(values, observed, statistics, eps=1.0e-6): ...
def residualize_future(future, observed, statistics, eps=1.0e-6): ...
def reconstruct_future(residual, observed, statistics): ...
```

Implementation requirements:

- time axis is always `-3` for both `[B,T,N,C]` and `[B,R,T,N,C]`;
- `mode="offset"` forces slope to zero;
- endpoint missing时使用线性拟合在 context endpoint 的估计值；
- scale 使用有效相邻一阶差分 RMS，无相邻差分时回退到 detrended RMS；
- invalid positions output exact zero；
- input shapes、finite values、positive `trend_length` 均做 defensive validation。

- [x] **Step 4: Add missing-value, affine-invariance and zero-scale tests**

```python
def test_residual_is_invariant_to_positive_affine_transform(self): ...
def test_missing_endpoint_uses_fitted_endpoint_without_nan(self): ...
def test_constant_context_produces_finite_residual(self): ...
```

- [x] **Step 5: Run focused tests and verify GREEN**

Run the same unittest command. Expected: all `test_trend_residual` tests pass.

## Task 2: Candidate scoring and aggregation

**Files:**
- Modify: `stanchor/retrieval/trend_residual.py`
- Modify: `tests/test_trend_residual.py`

- [x] **Step 1: Write failing tests for deployable Pearson ranking**

```python
def test_masked_pearson_prefers_matching_history_shape(self):
    query = torch.tensor([[[[0.0]], [[1.0]], [[0.0]]]])
    candidates = torch.tensor([[[[[0.0]], [[1.0]], [[0.0]]], [[[0.0]], [[-1.0]], [[0.0]]]]])
    scores, valid = masked_pearson_candidate_scores(...)
    self.assertGreater(float(scores[0, 0, 0]), float(scores[0, 0, 1]))
    self.assertTrue(bool(valid.all()))
```

- [x] **Step 2: Verify RED**

Expected: missing scoring API.

- [x] **Step 3: Implement candidate utilities**

Required API:

```python
def masked_pearson_candidate_scores(query, query_observed, candidates, candidate_observed, event_valid): ...
def masked_future_l1_scores(query, query_observed, candidates, candidate_observed, event_valid): ...
def match_selected_event_positions(event_ids, selected_event_ids, selected_valid): ...
def softmax_topk_weights(scores, valid, top_k, temperature, largest=True): ...
def weighted_candidate_mean(candidates, valid, weights): ...
def masked_spearman_rank_correlation(first_scores, second_scores, valid): ...
```

All candidate scores return `[B,N,R]`; invalid Pearson is `-inf` and invalid distance is `+inf`.

- [x] **Step 4: Add tests for causal-pool mapping, masked weighting and rank correlation**

```python
def test_match_selected_event_positions_maps_global_ids_to_pool_axis(self): ...
def test_weighted_candidate_mean_renormalizes_per_horizon_mask(self): ...
def test_spearman_is_positive_for_consistent_rankings_and_negative_when_reversed(self): ...
```

- [x] **Step 5: Run focused tests and verify GREEN**

Expected: all pure tensor and candidate tests pass.

## Task 3: Independent T0 diagnostic and CLI

**Files:**
- Create: `stanchor/diagnostics/trend_residual.py`
- Create: `scripts/diagnose_trend_residual.py`
- Create: `tests/test_trend_residual_diagnostics.py`

- [x] **Step 1: Write a failing diagnostic assembly test**

Test a synthetic candidate pool where raw futures have a level mismatch but trend residual reconstruction matches the query. Assert:

```python
self.assertGreater(raw_mae, trend_mae)
self.assertEqual(result["future_information_boundary"], "oracle_and_diagnostics_only")
```

- [x] **Step 2: Verify RED**

Expected: diagnostic module is missing.

- [x] **Step 3: Implement `diagnose_trend_residual_value`**

Signature:

```python
@torch.no_grad()
def diagnose_trend_residual_value(
    config: ExperimentConfig,
    checkpoint_path: str | Path,
    bank_path: str | Path,
    split: str = "val",
    trend_length: int = 12,
    max_batches: int | None = None,
) -> dict[str, Any]: ...
```

The diagnostic must:

- reuse `build_data_and_graph`, `load_pretrained_model`, `_validate_bank`, `TwoStageRetriever` and exact calendar filtering;
- load event contexts from `series` using Bank `context_end`，不扩展 schema；
- keep current E3 event pool/node selection fixed for learned payload comparisons；
- calculate fixed Pearson methods only from history；
- label future-oracle and rank correlation as non-deployable diagnostics；
- inverse-transform predictions before MAE/RMSE/MAPE；
- report common coverage, method metrics, rank correlation and oracle gaps。

- [x] **Step 4: Add CLI arguments**

```powershell
python scripts/diagnose_trend_residual.py `
  --config configs/metrla_e3_relation_v1.yaml `
  --checkpoint artifacts/metrla_e3_relation_seed42/pretrain_best_relation.pt `
  --bank artifacts/metrla_bank_e3_relation_relation `
  --split val `
  --trend-length 12 `
  --output artifacts/e5_t0_metrla_val.json
```

- [x] **Step 5: Run focused diagnostic tests and compileall**

Expected: tests pass and compileall exits `0`.

## Task 4: Smoke, formal runs and research decision

**Files:**
- Create: `doc/诊断报告合集/E5趋势残差T0诊断报告.md`
- Modify: `doc/优化方案合集/E5-Future-Guided趋势残差检索预训练方案.md`

- [x] **Step 1: Run one-batch METR-LA smoke**

Output to `tmp/e5_t0_metrla_smoke.json`. Verify finite metrics, causal coverage and all six methods, then delete the smoke JSON after validation.

- [x] **Step 2: Run full METR-LA validation T0**

Output: `artifacts/e5_t0_metrla_val.json`.

- [x] **Step 3: Run full PEMS-BAY validation T0**

Output: `artifacts/e5_t0_pemsbay_val.json` using the source E3 checkpoint and `pemsbay_bank_from_metrla_e3_relation`.

- [x] **Step 4: Apply decision rules**

- Keep `offset` if it improves retrieved future MAE by at least 1% over learned raw with equal coverage；
- Keep `trend` only if it improves over both raw and offset and does not rely on oracle；
- Enter T1 only if the retained representation improves both METR-LA and PEMS-BAY；
- Stop E5 trend branch if fixed/learned trend is worse on both datasets；
- If fixed trend is good but learned trend is not yet tested, continue only to FTR pretraining，不加新 backbone。

Applied result: keep `learned_offset_decay_topk` (`+4.72%` METR-LA, `+12.36%` PEMS-BAY); remove local scale transfer; stop learned/fixed linear trend branches. T1 仅实现 horizon-weighted offset future relation。

- [x] **Step 5: Run final verification**

```powershell
conda run -n research python -m unittest discover -s tests -v
conda run -n research python -m compileall -q stanchor scripts tests
```

Expected: all tests pass and compileall exits `0`.

Verified on 2026-08-02: `72` tests passed; `compileall` exited `0`; both formal JSON artifacts passed finite-value, future-boundary, coverage and decision-threshold assertions.

## Execution Note

当前 checkout 位于 `main` 且包含用户已有的未提交实验资产。由于 T0 依赖这些 Bank/config/artifact，本轮在当前目录执行，不创建 worktree、不提交 commit、不修改或回滚现有无关文件。
