# Cross-Dataset Retrieval Validation Status

This is the execution addendum for the cross-dataset retrieval transfer plan.

## Stage 0: completed

The four target datasets passed the basic finite-value, shape, node-count and graph checks.

| Dataset | Series shape | Nodes | Timestamp source | Status |
|---|---:|---:|---|---|
| PEMS-BAY | `[52116, 325, 1]` | 325 | HDF datetime index | Basic audit passed |
| PEMS04 | `[16992, 307, 3]` | 307 | Row-index inference | Basic audit passed; speed channel confirmation pending |
| PEMS07 | `[28224, 883, 1]` | 883 | Row-index inference | Basic audit passed; flow transfer is exploratory |
| PEMS08 | `[17856, 170, 3]` | 170 | Row-index inference | Basic audit passed; speed channel confirmation pending |

Formal artifacts:

- `artifacts/cross_dataset_stage0/summary.json`
- `artifacts/cross_dataset_stage0/pemsbay_audit.json`
- `artifacts/cross_dataset_stage0/pemsbay_graph_audit.json`
- `artifacts/cross_dataset_stage0/pems04_audit.json`
- `artifacts/cross_dataset_stage0/pems07_audit.json`
- `artifacts/cross_dataset_stage0/pems08_audit.json`

PEMS-BAY has one 65-minute timestamp gap caused by daylight-saving-time handling. PEMS04/07/08 have no original timestamps in the NPZ files, so their five-minute timeline is provisional until metadata is confirmed.

## Stage 1: PEMS-BAY completed

The source encoder is frozen. The target-domain Bank uses the PEMS-BAY training history, target scaler and target graph. The protocol is `weekday_radius1_overlap`, with `event_top_r=96` and `node_top_k=12`.

The legal candidate pool has mean `37.86`, median `38`, range `35..39`, and `100%` query coverage. Weekday offsets are `-1: 62011`, `0: 62094`, `+1: 61868`, with no other offsets.

| Retrieval method | MAE | RMSE |
|---|---:|---:|
| Source learned weighted Top-K | 2.0772 | 4.3462 |
| Source learned uniform Top-K | 2.1507 | 4.4804 |
| Random learned weighted Top-K | 2.3196 | 4.5516 |
| Raw-L1 Top-K | 2.1151 | 4.5449 |
| Oracle Top-1 | 1.1539 | 2.6640 |

Formal artifacts:

- `artifacts/cross_dataset_stage1/pemsbay_source_bank`
- `artifacts/cross_dataset_stage1/pemsbay_random_bank`
- `artifacts/cross_dataset_stage1/pemsbay_source_retrieval_val.json`
- `artifacts/cross_dataset_stage1/pemsbay_random_retrieval_val.json`

Source weighted Top-K improves over random by `0.2424` MAE (`10.45%`) and over raw-L1 by `0.0379` MAE (`1.79%`). The oracle gap remains, indicating usable but non-exhaustive candidate quality.

## Decision

PEMS-BAY passes Stage 1 and is eligible for a target-domain downstream comparison. PEMS04/07/08 are not yet Stage-1 retrieval results: the current project loader does not yet consume their NPZ contracts, and the speed-channel metadata is not fully confirmed. They must not be used for downstream Router training until that adapter and metadata check are complete.

The Stage-1 diagnostics are inference-only. Validation/test futures are used only for offline metrics and oracle analysis, never for query keys, candidate selection or Bank updates.
