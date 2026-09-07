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

## Stage 1: PEMS04 and PEMS08 completed

The earlier paragraph saying that PEMS04/07/08 were not yet supported is a
historical pre-adapter note; the results below supersede that status.

The NPZ transfer loader now selects one physical channel and creates the same
`[T,N,1]` contract as the HDF loader. PEMS04/PEMS08 use channel index 2,
which is the speed channel in the supplied files. Their CSV distance graphs
are converted to positive inverse-distance edge weights and receive explicit
self-loops before entering `GraphData`.

| Dataset / method | MAE | RMSE | Coverage |
|---|---:|---:|---:|
| PEMS04 source learned Top-12 | 2.2305 | 4.6998 | 100% |
| PEMS04 random learned Top-12 | 2.4286 | 4.9331 | 100% |
| PEMS04 raw-L1 Top-12 | 2.9211 | 5.8598 | 100% |
| PEMS04 oracle Top-1 | 1.4091 | 3.2153 | 100% |
| PEMS08 source learned Top-12 | 1.9257 | 4.6045 | 100% |
| PEMS08 random learned Top-12 | 2.0084 | 4.7236 | 100% |
| PEMS08 raw-L1 Top-12 | 2.3165 | 5.2283 | 100% |
| PEMS08 oracle Top-1 | 1.2167 | 3.3848 | 100% |

Both same-variable datasets pass the source-vs-random and source-vs-raw-L1
retrieval checks. PEMS04 has a source-vs-random gain of 0.1981 MAE (8.16%),
and PEMS08 has a gain of 0.0826 MAE (4.11%). The remaining oracle gaps show
that the candidate pool still contains useful future matches. Their legal
calendar pools are approximately 12.07 and 12.40 events per query, with
100% query coverage and contributions from offsets -1, 0 and +1.

Formal artifacts:

- `artifacts/cross_dataset_stage1/pems04_source_bank`
- `artifacts/cross_dataset_stage1/pems04_random_bank`
- `artifacts/cross_dataset_stage1/pems04_source_retrieval_val.json`
- `artifacts/cross_dataset_stage1/pems04_random_retrieval_val.json`
- `artifacts/cross_dataset_stage1/pems08_source_bank`
- `artifacts/cross_dataset_stage1/pems08_random_bank`
- `artifacts/cross_dataset_stage1/pems08_source_retrieval_val.json`
- `artifacts/cross_dataset_stage1/pems08_random_retrieval_val.json`

## Stage 1: PEMS07 pending

PEMS07 is a single-channel flow dataset and is therefore a separate
speed-to-flow transfer result. Its 883-node graph makes the current four-layer
source encoder Bank build substantially more expensive than PEMS04/08. The
local run reached array writes but timed out before writing a valid Bank
manifest after 3600 seconds; no PEMS07 metric is reported and the partial
directory is not a formal artifact. Run PEMS07 only on the 16-GB experiment
machine, and keep it separate from the same-variable fine-tuning decision.

## Current decision

PEMS-BAY, PEMS04 and PEMS08 pass the source retrieval screen and are eligible
for matched target-domain downstream comparisons. PEMS04 and PEMS08 are the
first candidates for the T0/T1/T2/T3 fine-tuning screen. PEMS07 remains
pending and is not part of the same-variable fine-tuning track until its
source retrieval result is available.
