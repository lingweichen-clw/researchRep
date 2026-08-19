# Mixed-Range History Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a history-conditioned mixed-range spatial route to the Latent48 encoder with a default budget of 10 selected nodes per target (4 first-order and 6 multi-hop/remote), while preserving the existing static local graph branch and v1 behavior by default.

**Architecture:** The encoder keeps its current static edge-sparse attention for all first-order graph neighbors. An optional route branch summarizes each node's history, computes low-rank directed query/key scores for all non-self nodes, adds a fixed two/three-hop random-walk prior, and selects a quota of first-order and remote nodes before gated value aggregation. The route is disabled for existing configurations and enabled only by the new v2 configuration.

**Tech Stack:** Python 3.10, PyTorch, unittest, YAML configuration, existing `GraphData` and `FactorizedSTEncoder` APIs.

---

### Task 1: Add the graph diffusion and route configuration contract

**Files:**
- Modify: `stanchor/config.py`
- Modify: `stanchor/data/graph.py`
- Test: `tests/test_graph_loading.py`

- [x] **Step 1: Write failing tests for self-loop-free diffusion and route validation**

Add tests that construct a three-node chain with self-loops, verify the diffusion prior has no self-loop contribution and has positive two-hop mass from node 0 to node 2, and verify invalid route quotas are rejected by `ExperimentConfig.validate()`.

- [x] **Step 2: Run the focused tests and confirm the expected failures**

Run:

```powershell
python -m unittest tests.test_graph_loading -v
```

Expected: failures because `GraphData.random_walk_diffusion_prior()` and the route fields do not yet exist.

- [x] **Step 3: Implement the minimal graph helper and configuration fields**

Add `GraphData.random_walk_diffusion_prior()` that removes self-loops, row-normalizes the weighted adjacency, returns `A_rw @ A_rw + A_rw @ A_rw @ A_rw`, and zeros the diagonal. Add these `ModelConfig` fields with defaults that preserve v1:

```python
route_enabled: bool = False
route_dim: int = 16
route_top_k: int = 10
route_local_quota: int = 4
route_prior_weight: float = 0.25
route_temperature: float = 0.1
route_gate_bias: float = -2.0
```

Validate positive dimensions and temperature, `0 <= route_local_quota <= route_top_k`, non-negative prior weight, and negative gate bias.

- [x] **Step 4: Run the focused tests and confirm they pass**

Run the same unittest command and expect PASS.

### Task 2: Implement mixed-range route attention

**Files:**
- Modify: `stanchor/models/encoder.py`
- Modify: `stanchor/models/pretraining.py`
- Test: `tests/test_encoder.py`

- [x] **Step 1: Write failing route behavior tests**

Create tests for:

```python
def test_route_selects_four_first_order_and_six_remote_nodes():
    # Use a graph with at least 5 direct neighbors and at least 8 remote nodes.
    # Assert the selected route indices contain four direct and six non-direct nodes.

def test_route_forward_preserves_shape_and_is_finite():
    # x: [B, P, N, D], assert output has the same shape and finite values.
```

- [x] **Step 2: Run the new tests and confirm they fail for the missing module**

Run:

```powershell
python -m unittest tests.test_encoder -v
```

Expected: import or attribute failures because the route module is not implemented.

- [x] **Step 3: Implement `MixedRangeRouteAttention`**

Add a module with:

```python
forward(x: Tensor, graph: GraphData, diffusion_prior: Tensor | None = None)
```

The implementation must:

- accept `x: [B, P, N, D]` and validate node/hidden dimensions;
- build `summary = cat(mean(x, dim=1), x[:, -1] - x[:, 0])` with shape `[B, N, 2D]`;
- project query/key to `route_dim` and score `[B, N, N]`;
- mask self nodes, add `route_prior_weight * log1p(diffusion_prior)`;
- select up to `route_local_quota` first-order nodes and the remaining `route_top_k - route_local_quota` remote nodes, filling scarce partitions from the other partition and masking only truly unavailable slots;
- aggregate values from all patches using the selected indices;
- apply a negative-bias sigmoid gate from the target and aggregated route states;
- return `[B, P, N, D]` and keep all route tensors finite.

Use explicit comments for `[B, P, N, D]`, `[B, N, N]`, and gathered `[B, P, N, K, D]` tensors. Do not add node ID embeddings or future inputs.

- [x] **Step 4: Integrate the route into `FactorizedSTBlock` and `FactorizedSTEncoder`**

Add route constructor arguments, compute the diffusion prior once per encoder forward, and combine:

```python
hidden = hidden + local_attention + route_gate * route_output
```

Leave `route_enabled=False` behavior identical to the current encoder path. Pass the new fields from `STAnchorPretrainModel`.

- [x] **Step 5: Run route tests and the existing encoder/pretraining tests**

Run:

```powershell
python -m unittest tests.test_encoder tests.test_pretraining_flow tests.test_pretraining_cli -v
```

Expected: PASS with no shape, finite-gradient, or checkpoint contract failures.

### Task 3: Add the v2 pretraining configuration and logging contract

**Files:**
- Create: `configs/metrla_e5_tgge_latent48_v2.yaml`
- Modify: `scripts/pretrain.py`
- Test: `tests/test_pretraining_cli.py`

- [x] **Step 1: Write a failing config/CLI test**

Assert the v2 YAML loads with `hidden_dim=80`, `route_enabled=true`, `route_top_k=10`, and `route_local_quota=4`, and that the pretraining startup log includes route parameters and the route parameter count.

- [x] **Step 2: Run the focused test and confirm it fails**

Run:

```powershell
python -m unittest tests.test_pretraining_cli -v
```

- [x] **Step 3: Add the v2 YAML and startup diagnostics**

Copy the current Global288 Latent48 protocol into the versioned v2 config, use the planned hidden width, and set the route fields above. Extend startup logging with `route_enabled`, `route_top_k`, `route_local_quota`, and `route_parameters`.

- [x] **Step 4: Run the CLI/config tests and a one-batch CUDA smoke test**

Run:

```powershell
python -m unittest tests.test_pretraining_cli tests.test_pretraining_flow -v
python scripts/pretrain.py --config configs/metrla_e5_tgge_latent48_v2.yaml --run-name mixed_route_smoke --epochs 1 --max-batches 1
```

Expected: the smoke run completes, prints the route contract, writes a checkpoint, and does not alter existing v1 artifacts.

### Task 4: Verification and handoff

**Files:**
- Modify: `doc/优化方案合集/E5-Latent48-TGGE-Structured-Error-Corrector-v2优化方案.md`
- Modify: `doc/优化方案合集/E5-Final-可泛化未来语义检索与风险感知校准统一优化方案.md`

- [x] **Step 1: Run the complete unit test suite (155 passed; one pre-existing missing profile config)**

Run:

```powershell
python -m unittest discover -s tests -v
python -m compileall -q stanchor scripts tests
```

- [x] **Step 2: Run a real one-batch CUDA forward/backward if CUDA is available**

Use the v2 configuration without writing a formal experiment directory; record shape, finite gradients, parameter count, and peak memory.

- [x] **Step 3: Update the two design documents**

Record the final `Top-10 = 4 first-order + 6 multi-hop/remote` rule, the fallback behavior for low-degree nodes, and the actual parameter count if it differs from the estimate.

- [x] **Step 4: Remove only smoke artifacts after validation**

Delete the newly created `mixed_route_smoke` directory after its logs and checkpoint contract have been verified; do not touch existing v1 or formal experiment artifacts.
