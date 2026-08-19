# Retrieval Acceleration and Graph WaveNet Integration Plan

> **For agentic workers:** This plan is executed inline in the current session with test-first checkpoints.

**Goal:** Accelerate the existing downstream retrieval path without changing candidate semantics or numerical outputs, and audit the supplied Graph WaveNet implementation for a compatible downstream adapter.

**Architecture:** Keep the public retrieval and OffsetDecay contracts unchanged. Add vectorized calendar candidate construction and target-node-only Bank reads behind the existing APIs, with fallback-equivalence tests against the current reference path. Treat Graph WaveNet as a separate adapter task using the official model's `[B,C,N,T]` convention converted at the project boundary.

**Tech Stack:** Python, PyTorch, NumPy memmap, pytest, YAML experiment configs.

---

### Task 1: Lock down retrieval equivalence

**Files:**
- Create: `tests/test_retrieval_acceleration.py`
- Modify: `stanchor/bank/storage.py`
- Modify: `stanchor/retrieval/retriever.py`
- Modify: `stanchor/retrieval/strategies.py`

- [ ] Write tests that compare reference and accelerated calendar IDs/masks and target-node OffsetDecay outputs on a synthetic Bank.
- [ ] Run the focused tests and observe failure because the accelerated entry points do not yet exist.
- [ ] Implement the smallest vectorized/cached paths while retaining the reference behavior for edge cases.
- [ ] Run the focused tests and the existing retrieval test suite.

### Task 2: Profile the real downstream path

**Files:**
- Modify: `stanchor/engine/target.py` only if instrumentation is needed.

- [ ] Add no persistent behavior changes; use a one-batch timing command to measure encoder, candidate search, aggregation, and STGCN separately.
- [ ] Confirm GPU peak memory and output equality on the same batch.

### Task 3: Audit Graph WaveNet source

**Files:**
- Read: `Graph-WaveNet/model.py`
- Read: `Graph-WaveNet/engine.py`
- Read: `Graph-WaveNet/train.py`
- Create: `docs/diagnostics/graphwavenet-adapter-audit.md`

- [ ] Record official tensor layouts, receptive field, adaptive adjacency, loss, and default hyperparameters.
- [ ] Map the adapter boundary to project tensors `[B,T,N,C] -> [B,C,N,T] -> [B,H,N,C]`.
- [ ] List the minimum files and config fields required before implementing the adapter.

### Task 4: Final verification

- [ ] Run focused retrieval tests.
- [ ] Run existing downstream/retrieval tests that do not require a full GPU training run.
- [ ] Report measured equivalence and any remaining Graph WaveNet implementation work without claiming it is integrated until a forward/shape test passes.
