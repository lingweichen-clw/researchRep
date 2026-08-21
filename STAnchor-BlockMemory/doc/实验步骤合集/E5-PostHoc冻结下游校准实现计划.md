# E5 PostHoc Frozen-Base Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Load an existing base-only downstream checkpoint, freeze its backbone, train only the ErrorAware calibrator, and compare the existing 32/8 capacity with the approved 64/16 capacity.

**Architecture:** Add a versioned `posthoc_frozen_base` target training protocol and a required `--base-checkpoint` input. The protocol copies only `backbone.*` tensors from a verified base-only checkpoint, records a deterministic fingerprint, trains only the risk head and additive fusion, and refuses to finish if the frozen backbone changes.

**Tech Stack:** Python 3.10, PyTorch, dataclass/YAML configuration, `unittest`, PowerShell background queues.

---

### Task 1: Define the configuration and checkpoint contract

**Files:**
- Modify: `stanchor/config.py`
- Modify: `scripts/train_downstream.py`
- Test: `tests/test_error_aware_fusion.py`

- [x] Add a failing config test that accepts only `staged_joint` and `posthoc_frozen_base`.
- [x] Add a failing CLI/engine contract test showing that PostHoc training requires a base checkpoint.
- [x] Add `training_protocol: str = "staged_joint"` to `TargetConfig` and validate the two values.
- [x] Add optional `--base-checkpoint` and pass it to `train_downstream`.

### Task 2: Load and verify a frozen base-only backbone

**Files:**
- Modify: `stanchor/engine/target.py`
- Test: `tests/test_error_aware_fusion.py`

- [x] Add a failing test with a real temporary checkpoint containing `downstream_mode=base_only` and `downstream_state_dict`.
- [x] Add a failing test rejecting non-base checkpoints and mismatched backbone shapes.
- [x] Implement `load_frozen_base_backbone` using `load_checkpoint`, an exact `backbone.` prefix filter, and strict `load_state_dict`.
- [x] Return provenance containing the resolved checkpoint path and loaded backbone fingerprint.

### Task 3: Add the PostHoc calibrator-only stage

**Files:**
- Modify: `stanchor/engine/target.py`
- Test: `tests/test_error_aware_fusion.py`

- [x] Add a failing stage test asserting that PostHoc trains only `StructuredErrorCorrector`.
- [x] Select only one `posthoc_calibrator` stage for `posthoc_frozen_base`; retain existing stage behavior for all old configs.
- [x] Require `learned_topk_error_aware`, reject base warm-up and calibrator warm-up in PostHoc mode, and use `target.epochs` as the calibrator budget.
- [x] Save `training_protocol`, base checkpoint provenance and base fingerprint in the downstream checkpoint.
- [x] Compare the backbone fingerprint before and after every epoch and fail if it changes.

### Task 4: Add the current Structured Error Corrector configs

**Files:**
- Create: versioned STGCN and GraphWaveNet error-aware configs

- [x] Base both configs on the pure Latent48 Global288 contract with `profile_dim=0`, `latent_dim=0`, `level_weight=0`, and `training_protocol=posthoc_frozen_base`.
- [x] Set the current Structured Error Corrector to `risk_hidden_dim=256`, `fusion_feature_hidden_dim=128` (224,142 parameters).
- [x] Make the queue require the existing pure Latent48 relation checkpoint, local Bank and controlled-init base-only checkpoint.
- [x] Refuse to overwrite output/log roots and run train, validation evaluation and branch diagnostics only for the current version.

### Task 5: Verify and launch

**Files:**
- Verify all modified and created files.

- [x] Run the new tests and confirm they fail before production implementation.
- [x] Run the new tests again after implementation and confirm they pass.
- [x] Run the full unittest suite.
- [x] Run `python -m compileall -q stanchor scripts tests`.
- [x] Parse the PowerShell queue as a script block.
- [x] Run a one-batch smoke queue in isolated smoke directories, then remove only the explicit smoke outputs.
- [x] Start the formal queue through hidden `powershell.exe`, record its PID and verify the first `.started` marker and stderr log.
