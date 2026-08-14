# E5 Latent48 Context-Conditioned FGDA Implementation Plan

> Execute this plan inline with test-first checkpoints. Do not run expensive pretraining until unit, integration, parameter, and checkpoint checks pass.

Goal: Implement the approved Context-Conditioned FGDA (CC-FGDA) adapter and produce an isolated pure Latent48 versus CC-FGDA experiment pair.

Architecture: Keep the pure 48-dimensional latent key, OffsetDecay relation teacher, symmetric distance normalization, static graph, masking protocol, and downstream interfaces. Add a context_conditioned dynamics adapter mode with a 48-dimensional conditioned low-rank residual branch, a 96-dimensional diagonal information-preserving skip, and an 8-group fusion gate.

Tech Stack: Python 3.10+, PyTorch 2.1+, existing STAnchorPretrainModel, YAML configs, unittest, and PowerShell queue scripts.

---

## Task 1: Freeze the implementation contract

Files:
- Modify: STAnchor-BlockMemory/stanchor/config.py
- Test: STAnchor-BlockMemory/tests/test_dynamics_adapter.py

- [x] Step 1: Write the failing configuration test.

Construct ModelConfig(dynamics_adapter_mode="context_conditioned", dynamics_bottleneck_dim=48, dynamics_gate_groups=8) and assert validation succeeds. Assert an unknown mode, zero groups, or a hidden dimension not divisible by the group count raises ValueError.

- [x] Step 2: Run RED.

Command:
python -m unittest tests.test_dynamics_adapter.DynamicsAdapterIntegrationTest.test_config_validates_context_conditioned_mode -v

Expected failure: context_conditioned and dynamics_gate_groups are not accepted.

- [x] Step 3: Implement the smallest configuration change.

Add dynamics_gate_groups: int = 8 to ModelConfig, accept context_conditioned, require dynamics_gate_groups > 0, and require hidden_dim % dynamics_gate_groups == 0. Preserve none, local, and local_graph behavior.

- [x] Step 4: Run GREEN.

Command:
python -m unittest tests.test_dynamics_adapter.DynamicsAdapterIntegrationTest.test_config_validates_context_conditioned_mode -v

## Task 2: Specify the CC-FGDA output with failing tests

Files:
- Modify: STAnchor-BlockMemory/tests/test_dynamics_adapter.py
- Modify: STAnchor-BlockMemory/stanchor/models/dynamics_adapter.py

- [x] Step 1: Write shape, identity, and condition-use tests.

Use hidden_dim=8, bottleneck_dim=4, and dynamics_gate_groups=2. Assert output.hidden is [1,2,3,8], output.modulation is [1,2,3,4], output.low_rank_residual is [1,2,3,8], output.direct_residual is [1,2,3,8], and output.fusion_gate is [1,2,3,2]. With zero-initialized output and direct paths, assert output.hidden equals hidden. After manually setting residual_up and direct_scale nonzero, change hidden while keeping history fixed and assert the modulation or final residual changes.

- [x] Step 2: Run RED.

Command:
python -m unittest tests.test_dynamics_adapter.HistoryDynamicsAdapterTest.test_context_conditioned_shapes_and_identity -v

Expected failure: the current output has no context-conditioned fields.

- [x] Step 3: Extend the output contract.

Add modulation, low_rank_residual, direct_residual, and grouped fusion_gate diagnostic fields. Keep residual as the final 96-dimensional residual and keep old local/local_graph shapes unchanged.

- [x] Step 4: Implement the context-conditioned branch.

For context_conditioned, implement this tensor flow without materializing an effective [B,P,N,96,96] matrix:

u = gelu(residual_down(F))
c = concat(layer_norm_z(hidden), layer_norm_f(F))
m = tanh(modulation_projection(c))
u_mod = u * (1.0 + m)
low_rank = residual_up(u_mod)
direct = direct_scale * F
residual = low_rank + direct
gate_input = concat(layer_norm_z(hidden), layer_norm_r(residual))
gate = sigmoid(fusion_gate_projection(gate_input))
adapted = hidden + expand_groups(gate) * residual

Use dynamics_gate_groups groups. Initialize residual_up, direct_scale, modulation projection, and fusion-gate weights so the initial adapter is an exact identity. Keep the existing local/local_graph branches unchanged.

- [x] Step 5: Run the adapter suite.

Command:
python -m unittest tests.test_dynamics_adapter.HistoryDynamicsAdapterTest -v

Expected result: old adapter tests and new context-conditioned tests pass.

## Task 3: Add interpretable diagnostics

Files:
- Modify: STAnchor-BlockMemory/stanchor/models/dynamics_adapter.py
- Modify: STAnchor-BlockMemory/stanchor/engine/pretrainer.py
- Test: STAnchor-BlockMemory/tests/test_dynamics_adapter.py

- [x] Step 1: Write the failing diagnostic test.

Assert summarize_adapter_output returns finite values for modulation_abs_mean, modulation_token_std, group_gate_mean, group_gate_std, low_rank_contribution_ratio, direct_contribution_ratio, and total_contribution_ratio.

- [x] Step 2: Run RED.

Command:
python -m unittest tests.test_dynamics_adapter.HistoryDynamicsAdapterTest.test_context_conditioned_diagnostics -v

- [x] Step 3: Implement no-grad summaries.

Compute summaries only over adapter_valid positions and return finite zeros when no valid position exists. Preserve old log keys fusion_gate_mean, spatial_gate_mean, and contribution_ratio.

- [x] Step 4: Extend PretrainEpochResult.

Aggregate the new scalar diagnostics over adapter batches and append them to epoch logs. Do not add a new loss term; OffsetDecay relation remains the only future-derived relation supervision.

- [x] Step 5: Run the integration diagnostic test.

Command:
python -m unittest tests.test_dynamics_adapter.DynamicsAdapterIntegrationTest.test_pretrain_epoch_reports_context_diagnostics -v

## Task 4: Integrate model state and an isolated production config

Files:
- Modify: STAnchor-BlockMemory/stanchor/models/pretraining.py only if strict-state loading needs adjustment
- Create: STAnchor-BlockMemory/configs/metrla_e5_final_latent48_cc_fgda_global288_v1.yaml
- Test: STAnchor-BlockMemory/tests/test_dynamics_adapter.py

- [x] Step 1: Write the checkpoint and parameter-budget test.

Assert retrieval_state_dict includes dynamics_adapter.* for CC-FGDA and excludes it for pure Latent48. Assert Global288 CC-FGDA increases parameters by 5%-10% relative to the no-adapter model.

- [x] Step 2: Run RED.

Command:
python -m unittest tests.test_dynamics_adapter.DynamicsAdapterIntegrationTest.test_context_conditioned_checkpoint_and_parameter_budget -v

- [x] Step 3: Preserve the existing model path.

Reuse the current clean/masked adapter calls. Ensure each branch derives dynamics from its own visible history and that retrieval checkpoints include the adapter. Do not modify the 48-dimensional key or teacher inputs.

- [x] Step 4: Create the YAML.

Copy the pure Latent48/SymNorm Global288 protocol and change only:
dynamics_adapter_mode: context_conditioned
dynamics_bottleneck_dim: 48
dynamics_gate_groups: 8
dynamics_gate_bias: -2.0

Use isolated names such as metrla_e5_final_latent48_cc_fgda_global288_seed42 for the run and Bank. Keep retrieval_dim: 48, profile_dim: 0, latent_dim: 0, profile_loss_weight: 0.0, relation_teacher_mode: offset_decay, and relation_distance_normalization: symmetric_geometric_mean.

- [x] Step 5: Run configuration tests.

Command:
python -m unittest tests.test_dynamics_adapter.DynamicsAdapterIntegrationTest.test_context_conditioned_checkpoint_and_parameter_budget tests.test_dynamics_adapter.DynamicsAdapterIntegrationTest.test_global288_cc_fgda_config_is_single_variable -v

## Task 5: Regression and one-batch verification

Files:
- Test: STAnchor-BlockMemory/tests/

- [x] Step 1: Run the adapter suite.

Command:
python -m unittest tests.test_dynamics_adapter -v

- [x] Step 2: Run all tests.

Command:
python -m unittest discover -s tests -p test_*.py -v

Expected result: old none, local, and local_graph contracts remain green.

- [x] Step 3: Run syntax and one-batch checks.

Commands:
python -m compileall -q stanchor scripts
python scripts/pretrain.py --config configs/metrla_e5_final_latent48_cc_fgda_global288_v1.yaml --max-batches 1

Verify finite losses, finite gradients, checkpoint creation, and new diagnostics. Smoke artifacts must not be used as formal evidence and should be removed after verification.

## Task 6: Formal experiment queue and handoff

Files:
- Create: STAnchor-BlockMemory/scripts/run_e5_latent48_cc_fgda_global288_queue.ps1
- Modify: this plan with final output paths and commands

- [x] Step 1: Add an isolated PowerShell queue.

Run in order: CC-FGDA pretraining, pure Latent48 pretraining or checkpoint verification, Bank construction, retrieval diagnostic, visualization, and downstream attribution. Each formal step writes .started and .completed markers under a unique log root and refuses to overwrite existing output directories.

- [x] Step 2: Validate queue syntax without training.

Command:
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "& { [scriptblock]::Create((Get-Content -Raw 'scripts/run_e5_latent48_cc_fgda_global288_queue.ps1')) | Out-Null; 'syntax-ok' }"

- [x] Step 3: Provide the experiment-machine command only after local verification.

Command:
$python = (Get-Command python).Source
$proc = Start-Process powershell.exe -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',(Resolve-Path 'scripts/run_e5_latent48_cc_fgda_global288_queue.ps1').Path,'-Python',$python) -WorkingDirectory (Get-Location).Path -WindowStyle Hidden -PassThru
$proc.Id

The handoff must include exact config, checkpoint, Bank, and log paths and must state that query future is never used by the encoder or retrieval stage.

## Verification Result

- Stage: implementation and local verification completed
- Production changes: CC-FGDA mode, 48-dimensional conditioned branch, 96-dimensional diagonal skip, 8-group gate, diagnostics, isolated YAML, and two-stage queue
- Full regression: 177 tests passed in the research environment
- Static verification: python -m compileall -q stanchor scripts tests passed
- Queue verification: PowerShell parser returned syntax-ok
- One-batch verification: CUDA forward/backward completed; total parameters 414,021, adapter parameters 22,185, adapter increase about 5.66%
- Future boundary: query future is not accepted by the adapter, key encoder, Bank construction, or candidate ranking

## Formal Experiment Commands

Run from the repository root in the environment that contains PyTorch.

1. Start both pure Latent48 and CC-FGDA Global288 pretraining:

~~~powershell
$python = (Get-Command python).Source
$proc = Start-Process powershell.exe -ArgumentList @(
  '-NoProfile',
  '-ExecutionPolicy', 'Bypass',
  '-File', (Resolve-Path 'scripts/run_e5_latent48_cc_fgda_global288_queue.ps1').Path,
  '-Python', $python,
  '-Stage', 'pretrain'
) -WorkingDirectory (Get-Location).Path -WindowStyle Hidden -PassThru
$proc.Id
~~~

This trains:

- configs/metrla_e5_final_latent48_global288_v1.yaml: pure 48-dimensional latent key, no FGDA;
- configs/metrla_e5_final_latent48_cc_fgda_global288_v1.yaml: pure 48-dimensional latent key plus CC-FGDA.

Both configurations use the same OffsetDecay teacher, symmetric geometric-mean distance normalization, exact-calendar candidate protocol, and the same static graph.

2. After both relation checkpoints exist, run Bank, retrieval diagnosis, visualization, and downstream attribution:

~~~powershell
$python = (Get-Command python).Source
$proc = Start-Process powershell.exe -ArgumentList @(
  '-NoProfile',
  '-ExecutionPolicy', 'Bypass',
  '-File', (Resolve-Path 'scripts/run_e5_latent48_cc_fgda_global288_queue.ps1').Path,
  '-Python', $python,
  '-Stage', 'posttrain'
) -WorkingDirectory (Get-Location).Path -WindowStyle Hidden -PassThru
$proc.Id
~~~

The post-training stage compares base_only, pretrained_offset_decay, and random_offset_decay for each encoder version, then exports retrieval diagnostics and figures. Formal output roots are protected against accidental overwrite.
