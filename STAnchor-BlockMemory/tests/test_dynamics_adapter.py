from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from stanchor.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    PretrainConfig,
    load_config,
)
from stanchor.data.graph import graph_from_dense
from stanchor.engine.pretrainer import run_pretrain_epoch
from stanchor.losses.pretraining import compute_pretraining_loss
from stanchor.models.dynamics_adapter import HistoryDynamicsAdapter
from stanchor.models.pretraining import STAnchorPretrainModel
from stanchor.utils import count_parameters


class HistoryDynamicsAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)
        adjacency = np.asarray(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        self.graph = graph_from_dense(adjacency)
        self.values = torch.tensor(
            [
                [
                    [[0.0], [0.0], [0.0]],
                    [[1.0], [3.0], [5.0]],
                    [[2.0], [6.0], [10.0]],
                    [[3.0], [9.0], [15.0]],
                ]
            ]
        )
        self.observed = torch.ones_like(self.values, dtype=torch.bool)
        self.hidden = torch.randn(1, 2, 3, 8)

    def _adapter(self, mode: str) -> HistoryDynamicsAdapter:
        return HistoryDynamicsAdapter(
            input_channels=1,
            patch_size=2,
            hidden_dim=8,
            bottleneck_dim=2,
            mode=mode,
            gate_bias=-2.0,
        )

    def _context_adapter(self) -> HistoryDynamicsAdapter:
        return HistoryDynamicsAdapter(
            input_channels=1,
            patch_size=2,
            hidden_dim=8,
            bottleneck_dim=4,
            mode="context_conditioned",
            gate_bias=-2.0,
            gate_groups=2,
        )

    def test_local_adapter_has_expected_shapes_and_identity_initialization(self) -> None:
        adapter = self._adapter("local")

        output = adapter(self.hidden, self.values, self.observed, self.graph)

        self.assertEqual(tuple(output.hidden.shape), (1, 2, 3, 8))
        self.assertEqual(tuple(output.local_dynamics.shape), (1, 2, 3, 8))
        self.assertEqual(tuple(output.local_valid.shape), (1, 2, 3))
        self.assertEqual(tuple(output.fusion_gate.shape), (1, 2, 3, 1))
        self.assertIsNone(output.graph_dynamics)
        self.assertIsNone(output.spatial_gate)
        self.assertTrue(torch.equal(output.hidden, self.hidden))
        self.assertTrue(torch.equal(output.residual, torch.zeros_like(output.residual)))

    def test_masked_values_cannot_change_local_dynamics(self) -> None:
        adapter = self._adapter("local")
        observed = self.observed.clone()
        observed[:, 1:3, 0, :] = False
        changed = self.values.clone()
        changed[:, 1:3, 0, :] = 10000.0

        original = adapter(self.hidden, self.values, observed, self.graph)
        perturbed = adapter(self.hidden, changed, observed, self.graph)

        self.assertTrue(torch.equal(original.local_valid, perturbed.local_valid))
        self.assertTrue(torch.equal(original.local_dynamics, perturbed.local_dynamics))
        self.assertTrue(torch.equal(original.hidden, perturbed.hidden))

    def test_fully_invalid_patch_forces_zero_residual_after_training(self) -> None:
        adapter = self._adapter("local")
        with torch.no_grad():
            adapter.residual_up.weight.fill_(0.5)
            adapter.residual_up.bias.fill_(1.0)
        observed = self.observed.clone()
        observed[:, :2, :, :] = False

        output = adapter(self.hidden, self.values, observed, self.graph)

        self.assertFalse(bool(output.adapter_valid[:, 0].any()))
        self.assertTrue(torch.equal(output.residual[:, 0], torch.zeros_like(output.residual[:, 0])))
        self.assertTrue(torch.equal(output.hidden[:, 0], self.hidden[:, 0]))

    def test_graph_aggregation_excludes_self_and_renormalizes_valid_neighbors(self) -> None:
        adapter = self._adapter("local_graph")
        observed = self.observed.clone()
        observed[:, :, 2, :] = False

        output = adapter(self.hidden, self.values, observed, self.graph)

        self.assertIsNotNone(output.graph_dynamics)
        self.assertIsNotNone(output.spatial_gate)
        assert output.graph_dynamics is not None
        # Node 0 has neighbors 1 and 2. Node 2 is invalid, so the normalized
        # graph aggregate must equal node 1's projected local dynamics.
        self.assertTrue(
            torch.allclose(
                output.graph_dynamics[:, :, 0],
                output.local_dynamics[:, :, 1],
                atol=1.0e-6,
            )
        )
        self.assertTrue(bool(output.graph_valid[:, :, 0].all()))

    def test_invalid_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "mode"):
            self._adapter("dynamic_attention")

    def test_context_conditioned_shapes_and_identity(self) -> None:
        adapter = self._context_adapter()

        output = adapter(self.hidden, self.values, self.observed, self.graph)

        self.assertEqual(tuple(output.hidden.shape), (1, 2, 3, 8))
        self.assertEqual(tuple(output.modulation.shape), (1, 2, 3, 4))
        self.assertEqual(tuple(output.low_rank_residual.shape), (1, 2, 3, 8))
        self.assertEqual(tuple(output.direct_residual.shape), (1, 2, 3, 8))
        self.assertEqual(tuple(output.fusion_gate.shape), (1, 2, 3, 2))
        self.assertTrue(torch.equal(output.hidden, self.hidden))
        self.assertTrue(torch.equal(output.residual, torch.zeros_like(output.residual)))

    def test_context_conditioned_modulation_depends_on_hidden_context(self) -> None:
        adapter = self._context_adapter()
        with torch.no_grad():
            adapter.modulation_projection.weight.normal_()
            adapter.modulation_projection.bias.normal_()
            adapter.residual_up.weight.normal_()
            adapter.residual_up.bias.normal_()

        original = adapter(self.hidden, self.values, self.observed, self.graph)
        changed_hidden = self.hidden + 3.0
        changed = adapter(changed_hidden, self.values, self.observed, self.graph)

        self.assertFalse(torch.equal(original.modulation, changed.modulation))
        self.assertFalse(torch.equal(original.low_rank_residual, changed.low_rank_residual))

    def test_context_conditioned_diagnostics(self) -> None:
        from stanchor.models.dynamics_adapter import summarize_adapter_output

        adapter = self._context_adapter()
        output = adapter(self.hidden, self.values, self.observed, self.graph)

        metrics = summarize_adapter_output(output)

        for name in (
            "modulation_abs_mean",
            "modulation_token_std",
            "group_gate_mean",
            "group_gate_std",
            "low_rank_contribution_ratio",
            "direct_contribution_ratio",
            "total_contribution_ratio",
        ):
            self.assertIn(name, metrics)
            self.assertTrue(np.isfinite(metrics[name]))

    def test_adapter_diagnostics_quantify_gate_and_hidden_contribution(self) -> None:
        from stanchor.models.dynamics_adapter import summarize_adapter_output

        adapter = self._adapter("local_graph")
        output = adapter(self.hidden, self.values, self.observed, self.graph)

        metrics = summarize_adapter_output(output)

        self.assertGreater(metrics["valid_fraction"], 0.0)
        self.assertAlmostEqual(
            metrics["fusion_gate_mean"],
            float(torch.sigmoid(torch.tensor(-2.0))),
            places=6,
        )
        self.assertAlmostEqual(
            metrics["spatial_gate_mean"],
            float(torch.sigmoid(torch.tensor(-2.0))),
            places=6,
        )
        self.assertEqual(metrics["contribution_ratio"], 0.0)


class DynamicsAdapterIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(29)
        self.batch = 6
        self.time = 12
        self.nodes = 4
        adjacency = np.ones((self.nodes, self.nodes), dtype=np.float32)
        self.graph = graph_from_dense(adjacency)
        self.neighbors = self.graph.dense_neighbors(include_self=False)
        base = torch.randn(self.batch, self.time + 12, self.nodes, 1)
        self.x = base[:, : self.time]
        self.future = base[:, self.time :]
        self.observed = torch.ones_like(self.x, dtype=torch.bool)
        self.weekday = torch.zeros((self.batch, self.time), dtype=torch.long)
        self.slot = torch.arange(self.time).view(1, -1).expand(self.batch, -1)

    @staticmethod
    def _model_config(mode: str) -> ModelConfig:
        return ModelConfig(
            patch_size=3,
            hidden_dim=16,
            retrieval_dim=8,
            num_heads=4,
            encoder_layers=1,
            dropout=0.0,
            dynamics_adapter_mode=mode,
            dynamics_bottleneck_dim=4,
            dynamics_gate_bias=-2.0,
        )

    def _model(self, mode: str) -> STAnchorPretrainModel:
        return STAnchorPretrainModel(
            self._model_config(mode),
            PretrainConfig(
                time_mask_ratio=0.25,
                time_mask_block_size=3,
                space_mask_ratio=0.25,
            ),
            context_length=self.time,
            slots_per_day=288,
        )

    def test_disabled_adapter_preserves_exact_previous_model_outputs(self) -> None:
        torch.manual_seed(101)
        baseline = self._model("none")
        torch.manual_seed(101)
        local = self._model("local")
        local.load_state_dict(
            {
                name: value
                for name, value in baseline.state_dict().items()
                if not name.startswith("dynamics_adapter.")
            },
            strict=False,
        )
        baseline.eval()
        local.eval()

        baseline_output = baseline.encode_clean(
            self.x, self.observed, self.weekday, self.slot, self.graph
        )
        local_output = local.encode_clean(
            self.x, self.observed, self.weekday, self.slot, self.graph
        )

        self.assertIsNone(baseline_output.dynamics)
        self.assertIsNotNone(local_output.dynamics)
        self.assertTrue(
            torch.equal(
                baseline_output.retrieval.node_keys,
                local_output.retrieval.node_keys,
            )
        )

    def test_masked_history_values_cannot_change_masked_branch(self) -> None:
        model = self._model("local_graph").eval()
        first = model.forward_pretrain(
            self.x,
            self.observed,
            self.weekday,
            self.slot,
            self.graph,
            self.neighbors,
            mask_task="time",
            generator=torch.Generator().manual_seed(7),
        )
        changed = self.x.clone()
        changed[first.mask.value_mask] = 10000.0
        second = model.forward_pretrain(
            changed,
            self.observed,
            self.weekday,
            self.slot,
            self.graph,
            self.neighbors,
            mask_task="time",
            generator=torch.Generator().manual_seed(7),
        )

        self.assertTrue(torch.equal(first.mask.patch_mask, second.mask.patch_mask))
        self.assertTrue(torch.equal(first.masked_hidden, second.masked_hidden))
        self.assertTrue(torch.equal(first.reconstruction, second.reconstruction))
        self.assertIsNotNone(first.masked_dynamics)
        assert first.masked_dynamics is not None
        masked_positions = first.mask.patch_mask.unsqueeze(-1).expand_as(
            first.masked_dynamics.residual
        )
        self.assertTrue(
            torch.equal(
                first.masked_dynamics.residual.masked_select(masked_positions),
                torch.zeros_like(
                    first.masked_dynamics.residual.masked_select(masked_positions)
                ),
            )
        )

    def test_offset_decay_relation_loss_reaches_adapter(self) -> None:
        model = self._model("local")
        output = model.forward_pretrain(
            self.x,
            self.observed,
            self.weekday,
            self.slot,
            self.graph,
            self.neighbors,
            mask_task="time",
            generator=torch.Generator().manual_seed(11),
        )
        losses = compute_pretraining_loss(
            output=output,
            future_model=self.future,
            observed_context=self.observed,
            observed_future=torch.ones_like(self.future, dtype=torch.bool),
            context_start=torch.arange(self.batch) * 40,
            future_end=torch.arange(self.batch) * 40 + 23,
            retrieval_weight=0.1,
            retrieval_temperature=0.1,
            positive_quantile=0.2,
            context_quantile=0.3,
            negative_quantile=0.7,
            hard_negative_weight=2.0,
            retrieval_loss_mode="relation",
            relation_teacher_temperature=0.1,
            relation_student_temperature=0.1,
            forecast_context=self.x,
            forecast_context_observed=self.observed,
            relation_teacher_mode="offset_decay",
            relation_distance_normalization="anchor_mean",
        )

        losses.total.backward()

        self.assertIsNotNone(model.dynamics_adapter)
        assert model.dynamics_adapter is not None
        gradient = model.dynamics_adapter.residual_up.weight.grad
        self.assertIsNotNone(gradient)
        assert gradient is not None
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_adapter_is_part_of_retrieval_checkpoint_and_costs_under_two_percent(self) -> None:
        production_common = dict(
            patch_size=12,
            hidden_dim=96,
            retrieval_dim=48,
            num_heads=4,
            encoder_layers=3,
            dropout=0.0,
            dynamics_bottleneck_dim=16,
            dynamics_gate_bias=-2.0,
        )
        pretrain_config = PretrainConfig(
            time_mask_ratio=0.25,
            time_mask_block_size=36,
            space_mask_ratio=0.25,
        )
        baseline = STAnchorPretrainModel(
            ModelConfig(**production_common, dynamics_adapter_mode="none"),
            pretrain_config,
            context_length=288,
            slots_per_day=288,
        )
        graph_adapter = STAnchorPretrainModel(
            ModelConfig(**production_common, dynamics_adapter_mode="local_graph"),
            pretrain_config,
            context_length=288,
            slots_per_day=288,
        )

        state = graph_adapter.retrieval_state_dict()
        self.assertTrue(any(name.startswith("dynamics_adapter.") for name in state))
        increase = (count_parameters(graph_adapter) - count_parameters(baseline)) / count_parameters(
            baseline
        )
        self.assertLess(increase, 0.02)

    def test_config_validates_adapter_mode_and_bottleneck(self) -> None:
        valid = ExperimentConfig(
            data=DataConfig(raw_path="data.h5", adjacency_path="adj.pkl"),
            model=ModelConfig(
                dynamics_adapter_mode="local_graph",
                dynamics_bottleneck_dim=16,
                dynamics_gate_bias=-2.0,
            ),
        )
        valid.validate()

        invalid_mode = ExperimentConfig(
            data=valid.data,
            model=ModelConfig(dynamics_adapter_mode="dynamic_attention"),
        )
        with self.assertRaisesRegex(ValueError, "dynamics_adapter_mode"):
            invalid_mode.validate()

        invalid_bottleneck = ExperimentConfig(
            data=valid.data,
            model=ModelConfig(
                hidden_dim=32,
                dynamics_adapter_mode="local",
                dynamics_bottleneck_dim=64,
            ),
        )
        with self.assertRaisesRegex(ValueError, "dynamics_bottleneck_dim"):
            invalid_bottleneck.validate()

    def test_config_validates_context_conditioned_mode(self) -> None:
        valid = ExperimentConfig(
            data=DataConfig(raw_path="data.h5", adjacency_path="adj.pkl"),
            model=ModelConfig(
                hidden_dim=32,
                num_heads=4,
                dynamics_adapter_mode="context_conditioned",
                dynamics_bottleneck_dim=16,
                dynamics_gate_groups=8,
            ),
        )
        valid.validate()

        invalid_groups = ExperimentConfig(
            data=valid.data,
            model=ModelConfig(
                hidden_dim=30,
                num_heads=5,
                dynamics_adapter_mode="context_conditioned",
                dynamics_bottleneck_dim=16,
                dynamics_gate_groups=8,
            ),
        )
        with self.assertRaisesRegex(ValueError, "dynamics_gate_groups"):
            invalid_groups.validate()

    def test_context_conditioned_checkpoint_and_parameter_budget(self) -> None:
        production_common = dict(
            patch_size=12,
            hidden_dim=96,
            retrieval_dim=48,
            num_heads=4,
            encoder_layers=3,
            dropout=0.0,
            dynamics_bottleneck_dim=48,
            dynamics_gate_groups=8,
            dynamics_gate_bias=-2.0,
        )
        pretrain_config = PretrainConfig(
            time_mask_ratio=0.25,
            time_mask_block_size=36,
            space_mask_ratio=0.25,
        )
        baseline = STAnchorPretrainModel(
            ModelConfig(**production_common, dynamics_adapter_mode="none"),
            pretrain_config,
            context_length=288,
            slots_per_day=288,
        )
        conditioned = STAnchorPretrainModel(
            ModelConfig(**production_common, dynamics_adapter_mode="context_conditioned"),
            pretrain_config,
            context_length=288,
            slots_per_day=288,
        )

        state = conditioned.retrieval_state_dict()
        self.assertTrue(any(name.startswith("dynamics_adapter.") for name in state))
        increase = (count_parameters(conditioned) - count_parameters(baseline)) / count_parameters(
            baseline
        )
        self.assertGreater(increase, 0.05)
        self.assertLess(increase, 0.10)

    def test_global288_cc_fgda_config_is_single_variable(self) -> None:
        project = Path(__file__).resolve().parents[1]
        config = load_config(
            project / "configs" / "metrla_e5_final_latent48_cc_fgda_global288_v1.yaml"
        )
        reference = load_config(project / "configs" / "metrla_e5_final_latent48_global288_v1.yaml")

        self.assertEqual(config.model.dynamics_adapter_mode, "context_conditioned")
        self.assertEqual(config.model.dynamics_bottleneck_dim, 48)
        self.assertEqual(config.model.dynamics_gate_groups, 8)
        model_without_new_fields = config.model.__dict__ | {
            "dynamics_adapter_mode": reference.model.dynamics_adapter_mode,
            "dynamics_bottleneck_dim": reference.model.dynamics_bottleneck_dim,
            "dynamics_gate_groups": reference.model.dynamics_gate_groups,
        }
        reference_model = reference.model.__dict__ | {"dynamics_gate_groups": 8}
        self.assertEqual(model_without_new_fields, reference_model)
        self.assertEqual(config.data, reference.data)
        self.assertEqual(config.pretrain, reference.pretrain)
        self.assertEqual(config.target, reference.target)

    def test_global288_ablation_configs_are_pure_latent_and_single_variable(self) -> None:
        project = Path(__file__).resolve().parents[1]
        expected = {
            "metrla_e5_final_latent48_global288_v1.yaml": "none",
            "metrla_e5_final_latent48_local_fgda_global288_v1.yaml": "local",
            "metrla_e5_final_latent48_local_graph_fgda_global288_v1.yaml": "local_graph",
        }
        loaded = {
            name: load_config(project / "configs" / name)
            for name in expected
        }
        reference = loaded["metrla_e5_final_latent48_global288_v1.yaml"]
        for name, mode in expected.items():
            config = loaded[name]
            self.assertEqual(config.model.dynamics_adapter_mode, mode)
            self.assertEqual(config.model.retrieval_dim, 48)
            self.assertEqual(config.model.profile_dim, 0)
            self.assertEqual(config.model.latent_dim, 0)
            self.assertEqual(config.pretrain.profile_loss_weight, 0.0)
            self.assertEqual(config.pretrain.relation_teacher_mode, "offset_decay")
            self.assertEqual(
                config.pretrain.relation_distance_normalization,
                "symmetric_geometric_mean",
            )
            model_without_mode = config.model.__dict__ | {
                "dynamics_adapter_mode": reference.model.dynamics_adapter_mode
            }
            self.assertEqual(model_without_mode, reference.model.__dict__)
            self.assertEqual(config.data, reference.data)
            self.assertEqual(config.pretrain, reference.pretrain)
            self.assertEqual(config.target, reference.target)

    def test_pretrain_epoch_reports_adapter_diagnostics(self) -> None:
        model = self._model("local_graph")
        pretrain = PretrainConfig(
            time_mask_ratio=0.25,
            time_mask_block_size=3,
            space_mask_ratio=0.25,
        )
        config = ExperimentConfig(
            data=DataConfig(raw_path="data.h5", adjacency_path="adj.pkl"),
            model=self._model_config("local_graph"),
            pretrain=pretrain,
        )
        loader = [
            {
                "retrieval_x": self.x,
                "retrieval_observed": self.observed,
                "retrieval_weekday": self.weekday,
                "retrieval_slot": self.slot,
                "x": self.x,
                "x_observed": self.observed,
                "y": self.future,
                "y_observed": torch.ones_like(self.future, dtype=torch.bool),
                "context_start": torch.arange(self.batch) * 40,
                "future_end": torch.arange(self.batch) * 40 + 23,
            }
        ]

        result = run_pretrain_epoch(
            model=model,
            loader=loader,
            graph=self.graph,
            neighbors=self.neighbors,
            config=config,
            device=torch.device("cpu"),
            optimizer=None,
            max_batches=1,
        )

        self.assertGreater(result.adapter_valid_fraction, 0.0)
        self.assertGreater(result.adapter_fusion_gate_mean, 0.0)
        self.assertGreater(result.adapter_spatial_gate_mean, 0.0)
        self.assertEqual(result.adapter_contribution_ratio, 0.0)


if __name__ == "__main__":
    unittest.main()
