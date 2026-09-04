from __future__ import annotations

import unittest

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset

from stanchor.config import DataConfig, ExperimentConfig, TargetConfig, load_config
from stanchor.data.graph import graph_from_dense
from stanchor.engine.target import (
    build_target_optimizer,
    build_target_scheduler,
    build_downstream_model,
    select_downstream_training_dataset,
    validate_downstream_bank_path,
    validate_downstream_pretrained_checkpoint,
)
from stanchor.modes import BASE_ONLY, LEARNED_TOPK_CONFIDENCE
from stanchor.models.stgcn import STGCNForecastBackbone, build_stgcn_gso


class STGCNBackboneTest(unittest.TestCase):
    def setUp(self) -> None:
        adjacency = np.array(
            [
                [0.0, 1.0, 0.2],
                [0.5, 0.0, 1.0],
                [0.2, 0.4, 0.0],
            ],
            dtype=np.float32,
        )
        self.graph = graph_from_dense(adjacency)

    def test_scaled_laplacian_gso_is_finite_and_symmetric(self) -> None:
        gso = build_stgcn_gso(self.graph)
        self.assertEqual(tuple(gso.shape), (3, 3))
        self.assertTrue(bool(torch.isfinite(gso).all()))
        self.assertTrue(torch.allclose(gso, gso.T, atol=1.0e-5))

    def test_forward_returns_project_forecast_layout_and_gradients(self) -> None:
        model = STGCNForecastBackbone(
            context_length=12,
            horizon=4,
            input_channels=1,
            output_channels=1,
            graph=self.graph,
            temporal_kernel=3,
            graph_kernel=3,
            block_num=1,
            hidden_channels=16,
            bottleneck_channels=4,
            output_hidden_channels=32,
            dropout=0.0,
        )
        x = torch.randn(2, 12, 3, 1)
        output = model(x)
        self.assertEqual(tuple(output.shape), (2, 4, 3, 1))
        self.assertTrue(bool(torch.isfinite(output).all()))
        output.square().mean().backward()
        self.assertIsNotNone(model.st_blocks[0].temporal1.conv.weight.grad)

    def test_downstream_factory_builds_stgcn_when_requested(self) -> None:
        config = ExperimentConfig(
            data=DataConfig(raw_path="data.h5", adjacency_path="adj.pkl"),
            target=TargetConfig(backbone_name="stgcn"),
        )
        model = build_downstream_model(config, self.graph)
        self.assertIsInstance(model.backbone, STGCNForecastBackbone)

    def test_stgcn_output_block_matches_reference_without_dropout(self) -> None:
        config = ExperimentConfig(
            data=DataConfig(raw_path="data.h5", adjacency_path="adj.pkl"),
            target=TargetConfig(backbone_name="stgcn"),
        )
        model = build_downstream_model(config, self.graph)

        self.assertFalse(
            any(isinstance(module, nn.Dropout) for module in model.backbone.output.modules())
        )

    def test_full_train_scope_does_not_discard_memory_partition(self) -> None:
        dataset = TensorDataset(torch.arange(10))
        config = ExperimentConfig(
            data=DataConfig(raw_path="data.h5", adjacency_path="adj.pkl"),
            target=TargetConfig(training_data_scope="full_train"),
        )

        selected, memory_events = select_downstream_training_dataset(dataset, config)

        self.assertIs(selected, dataset)
        self.assertEqual(memory_events, 7)

    def test_default_scope_preserves_existing_calibration_protocol(self) -> None:
        dataset = TensorDataset(torch.arange(10))
        config = ExperimentConfig(
            data=DataConfig(raw_path="data.h5", adjacency_path="adj.pkl"),
        )

        selected, memory_events = select_downstream_training_dataset(dataset, config)

        self.assertEqual(memory_events, 7)
        self.assertEqual(len(selected), 3)

    def test_reference_optimizer_and_scheduler_are_constructed(self) -> None:
        parameter = nn.Parameter(torch.ones(()))
        config = ExperimentConfig(
            data=DataConfig(raw_path="data.h5", adjacency_path="adj.pkl"),
            target=TargetConfig(
                optimizer_name="adam",
                scheduler_name="step_lr",
                scheduler_step_size=10,
                scheduler_gamma=0.95,
            ),
        )
        optimizer = build_target_optimizer(
            config,
            [{"params": [parameter], "lr": 1.0e-3}],
        )
        scheduler = build_target_scheduler(config, optimizer)

        self.assertIsInstance(optimizer, torch.optim.Adam)
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.StepLR)
        self.assertEqual(scheduler.step_size, 10)
        self.assertEqual(scheduler.gamma, 0.95)

    def test_current_stgcn_router_config_uses_formal_training_protocol(self) -> None:
        config = load_config("configs/formal_base_as_candidate_stgcn.yaml")

        self.assertEqual(config.target.training_data_scope, "full_train")
        self.assertEqual(config.target.batch_size, 32)
        self.assertEqual(config.target.epochs, 50)
        self.assertEqual(config.target.optimizer_name, "adam")
        self.assertEqual(config.target.scheduler_name, "step_lr")
        self.assertEqual(config.target.scheduler_step_size, 10)
        self.assertEqual(config.target.scheduler_gamma, 0.5)
        self.assertEqual(config.target.weight_decay, 1.0e-4)
        self.assertEqual(
            config.target.calibrator_arch, "retrieval_aware_mha_router"
        )

    def test_base_only_does_not_require_a_bank(self) -> None:
        self.assertIsNone(validate_downstream_bank_path(BASE_ONLY, None))

    def test_retrieval_mode_requires_a_bank(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires --bank"):
            validate_downstream_bank_path(LEARNED_TOPK_CONFIDENCE, None)

    def test_base_only_does_not_require_a_pretrained_checkpoint(self) -> None:
        self.assertIsNone(
            validate_downstream_pretrained_checkpoint(BASE_ONLY, None)
        )

    def test_retrieval_mode_requires_a_pretrained_checkpoint(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires --pretrained-checkpoint"):
            validate_downstream_pretrained_checkpoint(
                LEARNED_TOPK_CONFIDENCE,
                None,
            )


if __name__ == "__main__":
    unittest.main()
