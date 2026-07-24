from __future__ import annotations

import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from stanchor.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    PretrainConfig,
    RuntimeConfig,
)
from stanchor.data.dataset import TrafficSeries, TrafficWindowDataset
from stanchor.data.graph import graph_from_dense
from stanchor.data.normalization import NodeStandardScaler
from stanchor.engine.pretrainer import build_validation_loader, run_pretrain_epoch
from stanchor.losses.pretraining import compute_pretraining_loss
from stanchor.losses.pretraining import masked_reconstruction_loss
from stanchor.models.pretraining import STAnchorPretrainModel


class PretrainingFlowTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(3)
        self.batch, self.time, self.nodes, self.channels = 6, 12, 8, 1
        adjacency = np.zeros((self.nodes, self.nodes), dtype=np.float32)
        for node in range(self.nodes):
            adjacency[node, (node - 1) % self.nodes] = 1.0
            adjacency[node, (node + 1) % self.nodes] = 1.0
        self.graph = graph_from_dense(adjacency)
        self.neighbors = self.graph.dense_neighbors(include_self=False)
        self.model = STAnchorPretrainModel(
            ModelConfig(
                input_channels=1,
                output_channels=1,
                patch_size=3,
                hidden_dim=16,
                retrieval_dim=8,
                num_heads=4,
                encoder_layers=2,
                dropout=0.0,
            ),
            PretrainConfig(
                time_mask_ratio=0.25,
                space_mask_ratio=0.25,
            ),
            context_length=12,
            slots_per_day=288,
        )
        base = torch.randn(self.batch, self.time + 12, self.nodes, self.channels)
        self.x = base[:, : self.time]
        self.y = base[:, self.time :]
        self.observed = torch.ones_like(self.x, dtype=torch.bool)
        self.weekday = torch.arange(self.time).unsqueeze(0).expand(self.batch, -1) % 7
        self.slot = torch.arange(self.time).unsqueeze(0).expand(self.batch, -1)
        # Events are separated enough that every pair is a legal comparison.
        self.context_start = torch.arange(self.batch) * 40
        self.future_end = self.context_start + 23

    def _run_task(self, task: str) -> None:
        output = self.model.forward_pretrain(
            self.x,
            self.observed,
            self.weekday,
            self.slot,
            self.graph,
            self.neighbors,
            mask_task=task,
        )
        self.assertEqual(tuple(output.reconstruction.shape), tuple(self.x.shape))
        self.assertEqual(tuple(output.clean.retrieval.node_keys.shape), (self.batch, self.nodes, 8))
        self.assertEqual(tuple(output.clean.retrieval.event_keys.shape), (self.batch, 8))
        losses = compute_pretraining_loss(
            output=output,
            future_model=self.y,
            observed_context=self.observed,
            observed_future=torch.ones_like(self.y, dtype=torch.bool),
            context_start=self.context_start,
            future_end=self.future_end,
            retrieval_weight=0.1,
            retrieval_temperature=0.1,
            positive_quantile=0.2,
            context_quantile=0.3,
            negative_quantile=0.7,
            hard_negative_weight=2.0,
        )
        self.assertTrue(bool(torch.isfinite(losses.total)))
        self.assertGreater(losses.valid_retrieval_anchors, 0)
        self.model.zero_grad(set_to_none=True)
        losses.total.backward()
        encoder_grad = sum(
            float(parameter.grad.abs().sum())
            for parameter in self.model.encoder.parameters()
            if parameter.grad is not None
        )
        retrieval_grad = sum(
            float(parameter.grad.abs().sum())
            for parameter in self.model.retrieval_head.parameters()
            if parameter.grad is not None
        )
        reconstruction_grad = sum(
            float(parameter.grad.abs().sum())
            for parameter in self.model.reconstruction_head.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(encoder_grad, 0.0)
        self.assertGreater(retrieval_grad, 0.0)
        self.assertGreater(reconstruction_grad, 0.0)

    def test_time_mask_pretraining_flow(self) -> None:
        self._run_task("time")

    def test_space_mask_pretraining_flow(self) -> None:
        self._run_task("space")

    def test_future_relation_mode_reaches_model_keys(self) -> None:
        output = self.model.forward_pretrain(
            self.x,
            self.observed,
            self.weekday,
            self.slot,
            self.graph,
            self.neighbors,
            mask_task="time",
        )
        losses = compute_pretraining_loss(
            output=output,
            future_model=self.y,
            observed_context=self.observed,
            observed_future=torch.ones_like(self.y, dtype=torch.bool),
            context_start=self.context_start,
            future_end=self.future_end,
            retrieval_weight=0.1,
            retrieval_temperature=0.1,
            positive_quantile=0.2,
            context_quantile=0.3,
            negative_quantile=0.7,
            hard_negative_weight=2.0,
            retrieval_loss_mode="relation",
            relation_teacher_temperature=0.1,
            relation_student_temperature=0.1,
        )

        self.assertGreater(losses.relation_candidate_pairs, 0)
        self.assertGreater(losses.teacher_effective_support, 1.0)
        self.assertGreater(losses.student_effective_support, 1.0)
        self.assertTrue(bool(torch.isfinite(losses.total)))
        losses.total.backward()
        self.assertTrue(
            any(
                parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0
                for parameter in self.model.retrieval_head.parameters()
            )
        )

    def test_patch_size_one_keeps_twelve_tokens_and_masks_three_contiguous_steps(self) -> None:
        model = STAnchorPretrainModel(
            ModelConfig(
                patch_size=1,
                hidden_dim=16,
                retrieval_dim=8,
                num_heads=4,
                encoder_layers=1,
                dropout=0.0,
            ),
            PretrainConfig(
                time_mask_ratio=0.25,
                time_mask_block_size=3,
                space_mask_ratio=0.25,
            ),
            context_length=12,
            slots_per_day=288,
        )
        output = model.forward_pretrain(
            self.x,
            self.observed,
            self.weekday,
            self.slot,
            self.graph,
            self.neighbors,
            mask_task="time",
            generator=torch.Generator().manual_seed(23),
        )

        self.assertEqual(tuple(output.masked_hidden.shape), (self.batch, 12, self.nodes, 16))
        self.assertEqual(tuple(output.reconstruction.shape), tuple(self.x.shape))
        for sample_mask in output.mask.patch_mask[:, :, 0]:
            indices = torch.where(sample_mask)[0]
            self.assertEqual(indices.numel(), 3)
            self.assertTrue(torch.equal(indices, torch.arange(indices[0], indices[0] + 3)))

        losses = compute_pretraining_loss(
            output=output,
            future_model=self.y,
            observed_context=self.observed,
            observed_future=torch.ones_like(self.y, dtype=torch.bool),
            context_start=self.context_start,
            future_end=self.future_end,
            retrieval_weight=0.1,
            retrieval_temperature=0.1,
            positive_quantile=0.2,
            context_quantile=0.3,
            negative_quantile=0.7,
            hard_negative_weight=2.0,
        )
        losses.total.backward()
        self.assertTrue(bool(torch.isfinite(losses.total)))
        self.assertTrue(
            any(
                parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
                for parameter in model.encoder.parameters()
            )
        )

    def test_missing_future_node_is_excluded_from_retrieval_supervision(self) -> None:
        output = self.model.forward_pretrain(
            self.x,
            self.observed,
            self.weekday,
            self.slot,
            self.graph,
            self.neighbors,
            mask_task="time",
        )
        future_observed = torch.ones_like(self.y, dtype=torch.bool)
        future_observed[:, :, 0, :] = False
        losses = compute_pretraining_loss(
            output=output,
            future_model=self.y,
            observed_context=self.observed,
            observed_future=future_observed,
            context_start=self.context_start,
            future_end=self.future_end,
            retrieval_weight=0.1,
            retrieval_temperature=0.1,
            positive_quantile=0.2,
            context_quantile=0.3,
            negative_quantile=0.7,
            hard_negative_weight=2.0,
        )
        self.assertLessEqual(losses.valid_retrieval_anchors, self.batch * (self.nodes - 1))

    def test_empty_reconstruction_supervision_returns_connected_zero(self) -> None:
        prediction = torch.randn(2, 3, 4, 1, requires_grad=True)
        target = torch.zeros_like(prediction)
        loss = masked_reconstruction_loss(
            prediction,
            target,
            torch.ones_like(prediction, dtype=torch.bool),
            torch.zeros_like(prediction, dtype=torch.bool),
        )
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        self.assertIsNotNone(prediction.grad)

    def test_encoder_rejects_forecast_window_when_retrieval_window_is_longer(self) -> None:
        model = STAnchorPretrainModel(
            ModelConfig(
                patch_size=6,
                hidden_dim=16,
                retrieval_dim=8,
                num_heads=4,
                encoder_layers=1,
                dropout=0.0,
            ),
            PretrainConfig(time_mask_block_size=6),
            context_length=24,
            slots_per_day=288,
        )
        x = torch.randn(2, 12, self.nodes, 1)
        observed = torch.ones_like(x, dtype=torch.bool)
        weekday = torch.zeros((2, 12), dtype=torch.long)
        slot = torch.arange(12).unsqueeze(0).expand(2, -1)

        with self.assertRaisesRegex(ValueError, "configured context length"):
            model.encode_clean(x, observed, weekday, slot, self.graph)

    def test_pretrain_epoch_uses_retrieval_context_instead_of_forecast_context(self) -> None:
        length, nodes = 80, self.nodes
        values = (
            np.arange(length, dtype=np.float32)[:, None, None]
            + np.arange(nodes, dtype=np.float32)[None, :, None]
            + 1.0
        )
        observed = np.ones_like(values, dtype=bool)
        series = TrafficSeries(
            values=values,
            observed=observed,
            timestamps_ns=np.arange(length, dtype=np.int64),
            weekday=np.zeros(length, dtype=np.int64),
            slot=np.arange(length, dtype=np.int64) % 288,
            slots_per_day=288,
        )
        scaler = NodeStandardScaler.fit(values[:60], observed[:60])
        dataset = TrafficWindowDataset(
            series,
            scaler,
            0,
            60,
            context_length=12,
            horizon=12,
            retrieval_context_length=24,
        )
        config = ExperimentConfig(
            data=DataConfig(
                raw_path="unused.h5",
                adjacency_path="unused.pkl",
                context_length=12,
                retrieval_context_length=24,
                horizon=12,
            ),
            model=ModelConfig(
                patch_size=6,
                hidden_dim=16,
                retrieval_dim=8,
                num_heads=4,
                encoder_layers=1,
                dropout=0.0,
            ),
            pretrain=PretrainConfig(
                batch_size=6,
                time_mask_block_size=6,
                positive_quantile=0.2,
                context_quantile=0.3,
                negative_quantile=0.7,
            ),
            runtime=RuntimeConfig(device="cpu"),
        )
        model = STAnchorPretrainModel(
            config.model,
            config.pretrain,
            context_length=24,
            slots_per_day=288,
        )

        progress: list[tuple[int, int, float, float]] = []
        result = run_pretrain_epoch(
            model,
            DataLoader(dataset, batch_size=6, shuffle=False),
            self.graph,
            self.neighbors,
            config,
            torch.device("cpu"),
            optimizer=None,
            max_batches=1,
            progress_interval=1,
            progress_callback=lambda completed, total, elapsed, eta: progress.append(
                (completed, total, elapsed, eta)
            ),
        )

        self.assertEqual(result.batches, 1)
        self.assertGreater(result.reconstruction_positions, 0)
        self.assertEqual(len(progress), 1)
        self.assertEqual(progress[0][:2], (1, 1))
        self.assertGreaterEqual(progress[0][2], 0.0)
        self.assertGreaterEqual(progress[0][3], 0.0)

    def test_validation_loader_is_reproducible_and_contains_non_overlapping_events(self) -> None:
        length, nodes = 500, self.nodes
        values = np.ones((length, nodes, 1), dtype=np.float32)
        observed = np.ones_like(values, dtype=bool)
        series = TrafficSeries(
            values=values,
            observed=observed,
            timestamps_ns=np.arange(length, dtype=np.int64),
            weekday=np.zeros(length, dtype=np.int64),
            slot=np.arange(length, dtype=np.int64) % 288,
            slots_per_day=288,
        )
        scaler = NodeStandardScaler.fit(values[:400], observed[:400])
        dataset = TrafficWindowDataset(
            series,
            scaler,
            0,
            460,
            context_length=12,
            horizon=12,
            retrieval_context_length=24,
        )

        first = next(iter(build_validation_loader(dataset, 16, 0, seed=42)))
        second = next(iter(build_validation_loader(dataset, 16, 0, seed=42)))

        self.assertTrue(torch.equal(first["sample_id"], second["sample_id"]))
        non_overlap = (
            first["future_end"][:, None] < first["context_start"][None, :]
        ) | (
            first["future_end"][None, :] < first["context_start"][:, None]
        )
        self.assertTrue(bool(non_overlap.any()))

    def test_validation_loader_preserves_legacy_order_when_batch_spans_events(self) -> None:
        length, nodes = 100, self.nodes
        values = np.ones((length, nodes, 1), dtype=np.float32)
        observed = np.ones_like(values, dtype=bool)
        series = TrafficSeries(
            values=values,
            observed=observed,
            timestamps_ns=np.arange(length, dtype=np.int64),
            weekday=np.zeros(length, dtype=np.int64),
            slot=np.arange(length, dtype=np.int64) % 288,
            slots_per_day=288,
        )
        scaler = NodeStandardScaler.fit(values[:80], observed[:80])
        dataset = TrafficWindowDataset(
            series,
            scaler,
            0,
            80,
            context_length=12,
            horizon=12,
        )

        batch = next(iter(build_validation_loader(dataset, 32, 0, seed=42)))

        self.assertTrue(
            torch.equal(batch["sample_id"], torch.arange(11, 43, dtype=torch.long))
        )


if __name__ == "__main__":
    unittest.main()
