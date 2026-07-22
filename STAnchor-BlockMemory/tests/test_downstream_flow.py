from __future__ import annotations

import unittest

import torch

from stanchor.config import DataConfig, ExperimentConfig, TargetConfig
from stanchor.engine.target import checkpoint_downstream_mode
from stanchor.losses.downstream import compute_downstream_loss
from stanchor.models.downstream import (
    ConfidenceHead,
    LightweightForecastBackbone,
    SafeResidualFusion,
    STAnchorDownstreamModel,
    confidence_soft_target,
)
from stanchor.retrieval.retriever import AggregationOutput, NodeCandidates


class DownstreamFlowTest(unittest.TestCase):
    def test_config_rejects_unknown_downstream_mode(self) -> None:
        config = ExperimentConfig(
            data=DataConfig(raw_path="data.h5", adjacency_path="adj.pkl"),
            target=TargetConfig(downstream_mode="not_a_mode"),
        )

        with self.assertRaisesRegex(ValueError, "downstream mode"):
            config.validate()

    def test_legacy_checkpoint_defaults_to_existing_confidence_mode(self) -> None:
        self.assertEqual(
            checkpoint_downstream_mode({"config": {"target": {}}}),
            "learned_topk_confidence",
        )
        self.assertEqual(
            checkpoint_downstream_mode({"downstream_mode": "base_only"}),
            "base_only",
        )

    def _inputs(self, with_memory: bool = True):
        batch, time, horizon, nodes, channels, top_k = 2, 12, 3, 4, 1, 2
        x = torch.randn(batch, time, nodes, channels)
        candidate_valid = torch.full((batch, nodes, top_k), with_memory, dtype=torch.bool)
        candidates = NodeCandidates(
            event_ids=torch.zeros((batch, nodes, top_k), dtype=torch.long),
            total_scores=torch.tensor([0.9, 0.7]).view(1, 1, 2).expand(batch, nodes, -1),
            shape_scores=torch.tensor([0.8, 0.6]).view(1, 1, 2).expand(batch, nodes, -1),
            level_distances=torch.tensor([0.1, 0.2]).view(1, 1, 2).expand(batch, nodes, -1),
            weights=torch.tensor([0.7, 0.3]).view(1, 1, 2).expand(batch, nodes, -1),
            valid=candidate_valid,
        )
        memory_valid = torch.full((batch, horizon, nodes, channels), with_memory, dtype=torch.bool)
        aggregation = AggregationOutput(
            prediction=torch.randn(batch, horizon, nodes, channels),
            variance=torch.rand(batch, horizon, nodes, channels),
            valid=memory_valid,
            candidate_futures=torch.randn(batch, horizon, nodes, top_k, channels),
            candidate_masks=torch.full((batch, horizon, nodes, top_k, channels), with_memory),
        )
        model = STAnchorDownstreamModel(
            LightweightForecastBackbone(time, horizon, channels, channels, hidden_dim=16, dropout=0.0),
            ConfidenceHead(hidden_dim=8),
            SafeResidualFusion(horizon, initial_max_weight=0.1),
            confidence_level_temperature=1.0,
        )
        return x, candidates, aggregation, model

    def test_no_candidate_is_exact_base_fallback(self) -> None:
        x, candidates, aggregation, model = self._inputs(with_memory=False)
        output = model(x, candidates, aggregation)
        self.assertTrue(torch.equal(output.final_prediction, output.base_prediction))
        self.assertTrue(torch.equal(output.fusion_weight, torch.zeros_like(output.fusion_weight)))
        self.assertTrue(torch.equal(output.confidence, torch.zeros_like(output.confidence)))

    def test_downstream_shapes_and_gradients(self) -> None:
        x, candidates, aggregation, model = self._inputs(with_memory=True)
        output = model(x, candidates, aggregation)
        self.assertEqual(tuple(output.confidence_features.shape), (2, 3, 4, 6))
        self.assertEqual(tuple(output.final_prediction.shape), (2, 3, 4, 1))
        self.assertTrue(bool((output.fusion_weight >= 0).all()))
        self.assertTrue(bool((output.fusion_weight <= 1).all()))
        target = torch.randn_like(output.final_prediction)
        losses = compute_downstream_loss(
            output,
            target,
            torch.ones_like(target, dtype=torch.bool),
            confidence_weight=1.0,
            help_margin=0.0,
            help_temperature=0.1,
        )
        self.assertTrue(bool(torch.isfinite(losses.total)))
        losses.total.backward()
        self.assertIsNotNone(model.backbone.network[0].weight.grad)
        self.assertIsNotNone(model.confidence_head.network[0].weight.grad)
        self.assertIsNotNone(model.fusion.horizon_logits.grad)

    def test_confidence_target_rewards_better_memory(self) -> None:
        target = torch.zeros((1, 2, 1, 1))
        base = torch.ones_like(target)
        memory = torch.full_like(target, 0.1)
        valid = torch.ones_like(target, dtype=torch.bool)
        soft = confidence_soft_target(base, memory, target, valid, margin=0.0, temperature=0.1)
        self.assertTrue(bool((soft > 0.5).all()))

    def test_base_only_mode_does_not_require_memory(self) -> None:
        x, _, _, _ = self._inputs(with_memory=True)
        model = STAnchorDownstreamModel(
            LightweightForecastBackbone(12, 3, 1, 1, hidden_dim=16, dropout=0.0),
            ConfidenceHead(hidden_dim=8),
            SafeResidualFusion(3, initial_max_weight=0.1),
            confidence_level_temperature=1.0,
            mode="base_only",
        )

        output = model(x, None, None)

        self.assertTrue(torch.equal(output.final_prediction, output.base_prediction))
        self.assertTrue(torch.equal(output.fusion_weight, torch.zeros_like(output.fusion_weight)))
        self.assertFalse(bool(output.memory_valid.any()))

    def test_horizon_only_mode_uses_memory_without_confidence_head_gradient(self) -> None:
        x, candidates, aggregation, _ = self._inputs(with_memory=True)
        model = STAnchorDownstreamModel(
            LightweightForecastBackbone(12, 3, 1, 1, hidden_dim=16, dropout=0.0),
            ConfidenceHead(hidden_dim=8),
            SafeResidualFusion(3, initial_max_weight=0.1),
            confidence_level_temperature=1.0,
            mode="learned_topk_horizon",
        )

        output = model(x, candidates, aggregation)
        expected = output.base_prediction + 0.1 * (
            output.memory_prediction - output.base_prediction
        )
        self.assertTrue(torch.allclose(output.final_prediction, expected, atol=1.0e-6))
        self.assertTrue(torch.equal(output.confidence, torch.ones_like(output.confidence)))
        target = torch.randn_like(output.final_prediction)
        losses = compute_downstream_loss(
            output,
            target,
            torch.ones_like(target, dtype=torch.bool),
            confidence_weight=1.0,
            help_margin=0.0,
            help_temperature=0.1,
            use_confidence=False,
        )
        losses.total.backward()
        self.assertIsNone(model.confidence_head.network[0].weight.grad)
        self.assertIsNotNone(model.fusion.horizon_logits.grad)


if __name__ == "__main__":
    unittest.main()
