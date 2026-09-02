from __future__ import annotations

import unittest

import torch

from stanchor.config import DataConfig, ExperimentConfig, TargetConfig
from stanchor.engine.target import (
    build_downstream_model,
    checkpoint_bank_level_weight,
    checkpoint_candidate_protocol,
    checkpoint_downstream_mode,
    validate_evaluation_bank_path,
    should_stop_target_stage,
)
from stanchor.losses.downstream import compute_downstream_loss
from stanchor.modes import LEARNED_TOPK_OFFSET_DECAY_HORIZON
from stanchor.models.downstream import (
    ConfidenceHead,
    DownstreamOutput,
    LightweightForecastBackbone,
    SafeResidualFusion,
    STAnchorDownstreamModel,
    confidence_soft_target,
)
from stanchor.retrieval.retriever import AggregationOutput, NodeCandidates
from scripts.train_downstream import build_parser


class DownstreamFlowTest(unittest.TestCase):
    def test_forecast_loss_accepts_physical_space_inputs(self) -> None:
        zeros = torch.zeros(1, 1, 2, 1)
        output = DownstreamOutput(
            base_prediction=zeros,
            memory_prediction=zeros,
            confidence_features=zeros,
            confidence=zeros,
            fusion_weight=zeros,
            final_prediction=zeros,
            memory_valid=torch.ones_like(zeros, dtype=torch.bool),
        )
        physical_prediction = torch.tensor([[[[2.0], [4.0]]]])
        losses = compute_downstream_loss(
            output,
            target=zeros,
            observed=torch.ones_like(zeros, dtype=torch.bool),
            confidence_weight=0.0,
            help_margin=0.0,
            help_temperature=0.1,
            use_confidence=False,
            forecast_prediction=physical_prediction,
            forecast_target=zeros,
        )
        self.assertAlmostEqual(float(losses.forecast), 3.0, places=6)

    def test_target_early_stopping_can_be_disabled_for_formal_runs(self) -> None:
        self.assertFalse(should_stop_target_stage(50, 10, enabled=False))
        self.assertTrue(should_stop_target_stage(10, 10, enabled=True))

    def test_base_only_evaluation_does_not_require_a_bank(self) -> None:
        self.assertIsNone(validate_evaluation_bank_path("base_only", None))
        with self.assertRaisesRegex(ValueError, "requires --bank"):
            validate_evaluation_bank_path("learned_topk_error_aware", None)

    def test_downstream_cli_exposes_formal_no_early_stopping_switch(self) -> None:
        args = build_parser().parse_args(
            [
                "--config",
                "config.yaml",
                "--disable-early-stopping",
            ]
        )
        self.assertTrue(args.disable_early_stopping)

    def test_config_rejects_unknown_downstream_mode(self) -> None:
        config = ExperimentConfig(
            data=DataConfig(raw_path="data.h5", adjacency_path="adj.pkl"),
            target=TargetConfig(downstream_mode="not_a_mode"),
        )

        with self.assertRaisesRegex(ValueError, "downstream mode"):
            config.validate()

    def test_config_rejects_unknown_candidate_protocol(self) -> None:
        config = ExperimentConfig(
            data=DataConfig(raw_path="data.h5", adjacency_path="adj.pkl"),
            target=TargetConfig(candidate_protocol="future_oracle"),
        )

        with self.assertRaisesRegex(ValueError, "candidate protocol"):
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

    def test_legacy_mode_does_not_add_error_aware_checkpoint_parameters(self) -> None:
        config = ExperimentConfig(
            data=DataConfig(raw_path="data.h5", adjacency_path="adj.pkl"),
            target=TargetConfig(downstream_mode="learned_topk_confidence"),
        )
        model = build_downstream_model(config)
        self.assertIsNone(model.error_corrector)
        self.assertFalse(
            any(name.startswith("error_corrector") for name in model.state_dict())
        )

    def test_candidate_protocol_defaults_to_exact_calendar(self) -> None:
        self.assertEqual(
            checkpoint_candidate_protocol({"config": {"target": {}}}),
            "exact_calendar",
        )
        self.assertEqual(
            checkpoint_candidate_protocol({"candidate_protocol": "relaxed_calendar"}),
            "relaxed_calendar",
        )

    def test_candidate_protocol_rejects_unknown_checkpoint_value(self) -> None:
        with self.assertRaisesRegex(ValueError, "candidate protocol"):
            checkpoint_candidate_protocol({"candidate_protocol": "future_oracle"})

    def test_candidate_protocol_rejects_explicit_evaluation_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "differs from checkpoint"):
            checkpoint_candidate_protocol(
                {"candidate_protocol": "relaxed_calendar"},
                expected="exact_calendar",
            )

    def test_bank_level_weight_is_restored_from_training_checkpoint(self) -> None:
        checkpoint = {"config": {"bank": {"level_weight": 0.0}}}

        self.assertEqual(checkpoint_bank_level_weight(checkpoint, default=0.25), 0.0)
        self.assertEqual(checkpoint_bank_level_weight({}, default=0.25), 0.25)

        with self.assertRaisesRegex(ValueError, "level_weight"):
            checkpoint_bank_level_weight(
                {"config": {"bank": {"level_weight": -1.0}}},
                default=0.25,
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

    def test_offset_decay_horizon_mode_uses_memory_without_confidence_head_gradient(self) -> None:
        x, candidates, aggregation, _ = self._inputs(with_memory=True)
        model = STAnchorDownstreamModel(
            LightweightForecastBackbone(12, 3, 1, 1, hidden_dim=16, dropout=0.0),
            ConfidenceHead(hidden_dim=8),
            SafeResidualFusion(3, initial_max_weight=0.1),
            confidence_level_temperature=1.0,
            mode=LEARNED_TOPK_OFFSET_DECAY_HORIZON,
        )

        output = model(x, candidates, aggregation)

        self.assertTrue(torch.equal(output.confidence, torch.ones_like(output.confidence)))
        self.assertTrue(torch.equal(output.confidence_features, torch.zeros_like(output.confidence_features)))
        losses = compute_downstream_loss(
            output,
            torch.randn_like(output.final_prediction),
            torch.ones_like(output.final_prediction, dtype=torch.bool),
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
