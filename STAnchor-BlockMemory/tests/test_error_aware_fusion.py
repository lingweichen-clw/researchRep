from __future__ import annotations

import unittest

import torch

from stanchor.config import DataConfig, ExperimentConfig, TargetConfig
from stanchor.engine.target import configure_error_aware_stage
from stanchor.losses.downstream import (
    build_blend_target,
    build_huber_risk_target,
    compute_downstream_loss,
)
from stanchor.models.downstream import (
    ConfidenceHead,
    ErrorAwareAdditiveFusion,
    LightweightForecastBackbone,
    PredictedBaseRisk,
    SafeResidualFusion,
    STAnchorDownstreamModel,
    build_error_aware_features,
)
from stanchor.retrieval.retriever import AggregationOutput, NodeCandidates


class ErrorAwareFusionTest(unittest.TestCase):
    def _inputs(self, valid: bool = True):
        batch, time, horizon, nodes, channels, top_k = 2, 12, 3, 4, 1, 2
        x = torch.randn(batch, time, nodes, channels)
        base = torch.randn(batch, horizon, nodes, channels)
        candidate_valid = torch.full((batch, nodes, top_k), valid, dtype=torch.bool)
        candidates = NodeCandidates(
            event_ids=torch.zeros(batch, nodes, top_k, dtype=torch.long),
            total_scores=torch.tensor([0.9, 0.7]).view(1, 1, 2).expand(batch, nodes, -1),
            shape_scores=torch.tensor([0.8, 0.6]).view(1, 1, 2).expand(batch, nodes, -1),
            level_distances=torch.tensor([0.1, 0.2]).view(1, 1, 2).expand(batch, nodes, -1),
            weights=torch.tensor([0.7, 0.3]).view(1, 1, 2).expand(batch, nodes, -1),
            valid=candidate_valid,
            profile_scores=torch.tensor([0.75, 0.55]).view(1, 1, 2).expand(batch, nodes, -1),
            latent_scores=torch.tensor([0.82, 0.61]).view(1, 1, 2).expand(batch, nodes, -1),
        )
        memory_valid = torch.full((batch, horizon, nodes, channels), valid, dtype=torch.bool)
        candidate_futures = torch.stack((base + 0.5, base + 0.2), dim=3)
        aggregation = AggregationOutput(
            prediction=0.7 * candidate_futures[:, :, :, 0] + 0.3 * candidate_futures[:, :, :, 1],
            variance=torch.full_like(base, 0.02),
            valid=memory_valid,
            candidate_futures=candidate_futures,
            candidate_masks=torch.full_like(candidate_futures, valid, dtype=torch.bool),
        )
        return x, base, candidates, aggregation

    def test_risk_head_and_ten_features_have_expected_shapes(self) -> None:
        x, base, candidates, aggregation = self._inputs()
        risk_head = PredictedBaseRisk(12, 3, 1, hidden_dim=8)
        risk = risk_head(x, base)
        features, memory_valid = build_error_aware_features(
            candidates, aggregation, base, risk, level_temperature=1.0
        )
        self.assertEqual(tuple(risk.shape), (2, 3, 4, 1))
        self.assertEqual(tuple(features.shape), (2, 3, 4, 10))
        self.assertTrue(bool(memory_valid.all()))
        self.assertTrue(bool(torch.isfinite(features).all()))

    def test_additive_fusion_starts_at_point_one_and_returns_contributions(self) -> None:
        x, base, candidates, aggregation = self._inputs()
        risk = PredictedBaseRisk(12, 3, 1, hidden_dim=8)(x, base)
        features, memory_valid = build_error_aware_features(
            candidates, aggregation, base, risk, level_temperature=1.0
        )
        fusion = ErrorAwareAdditiveFusion(num_features=10, hidden_dim=8, initial_weight=0.1)
        final, weight, contributions = fusion(
            base, aggregation.prediction, features, memory_valid
        )
        self.assertTrue(torch.allclose(weight, torch.full_like(weight, 0.1), atol=1.0e-6))
        self.assertEqual(tuple(contributions.shape), (2, 3, 4, 10))
        self.assertTrue(torch.allclose(final, base + 0.1 * (aggregation.prediction - base)))

    def test_no_memory_is_exact_base_fallback(self) -> None:
        x, base, candidates, aggregation = self._inputs(valid=False)
        risk = PredictedBaseRisk(12, 3, 1, hidden_dim=8)(x, base)
        features, memory_valid = build_error_aware_features(
            candidates, aggregation, base, risk, level_temperature=1.0
        )
        final, weight, _ = ErrorAwareAdditiveFusion(10, 8)(
            base, aggregation.prediction, features, memory_valid
        )
        self.assertTrue(torch.equal(final, base))
        self.assertTrue(torch.equal(weight, torch.zeros_like(weight)))

    def test_risk_and_oracle_blend_targets_match_definitions(self) -> None:
        base = torch.tensor([[[[2.0]]]])
        memory = torch.tensor([[[[0.0]]]])
        target = torch.tensor([[[[1.0]]]])
        observed = torch.ones_like(target, dtype=torch.bool)
        valid = torch.ones_like(target, dtype=torch.bool)
        risk_target, risk_valid = build_huber_risk_target(base, target, observed)
        blend_target, blend_valid = build_blend_target(
            base, memory, target, observed, valid, minimum_direction_norm=1.0e-6
        )
        self.assertAlmostEqual(float(risk_target), 0.5, places=6)
        self.assertTrue(bool(risk_valid.all()))
        self.assertAlmostEqual(float(blend_target), 0.5, places=6)
        self.assertTrue(bool(blend_valid.all()))

    def test_full_error_aware_mode_has_finite_joint_gradients(self) -> None:
        ExperimentConfig(
            data=DataConfig(raw_path="data.h5", adjacency_path="adj.pkl"),
            target=TargetConfig(downstream_mode="learned_topk_error_aware"),
        ).validate()
        x, _, candidates, aggregation = self._inputs()
        model = STAnchorDownstreamModel(
            backbone=LightweightForecastBackbone(12, 3, 1, 1, 16, dropout=0.0),
            confidence_head=ConfidenceHead(8),
            fusion=SafeResidualFusion(3),
            confidence_level_temperature=1.0,
            mode="learned_topk_error_aware",
            risk_head=PredictedBaseRisk(12, 3, 1, hidden_dim=8),
            error_aware_fusion=ErrorAwareAdditiveFusion(10, 8),
        )
        output = model(x, candidates, aggregation)
        target = torch.randn_like(output.final_prediction)
        losses = compute_downstream_loss(
            output,
            target,
            torch.ones_like(target, dtype=torch.bool),
            confidence_weight=0.0,
            help_margin=0.0,
            help_temperature=0.1,
            use_confidence=False,
            use_error_aware=True,
            risk_weight=0.1,
            blend_weight=0.1,
        )
        self.assertTrue(bool(torch.isfinite(losses.total)))
        losses.total.backward()
        self.assertIsNotNone(model.backbone.network[0].weight.grad)
        self.assertIsNotNone(model.risk_head.network[0].weight.grad)
        self.assertIsNotNone(model.error_aware_fusion.shape_functions[0][0].weight.grad)

    def test_training_stages_freeze_the_expected_modules(self) -> None:
        x, _, candidates, aggregation = self._inputs()
        del x, candidates, aggregation
        model = STAnchorDownstreamModel(
            LightweightForecastBackbone(12, 3, 1, 1, 16, dropout=0.0),
            ConfidenceHead(8),
            SafeResidualFusion(3),
            1.0,
            mode="learned_topk_error_aware",
            risk_head=PredictedBaseRisk(12, 3, 1, 8),
            error_aware_fusion=ErrorAwareAdditiveFusion(10, 8),
        )
        base_groups = configure_error_aware_stage(model, "base")
        self.assertEqual([group["role"] for group in base_groups], ["backbone"])
        self.assertTrue(all(parameter.requires_grad for parameter in model.backbone.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in model.risk_head.parameters()))
        calibrator_groups = configure_error_aware_stage(model, "calibrator")
        self.assertEqual([group["role"] for group in calibrator_groups], ["calibrator"])
        self.assertFalse(any(parameter.requires_grad for parameter in model.backbone.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.risk_head.parameters()))
        model.train()
        self.assertFalse(model.backbone.training)
        self.assertTrue(model.risk_head.training)
        self.assertTrue(model.error_aware_fusion.training)
        joint_groups = configure_error_aware_stage(model, "joint")
        model.train()
        self.assertEqual(
            [group["role"] for group in joint_groups], ["backbone", "calibrator"]
        )
        self.assertTrue(model.backbone.training)

    def test_base_risk_is_supervised_even_when_memory_is_missing(self) -> None:
        x, _, candidates, aggregation = self._inputs(valid=False)
        model = STAnchorDownstreamModel(
            LightweightForecastBackbone(12, 3, 1, 1, 16, dropout=0.0),
            ConfidenceHead(8),
            SafeResidualFusion(3),
            1.0,
            mode="learned_topk_error_aware",
            risk_head=PredictedBaseRisk(12, 3, 1, 8),
            error_aware_fusion=ErrorAwareAdditiveFusion(10, 8),
        )
        output = model(x, candidates, aggregation)
        target = torch.randn_like(output.final_prediction)
        losses = compute_downstream_loss(
            output,
            target,
            torch.ones_like(target, dtype=torch.bool),
            confidence_weight=0.0,
            help_margin=0.0,
            help_temperature=0.1,
            use_confidence=False,
            use_error_aware=True,
            risk_weight=0.1,
            blend_weight=0.1,
        )
        self.assertIsNotNone(losses.risk)
        self.assertGreater(float(losses.risk.detach()), 0.0)
        self.assertEqual(float(losses.blend.detach()), 0.0)


if __name__ == "__main__":
    unittest.main()
