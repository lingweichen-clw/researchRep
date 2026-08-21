from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from stanchor.config import DataConfig, ExperimentConfig, TargetConfig, load_config
from stanchor.engine import target as target_engine
from stanchor.engine.target import configure_error_aware_stage
from stanchor.losses.downstream import (
    build_blend_target,
    build_huber_risk_target,
    compute_downstream_loss,
)
from stanchor.models.downstream import (
    ConfidenceHead,
    LightweightForecastBackbone,
    SafeResidualFusion,
    STAnchorDownstreamModel,
    StructuredErrorCorrector,
    build_error_aware_features,
)
from stanchor.retrieval.retriever import AggregationOutput, NodeCandidates


class ErrorAwareFusionTest(unittest.TestCase):
    @staticmethod
    def _model(
        mode: str = "learned_topk_error_aware",
        backbone_hidden_dim: int = 16,
        risk_hidden_dim: int = 256,
        fusion_hidden_dim: int = 128,
    ) -> STAnchorDownstreamModel:
        return STAnchorDownstreamModel(
            LightweightForecastBackbone(
                12, 3, 1, 1, backbone_hidden_dim, dropout=0.0
            ),
            ConfidenceHead(8),
            SafeResidualFusion(3),
            1.0,
            mode=mode,
            error_corrector=(
                StructuredErrorCorrector(12, 3, 1, risk_hidden_dim, fusion_hidden_dim)
                if mode == "learned_topk_error_aware"
                else None
            ),
        )

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

    def test_latent48_error_aware_features_have_nine_nonredundant_shapes(self) -> None:
        x, base, candidates, aggregation = self._inputs()
        corrector = StructuredErrorCorrector(12, 3, 1)
        risk, _ = corrector.predict_risk(x, base)
        features, memory_valid = build_error_aware_features(
            candidates, aggregation, base, risk, level_temperature=1.0
        )
        self.assertEqual(tuple(risk.shape), (2, 3, 4, 1))
        self.assertEqual(tuple(features.shape), (2, 3, 4, 9))
        self.assertTrue(bool(memory_valid.all()))
        self.assertTrue(bool(torch.isfinite(features).all()))

    def test_additive_fusion_starts_at_point_one_and_returns_contributions(self) -> None:
        x, base, candidates, aggregation = self._inputs()
        corrector = StructuredErrorCorrector(12, 3, 1)
        risk, _ = corrector.predict_risk(x, base)
        features, memory_valid = build_error_aware_features(
            candidates, aggregation, base, risk, level_temperature=1.0
        )
        fusion = StructuredErrorCorrector(12, 3, 1)
        final, weight, contributions = fusion(
            x, base, aggregation.prediction, features, memory_valid
        )
        self.assertTrue(bool(torch.isfinite(weight).all()))
        self.assertTrue(bool((weight >= 0).all()))
        self.assertTrue(bool((weight <= 1).all()))
        self.assertEqual(tuple(contributions.shape), (2, 3, 4, 9))
        self.assertTrue(bool(torch.isfinite(final).all()))

    def test_no_memory_is_exact_base_fallback(self) -> None:
        x, base, candidates, aggregation = self._inputs(valid=False)
        corrector = StructuredErrorCorrector(12, 3, 1)
        risk, _ = corrector.predict_risk(x, base)
        features, memory_valid = build_error_aware_features(
            candidates, aggregation, base, risk, level_temperature=1.0
        )
        final, weight, _ = StructuredErrorCorrector(12, 3, 1)(
            x, base, aggregation.prediction, features, memory_valid
        )
        self.assertTrue(torch.equal(final, base))
        self.assertTrue(torch.equal(weight, torch.zeros_like(weight)))

    def test_structured_corrector_matches_documented_224k_architecture(self) -> None:
        corrector = StructuredErrorCorrector(12, 12, 1)
        parameters = sum(parameter.numel() for parameter in corrector.parameters())
        self.assertEqual(corrector.risk_repr_dim, 128)
        self.assertEqual(corrector.joint_dim, 256)
        self.assertEqual(corrector.evidence_encoder[0].out_features, 128)
        self.assertEqual(corrector.shape_functions[0][0].out_features, 32)
        self.assertEqual(corrector.output[0].in_features, 256)
        self.assertEqual(corrector.output[0].out_features, 128)
        self.assertEqual(parameters, 224_142)

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
            error_corrector=StructuredErrorCorrector(12, 3, 1, 256, 128),
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
        self.assertIsNotNone(model.error_corrector.risk_encoder[0].weight.grad)
        self.assertIsNotNone(model.error_corrector.shape_functions[0][0].weight.grad)

    def test_training_stages_freeze_the_expected_modules(self) -> None:
        x, _, candidates, aggregation = self._inputs()
        del x, candidates, aggregation
        model = STAnchorDownstreamModel(
            LightweightForecastBackbone(12, 3, 1, 1, 16, dropout=0.0),
            ConfidenceHead(8),
            SafeResidualFusion(3),
            1.0,
            mode="learned_topk_error_aware",
            error_corrector=StructuredErrorCorrector(12, 3, 1, 256, 128),
        )
        base_groups = configure_error_aware_stage(model, "base")
        self.assertEqual([group["role"] for group in base_groups], ["backbone"])
        self.assertTrue(all(parameter.requires_grad for parameter in model.backbone.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in model.error_corrector.parameters()))
        calibrator_groups = configure_error_aware_stage(model, "calibrator")
        self.assertEqual([group["role"] for group in calibrator_groups], ["calibrator"])
        self.assertFalse(any(parameter.requires_grad for parameter in model.backbone.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.error_corrector.parameters()))
        model.train()
        self.assertFalse(model.backbone.training)
        self.assertTrue(model.error_corrector.training)
        joint_groups = configure_error_aware_stage(model, "joint")
        model.train()
        self.assertEqual(
            [group["role"] for group in joint_groups], ["backbone", "calibrator"]
        )
        self.assertTrue(model.backbone.training)

    def test_posthoc_protocol_requires_error_aware_mode_and_zero_warmups(self) -> None:
        valid = ExperimentConfig(
            data=DataConfig(raw_path="data.h5", adjacency_path="adj.pkl"),
            target=TargetConfig(
                downstream_mode="learned_topk_error_aware",
                training_protocol="posthoc_frozen_base",
                base_warmup_epochs=0,
                calibrator_warmup_epochs=0,
            ),
        )
        valid.validate()
        with self.assertRaisesRegex(ValueError, "training_protocol"):
            ExperimentConfig(
                data=DataConfig(raw_path="data.h5", adjacency_path="adj.pkl"),
                target=TargetConfig(training_protocol="unknown"),
            ).validate()
        with self.assertRaisesRegex(ValueError, "learned_topk_error_aware"):
            ExperimentConfig(
                data=DataConfig(raw_path="data.h5", adjacency_path="adj.pkl"),
                target=TargetConfig(
                    downstream_mode="base_only",
                    training_protocol="posthoc_frozen_base",
                    calibrator_warmup_epochs=0,
                ),
            ).validate()
        with self.assertRaisesRegex(ValueError, "warmup"):
            ExperimentConfig(
                data=DataConfig(raw_path="data.h5", adjacency_path="adj.pkl"),
                target=TargetConfig(
                    downstream_mode="learned_topk_error_aware",
                    training_protocol="posthoc_frozen_base",
                    base_warmup_epochs=1,
                    calibrator_warmup_epochs=0,
                ),
            ).validate()

    def test_posthoc_protocol_selects_only_calibrator_stage(self) -> None:
        config = ExperimentConfig(
            data=DataConfig(raw_path="data.h5", adjacency_path="adj.pkl"),
            target=TargetConfig(
                downstream_mode="learned_topk_error_aware",
                training_protocol="posthoc_frozen_base",
                epochs=17,
                base_warmup_epochs=0,
                calibrator_warmup_epochs=0,
            ),
        )
        self.assertEqual(
            target_engine.build_downstream_training_stages(
                config, base_checkpoint_path=Path("base.pt")
            ),
            [("posthoc_calibrator", 17)],
        )
        with self.assertRaisesRegex(ValueError, "base checkpoint"):
            target_engine.build_downstream_training_stages(
                config, base_checkpoint_path=None
            )
        model = self._model()
        groups = configure_error_aware_stage(model, "posthoc_calibrator")
        self.assertEqual([group["role"] for group in groups], ["calibrator"])
        self.assertFalse(
            any(parameter.requires_grad for parameter in model.backbone.parameters())
        )

    def test_frozen_base_checkpoint_loads_only_matching_base_backbone(self) -> None:
        source = self._model(mode="base_only")
        destination = self._model()
        with torch.no_grad():
            for parameter in source.backbone.parameters():
                parameter.fill_(0.25)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "base.pt"
            torch.save(
                {
                    "downstream_mode": "base_only",
                    "downstream_state_dict": source.state_dict(),
                },
                checkpoint_path,
            )
            provenance = target_engine.load_frozen_base_backbone(
                destination, checkpoint_path, torch.device("cpu")
            )
        for expected, actual in zip(
            source.backbone.parameters(), destination.backbone.parameters()
        ):
            self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(provenance["fingerprint"], target_engine._state_dict_fingerprint(destination.backbone))
        self.assertTrue(provenance["path"].endswith("base.pt"))
        self.assertFalse(
            any(parameter.requires_grad for parameter in destination.backbone.parameters())
        )

    def test_frozen_base_checkpoint_rejects_wrong_mode_and_shape(self) -> None:
        destination = self._model()
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            wrong_mode_path = directory_path / "wrong_mode.pt"
            torch.save(
                {
                    "downstream_mode": "learned_topk_error_aware",
                    "downstream_state_dict": destination.state_dict(),
                },
                wrong_mode_path,
            )
            with self.assertRaisesRegex(ValueError, "base_only"):
                target_engine.load_frozen_base_backbone(
                    destination, wrong_mode_path, torch.device("cpu")
                )

            wrong_shape_path = directory_path / "wrong_shape.pt"
            wrong_shape = self._model(
                mode="base_only", backbone_hidden_dim=8
            )
            torch.save(
                {
                    "downstream_mode": "base_only",
                    "downstream_state_dict": wrong_shape.state_dict(),
                },
                wrong_shape_path,
            )
            with self.assertRaisesRegex(ValueError, "incompatible backbone"):
                target_engine.load_frozen_base_backbone(
                    destination, wrong_shape_path, torch.device("cpu")
                )

    def test_posthoc_capacity_configs_have_exact_calibrator_parameters(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        expected = {
            "metrla_stgcn_tgge_v3_error_aware_posthoc_v1.yaml": 224142,
            "metrla_graphwavenet_tgge_v3_error_aware_posthoc_v1.yaml": 224142,
        }
        for config_name, expected_parameters in expected.items():
            config = load_config(project_root / "configs" / config_name)
            self.assertEqual(
                config.target.training_protocol, "posthoc_frozen_base"
            )
            self.assertEqual(
                config.target.downstream_mode, "learned_topk_error_aware"
            )
            model = StructuredErrorCorrector(
                config.data.context_length,
                config.data.horizon,
                config.model.input_channels,
                config.target.risk_hidden_dim,
                config.target.fusion_feature_hidden_dim,
            )
            calibrator_parameters = sum(parameter.numel() for parameter in model.parameters())
            self.assertEqual(calibrator_parameters, expected_parameters)

    def test_base_risk_is_supervised_even_when_memory_is_missing(self) -> None:
        x, _, candidates, aggregation = self._inputs(valid=False)
        model = STAnchorDownstreamModel(
            LightweightForecastBackbone(12, 3, 1, 1, 16, dropout=0.0),
            ConfidenceHead(8),
            SafeResidualFusion(3),
            1.0,
            mode="learned_topk_error_aware",
            error_corrector=StructuredErrorCorrector(12, 3, 1, 256, 128),
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
