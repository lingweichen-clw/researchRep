from __future__ import annotations

import unittest

from stanchor.config import DataConfig, ExperimentConfig, TargetConfig
from stanchor.engine.target import build_downstream_model, configure_error_aware_stage
from stanchor.models.retrieval_router import RetrievalAwareMHAResidualRouter


class CurrentCalibratorConstructionTest(unittest.TestCase):
    def test_base_as_candidate_builds_only_current_router(self) -> None:
        config = ExperimentConfig(
            data=DataConfig(raw_path="data.h5", adjacency_path="adj.pkl"),
            target=TargetConfig(
                downstream_mode="learned_topk_error_aware",
                validation_correction_variant="base_as_candidate",
                calibrator_arch="retrieval_aware_mha_router",
            ),
        )
        model = build_downstream_model(config)
        self.assertIsNone(model.horizon_aggregator)
        self.assertIsInstance(model.error_corrector, RetrievalAwareMHAResidualRouter)
        groups = configure_error_aware_stage(model, "posthoc_calibrator")
        self.assertEqual([group["role"] for group in groups], ["calibrator"])

    def test_obsolete_calibrator_arch_is_not_executable(self) -> None:
        config = ExperimentConfig(
            data=DataConfig(raw_path="data.h5", adjacency_path="adj.pkl"),
            target=TargetConfig(
                downstream_mode="learned_topk_error_aware",
                validation_correction_variant="base_as_candidate",
                calibrator_arch="transformer_candidate_router",
            ),
        )
        with self.assertRaisesRegex(ValueError, "retrieval_aware_mha_router"):
            build_downstream_model(config)


if __name__ == "__main__":
    unittest.main()
