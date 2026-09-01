from __future__ import annotations

import unittest

from stanchor.config import DataConfig, ExperimentConfig, TargetConfig
from stanchor.engine.target import build_downstream_model, configure_error_aware_stage


class HorizonAggregatorTypeTest(unittest.TestCase):
    def test_base_as_candidate_uses_none_horizon_aggregator(self) -> None:
        config = ExperimentConfig(
            data=DataConfig(raw_path="data.h5", adjacency_path="adj.pkl"),
            target=TargetConfig(
                downstream_mode="learned_topk_error_aware",
                validation_correction_variant="base_as_candidate",
                calibrator_arch="transformer_candidate_router",
            ),
        )
        model = build_downstream_model(config)
        self.assertIsNone(model.horizon_aggregator)
        groups = configure_error_aware_stage(model, "posthoc_calibrator")
        self.assertEqual([group["role"] for group in groups], ["calibrator"])


if __name__ == "__main__":
    unittest.main()