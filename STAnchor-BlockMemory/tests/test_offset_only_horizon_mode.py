from __future__ import annotations

import unittest

from torch import nn

from stanchor.engine.target import configure_error_aware_stage
from stanchor.models.downstream import (
    ConfidenceHead,
    SafeResidualFusion,
    STAnchorDownstreamModel,
)


class OffsetOnlyHorizonModeTest(unittest.TestCase):
    def test_mode_is_registered_as_horizon_memory_mode(self) -> None:
        from stanchor.modes import (
            DOWNSTREAM_MODES,
            HORIZON_ONLY_MODES,
            LEARNED_TOPK_OFFSET_ONLY_HORIZON,
            MEMORY_MODES,
            validate_downstream_mode,
        )

        self.assertEqual(
            LEARNED_TOPK_OFFSET_ONLY_HORIZON,
            "learned_topk_offset_only_horizon",
        )
        self.assertIn(LEARNED_TOPK_OFFSET_ONLY_HORIZON, DOWNSTREAM_MODES)
        self.assertIn(LEARNED_TOPK_OFFSET_ONLY_HORIZON, HORIZON_ONLY_MODES)
        self.assertIn(LEARNED_TOPK_OFFSET_ONLY_HORIZON, MEMORY_MODES)
        self.assertEqual(
            validate_downstream_mode(LEARNED_TOPK_OFFSET_ONLY_HORIZON),
            LEARNED_TOPK_OFFSET_ONLY_HORIZON,
        )

    def test_posthoc_stage_trains_only_simple_horizon_fusion(self) -> None:
        from stanchor.modes import LEARNED_TOPK_OFFSET_ONLY_HORIZON

        model = STAnchorDownstreamModel(
            backbone=nn.Linear(1, 1),
            confidence_head=ConfidenceHead(4),
            fusion=SafeResidualFusion(horizon=12),
            confidence_level_temperature=1.0,
            mode=LEARNED_TOPK_OFFSET_ONLY_HORIZON,
        )

        groups = configure_error_aware_stage(model, 'posthoc_calibrator')

        self.assertEqual([group['role'] for group in groups], ['fusion'])
        self.assertTrue(all(parameter.requires_grad for parameter in model.fusion.parameters()))
        self.assertFalse(any(parameter.requires_grad for parameter in model.backbone.parameters()))
        self.assertFalse(
            any(parameter.requires_grad for parameter in model.confidence_head.parameters())
        )


if __name__ == "__main__":
    unittest.main()
