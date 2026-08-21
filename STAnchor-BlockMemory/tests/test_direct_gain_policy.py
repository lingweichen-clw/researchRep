from __future__ import annotations

import unittest

import torch

from stanchor.diagnostics.direct_gain import build_direct_gain_features
from stanchor.retrieval.retriever import AggregationOutput, NodeCandidates


class DirectGainFeatureTest(unittest.TestCase):
    def test_signed_and_horizon_specific_candidate_features_are_preserved(self) -> None:
        base = torch.zeros(1, 2, 1, 1)
        futures = torch.tensor([1.0, 3.0, -2.0, -4.0]).view(1, 2, 1, 2, 1)
        candidates = NodeCandidates(
            event_ids=torch.tensor([[[0, 1]]]),
            total_scores=torch.tensor([[[1.0, 0.5]]]),
            shape_scores=torch.tensor([[[1.0, 0.5]]]),
            level_distances=torch.zeros(1, 1, 2),
            weights=torch.tensor([[[0.75, 0.25]]]),
            valid=torch.ones(1, 1, 2, dtype=torch.bool),
        )
        aggregation = AggregationOutput(
            prediction=torch.tensor([[[[1.5]], [[-2.5]]]]),
            variance=torch.ones_like(base),
            valid=torch.ones_like(base, dtype=torch.bool),
            candidate_futures=futures,
            candidate_masks=torch.ones_like(futures, dtype=torch.bool),
        )
        features = build_direct_gain_features(torch.zeros(1, 2, 1, 9), candidates, aggregation, base)
        self.assertEqual(tuple(features.shape), (1, 2, 1, 14))
        self.assertGreater(float(features[0, 0, 0, 11]), 0.0)
        self.assertLess(float(features[0, 1, 0, 11]), 0.0)
        self.assertNotEqual(float(features[0, 0, 0, 9]), float(features[0, 1, 0, 9]))


if __name__ == "__main__":
    unittest.main()
