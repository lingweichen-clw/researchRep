from __future__ import annotations

import unittest

import torch

from stanchor.diagnostics.teacher_metric_diagnostic import (
    candidate_distance,
    distribution_asymmetry,
    effective_support_from_distance,
    near_tie_collision_rate,
    symmetric_candidate_distance,
    symmetric_geometric_mean_normalize_distances,
    topk_jaccard,
)


class TeacherMetricDiagnosticTest(unittest.TestCase):
    def test_symmetric_geometric_mean_normalization_is_symmetric(self) -> None:
        distances = torch.tensor(
            [
                [
                    [0.0, 0.0],
                    [2.0, 4.0],
                    [8.0, 2.0],
                ],
                [
                    [2.0, 4.0],
                    [0.0, 0.0],
                    [4.0, 6.0],
                ],
                [
                    [8.0, 2.0],
                    [4.0, 6.0],
                    [0.0, 0.0],
                ],
            ]
        )
        valid = ~torch.eye(3, dtype=torch.bool).unsqueeze(-1).expand_as(distances)

        normalized, normalized_valid = symmetric_geometric_mean_normalize_distances(
            distances,
            valid,
        )

        self.assertTrue(torch.equal(normalized_valid, normalized_valid.transpose(0, 1)))
        self.assertTrue(torch.allclose(normalized, normalized.transpose(0, 1), atol=1.0e-7))

    def test_symmetric_geometric_mean_normalization_is_globally_scale_invariant(self) -> None:
        distances = torch.tensor(
            [
                [[0.0], [2.0], [8.0]],
                [[2.0], [0.0], [4.0]],
                [[8.0], [4.0], [0.0]],
            ]
        )
        valid = ~torch.eye(3, dtype=torch.bool).unsqueeze(-1)

        normalized, _ = symmetric_geometric_mean_normalize_distances(
            distances,
            valid,
            eps=1.0e-12,
        )
        scaled, _ = symmetric_geometric_mean_normalize_distances(
            17.0 * distances,
            valid,
            eps=1.0e-12,
        )

        self.assertTrue(torch.allclose(normalized, scaled, atol=1.0e-6))

    def test_symmetric_geometric_mean_normalization_masks_invalid_and_empty_events(self) -> None:
        distances = torch.tensor(
            [
                [[0.0], [3.0], [7.0]],
                [[3.0], [0.0], [5.0]],
                [[7.0], [5.0], [0.0]],
            ]
        )
        valid = torch.tensor(
            [
                [[False], [True], [False]],
                [[True], [False], [False]],
                [[False], [False], [False]],
            ]
        )

        normalized, normalized_valid = symmetric_geometric_mean_normalize_distances(
            distances,
            valid,
        )

        self.assertTrue(bool(normalized_valid[0, 1, 0]))
        self.assertFalse(bool(normalized_valid[0, 2, 0]))
        self.assertFalse(bool(normalized_valid[2].any()))
        self.assertTrue(torch.isfinite(normalized).all())
        self.assertTrue(torch.equal(normalized.masked_select(~normalized_valid), torch.zeros(7)))

    def test_symmetric_candidate_distance_uses_candidate_to_event_scales(self) -> None:
        query = torch.tensor([[[[0.0]]]])
        candidates = torch.tensor([[[[[2.0]]], [[[10.0]]]]])
        query_observed = torch.ones_like(query, dtype=torch.bool)
        candidate_observed = torch.ones_like(candidates, dtype=torch.bool)
        event_valid = torch.ones(1, 2, dtype=torch.bool)

        normalized, valid = symmetric_candidate_distance(
            query,
            query_observed,
            candidates,
            candidate_observed,
            event_valid,
        )

        # Common event set {0, 2, 10}: mu_q=6, mu_2=5, mu_10=9.
        expected = torch.tensor(
            [[[2.0 / (6.0 * 5.0) ** 0.5, 10.0 / (6.0 * 9.0) ** 0.5]]]
        )
        self.assertTrue(torch.equal(valid, torch.ones_like(valid)))
        self.assertTrue(torch.allclose(normalized, expected, atol=1.0e-6))

    def test_candidate_distance_clips_single_horizon_outlier(self) -> None:
        query = torch.zeros(1, 3, 1, 1)
        candidates = torch.tensor([[[[[0.0]], [[0.0]], [[0.0]]]]])
        candidates = candidates.view(1, 1, 3, 1, 1)
        query[:, 1] = 10.0
        observed = torch.ones_like(query, dtype=torch.bool)
        candidate_observed = torch.ones_like(candidates, dtype=torch.bool)
        valid = torch.ones(1, 1, dtype=torch.bool)

        raw, raw_valid = candidate_distance(
            query,
            observed,
            candidates,
            candidate_observed,
            valid,
        )
        clipped, clipped_valid = candidate_distance(
            query,
            observed,
            candidates,
            candidate_observed,
            valid,
            clip_delta=2.0,
        )

        self.assertAlmostEqual(float(raw[0, 0, 0]), 10.0 / 3.0, places=5)
        self.assertAlmostEqual(float(clipped[0, 0, 0]), 2.0 / 3.0, places=5)
        self.assertTrue(torch.equal(raw_valid, clipped_valid))

    def test_topk_jaccard_is_one_for_same_support_and_zero_for_disjoint_support(self) -> None:
        clean = torch.tensor([[[0, 1, 2, 3, 4]]])
        same = torch.tensor([[[4, 3, 2, 1, 0]]])
        disjoint = torch.tensor([[[5, 6, 7, 8, 9]]])

        self.assertAlmostEqual(float(topk_jaccard(clean, same).mean()), 1.0)
        self.assertAlmostEqual(float(topk_jaccard(clean, disjoint).mean()), 0.0)

    def test_effective_support_matches_uniform_and_one_hot_limits(self) -> None:
        uniform_distance = torch.zeros(1, 1, 4)
        valid = torch.ones(1, 1, 4, dtype=torch.bool)
        support_uniform = effective_support_from_distance(uniform_distance, valid, 0.1)
        self.assertAlmostEqual(float(support_uniform[0, 0]), 4.0, places=5)

        one_hot_distance = torch.tensor([[[0.0, 100.0, 100.0, 100.0]]])
        support_one_hot = effective_support_from_distance(one_hot_distance, valid, 0.1)
        self.assertLess(float(support_one_hot[0, 0]), 1.01)

    def test_collision_rate_detects_od_near_tie_with_increment_separation(self) -> None:
        od = torch.tensor([[[1.00, 1.04, 0.10]]])
        increment = torch.tensor([[[0.10, 0.60, 0.20]]])
        valid = torch.ones_like(od, dtype=torch.bool)

        rate, pairs = near_tie_collision_rate(
            od,
            increment,
            valid,
            od_tolerance=0.05,
            increment_gap=0.25,
        )

        self.assertAlmostEqual(float(rate), 1.0)
        self.assertEqual(int(pairs), 1)

    def test_distribution_asymmetry_rejects_batches_without_symmetric_pairs(self) -> None:
        probabilities = torch.zeros(2, 2, 1)
        valid = torch.zeros_like(probabilities, dtype=torch.bool)

        with self.assertRaisesRegex(ValueError, "symmetric pair"):
            distribution_asymmetry(probabilities, valid)


if __name__ == "__main__":
    unittest.main()
