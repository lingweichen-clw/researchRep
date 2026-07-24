from __future__ import annotations

import unittest

import torch

from stanchor.data.normalization import WindowStatistics
from stanchor.losses.pretraining import (
    build_future_relation_targets,
    future_relation_retrieval_loss,
)


class FutureRelationLossTest(unittest.TestCase):
    def setUp(self) -> None:
        self.batch = 4
        self.future = torch.tensor(
            [
                [[0.0]],
                [[0.1]],
                [[1.0]],
                [[2.0]],
            ],
            dtype=torch.float32,
        ).view(self.batch, 1, 1, 1).expand(-1, 2, -1, -1)
        self.statistics = WindowStatistics(
            normalized=torch.zeros(self.batch, 3, 1, 1),
            level_features=torch.zeros(self.batch, 1, 4),
            level_valid=torch.ones(self.batch, 1, 1, dtype=torch.bool),
            mean=torch.zeros(self.batch, 1, 1),
            std=torch.ones(self.batch, 1, 1),
        )
        self.future_observed = torch.ones_like(self.future, dtype=torch.bool)
        self.context_start = torch.tensor([0, 20, 40, 60])
        self.future_end = self.context_start + 11

    def test_teacher_distribution_prefers_closer_future(self) -> None:
        targets = build_future_relation_targets(
            self.future,
            self.statistics,
            self.future_observed,
            self.context_start,
            self.future_end,
            teacher_temperature=0.1,
        )

        # For query 0, sample 1 is closer than samples 2 and 3.
        self.assertGreater(
            float(targets.teacher_distribution[0, 1, 0]),
            float(targets.teacher_distribution[0, 2, 0]),
        )
        self.assertGreater(
            float(targets.teacher_distribution[0, 2, 0]),
            float(targets.teacher_distribution[0, 3, 0]),
        )
        self.assertEqual(tuple(targets.future_distance.shape), (4, 4, 1))
        self.assertEqual(tuple(targets.candidate_mask.shape), (4, 4, 1))
        self.assertEqual(tuple(targets.valid_anchors.shape), (4, 1))
        self.assertTrue(bool(targets.valid_anchors.all()))

    def test_relation_loss_has_finite_value_and_gradient(self) -> None:
        keys = torch.randn(self.batch, 1, 3, requires_grad=True)
        result = future_relation_retrieval_loss(
            keys,
            self.future,
            self.statistics,
            self.future_observed,
            self.context_start,
            self.future_end,
            teacher_temperature=0.1,
            student_temperature=0.1,
        )

        self.assertEqual(result.valid_anchors, self.batch)
        self.assertEqual(result.candidate_pairs, self.batch * (self.batch - 1))
        self.assertTrue(bool(torch.isfinite(result.loss)))
        self.assertGreater(result.teacher_effective_support, 1.0)
        result.loss.backward()
        self.assertIsNotNone(keys.grad)
        self.assertTrue(bool(torch.isfinite(keys.grad).all()))
        self.assertGreater(float(keys.grad.abs().sum()), 0.0)

    def test_single_candidate_is_excluded_from_relation_gradient(self) -> None:
        future = self.future[:2]
        statistics = WindowStatistics(
            normalized=self.statistics.normalized[:2],
            level_features=self.statistics.level_features[:2],
            level_valid=self.statistics.level_valid[:2],
            mean=self.statistics.mean[:2],
            std=self.statistics.std[:2],
        )
        keys = torch.randn(2, 1, 3, requires_grad=True)
        result = future_relation_retrieval_loss(
            keys,
            future,
            statistics,
            self.future_observed[:2],
            self.context_start[:2],
            self.future_end[:2],
            teacher_temperature=0.1,
            student_temperature=0.1,
        )

        self.assertEqual(result.valid_anchors, 0)
        self.assertEqual(result.candidate_pairs, 0)
        self.assertEqual(float(result.loss.detach()), 0.0)
        result.loss.backward()
        self.assertIsNotNone(keys.grad)
        self.assertEqual(float(keys.grad.abs().sum()), 0.0)

    def test_missing_or_overlapping_future_is_not_a_candidate(self) -> None:
        observed = self.future_observed.clone()
        observed[1, :, 0, :] = False
        overlapping_end = self.future_end.clone()
        overlapping_end[2] = self.context_start[3]
        targets = build_future_relation_targets(
            self.future,
            self.statistics,
            observed,
            self.context_start,
            overlapping_end,
            teacher_temperature=0.1,
        )

        self.assertFalse(bool(targets.candidate_mask[:, 1, 0].any()))
        self.assertFalse(bool(targets.candidate_mask[2, 3, 0]))
        self.assertEqual(int(targets.candidate_mask.sum()), 4)


if __name__ == "__main__":
    unittest.main()
