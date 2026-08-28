from __future__ import annotations

import math
import unittest

import torch

from stanchor.config import DataConfig, ExperimentConfig, PretrainConfig
from stanchor.data.normalization import WindowStatistics
from stanchor.losses.pretraining import (
    anchor_mean_normalize_distances,
    build_future_increment,
    build_future_relation_targets,
    build_offset_decay_signature,
    future_relation_retrieval_loss,
    hard_mirage_ranking_loss,
    offset_decay_hard_negative_retrieval_loss,
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

    def test_config_enforces_teacher_mode_contract(self) -> None:
        ExperimentConfig(
            data=DataConfig(raw_path="data.h5", adjacency_path="adj.pkl"),
            pretrain=PretrainConfig(),
        ).validate()
        ExperimentConfig(
            data=DataConfig(raw_path="data.h5", adjacency_path="adj.pkl"),
            pretrain=PretrainConfig(
                retrieval_loss_mode="relation",
                relation_teacher_mode="offset_decay",
                relation_distance_normalization="anchor_mean",
            ),
        ).validate()
        with self.assertRaisesRegex(ValueError, "anchor_mean"):
            ExperimentConfig(
                data=DataConfig(raw_path="data.h5", adjacency_path="adj.pkl"),
                pretrain=PretrainConfig(
                    retrieval_loss_mode="relation",
                    relation_teacher_mode="offset_decay",
                ),
            ).validate()

        with self.assertRaisesRegex(ValueError, "rank_positive_count"):
            ExperimentConfig(
                data=DataConfig(raw_path="data.h5", adjacency_path="adj.pkl"),
                pretrain=PretrainConfig(
                    retrieval_loss_mode="relation",
                    rank_loss_weight=0.05,
                    rank_positive_count=0,
                ),
            ).validate()
        with self.assertRaisesRegex(ValueError, "0.5"):
            ExperimentConfig(
                data=DataConfig(raw_path="data.h5", adjacency_path="adj.pkl"),
                pretrain=PretrainConfig(
                    retrieval_loss_mode="relation",
                    relation_teacher_mode="offset_decay_increment",
                    relation_distance_normalization="anchor_mean",
                    future_increment_weight=0.25,
                ),
            ).validate()

    def test_hard_mirage_rank_prefers_correct_candidate_order(self) -> None:
        distance = torch.tensor(
            [
                [[0.0], [0.1], [0.2], [1.0]],
                [[0.1], [0.0], [0.3], [1.1]],
                [[0.2], [0.3], [0.0], [1.2]],
                [[1.0], [1.1], [1.2], [0.0]],
            ]
        )
        candidate_mask = ~torch.eye(4, dtype=torch.bool).unsqueeze(-1)
        correct_similarity = -distance
        wrong_similarity = correct_similarity.clone()
        wrong_similarity[0, 3, 0] = 1.0

        correct = hard_mirage_ranking_loss(
            student_similarity=correct_similarity,
            future_distance=distance,
            candidate_mask=candidate_mask,
            positive_count=2,
            negative_count=2,
            future_gap=0.05,
            margin=0.05,
            temperature=0.1,
        )
        wrong = hard_mirage_ranking_loss(
            student_similarity=wrong_similarity,
            future_distance=distance,
            candidate_mask=candidate_mask,
            positive_count=2,
            negative_count=2,
            future_gap=0.05,
            margin=0.05,
            temperature=0.1,
        )

        self.assertGreater(correct.valid_pairs, 0)
        self.assertEqual(correct.valid_pairs, wrong.valid_pairs)
        self.assertLess(float(correct.loss), float(wrong.loss))

    def test_relation_loss_reports_rank_loss_and_finite_gradient(self) -> None:
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
            rank_loss_weight=0.05,
            rank_positive_count=2,
            rank_negative_count=2,
            rank_future_gap=0.05,
            rank_margin=0.05,
            rank_temperature=0.1,
        )

        self.assertGreater(result.rank_pairs, 0)
        self.assertTrue(bool(torch.isfinite(result.rank_loss)))
        result.loss.backward()
        self.assertIsNotNone(keys.grad)
        self.assertTrue(bool(torch.isfinite(keys.grad).all()))
    def test_offset_decay_signature_uses_endpoint_and_linear_horizon_decay(self) -> None:
        forecast_context = torch.tensor([[[[1.0]], [[2.0]], [[3.0]]]])
        context_observed = torch.ones_like(forecast_context, dtype=torch.bool)
        future = torch.tensor([[[[4.0]], [[5.0]], [[6.0]]]])
        future_observed = torch.ones_like(future, dtype=torch.bool)

        signature, valid = build_offset_decay_signature(
            future,
            future_observed,
            forecast_context,
            context_observed,
        )

        expected = torch.tensor([[[[1.0]], [[3.5]], [[6.0]]]])
        self.assertTrue(torch.allclose(signature, expected))
        self.assertTrue(torch.equal(valid, future_observed))

    def test_offset_decay_signature_falls_back_to_visible_mean_endpoint(self) -> None:
        forecast_context = torch.tensor([[[[1.0]], [[3.0]], [[99.0]]]])
        context_observed = torch.tensor([[[[True]], [[True]], [[False]]]])
        future = torch.tensor([[[[4.0]], [[5.0]], [[6.0]]]])
        future_observed = torch.ones_like(future, dtype=torch.bool)

        signature, valid = build_offset_decay_signature(
            future,
            future_observed,
            forecast_context,
            context_observed,
        )

        expected = torch.tensor([[[[2.0]], [[4.0]], [[6.0]]]])
        self.assertTrue(torch.allclose(signature, expected))
        self.assertTrue(torch.equal(valid, future_observed))

    def test_future_increment_starts_from_endpoint_then_uses_adjacent_future(self) -> None:
        endpoint = torch.tensor([[[3.0]]])
        endpoint_valid = torch.ones_like(endpoint, dtype=torch.bool)
        future = torch.tensor([[[[4.0]], [[7.0]], [[6.0]]]])
        future_observed = torch.ones_like(future, dtype=torch.bool)

        increment, valid = build_future_increment(
            future,
            future_observed,
            endpoint,
            endpoint_valid,
        )

        expected = torch.tensor([[[[1.0]], [[3.0]], [[-1.0]]]])
        self.assertTrue(torch.allclose(increment, expected))
        self.assertTrue(torch.equal(valid, future_observed))

    def test_anchor_mean_normalization_preserves_order_and_masks_invalid_pairs(self) -> None:
        distances = torch.tensor([[[0.0], [2.0], [4.0]]])
        valid = torch.tensor([[[False], [True], [True]]])

        normalized = anchor_mean_normalize_distances(distances, valid)

        self.assertTrue(torch.allclose(normalized[0, :, 0], torch.tensor([0.0, 2.0 / 3.0, 4.0 / 3.0])))
        self.assertTrue(bool(torch.isfinite(normalized).all()))

    def test_offset_decay_teacher_uses_forecast_endpoint_not_context_statistics(self) -> None:
        future = torch.tensor(
            [
                [[1.0], [100.0]],
                [[11.0], [101.0]],
                [[13.0], [102.0]],
            ]
        ).unsqueeze(-1)
        forecast_context = torch.tensor([[[[0.0]]], [[[10.0]]], [[[20.0]]]])
        context_observed = torch.ones_like(forecast_context, dtype=torch.bool)
        observed = torch.ones_like(future, dtype=torch.bool)
        starts = torch.tensor([0, 20, 40])
        ends = starts + 1
        statistics = WindowStatistics(
            normalized=torch.zeros(3, 1, 1, 1),
            level_features=torch.zeros(3, 1, 4),
            level_valid=torch.ones(3, 1, 1, dtype=torch.bool),
            mean=torch.zeros(3, 1, 1),
            std=torch.ones(3, 1, 1),
        )

        targets = build_future_relation_targets(
            future,
            statistics,
            observed,
            starts,
            ends,
            teacher_temperature=0.1,
            relation_teacher_mode="offset_decay",
            forecast_context=forecast_context,
            forecast_context_observed=context_observed,
            relation_distance_normalization="anchor_mean",
        )

        self.assertGreater(
            float(targets.teacher_distribution[0, 1, 0]),
            float(targets.teacher_distribution[0, 2, 0]),
        )
        self.assertTrue(bool(torch.isfinite(targets.teacher_distribution).all()))

    def test_increment_teacher_adds_future_dynamics_relation(self) -> None:
        future = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 2.0],
                [2.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        ).view(3, 3, 1, 1)
        observed = torch.ones_like(future, dtype=torch.bool)
        forecast_context = torch.zeros(3, 2, 1, 1)
        context_observed = torch.ones_like(forecast_context, dtype=torch.bool)
        starts = torch.tensor([0, 20, 40])
        ends = starts + 2
        statistics = WindowStatistics(
            normalized=torch.zeros(3, 2, 1, 1),
            level_features=torch.zeros(3, 1, 4),
            level_valid=torch.ones(3, 1, 1, dtype=torch.bool),
            mean=torch.zeros(3, 1, 1),
            std=torch.ones(3, 1, 1),
        )

        offset_targets = build_future_relation_targets(
            future,
            statistics,
            observed,
            starts,
            ends,
            relation_teacher_mode="offset_decay",
            forecast_context=forecast_context,
            forecast_context_observed=context_observed,
            relation_distance_normalization="anchor_mean",
        )
        increment_targets = build_future_relation_targets(
            future,
            statistics,
            observed,
            starts,
            ends,
            relation_teacher_mode="offset_decay_increment",
            forecast_context=forecast_context,
            forecast_context_observed=context_observed,
            relation_distance_normalization="anchor_mean",
            future_increment_weight=0.5,
        )

        self.assertAlmostEqual(
            float(offset_targets.teacher_distribution[0, 1, 0]),
            float(offset_targets.teacher_distribution[0, 2, 0]),
            places=6,
        )
        self.assertGreater(
            float(increment_targets.teacher_distribution[0, 1, 0]),
            float(increment_targets.teacher_distribution[0, 2, 0]),
        )
        self.assertTrue(bool(torch.isfinite(increment_targets.future_distance).all()))

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

    def test_hn_offset_decay_reports_effective_support(self) -> None:
        keys = torch.randn(self.batch, 1, 3, requires_grad=True)
        context_normalized = torch.zeros(self.batch, 3, 1, 1)
        context_observed = torch.ones_like(context_normalized, dtype=torch.bool)
        forecast_context = torch.zeros_like(context_normalized)
        result = offset_decay_hard_negative_retrieval_loss(
            node_keys=keys,
            context_normalized=context_normalized,
            future_model=self.future,
            context_statistics=self.statistics,
            context_observed=context_observed,
            future_observed=self.future_observed,
            context_start=self.context_start,
            future_end=self.future_end,
            teacher_temperature=0.1,
            student_temperature=0.1,
            relation_teacher_mode="offset_decay",
            forecast_context=forecast_context,
            forecast_context_observed=context_observed,
            relation_distance_normalization="anchor_mean",
        )

        self.assertGreater(result.candidate_pairs, 0)
        self.assertGreater(result.valid_anchors, 0)
        self.assertTrue(math.isfinite(result.teacher_effective_support))
        self.assertTrue(math.isfinite(result.student_effective_support))
        self.assertGreater(result.teacher_effective_support, 1.0)
        self.assertGreater(result.student_effective_support, 1.0)
        self.assertTrue(bool(torch.isfinite(result.loss)))
        result.loss.backward()
        self.assertIsNotNone(keys.grad)
        self.assertTrue(bool(torch.isfinite(keys.grad).all()))

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
