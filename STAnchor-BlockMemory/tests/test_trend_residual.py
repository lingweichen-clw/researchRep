from __future__ import annotations

import unittest

import torch

from stanchor.retrieval.trend_residual import (
    estimate_local_trend,
    masked_pearson_candidate_scores,
    masked_spearman_rank_correlation,
    match_selected_event_positions,
    reconstruct_future,
    residualize_future,
    softmax_topk_weights,
    weighted_candidate_mean,
)


class TrendResidualTest(unittest.TestCase):
    def test_trend_residual_reconstructs_query_local_future(self) -> None:
        candidate_context = torch.tensor([[[[10.0]], [[12.0]], [[14.0]]]])
        candidate_future = torch.tensor([[[[17.0]], [[19.0]]]])
        query_context = torch.tensor([[[[100.0]], [[103.0]], [[106.0]]]])
        context_observed = torch.ones_like(candidate_context, dtype=torch.bool)
        future_observed = torch.ones_like(candidate_future, dtype=torch.bool)

        candidate_stats = estimate_local_trend(
            candidate_context,
            context_observed,
            trend_length=3,
        )
        query_stats = estimate_local_trend(
            query_context,
            torch.ones_like(query_context, dtype=torch.bool),
            trend_length=3,
        )
        residual, valid = residualize_future(
            candidate_future,
            future_observed,
            candidate_stats,
        )
        reconstructed, reconstructed_valid = reconstruct_future(
            residual,
            valid,
            query_stats,
        )

        self.assertTrue(
            torch.allclose(
                reconstructed.flatten(),
                torch.tensor([110.5, 113.5]),
                atol=1.0e-5,
            )
        )
        self.assertTrue(bool(reconstructed_valid.all()))

    def test_residual_is_invariant_to_positive_affine_transform(self) -> None:
        context = torch.tensor([[[[2.0]], [[3.0]], [[5.0]], [[8.0]]]])
        future = torch.tensor([[[[10.0]], [[13.0]]]])
        context_observed = torch.ones_like(context, dtype=torch.bool)
        future_observed = torch.ones_like(future, dtype=torch.bool)

        stats = estimate_local_trend(context, context_observed, trend_length=4)
        residual, valid = residualize_future(future, future_observed, stats)

        transformed_context = 7.0 * context + 11.0
        transformed_future = 7.0 * future + 11.0
        transformed_stats = estimate_local_trend(
            transformed_context,
            context_observed,
            trend_length=4,
        )
        transformed_residual, transformed_valid = residualize_future(
            transformed_future,
            future_observed,
            transformed_stats,
        )

        self.assertTrue(torch.equal(valid, transformed_valid))
        self.assertTrue(torch.allclose(residual, transformed_residual, atol=1.0e-5))

    def test_missing_endpoint_uses_fitted_endpoint_without_nan(self) -> None:
        context = torch.tensor([[[[1.0]], [[3.0]], [[0.0]]]])
        observed = torch.tensor([[[[True]], [[True]], [[False]]]])
        future = torch.tensor([[[[7.0]]]])

        stats = estimate_local_trend(context, observed, trend_length=3)
        residual, valid = residualize_future(
            future,
            torch.ones_like(future, dtype=torch.bool),
            stats,
        )

        self.assertAlmostEqual(float(stats.level.item()), 5.0, places=5)
        self.assertAlmostEqual(float(stats.slope.item()), 2.0, places=5)
        self.assertTrue(bool(valid.all()))
        self.assertTrue(bool(torch.isfinite(residual).all()))
        self.assertTrue(torch.allclose(residual, torch.zeros_like(residual), atol=1.0e-5))

    def test_constant_context_produces_finite_residual(self) -> None:
        context = torch.full((1, 4, 1, 1), 5.0)
        future = torch.full((1, 2, 1, 1), 5.0)
        observed = torch.ones_like(context, dtype=torch.bool)

        stats = estimate_local_trend(context, observed, trend_length=4)
        residual, valid = residualize_future(
            future,
            torch.ones_like(future, dtype=torch.bool),
            stats,
        )

        self.assertTrue(bool(valid.all()))
        self.assertGreater(float(stats.scale.item()), 0.0)
        self.assertTrue(bool(torch.isfinite(residual).all()))
        self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))

    def test_masked_pearson_prefers_matching_history_shape(self) -> None:
        query = torch.tensor([[[[0.0]], [[1.0]], [[0.0]]]])
        query_observed = torch.ones_like(query, dtype=torch.bool)
        candidates = torch.tensor(
            [
                [
                    [[[0.0]], [[1.0]], [[0.0]]],
                    [[[0.0]], [[-1.0]], [[0.0]]],
                ]
            ]
        )
        candidate_observed = torch.ones_like(candidates, dtype=torch.bool)

        scores, valid = masked_pearson_candidate_scores(
            query,
            query_observed,
            candidates,
            candidate_observed,
            torch.tensor([[True, True]]),
        )

        self.assertEqual(tuple(scores.shape), (1, 1, 2))
        self.assertTrue(bool(valid.all()))
        self.assertGreater(float(scores[0, 0, 0]), float(scores[0, 0, 1]))
        self.assertTrue(torch.allclose(scores[0, 0], torch.tensor([1.0, -1.0]), atol=1.0e-5))

    def test_match_selected_event_positions_maps_global_ids_to_pool_axis(self) -> None:
        event_ids = torch.tensor([[10, 20, 30, -1]])
        selected_event_ids = torch.tensor([[[30, 10], [20, 30]]])
        selected_valid = torch.tensor([[[True, True], [True, False]]])

        positions, valid = match_selected_event_positions(
            event_ids,
            selected_event_ids,
            selected_valid,
        )

        self.assertTrue(torch.equal(positions, torch.tensor([[[2, 0], [1, 2]]])))
        self.assertTrue(torch.equal(valid, selected_valid))

    def test_weighted_candidate_mean_renormalizes_per_horizon_mask(self) -> None:
        candidates = torch.tensor(
            [
                [
                    [[[1.0], [3.0]]],
                ],
                [
                    [[[2.0], [100.0]]],
                ],
            ]
        ).permute(1, 0, 2, 3, 4)
        valid = torch.tensor(
            [
                [
                    [[[True], [True]]],
                ],
                [
                    [[[True], [False]]],
                ],
            ]
        ).permute(1, 0, 2, 3, 4)
        weights = torch.tensor([[[0.25, 0.75]]])

        prediction, prediction_valid = weighted_candidate_mean(candidates, valid, weights)

        self.assertTrue(bool(prediction_valid.all()))
        self.assertTrue(torch.allclose(prediction.flatten(), torch.tensor([2.5, 2.0])))

    def test_softmax_topk_weights_selects_only_valid_candidates(self) -> None:
        scores = torch.tensor([[[0.9, 0.8, -10.0]]])
        valid = torch.tensor([[[True, True, False]]])

        selected, selected_valid, weights = softmax_topk_weights(
            scores,
            valid,
            top_k=2,
            temperature=0.1,
            largest=True,
        )

        self.assertTrue(torch.equal(selected, torch.tensor([[[0, 1]]])))
        self.assertTrue(bool(selected_valid.all()))
        self.assertAlmostEqual(float(weights.sum().item()), 1.0, places=6)
        self.assertGreater(float(weights[0, 0, 0]), float(weights[0, 0, 1]))

    def test_spearman_detects_consistent_and_reversed_rankings(self) -> None:
        first = torch.tensor([[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]])
        second = torch.tensor([[[10.0, 20.0, 30.0], [30.0, 20.0, 10.0]]])
        valid = torch.ones_like(first, dtype=torch.bool)

        correlation, correlation_valid = masked_spearman_rank_correlation(
            first,
            second,
            valid,
        )

        self.assertTrue(bool(correlation_valid.all()))
        self.assertTrue(torch.allclose(correlation, torch.tensor([[1.0, -1.0]]), atol=1.0e-5))


if __name__ == "__main__":
    unittest.main()
