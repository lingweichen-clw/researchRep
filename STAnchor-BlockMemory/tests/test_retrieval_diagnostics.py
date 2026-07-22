from __future__ import annotations

import unittest

import torch

from stanchor.diagnostics.retrieval import (
    effective_support_size,
    masked_candidate_mean,
    raw_l1_candidate_scores,
    select_candidate_set_by_node,
    select_oracle_future,
)


class RetrievalDiagnosticsTest(unittest.TestCase):
    def test_effective_support_size_matches_uniform_candidate_count(self) -> None:
        weights = torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0],
                    [0.5, 0.5, 0.0],
                    [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                ]
            ]
        )
        valid = weights > 0

        result = effective_support_size(weights, valid)

        self.assertTrue(torch.allclose(result, torch.tensor([[1.0, 2.0, 3.0]])))

    def test_masked_candidate_mean_ignores_missing_future_values(self) -> None:
        # candidates: [B, H, N, R, C]
        candidates = torch.tensor([[[[[1.0], [3.0], [100.0]]], [[[2.0], [4.0], [200.0]]]]])
        valid = torch.tensor([[[[[True], [True], [False]]], [[[True], [False], [False]]]]])

        prediction, prediction_valid = masked_candidate_mean(candidates, valid)

        self.assertTrue(torch.equal(prediction_valid, torch.ones_like(prediction_valid)))
        self.assertTrue(torch.allclose(prediction.flatten(), torch.tensor([2.0, 2.0])))

    def test_raw_l1_scores_compare_the_same_node_across_events(self) -> None:
        query = torch.tensor([[[[0.0], [10.0]], [[1.0], [12.0]]]])
        query_observed = torch.ones_like(query, dtype=torch.bool)
        candidates = torch.tensor(
            [
                [
                    [[[0.0], [20.0]], [[1.0], [20.0]]],
                    [[[2.0], [10.0]], [[3.0], [12.0]]],
                ]
            ]
        )
        candidate_observed = torch.ones_like(candidates, dtype=torch.bool)
        event_valid = torch.tensor([[True, True]])

        scores, valid = raw_l1_candidate_scores(
            query,
            query_observed,
            candidates,
            candidate_observed,
            event_valid,
        )

        self.assertTrue(bool(valid.all()))
        self.assertTrue(torch.allclose(scores[0, 0], torch.tensor([0.0, 2.0])))
        self.assertTrue(torch.allclose(scores[0, 1], torch.tensor([9.0, 0.0])))

    def test_oracle_selects_one_candidate_per_node_using_full_future(self) -> None:
        candidates = torch.tensor(
            [
                [
                    [[[0.0], [2.0]], [[10.0], [13.0]]],
                    [[[0.0], [4.0]], [[10.0], [15.0]]],
                ]
            ]
        )
        candidate_valid = torch.ones_like(candidates, dtype=torch.bool)
        target = torch.tensor([[[[1.0], [13.0]], [[3.0], [15.0]]]])
        target_observed = torch.ones_like(target, dtype=torch.bool)
        event_valid = torch.tensor([[True, True]])

        prediction, prediction_valid, selected = select_oracle_future(
            candidates,
            candidate_valid,
            target,
            target_observed,
            event_valid,
        )

        self.assertTrue(torch.equal(selected, torch.tensor([[1, 1]])))
        self.assertTrue(bool(prediction_valid.all()))
        self.assertTrue(
            torch.allclose(
                prediction,
                torch.tensor([[[[2.0], [13.0]], [[4.0], [15.0]]]]),
            )
        )

    def test_select_candidate_set_uses_independent_node_indices(self) -> None:
        candidates = torch.tensor(
            [
                [
                    [[[1.0], [2.0], [3.0]], [[10.0], [20.0], [30.0]]],
                ]
            ]
        )
        candidate_valid = torch.ones_like(candidates, dtype=torch.bool)
        selected = torch.tensor([[[2, 0], [1, 2]]])
        selected_valid = torch.ones_like(selected, dtype=torch.bool)

        values, valid = select_candidate_set_by_node(
            candidates,
            candidate_valid,
            selected,
            selected_valid,
        )

        self.assertTrue(bool(valid.all()))
        self.assertTrue(
            torch.allclose(
                values,
                torch.tensor([[[[[3.0], [1.0]], [[20.0], [30.0]]]]]),
            )
        )


if __name__ == "__main__":
    unittest.main()
