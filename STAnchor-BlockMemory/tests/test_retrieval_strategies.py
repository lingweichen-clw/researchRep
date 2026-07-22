from __future__ import annotations

import unittest

import numpy as np
import torch

from stanchor.retrieval.strategies import (
    calendar_event_candidates,
    uniform_candidate_aggregation,
)


class RetrievalStrategiesTest(unittest.TestCase):
    def test_calendar_candidates_keep_all_and_only_causal_events(self) -> None:
        class Calendar:
            @staticmethod
            def lookup(weekday: int, slot: int) -> np.ndarray:
                self.assertEqual((weekday, slot), (2, 10))
                return np.asarray([0, 1, 2], dtype=np.int64)

        class Bank:
            calendar = Calendar()
            future_end = np.asarray([5, 15, 25], dtype=np.int64)

        result = calendar_event_candidates(
            Bank(),
            weekday=torch.tensor([2]),
            slot=torch.tensor([10]),
            context_start=torch.tensor([20]),
            max_candidates=4,
            device=torch.device("cpu"),
        )

        self.assertTrue(torch.equal(result.event_ids, torch.tensor([[0, 1, -1, -1]])))
        self.assertTrue(torch.equal(result.valid, torch.tensor([[True, True, False, False]])))

    def test_calendar_candidates_reject_silent_pool_truncation(self) -> None:
        class Calendar:
            @staticmethod
            def lookup(weekday: int, slot: int) -> np.ndarray:
                return np.asarray([0, 1], dtype=np.int64)

        class Bank:
            calendar = Calendar()
            future_end = np.asarray([1, 2], dtype=np.int64)

        with self.assertRaisesRegex(ValueError, "event_top_r"):
            calendar_event_candidates(
                Bank(),
                weekday=torch.tensor([0]),
                slot=torch.tensor([0]),
                context_start=torch.tensor([10]),
                max_candidates=1,
                device=torch.device("cpu"),
            )

    def test_uniform_aggregation_ignores_missing_candidate_and_computes_variance(self) -> None:
        # [B, H, N, K, C]
        candidates = torch.tensor([[[[[1.0], [3.0], [100.0]]]]])
        valid = torch.tensor([[[[[True], [True], [False]]]]])

        result = uniform_candidate_aggregation(candidates, valid)

        self.assertTrue(torch.allclose(result.prediction, torch.tensor([[[[2.0]]]])))
        self.assertTrue(torch.allclose(result.variance, torch.tensor([[[[1.0]]]])))
        self.assertTrue(bool(result.valid.all()))
        self.assertTrue(torch.equal(result.candidate_futures, candidates))
        self.assertTrue(torch.equal(result.candidate_masks, valid))


if __name__ == "__main__":
    unittest.main()
