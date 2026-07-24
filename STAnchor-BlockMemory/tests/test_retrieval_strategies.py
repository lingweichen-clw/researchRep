from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from stanchor.retrieval.strategies import (
    calendar_event_candidates,
    candidate_contexts,
    uniform_candidate_aggregation,
)


class RetrievalStrategiesTest(unittest.TestCase):
    def test_candidate_contexts_use_forecast_tail_of_long_retrieval_window(self) -> None:
        values = np.arange(30, dtype=np.float32).reshape(30, 1, 1)
        observed = np.ones_like(values, dtype=bool)
        series = SimpleNamespace(values=values, observed=observed)
        scaler = SimpleNamespace(
            mean=np.zeros((1, 1), dtype=np.float32),
            std=np.ones((1, 1), dtype=np.float32),
            eps=1.0e-6,
        )
        bank = SimpleNamespace(
            context_start=np.asarray([0], dtype=np.int64),
            context_end=np.asarray([23], dtype=np.int64),
        )

        contexts, context_observed = candidate_contexts(
            bank,
            event_ids=torch.tensor([[0]]),
            series=series,
            scaler=scaler,
            context_length=12,
            device=torch.device("cpu"),
        )

        expected = torch.arange(12, 24, dtype=torch.float32).view(1, 1, 12, 1, 1)
        self.assertTrue(torch.allclose(contexts, expected, atol=1.0e-5))
        self.assertTrue(bool(context_observed.all()))

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
