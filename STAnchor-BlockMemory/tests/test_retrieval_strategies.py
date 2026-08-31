from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from stanchor.retrieval.strategies import (
    calendar_event_candidates,
    candidate_contexts,
    offset_decay_aggregation,
    uniform_candidate_aggregation,
)
from stanchor.retrieval.retriever import NodeCandidates


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

    def test_relaxed_calendar_candidates_include_adjacent_slots_without_duplicates(self) -> None:
        class Calendar:
            @staticmethod
            def lookup(weekday: int, slot: int) -> np.ndarray:
                self.assertEqual(weekday, 2)
                return {
                    9: np.asarray([0, 1], dtype=np.int64),
                    10: np.asarray([1, 2], dtype=np.int64),
                    11: np.asarray([3], dtype=np.int64),
                }[slot]

        class Bank:
            calendar = Calendar()
            future_end = np.asarray([5, 6, 7, 25], dtype=np.int64)
            manifest = SimpleNamespace(slots_per_day=288)

        result = calendar_event_candidates(
            Bank(),
            weekday=torch.tensor([2]),
            slot=torch.tensor([10]),
            context_start=torch.tensor([20]),
            max_candidates=5,
            device=torch.device("cpu"),
            candidate_protocol="relaxed_calendar",
        )

        self.assertTrue(torch.equal(result.event_ids, torch.tensor([[0, 1, 2, -1, -1]])))
        self.assertTrue(torch.equal(result.valid, torch.tensor([[True, True, True, False, False]])))

    def test_relaxed_calendar_diverse_removes_overlapping_windows(self) -> None:
        class Calendar:
            @staticmethod
            def lookup(weekday: int, slot: int) -> np.ndarray:
                return {
                    9: np.asarray([0, 1], dtype=np.int64),
                    10: np.asarray([2], dtype=np.int64),
                    11: np.asarray([3], dtype=np.int64),
                }[slot]

        class Bank:
            calendar = Calendar()
            future_end = np.asarray([10, 20, 30, 40], dtype=np.int64)
            context_end = np.asarray([100, 110, 450, 760], dtype=np.int64)
            manifest = SimpleNamespace(
                slots_per_day=288,
                context_length=288,
                horizon=12,
            )

        result = calendar_event_candidates(
            Bank(),
            weekday=torch.tensor([2]),
            slot=torch.tensor([10]),
            context_start=torch.tensor([1000]),
            max_candidates=5,
            device=torch.device("cpu"),
            candidate_protocol="relaxed_calendar_diverse",
        )

        self.assertTrue(torch.equal(result.event_ids, torch.tensor([[2, 0, 3, -1, -1]])))
        self.assertTrue(torch.equal(result.valid, torch.tensor([[True, True, True, False, False]])))

    def test_weekday_radius1_overlap_uses_adjacent_weekdays_and_context_end_boundary(self) -> None:
        class Calendar:
            @staticmethod
            def lookup(weekday: int, slot: int) -> np.ndarray:
                return {
                    (1, 84): np.asarray([0], dtype=np.int64),
                    (2, 84): np.asarray([1], dtype=np.int64),
                    (3, 84): np.asarray([2, 3], dtype=np.int64),
                }.get((weekday, slot), np.asarray([], dtype=np.int64))

        class Bank:
            calendar = Calendar()
            future_end = np.asarray([90, 100, 110, 130], dtype=np.int64)
            context_end = np.asarray([78, 88, 98, 118], dtype=np.int64)
            manifest = SimpleNamespace(slots_per_day=288, context_length=21)

        result = calendar_event_candidates(
            Bank(),
            weekday=torch.tensor([2]),
            slot=torch.tensor([84]),
            context_start=torch.tensor([90]),
            max_candidates=8,
            device=torch.device("cpu"),
            candidate_protocol="weekday_radius1_overlap",
        )

        self.assertTrue(torch.equal(result.event_ids, torch.tensor([[0, 1, 2, -1, -1, -1, -1, -1]])))
        self.assertTrue(torch.equal(result.valid, torch.tensor([[True, True, True, False, False, False, False, False]])))
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

    def test_offset_decay_aggregation_aligns_near_horizon_and_returns_to_raw(self) -> None:
        values = np.concatenate(
            (
                np.full((12, 1, 1), 2.0, dtype=np.float32),
                np.full((12, 1, 1), 12.0, dtype=np.float32),
            ),
            axis=0,
        )
        observed = np.ones_like(values, dtype=bool)
        series = SimpleNamespace(values=values, observed=observed)
        scaler = SimpleNamespace(
            mean=np.zeros((1, 1), dtype=np.float32),
            std=np.ones((1, 1), dtype=np.float32),
            eps=1.0e-6,
        )
        bank = SimpleNamespace(
            context_end=np.asarray([11, 23], dtype=np.int64),
            future_values=np.asarray(
                [
                    [[[3.0]], [[4.0]]],
                    [[[13.0]], [[14.0]]],
                ],
                dtype=np.float32,
            ),
            future_masks=np.ones((2, 2, 1, 1), dtype=np.uint8),
        )
        candidates = NodeCandidates(
            event_ids=torch.tensor([[[0, 1]]]),
            total_scores=torch.ones(1, 1, 2),
            shape_scores=torch.ones(1, 1, 2),
            level_distances=torch.zeros(1, 1, 2),
            weights=torch.tensor([[[0.5, 0.5]]]),
            valid=torch.ones(1, 1, 2, dtype=torch.bool),
        )
        query = torch.full((1, 12, 1, 1), 10.0)
        query_observed = torch.ones_like(query, dtype=torch.bool)

        result = offset_decay_aggregation(
            candidates,
            query,
            query_observed,
            bank,
            series,
            scaler,
            context_length=12,
            device=torch.device("cpu"),
        )

        expected = torch.tensor([[[[11.0]], [[9.0]]]])
        self.assertTrue(torch.allclose(result.prediction, expected, atol=1.0e-6))
        self.assertTrue(bool(result.valid.all()))
        self.assertTrue(torch.allclose(result.candidate_futures[:, 0], torch.tensor([[[[11.0], [11.0]]]])))

    def test_offset_decay_aggregation_has_exact_empty_candidate_fallback_mask(self) -> None:
        values = np.full((12, 1, 1), 2.0, dtype=np.float32)
        series = SimpleNamespace(values=values, observed=np.ones_like(values, dtype=bool))
        scaler = SimpleNamespace(
            mean=np.zeros((1, 1), dtype=np.float32),
            std=np.ones((1, 1), dtype=np.float32),
            eps=1.0e-6,
        )
        bank = SimpleNamespace(
            context_end=np.asarray([11], dtype=np.int64),
            future_values=np.asarray([[[[3.0]], [[4.0]]]], dtype=np.float32),
            future_masks=np.ones((1, 2, 1, 1), dtype=np.uint8),
        )
        candidates = NodeCandidates(
            event_ids=torch.tensor([[[0]]]),
            total_scores=torch.full((1, 1, 1), -torch.inf),
            shape_scores=torch.zeros(1, 1, 1),
            level_distances=torch.zeros(1, 1, 1),
            weights=torch.zeros(1, 1, 1),
            valid=torch.zeros(1, 1, 1, dtype=torch.bool),
        )
        query = torch.full((1, 12, 1, 1), 10.0)

        result = offset_decay_aggregation(
            candidates,
            query,
            torch.ones_like(query, dtype=torch.bool),
            bank,
            series,
            scaler,
            context_length=12,
            device=torch.device("cpu"),
        )

        self.assertTrue(torch.equal(result.prediction, torch.zeros_like(result.prediction)))
        self.assertFalse(bool(result.valid.any()))


if __name__ == "__main__":
    unittest.main()
