from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from stanchor.bank.storage import CalendarIndex
from stanchor.retrieval.retriever import NodeCandidates
from stanchor.retrieval.strategies import (
    candidate_contexts_for_nodes,
    event_candidate_futures_for_nodes,
)


class RetrievalAccelerationTest(unittest.TestCase):
    def test_vectorized_exact_search_preserves_reference_order_and_padding(self) -> None:
        event_keys = np.asarray(
            [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [0.6, 0.8]],
            dtype=np.float32,
        )
        calendar = CalendarIndex.build(
            weekday=np.asarray([0, 0, 0, 1]),
            slot=np.asarray([2, 2, 2, 2]),
            slots_per_day=4,
        )
        bank = SimpleNamespace(
            event_keys_memory=event_keys,
            future_end=np.asarray([5, 15, 25, 35], dtype=np.int64),
            calendar=calendar,
            manifest=SimpleNamespace(slots_per_day=4),
        )
        retriever = __import__(
            "stanchor.retrieval.retriever", fromlist=["TwoStageRetriever"]
        ).TwoStageRetriever(
            bank,
            event_top_r=4,
            node_top_k=1,
            level_weight=0.0,
            level_temperature=1.0,
            search_temperature=0.1,
            device=torch.device("cpu"),
        )
        queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        weekdays = torch.tensor([0, 1])
        slots = torch.tensor([2, 2])
        context_start = torch.tensor([30, 100])
        actual = retriever.search_events(queries, weekdays, slots, context_start)

        expected_ids = torch.full((2, 4), -1, dtype=torch.long)
        expected_scores = torch.full((2, 4), -torch.inf)
        expected_valid = torch.zeros((2, 4), dtype=torch.bool)
        for row in range(2):
            legal = calendar.lookup(int(weekdays[row]), int(slots[row]))
            legal = legal[bank.future_end[legal] < int(context_start[row])]
            if legal.size:
                score = queries[row] @ torch.from_numpy(event_keys[legal]).T
                count = min(4, legal.size)
                top_scores, local = torch.topk(score, count)
                expected_ids[row, :count] = torch.from_numpy(legal[local.numpy()])
                expected_scores[row, :count] = top_scores
                expected_valid[row, :count] = True
        self.assertTrue(torch.equal(actual.event_ids, expected_ids))
        self.assertTrue(torch.allclose(actual.scores, expected_scores, equal_nan=True))
        self.assertTrue(torch.equal(actual.valid, expected_valid))

    def test_target_node_reads_return_reference_values(self) -> None:
        """Direct target-node reads match the legacy all-node gather values."""
        values = np.arange(40, dtype=np.float32).reshape(10, 4, 1)
        observed = np.ones_like(values, dtype=bool)
        bank = SimpleNamespace(
            context_end=np.asarray([5, 7], dtype=np.int64),
            future_values=np.arange(2 * 2 * 4, dtype=np.float32).reshape(2, 2, 4, 1),
            future_masks=np.ones((2, 2, 4, 1), dtype=np.uint8),
        )
        series = SimpleNamespace(values=values, observed=observed)
        scaler = SimpleNamespace(
            mean=np.zeros((4, 1), dtype=np.float32),
            std=np.ones((4, 1), dtype=np.float32),
            eps=1.0e-6,
        )
        event_ids = torch.tensor([[[0, 1], [1, 0]]], dtype=torch.long)
        valid = torch.tensor([[[True, True], [True, False]]])
        node_ids = torch.tensor([[[0, 2], [1, 3]]], dtype=torch.long)

        direct_context, direct_context_valid = candidate_contexts_for_nodes(
            bank, event_ids, node_ids, series, scaler, 3, torch.device("cpu")
        )
        direct_future, direct_future_valid = event_candidate_futures_for_nodes(
            bank, event_ids, valid, node_ids, torch.device("cpu")
        )
        safe_ids = event_ids.cpu().numpy().reshape(-1, 2)
        legacy_context = []
        legacy_future = []
        for row in safe_ids:
            ends = bank.context_end[row]
            indices = ends[:, None] - 3 + 1 + np.arange(3)[None, :]
            context = values[indices]
            legacy_context.append(context)
            legacy_future.append(bank.future_values[row])
        legacy_context = np.asarray(legacy_context).reshape(1, 2, 2, 3, 4, 1)
        legacy_future = np.asarray(legacy_future).reshape(1, 2, 2, 2, 4, 1)
        expected_context = np.stack(
            [legacy_context[0, n, k, :, node_ids[0, n, k], :] for n in range(2) for k in range(2)]
        ).reshape(1, 2, 2, 3, 1).transpose(0, 1, 3, 2, 4)
        expected_future = np.stack(
            [legacy_future[0, n, k, :, node_ids[0, n, k], :] for n in range(2) for k in range(2)]
        ).reshape(1, 2, 2, 2, 1).transpose(0, 3, 1, 2, 4)
        self.assertTrue(torch.allclose(direct_context, torch.from_numpy(expected_context)))
        self.assertTrue(torch.equal(direct_context_valid, torch.ones_like(direct_context_valid)))
        self.assertTrue(torch.equal(direct_future, torch.from_numpy(expected_future)))
        self.assertTrue(torch.equal(direct_future_valid, valid[:, None, :, :, None].expand_as(direct_future_valid)))

    def test_target_node_reads_return_only_requested_nodes(self) -> None:
        values = np.arange(40, dtype=np.float32).reshape(10, 4, 1)
        observed = np.ones_like(values, dtype=bool)
        bank = SimpleNamespace(
            context_end=np.asarray([5, 7], dtype=np.int64),
            future_values=np.arange(2 * 2 * 4, dtype=np.float32).reshape(2, 2, 4, 1),
            future_masks=np.ones((2, 2, 4, 1), dtype=np.uint8),
        )
        series = SimpleNamespace(values=values, observed=observed)
        scaler = SimpleNamespace(
            mean=np.zeros((4, 1), dtype=np.float32),
            std=np.ones((4, 1), dtype=np.float32),
            eps=1.0e-6,
        )
        event_ids = torch.tensor([[[0, 1], [1, 0]]], dtype=torch.long)
        valid = torch.tensor([[[True, True], [True, False]]])
        node_ids = torch.tensor([[[0, 2], [1, 3]]], dtype=torch.long)

        contexts, context_valid = candidate_contexts_for_nodes(
            bank,
            event_ids,
            node_ids,
            series,
            scaler,
            context_length=3,
            device=torch.device("cpu"),
        )
        futures, future_valid = event_candidate_futures_for_nodes(
            bank,
            event_ids,
            valid,
            node_ids,
            device=torch.device("cpu"),
        )

        self.assertEqual(tuple(contexts.shape), (1, 2, 3, 2, 1))
        self.assertEqual(tuple(futures.shape), (1, 2, 2, 2, 1))
        self.assertTrue(bool(context_valid.all()))
        self.assertTrue(
            torch.equal(
                future_valid,
                valid[:, None, :, :, None].expand_as(future_valid),
            )
        )
        self.assertTrue(torch.allclose(contexts[0, 0, :, 0, 0], torch.tensor([12.0, 16.0, 20.0])))
        self.assertTrue(torch.allclose(contexts[0, 1, :, 0, 0], torch.tensor([21.0, 25.0, 29.0])))
        self.assertTrue(torch.equal(futures[0, :, 0, 0, 0], torch.tensor([0.0, 4.0])))


if __name__ == "__main__":
    unittest.main()
