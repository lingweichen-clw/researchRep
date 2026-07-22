from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from stanchor.bank.schema import BankManifest
from stanchor.bank.storage import BankWriter, MemoryBank
from stanchor.retrieval.retriever import TwoStageRetriever


class BankAndRetrievalTest(unittest.TestCase):
    def _build_bank(self, path: Path) -> MemoryBank:
        events, nodes, horizon, channels, dim = 6, 3, 2, 1, 2
        manifest = BankManifest(
            schema_version=1,
            dataset_name="synthetic",
            num_events=events,
            num_nodes=nodes,
            context_length=4,
            horizon=horizon,
            channels=channels,
            retrieval_dim=dim,
            slots_per_day=4,
            key_dtype="float32",
            future_dtype="float32",
            encoder_fingerprint="encoder",
            graph_fingerprint="graph",
            scaler={"mean": [[0.0]] * nodes, "std": [[1.0]] * nodes, "eps": 1.0e-6},
        )
        event_keys = np.array(
            [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0], [0.8, 0.2], [0.7, 0.3]],
            dtype=np.float32,
        )
        event_keys /= np.linalg.norm(event_keys, axis=1, keepdims=True)
        node_keys = np.repeat(event_keys[:, None, :], nodes, axis=1)
        future = np.zeros((events, horizon, nodes, channels), dtype=np.float32)
        for event in range(events):
            future[event] = event + 1
        writer = BankWriter(path, manifest)
        writer.write(
            {
                "event_keys": event_keys,
                "node_keys": node_keys,
                "future_values": future,
                "future_masks": np.ones_like(future, dtype=np.uint8),
                "level_features": np.zeros((events, nodes, 4), dtype=np.float32),
                "weekday": np.zeros(events, dtype=np.int16),
                "slot": np.zeros(events, dtype=np.int16),
                "context_start": np.arange(events, dtype=np.int64) * 10,
                "context_end": np.arange(events, dtype=np.int64) * 10 + 3,
                "future_end": np.arange(events, dtype=np.int64) * 10 + 5,
                "sample_id": np.arange(events, dtype=np.int64),
            }
        )
        writer.finalize()
        return MemoryBank(path)

    def test_storage_and_exact_retrieval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self._build_bank(Path(directory)) as bank:
                self.assertIsInstance(bank.node_keys, np.memmap)
                self.assertEqual(bank.calendar.lookup(0, 0).size, 6)
                retriever = TwoStageRetriever(
                    bank,
                    event_top_r=3,
                    node_top_k=2,
                    level_weight=0.0,
                    level_temperature=1.0,
                    search_temperature=0.1,
                    device=torch.device("cpu"),
                )
                query_event = torch.tensor([[1.0, 0.0]])
                query_node = query_event[:, None, :].expand(1, 3, 2)
                events, nodes, aggregation = retriever.retrieve(
                    query_event,
                    query_node,
                    torch.zeros((1, 3, 4)),
                    weekday=torch.tensor([0]),
                    slot=torch.tensor([0]),
                    context_start=torch.tensor([36]),
                )
                # Only events 0-3 have completed futures; event 0 is the closest.
                self.assertEqual(int(events.event_ids[0, 0]), 0)
                self.assertEqual(tuple(nodes.event_ids.shape), (1, 3, 2))
                self.assertEqual(tuple(aggregation.prediction.shape), (1, 2, 3, 1))
                self.assertTrue(bool(aggregation.valid.all()))
                self.assertTrue(bool((aggregation.prediction >= 1.0).all()))
                self.assertTrue(bool((aggregation.prediction <= 2.0).all()))

    def test_causal_filter_can_return_no_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self._build_bank(Path(directory)) as bank:
                retriever = TwoStageRetriever(bank, 3, 2, 0.0, 1.0, 0.1, torch.device("cpu"))
                events = retriever.search_events(
                    torch.tensor([[1.0, 0.0]]),
                    torch.tensor([0]),
                    torch.tensor([0]),
                    torch.tensor([0]),
                )
                self.assertFalse(bool(events.valid.any()))


if __name__ == "__main__":
    unittest.main()
