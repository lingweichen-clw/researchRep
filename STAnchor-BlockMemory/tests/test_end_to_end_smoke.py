from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from stanchor.bank.builder import build_memory_bank
from stanchor.bank.storage import MemoryBank
from stanchor.config import ModelConfig, PretrainConfig
from stanchor.data.dataset import TrafficSeries, TrafficWindowDataset
from stanchor.data.graph import graph_from_dense
from stanchor.data.normalization import NodeStandardScaler
from stanchor.models.downstream import (
    ConfidenceHead,
    LightweightForecastBackbone,
    SafeResidualFusion,
    STAnchorDownstreamModel,
)
from stanchor.models.pretraining import STAnchorPretrainModel
from stanchor.retrieval.retriever import TwoStageRetriever


class EndToEndSmokeTest(unittest.TestCase):
    def test_encode_build_retrieve_and_fuse(self) -> None:
        torch.manual_seed(11)
        length, nodes, context, horizon = 100, 6, 12, 4
        time = np.arange(length, dtype=np.float32)[:, None, None]
        node = np.arange(nodes, dtype=np.float32)[None, :, None]
        values = np.sin(time / 6.0) + node / 10.0 + 5.0
        observed = np.ones_like(values, dtype=bool)
        series = TrafficSeries(
            values=values.astype(np.float32),
            observed=observed,
            timestamps_ns=np.arange(length, dtype=np.int64),
            weekday=np.zeros(length, dtype=np.int64),
            slot=np.arange(length, dtype=np.int64) % 4,
            slots_per_day=4,
        )
        scaler = NodeStandardScaler.fit(values[:70], observed[:70])
        dataset = TrafficWindowDataset(series, scaler, 0, 100, context, horizon)
        adjacency = np.zeros((nodes, nodes), dtype=np.float32)
        for index in range(nodes):
            adjacency[index, (index - 1) % nodes] = 1.0
            adjacency[index, (index + 1) % nodes] = 1.0
        graph = graph_from_dense(adjacency)
        model = STAnchorPretrainModel(
            ModelConfig(
                patch_size=3,
                hidden_dim=16,
                retrieval_dim=8,
                num_heads=4,
                encoder_layers=1,
                dropout=0.0,
            ),
            PretrainConfig(),
            context_length=context,
            slots_per_day=4,
        )
        with tempfile.TemporaryDirectory() as directory:
            build_memory_bank(
                model,
                Subset(dataset, range(20)),
                graph,
                scaler,
                Path(directory),
                "synthetic",
                batch_size=5,
                num_workers=0,
                device=torch.device("cpu"),
                key_dtype="float32",
            )
            with MemoryBank(directory) as bank:
                retriever = TwoStageRetriever(bank, 8, 3, 0.1, 1.0, 0.2, torch.device("cpu"))
                # Index 40 is causally later than every bank event and has a calendar match.
                query = next(iter(DataLoader(Subset(dataset, [40]), batch_size=1)))
                encoding = model.encode_clean(
                    query["x"], query["x_observed"], query["weekday"], query["slot"], graph
                )
                _, candidates, aggregation = retriever.retrieve(
                    encoding.retrieval.event_keys,
                    encoding.retrieval.node_keys,
                    encoding.statistics.level_features,
                    query["query_weekday"],
                    query["query_slot"],
                    query["context_start"],
                )
                downstream = STAnchorDownstreamModel(
                    LightweightForecastBackbone(context, horizon, 1, 1, 16, dropout=0.0),
                    ConfidenceHead(8),
                    SafeResidualFusion(horizon),
                    confidence_level_temperature=1.0,
                )
                output = downstream(query["x"], candidates, aggregation)
                self.assertEqual(tuple(output.final_prediction.shape), (1, horizon, nodes, 1))
                self.assertTrue(bool(output.memory_valid.any()))
                self.assertTrue(bool(torch.isfinite(output.final_prediction).all()))


if __name__ == "__main__":
    unittest.main()

