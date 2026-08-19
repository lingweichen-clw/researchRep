from __future__ import annotations

import unittest

import numpy as np
import torch

from stanchor.data.graph import graph_from_dense
from stanchor.models.graph_wavenet import GraphWaveNetForecastBackbone
from stanchor.config import DataConfig, ExperimentConfig, TargetConfig
from stanchor.engine.target import build_downstream_model


class GraphWaveNetBackboneTest(unittest.TestCase):
    def test_forward_uses_project_layout_and_has_gradients(self) -> None:
        graph = graph_from_dense(
            np.asarray(
                [[0.0, 1.0, 0.2], [0.5, 0.0, 1.0], [0.2, 0.4, 0.0]],
                dtype=np.float32,
            )
        )
        model = GraphWaveNetForecastBackbone(
            context_length=12,
            horizon=4,
            input_channels=1,
            output_channels=1,
            graph=graph,
            residual_channels=8,
            dilation_channels=8,
            skip_channels=16,
            end_channels=32,
            blocks=1,
            layers=2,
            dropout=0.0,
        )
        output = model(torch.randn(2, 12, 3, 1))
        self.assertEqual(tuple(output.shape), (2, 4, 3, 1))
        self.assertTrue(bool(torch.isfinite(output).all()))
        output.square().mean().backward()
        self.assertIsNotNone(model.network.start_conv.weight.grad)

    def test_downstream_factory_builds_graph_wavenet(self) -> None:
        graph = graph_from_dense(np.eye(3, dtype=np.float32))
        config = ExperimentConfig(
            data=DataConfig(raw_path="data.h5", adjacency_path="adj.pkl"),
            target=TargetConfig(backbone_name="graph_wavenet"),
        )
        model = build_downstream_model(config, graph)
        self.assertIsInstance(model.backbone, GraphWaveNetForecastBackbone)


if __name__ == "__main__":
    unittest.main()
