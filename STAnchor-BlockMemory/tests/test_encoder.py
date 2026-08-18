from __future__ import annotations

import unittest

import numpy as np
import torch

from stanchor.data.graph import graph_from_dense
from stanchor.models.encoder import FactorizedSTEncoder, MixedRangeRouteAttention


def _route_graph() -> object:
    nodes = 16
    adjacency = np.eye(nodes, dtype=np.float32)
    adjacency[0, 1:6] = 1.0
    adjacency[1:6, 0] = 1.0
    return graph_from_dense(adjacency)


class MixedRangeRouteAttentionTest(unittest.TestCase):
    def test_route_selects_four_first_order_and_six_remote_nodes(self) -> None:
        graph = _route_graph()
        module = MixedRangeRouteAttention(
            hidden_dim=8,
            route_dim=4,
            route_top_k=10,
            route_local_quota=4,
            prior_weight=0.25,
            temperature=0.1,
            gate_bias=-2.0,
        )
        scores = torch.randn(1, graph.num_nodes, graph.num_nodes)

        indices, valid, local = module.select_indices(scores, graph)

        self.assertEqual(indices.shape, (1, graph.num_nodes, 10))
        self.assertTrue(bool(valid[0, 0].all()))
        self.assertEqual(int(local[0, 0].sum().item()), 4)
        self.assertEqual(int((~local[0, 0]).sum().item()), 6)

    def test_route_forward_preserves_shape_and_is_finite(self) -> None:
        graph = _route_graph()
        module = MixedRangeRouteAttention(
            hidden_dim=8,
            route_dim=4,
            route_top_k=10,
            route_local_quota=4,
            prior_weight=0.25,
            temperature=0.1,
            gate_bias=-2.0,
        )
        tokens = torch.randn(2, 5, graph.num_nodes, 8)
        prior = graph.random_walk_diffusion_prior()

        output = module(tokens, graph, prior)

        self.assertEqual(tuple(output.shape), tuple(tokens.shape))
        self.assertTrue(bool(torch.isfinite(output).all()))

    def test_route_value_path_is_low_rank(self) -> None:
        module = MixedRangeRouteAttention(
            hidden_dim=80,
            route_dim=16,
            route_top_k=10,
            route_local_quota=4,
        )

        self.assertEqual(module.value_down.out_features, 16)
        self.assertEqual(module.value_up.in_features, 16)
        self.assertLess(sum(parameter.numel() for parameter in module.parameters()), 10_000)

    def test_encoder_route_branch_is_connected_and_receives_gradient(self) -> None:
        graph = _route_graph()
        encoder = FactorizedSTEncoder(
            hidden_dim=8,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
            route_enabled=True,
            route_dim=4,
            route_top_k=10,
            route_local_quota=4,
        )
        tokens = torch.randn(2, 5, graph.num_nodes, 8, requires_grad=True)

        output = encoder(tokens, graph)
        output.square().mean().backward()

        self.assertEqual(tuple(output.shape), tuple(tokens.shape))
        self.assertTrue(bool(torch.isfinite(output).all()))
        route_parameters = [
            parameter
            for name, parameter in encoder.named_parameters()
            if "route_attention" in name
        ]
        self.assertTrue(route_parameters)
        self.assertTrue(
            any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in route_parameters)
        )


if __name__ == "__main__":
    unittest.main()
