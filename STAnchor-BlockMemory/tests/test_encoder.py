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
    @staticmethod
    def _reference_selection(module, scores, graph):
        batch, nodes, _ = scores.shape
        effective_k = min(module.route_top_k, max(nodes - 1, 0))
        indices = torch.zeros((batch, nodes, effective_k), dtype=torch.long)
        valid = torch.zeros((batch, nodes, effective_k), dtype=torch.bool)
        local_slots = torch.zeros_like(valid)
        direct = graph.dense_neighbors(include_self=False)
        remote = ~direct
        remote.fill_diagonal_(False)
        local_quota = min(module.route_local_quota, effective_k)
        remote_quota = effective_k - local_quota
        for target in range(nodes):
            local_ids = torch.where(direct[target])[0]
            remote_ids = torch.where(remote[target])[0]
            local_take = min(local_quota, int(local_ids.numel()))
            remote_take = min(remote_quota, int(remote_ids.numel()))
            local_deficit = local_quota - local_take
            remote_deficit = remote_quota - remote_take
            remote_take = min(int(remote_ids.numel()), remote_take + local_deficit)
            local_take = min(int(local_ids.numel()), local_take + remote_deficit)
            position = 0
            if local_take:
                values = scores[:, target, local_ids]
                _, order = torch.topk(values, local_take, dim=-1)
                indices[:, target, position : position + local_take] = local_ids[order]
                valid[:, target, position : position + local_take] = True
                local_slots[:, target, position : position + local_take] = True
                position += local_take
            if remote_take:
                values = scores[:, target, remote_ids]
                _, order = torch.topk(values, remote_take, dim=-1)
                indices[:, target, position : position + remote_take] = remote_ids[order]
                valid[:, target, position : position + remote_take] = True
        return indices, valid, local_slots

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

    def test_vectorized_route_matches_reference(self) -> None:
        adjacency = np.eye(12, dtype=np.float32)
        for node in range(12):
            adjacency[node, (node - 1) % 12] = 1.0
            adjacency[node, (node + 1) % 12] = 1.0
        graph = graph_from_dense(adjacency)
        module = MixedRangeRouteAttention(
            hidden_dim=8,
            route_dim=4,
            route_top_k=10,
            route_local_quota=4,
        )
        scores = torch.randn(3, graph.num_nodes, graph.num_nodes)
        expected = self._reference_selection(module, scores, graph)

        actual = module.select_indices_vectorized(scores, graph)

        for actual_tensor, expected_tensor in zip(actual, expected):
            self.assertTrue(torch.equal(actual_tensor, expected_tensor))

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
