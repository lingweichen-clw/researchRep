from __future__ import annotations

import unittest

import torch

from stanchor.diagnostics.cfdp_probe import (
    HorizonSpecificPoolingProbe,
    SharedPooledLinearProbe,
    SharedPooledMLPProbe,
    masked_profile_loss,
    profile_relation_metrics,
)


class CFDPProbeTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)
        self.batch = 5
        self.tokens = 12
        self.nodes = 4
        self.hidden_dim = 16
        self.profile_dim = 12
        self.hidden = torch.randn(
            self.batch,
            self.tokens,
            self.nodes,
            self.hidden_dim,
        )
        self.pooled = self.hidden.mean(dim=1)

    def test_shared_pooled_probes_return_node_profiles(self) -> None:
        linear = SharedPooledLinearProbe(self.hidden_dim, self.profile_dim)
        mlp = SharedPooledMLPProbe(self.hidden_dim, self.profile_dim)

        self.assertEqual(
            tuple(linear(self.pooled).shape),
            (self.batch, self.nodes, self.profile_dim),
        )
        self.assertEqual(
            tuple(mlp(self.pooled).shape),
            (self.batch, self.nodes, self.profile_dim),
        )

    def test_horizon_specific_probe_normalizes_each_horizon_over_tokens(self) -> None:
        probe = HorizonSpecificPoolingProbe(self.hidden_dim, self.profile_dim)

        prediction, weights = probe(self.hidden, return_weights=True)

        self.assertEqual(
            tuple(prediction.shape),
            (self.batch, self.nodes, self.profile_dim),
        )
        self.assertEqual(
            tuple(weights.shape),
            (self.batch, self.profile_dim, self.tokens, self.nodes),
        )
        self.assertTrue(
            torch.allclose(
                weights.sum(dim=2),
                torch.ones(self.batch, self.profile_dim, self.nodes),
                atol=1.0e-6,
            )
        )

    def test_masked_profile_loss_ignores_invalid_positions(self) -> None:
        prediction = torch.tensor([[[1.0, 99.0]]], requires_grad=True)
        teacher = torch.tensor([[[0.0, 0.0]]])
        valid = torch.tensor([[[True, False]]])

        loss = masked_profile_loss(prediction, teacher, valid)

        self.assertAlmostEqual(float(loss.detach()), 0.5)
        loss.backward()
        self.assertAlmostEqual(float(prediction.grad[0, 0, 1]), 0.0)

    def test_teacher_profile_is_a_perfect_prediction_but_not_necessarily_perfect_geometry(self) -> None:
        teacher = torch.randn(self.batch, self.nodes, self.profile_dim)
        valid = torch.ones_like(teacher, dtype=torch.bool)
        candidate_mask = ~torch.eye(self.batch, dtype=torch.bool).unsqueeze(-1)
        candidate_mask = candidate_mask.expand(-1, -1, self.nodes)
        od_distance = torch.cdist(
            teacher.permute(1, 0, 2),
            teacher.permute(1, 0, 2),
            p=1,
        ).permute(1, 2, 0) / float(self.profile_dim)

        metrics = profile_relation_metrics(
            teacher,
            teacher,
            valid,
            od_distance,
            candidate_mask,
            top_k=3,
        )

        self.assertAlmostEqual(metrics["profile_mae"], 0.0)
        self.assertAlmostEqual(metrics["profile_cosine"], 1.0, places=6)
        self.assertIn("profile_mae_relation_spearman", metrics)
        self.assertIn("od_relation_spearman", metrics)

    def test_invalid_probe_shapes_are_rejected(self) -> None:
        probe = HorizonSpecificPoolingProbe(self.hidden_dim, self.profile_dim)
        with self.assertRaisesRegex(ValueError, "hidden"):
            probe(torch.randn(self.batch, self.tokens, self.hidden_dim))


if __name__ == "__main__":
    unittest.main()
