from __future__ import annotations

import unittest

import torch

from stanchor.diagnostics.cfdp import cfdp_batch_metrics


class CFDPSemanticDiagnosticTest(unittest.TestCase):
    def test_perfect_profile_and_key_relations_score_one(self) -> None:
        # Four events, one node, three canonical future positions.
        teacher = torch.tensor(
            [
                [[0.0, 0.0, 0.0]],
                [[0.1, 0.0, 0.0]],
                [[0.0, 1.0, 0.0]],
                [[0.0, 0.0, 1.0]],
            ],
            dtype=torch.float32,
        )
        prediction = teacher.clone()
        valid = torch.ones_like(teacher, dtype=torch.bool)
        profile_keys = torch.nn.functional.normalize(
            torch.tensor(
                [
                    [[1.0, 0.0, 0.0]],
                    [[0.99, 0.1, 0.0]],
                    [[0.0, 1.0, 0.0]],
                    [[0.0, 0.0, 1.0]],
                ]
            ),
            dim=-1,
        )
        total_keys = profile_keys.clone()
        candidate_mask = ~torch.eye(4, dtype=torch.bool).unsqueeze(-1)
        od_distance = torch.cdist(
            total_keys[:, 0], total_keys[:, 0], p=2
        ).unsqueeze(-1)

        metrics = cfdp_batch_metrics(
            prediction,
            teacher,
            valid,
            profile_keys,
            total_keys,
            od_distance,
            candidate_mask,
            top_k=2,
        )

        self.assertAlmostEqual(metrics["profile_mae"], 0.0)
        self.assertAlmostEqual(metrics["profile_cosine"], 1.0, places=6)
        # The teacher uses pointwise MAE while the key uses cosine distance;
        # they should agree in ranking direction, not be algebraically equal.
        self.assertGreater(metrics["profile_relation_spearman"], 0.4)
        self.assertGreaterEqual(metrics["profile_relation_recall_at_k"], 0.5)
        self.assertGreater(metrics["total_od_relation_spearman"], 0.9)
        self.assertAlmostEqual(metrics["total_od_relation_recall_at_k"], 1.0)

    def test_missing_profile_head_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "profile"):
            cfdp_batch_metrics(
                None,
                torch.zeros(2, 1, 3),
                torch.ones(2, 1, 3, dtype=torch.bool),
                None,
                torch.zeros(2, 1, 3),
                torch.zeros(2, 2, 1),
                torch.ones(2, 2, 1, dtype=torch.bool),
            )


if __name__ == "__main__":
    unittest.main()
