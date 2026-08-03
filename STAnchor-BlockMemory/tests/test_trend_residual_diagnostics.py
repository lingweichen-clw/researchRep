from __future__ import annotations

import unittest
import subprocess
import sys
from pathlib import Path

import torch

from stanchor.diagnostics.trend_residual import (
    assemble_trend_residual_methods,
    diagnose_trend_residual_value,
)


class TrendResidualDiagnosticsTest(unittest.TestCase):
    def test_cli_exposes_trend_length_argument(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [sys.executable, "scripts/diagnose_trend_residual.py", "--help"],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--trend-length", completed.stdout)

    def test_diagnostic_rejects_non_evaluation_split_before_io(self) -> None:
        with self.assertRaisesRegex(ValueError, "split must be val or test"):
            diagnose_trend_residual_value(
                config=None,
                checkpoint_path="missing.pt",
                bank_path="missing-bank",
                split="train",
            )

    def test_trend_payload_removes_candidate_level_and_slope_mismatch(self) -> None:
        query_context = torch.tensor([[[[100.0]], [[103.0]], [[106.0]]]])
        query_observed = torch.ones_like(query_context, dtype=torch.bool)
        target = torch.tensor([[[[110.0]], [[113.0]]]])
        target_observed = torch.ones_like(target, dtype=torch.bool)
        candidate_contexts = torch.tensor(
            [[[[[10.0]], [[12.0]], [[14.0]]]]]
        )
        candidate_context_observed = torch.ones_like(candidate_contexts, dtype=torch.bool)
        candidate_futures = torch.tensor([[[[[17.0]]], [[[19.0]]]]])
        candidate_future_observed = torch.ones_like(candidate_futures, dtype=torch.bool)

        result = assemble_trend_residual_methods(
            query_context=query_context,
            query_observed=query_observed,
            target=target,
            target_observed=target_observed,
            candidate_contexts=candidate_contexts,
            candidate_context_observed=candidate_context_observed,
            candidate_futures=candidate_futures,
            candidate_future_observed=candidate_future_observed,
            event_ids=torch.tensor([[7]]),
            event_valid=torch.tensor([[True]]),
            learned_event_ids=torch.tensor([[[7]]]),
            learned_valid=torch.tensor([[[True]]]),
            learned_weights=torch.tensor([[[1.0]]]),
            trend_length=3,
            fixed_top_k=1,
            temperature=0.1,
        )

        raw = result["predictions"]["learned_raw_topk"]
        trend = result["predictions"]["learned_trend_topk"]
        raw_mae = (raw - target).abs().mean()
        trend_mae = (trend - target).abs().mean()

        self.assertGreater(float(raw_mae), float(trend_mae))
        self.assertTrue(torch.allclose(trend, target, atol=1.0e-5))
        self.assertEqual(
            result["future_information_boundary"],
            "oracle_and_rank_diagnostics_only",
        )

    def test_trend_payload_does_not_amplify_a_flat_candidate_scale(self) -> None:
        query_context = torch.tensor([[[[0.0]], [[1.0]], [[2.0]]]])
        target = torch.tensor([[[[4.0]], [[5.0]]]])
        candidate_contexts = torch.tensor(
            [[[[[10.0]], [[10.0]], [[10.0]]]]]
        )
        candidate_futures = torch.tensor([[[[[11.0]]], [[[11.0]]]]])

        result = assemble_trend_residual_methods(
            query_context=query_context,
            query_observed=torch.ones_like(query_context, dtype=torch.bool),
            target=target,
            target_observed=torch.ones_like(target, dtype=torch.bool),
            candidate_contexts=candidate_contexts,
            candidate_context_observed=torch.ones_like(
                candidate_contexts, dtype=torch.bool
            ),
            candidate_futures=candidate_futures,
            candidate_future_observed=torch.ones_like(
                candidate_futures, dtype=torch.bool
            ),
            event_ids=torch.tensor([[7]]),
            event_valid=torch.tensor([[True]]),
            learned_event_ids=torch.tensor([[[7]]]),
            learned_valid=torch.tensor([[[True]]]),
            learned_weights=torch.tensor([[[1.0]]]),
            trend_length=3,
            fixed_top_k=1,
            temperature=0.1,
        )

        trend = result["predictions"]["learned_trend_topk"]
        self.assertTrue(torch.allclose(trend, target, atol=1.0e-5))
        self.assertEqual(result["payload_scale_mode"], "unit")

    def test_offset_decay_uses_query_level_near_term_and_raw_at_horizon_end(self) -> None:
        query_context = torch.tensor([[[[20.0]], [[20.0]], [[20.0]]]])
        target = torch.zeros((1, 3, 1, 1))
        candidate_contexts = torch.tensor(
            [[[[[10.0]], [[10.0]], [[10.0]]]]]
        )
        candidate_futures = torch.tensor(
            [[[[[11.0]]], [[[12.0]]], [[[13.0]]]]]
        )

        result = assemble_trend_residual_methods(
            query_context=query_context,
            query_observed=torch.ones_like(query_context, dtype=torch.bool),
            target=target,
            target_observed=torch.ones_like(target, dtype=torch.bool),
            candidate_contexts=candidate_contexts,
            candidate_context_observed=torch.ones_like(
                candidate_contexts, dtype=torch.bool
            ),
            candidate_futures=candidate_futures,
            candidate_future_observed=torch.ones_like(
                candidate_futures, dtype=torch.bool
            ),
            event_ids=torch.tensor([[7]]),
            event_valid=torch.tensor([[True]]),
            learned_event_ids=torch.tensor([[[7]]]),
            learned_valid=torch.tensor([[[True]]]),
            learned_weights=torch.tensor([[[1.0]]]),
            trend_length=3,
            fixed_top_k=1,
            temperature=0.1,
        )

        prediction = result["predictions"]["learned_offset_decay_topk"]
        expected = torch.tensor([[[[21.0]], [[17.0]], [[13.0]]]])
        self.assertTrue(torch.allclose(prediction, expected, atol=1.0e-5))

    def test_assembly_rejects_query_future_as_a_deployable_selector(self) -> None:
        query_context = torch.zeros((1, 3, 1, 1))
        target = torch.zeros((1, 2, 1, 1))
        candidate_contexts = torch.zeros((1, 2, 3, 1, 1))
        candidate_futures = torch.zeros((1, 2, 1, 2, 1))

        result = assemble_trend_residual_methods(
            query_context=query_context,
            query_observed=torch.ones_like(query_context, dtype=torch.bool),
            target=target,
            target_observed=torch.ones_like(target, dtype=torch.bool),
            candidate_contexts=candidate_contexts,
            candidate_context_observed=torch.ones_like(candidate_contexts, dtype=torch.bool),
            candidate_futures=candidate_futures,
            candidate_future_observed=torch.ones_like(candidate_futures, dtype=torch.bool),
            event_ids=torch.tensor([[1, 2]]),
            event_valid=torch.tensor([[True, True]]),
            learned_event_ids=torch.tensor([[[1]]]),
            learned_valid=torch.tensor([[[True]]]),
            learned_weights=torch.tensor([[[1.0]]]),
            trend_length=3,
            fixed_top_k=1,
            temperature=0.1,
        )

        self.assertNotIn("future_oracle_trend_top1", result["deployable_methods"])
        self.assertIn("future_oracle_trend_top1", result["diagnostic_only_methods"])


if __name__ == "__main__":
    unittest.main()
