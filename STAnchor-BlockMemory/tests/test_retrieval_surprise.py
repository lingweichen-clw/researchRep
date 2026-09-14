from __future__ import annotations

import subprocess
import sys
import unittest

import numpy as np
import torch

from stanchor.diagnostics.retrieval_surprise import (
    aligned_anchor_ranking_metrics,
    persistence_surprise,
    stratified_forecast_metrics,
    stratified_scalar_summary,
    surprise_strata_masks,
    surprise_thresholds,
)
from stanchor.diagnostics.retrieval_visualization import anchor_wise_ranking_metrics


class RetrievalSurpriseDiagnosticTest(unittest.TestCase):
    def test_persistence_surprise_uses_last_value_and_visible_mean_fallback(self) -> None:
        context = torch.tensor(
            [
                [
                    [[1.0], [1.0]],
                    [[2.0], [9.0]],
                ]
            ]
        )
        context_observed = torch.tensor(
            [
                [
                    [[True], [True]],
                    [[True], [False]],
                ]
            ]
        )
        future = torch.tensor(
            [
                [
                    [[3.0], [3.0]],
                    [[4.0], [5.0]],
                ]
            ]
        )
        future_observed = torch.ones_like(future, dtype=torch.bool)

        surprise, valid = persistence_surprise(
            context,
            context_observed,
            future,
            future_observed,
        )

        self.assertTrue(torch.equal(valid, torch.tensor([[True, True]])))
        self.assertTrue(torch.allclose(surprise, torch.tensor([[1.5, 3.0]])))

    def test_surprise_strata_are_shared_disjoint_at_q80_and_nested_at_q95(self) -> None:
        surprise = torch.arange(1.0, 21.0).reshape(4, 5)
        valid = torch.ones_like(surprise, dtype=torch.bool)

        thresholds = surprise_thresholds(surprise, valid)
        strata = surprise_strata_masks(surprise, valid, thresholds)

        self.assertFalse(bool((strata["ordinary_80"] & strata["surprise_top20"]).any()))
        self.assertTrue(torch.equal(strata["all"], strata["ordinary_80"] | strata["surprise_top20"]))
        self.assertTrue(bool((strata["surprise_top5"] <= strata["surprise_top20"]).all()))
        self.assertEqual(int(strata["all"].sum()), 20)

    def test_stratified_forecast_metrics_select_query_node_across_all_horizons(self) -> None:
        target = torch.zeros((1, 2, 2, 1))
        prediction = torch.tensor([[[[1.0], [10.0]], [[3.0], [10.0]]]])
        observed = torch.ones_like(target, dtype=torch.bool)
        strata = {
            "ordinary_80": torch.tensor([[True, False]]),
            "surprise_top20": torch.tensor([[False, True]]),
        }

        result = stratified_forecast_metrics(prediction, target, observed, strata)

        self.assertAlmostEqual(result["ordinary_80"]["mae"], 2.0)
        self.assertAlmostEqual(result["surprise_top20"]["mae"], 10.0)
        self.assertEqual(result["ordinary_80"]["count"], 2)

    def test_aligned_ranking_metrics_preserve_anchor_positions(self) -> None:
        key = np.asarray(
            [
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                    [5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
                ]
            ],
            dtype=np.float64,
        )
        teacher = np.asarray(
            [
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                ]
            ],
            dtype=np.float64,
        )
        valid = np.ones_like(key, dtype=bool)

        result = aligned_anchor_ranking_metrics(key, teacher, valid, k=5)

        self.assertEqual(result["spearman"].shape, (1, 2))
        self.assertTrue(np.allclose(result["spearman"], [[1.0, -1.0]]))
        self.assertTrue(np.allclose(result["recall_at_5"], [[1.0, 0.8]]))

    def test_aligned_ranking_metrics_match_existing_aggregate_semantics(self) -> None:
        generator = np.random.default_rng(17)
        key = generator.uniform(0.0, 2.0, size=(2, 3, 8))
        teacher = generator.uniform(0.0, 2.0, size=(2, 3, 8))
        valid = generator.uniform(size=(2, 3, 8)) > 0.15

        aligned = aligned_anchor_ranking_metrics(key, teacher, valid, k=5)
        reference = anchor_wise_ranking_metrics(key, teacher, valid, ndcg_k=5)

        self.assertAlmostEqual(
            float(np.nanmean(aligned["spearman"])),
            reference["spearman_mean"],
            places=12,
        )
        self.assertAlmostEqual(
            float(np.nanmean(aligned["recall_at_5"])),
            reference["recall_at_5_mean"],
            places=12,
        )

    def test_scalar_summary_uses_same_query_node_mask(self) -> None:
        values = np.asarray([[1.0, 2.0], [3.0, np.nan]])
        strata = {
            "all": torch.tensor([[True, True], [True, True]]),
            "top": torch.tensor([[False, True], [True, False]]),
        }

        result = stratified_scalar_summary(values, strata)

        self.assertEqual(result["all"]["count"], 3)
        self.assertAlmostEqual(result["all"]["mean"], 2.0)
        self.assertAlmostEqual(result["top"]["mean"], 2.5)

    def test_cli_exposes_both_frozen_versions(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/diagnose_retrieval_surprise.py", "--help"],
            capture_output=True,
            check=False,
            text=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--offset-only-checkpoint", result.stdout)
        self.assertIn("--offset-decay-checkpoint", result.stdout)
        self.assertIn("--max-queries", result.stdout)


if __name__ == "__main__":
    unittest.main()
