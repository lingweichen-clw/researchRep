from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from stanchor.config import DataConfig, ExperimentConfig, TargetConfig
from stanchor.diagnostics.downstream import (
    DownstreamDiagnosticAccumulator,
    binary_confidence_metrics,
    confidence_quartile_gains,
    diagnose_downstream_checkpoint,
    distribution_summary,
)


class DownstreamDiagnosticsTest(unittest.TestCase):
    def test_checkpoint_diagnostic_uses_mode_specific_retrieval(self) -> None:
        class RetrievalReached(RuntimeError):
            pass

        sample = {
            "x": torch.zeros(12, 2, 1),
            "x_observed": torch.ones(12, 2, 1, dtype=torch.bool),
            "weekday": torch.zeros(12, dtype=torch.long),
            "slot": torch.arange(12, dtype=torch.long),
            "query_weekday": torch.tensor(0, dtype=torch.long),
            "query_slot": torch.tensor(12, dtype=torch.long),
            "context_start": torch.tensor(0, dtype=torch.long),
            "y": torch.zeros(12, 2, 1),
            "y_observed": torch.ones(12, 2, 1, dtype=torch.bool),
        }
        scaler = MagicMock()
        scaler.state_dict.return_value = {"mean": [0.0], "std": [1.0]}
        data = SimpleNamespace(
            val=[sample],
            series=SimpleNamespace(slots_per_day=288),
            scaler=scaler,
        )
        graph = MagicMock()
        graph.to.return_value = graph
        pretrained = MagicMock()
        downstream = MagicMock()
        bank = MagicMock()
        bank.__enter__.return_value = bank
        bank.__exit__.return_value = False
        bank.manifest.dataset_name = "METR-LA"
        checkpoint = {
            "downstream_mode": "weekly_mean_horizon",
            "downstream_state_dict": {},
            "bank_manifest": bank.manifest.to_dict(),
        }

        with (
            patch("stanchor.diagnostics.downstream.resolve_device", return_value=torch.device("cpu")),
            patch("stanchor.diagnostics.downstream.build_data_and_graph", return_value=(data, graph)),
            patch(
                "stanchor.diagnostics.downstream.load_pretrained_model",
                return_value=(pretrained, {}),
            ),
            patch("stanchor.diagnostics.downstream.load_checkpoint", return_value=checkpoint),
            patch("stanchor.diagnostics.downstream.build_downstream_model", return_value=downstream),
            patch("stanchor.diagnostics.downstream.MemoryBank", return_value=bank),
            patch("stanchor.diagnostics.downstream._validate_bank"),
            patch("stanchor.diagnostics.downstream.TwoStageRetriever"),
            patch(
                "stanchor.diagnostics.downstream.retrieve_for_downstream_mode",
                side_effect=RetrievalReached,
            ) as retrieval,
        ):
            with self.assertRaises(RetrievalReached):
                diagnose_downstream_checkpoint(
                    ExperimentConfig(
                        data=DataConfig(raw_path="data.h5", adjacency_path="adj.pkl"),
                        target=TargetConfig(downstream_mode="base_only"),
                    ),
                    pretrained_checkpoint="pretrain.pt",
                    downstream_checkpoint="downstream.pt",
                    bank_path="bank",
                )

        self.assertEqual(retrieval.call_args.args[0], "weekly_mean_horizon")

    def test_accumulator_compares_memory_on_common_valid_positions(self) -> None:
        target = torch.full((1, 2, 2, 1), 10.0)
        base = torch.full_like(target, 12.0)
        memory = torch.tensor([[[[13.0], [11.0]], [[13.0], [11.0]]]])
        final = torch.full_like(target, 11.5)
        observed = torch.ones_like(target, dtype=torch.bool)
        memory_valid = torch.ones_like(target, dtype=torch.bool)
        confidence = torch.tensor([[[[0.1], [0.9]], [[0.2], [0.8]]]])
        fusion_weight = confidence * 0.1
        accumulator = DownstreamDiagnosticAccumulator(horizon=2)

        accumulator.update(
            base,
            memory,
            final,
            target,
            observed,
            memory_valid,
            confidence,
            fusion_weight,
        )
        result = accumulator.compute()

        self.assertAlmostEqual(result["branches"]["base"]["mae"], 2.0)
        self.assertAlmostEqual(result["branches"]["base_memory_common"]["mae"], 2.0)
        self.assertAlmostEqual(result["branches"]["memory"]["mae"], 2.0)
        self.assertAlmostEqual(result["branches"]["final"]["mae"], 1.5)
        self.assertAlmostEqual(result["gains"]["final_vs_base_percent"], 25.0)
        self.assertAlmostEqual(result["confidence_quality"]["auroc"], 1.0)
        self.assertAlmostEqual(result["confidence_quality"]["prevalence"], 0.5)

    def test_binary_metrics_are_exact_for_perfect_ranking(self) -> None:
        confidence = np.asarray([0.9, 0.8, 0.2, 0.1], dtype=np.float64)
        helpful = np.asarray([1, 1, 0, 0], dtype=bool)

        result = binary_confidence_metrics(confidence, helpful)

        self.assertAlmostEqual(result["auroc"], 1.0)
        self.assertAlmostEqual(result["auprc"], 1.0)
        self.assertAlmostEqual(result["brier"], 0.025)
        self.assertAlmostEqual(result["ece"], 0.15)
        self.assertEqual(result["ece_bins"], 10)
        self.assertAlmostEqual(result["prevalence"], 0.5)
        self.assertAlmostEqual(result["constant_brier"], 0.25)

    def test_constant_confidence_has_random_ranking_baselines(self) -> None:
        confidence = np.full(4, 0.5, dtype=np.float64)
        helpful = np.asarray([1, 0, 1, 0], dtype=bool)

        result = binary_confidence_metrics(confidence, helpful)

        self.assertAlmostEqual(result["auroc"], 0.5)
        self.assertAlmostEqual(result["auprc"], 0.5)
        self.assertAlmostEqual(result["brier"], 0.25)
        self.assertAlmostEqual(result["ece"], 0.0)

    def test_quartile_gain_increases_with_confidence(self) -> None:
        confidence = np.asarray([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
        base_error = np.full(8, 2.0)
        memory_error = np.asarray([3.0, 3.0, 2.5, 2.5, 1.5, 1.5, 1.0, 1.0])

        groups = confidence_quartile_gains(confidence, base_error, memory_error)

        self.assertEqual([group["count"] for group in groups], [2, 2, 2, 2])
        self.assertEqual(
            [group["absolute_memory_gain"] for group in groups],
            [-1.0, -0.5, 0.5, 1.0],
        )
        summary = distribution_summary(confidence)
        self.assertAlmostEqual(summary["mean"], 0.5)
        self.assertAlmostEqual(summary["median"], 0.5)


if __name__ == "__main__":
    unittest.main()
