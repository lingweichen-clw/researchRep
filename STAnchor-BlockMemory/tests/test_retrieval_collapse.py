from __future__ import annotations

import unittest
import subprocess
import sys

import numpy as np
import torch

from stanchor.config import ModelConfig, PretrainConfig
from stanchor.data.graph import graph_from_dense
from stanchor.data.normalization import normalize_window
from stanchor.diagnostics.retrieval_collapse import (
    _aggregation_for_version,
    encode_components,
    key_geometry_summary,
    mean_cosine_change,
    pooling_effective_patch_count,
    reverse_temporal_patches,
    select_topk_event_ids,
    weekly_donor_pairs,
    shift_calendar,
    topk_jaccard,
)
from stanchor.models.pretraining import STAnchorPretrainModel


class RetrievalCollapseDiagnosticTest(unittest.TestCase):
    def test_collapse_diagnostic_uses_matching_payload(self) -> None:
        decay_fn, decay_name = _aggregation_for_version("hn_offset_decay_v2")
        offset_fn, offset_name = _aggregation_for_version("hn_offset_only_v1")

        self.assertEqual(decay_name, "offset_decay")
        self.assertEqual(offset_name, "offset_only")
        self.assertNotEqual(decay_fn, offset_fn)

    def setUp(self) -> None:
        torch.manual_seed(7)
        nodes = 3
        adjacency = np.eye(nodes, dtype=np.float32)
        self.graph = graph_from_dense(adjacency)
        self.model = STAnchorPretrainModel(
            ModelConfig(
                input_channels=1,
                output_channels=1,
                patch_size=2,
                hidden_dim=8,
                retrieval_dim=4,
                num_heads=2,
                encoder_layers=1,
                dropout=0.0,
                adapter_bottleneck_dim=4,
            ),
            PretrainConfig(time_mask_block_size=2),
            context_length=4,
            slots_per_day=8,
        ).eval()
        self.x = torch.randn(2, 4, nodes, 1)
        self.observed = torch.ones_like(self.x, dtype=torch.bool)
        self.weekday = torch.tensor([[0, 0, 0, 0], [2, 2, 2, 2]])
        self.slot = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])

    def test_component_encoding_reproduces_clean_encoding(self) -> None:
        clean = self.model.encode_clean(
            self.x,
            self.observed,
            self.weekday,
            self.slot,
            self.graph,
        )
        statistics = normalize_window(self.x, self.observed)

        component = encode_components(
            self.model,
            normalized=statistics.normalized,
            level_features=statistics.level_features,
            level_valid=statistics.level_valid,
            weekday=self.weekday,
            slot=self.slot,
            observed=self.observed,
            graph=self.graph,
        )

        self.assertTrue(torch.allclose(component.hidden, clean.hidden, atol=1.0e-6))
        self.assertTrue(
            torch.allclose(
                component.retrieval.node_keys,
                clean.retrieval.node_keys,
                atol=1.0e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                component.retrieval.pooling_weights,
                clean.retrieval.pooling_weights,
                atol=1.0e-6,
            )
        )

    def test_reverse_temporal_patches_preserves_values_inside_each_patch(self) -> None:
        values = torch.tensor([0.0, 1.0, 2.0, 3.0]).reshape(1, 4, 1, 1)

        result = reverse_temporal_patches(values, patch_size=2)

        self.assertTrue(
            torch.equal(
                result.flatten(),
                torch.tensor([2.0, 3.0, 0.0, 1.0]),
            )
        )

    def test_shift_calendar_moves_a_consistent_weekly_timeline(self) -> None:
        weekday = torch.tensor([[6, 6, 0, 0]])
        slot = torch.tensor([[6, 7, 0, 1]])

        shifted_weekday, shifted_slot = shift_calendar(
            weekday,
            slot,
            slots_per_day=8,
            shift_slots=4,
        )

        self.assertTrue(torch.equal(shifted_weekday, torch.tensor([[0, 0, 0, 0]])))
        self.assertTrue(torch.equal(shifted_slot, torch.tensor([[2, 3, 4, 5]])))

    def test_effective_patch_count_distinguishes_uniform_and_one_patch_pooling(self) -> None:
        weights = torch.tensor(
            [
                [
                    [0.25, 1.0],
                    [0.25, 0.0],
                    [0.25, 0.0],
                    [0.25, 0.0],
                ]
            ]
        )

        result = pooling_effective_patch_count(weights)

        self.assertTrue(torch.allclose(result, torch.tensor([[4.0, 1.0]])))

    def test_mean_cosine_change_is_zero_for_identical_and_one_for_orthogonal(self) -> None:
        reference = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        identical = reference.clone()
        orthogonal = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])

        self.assertAlmostEqual(mean_cosine_change(reference, identical), 0.0)
        self.assertAlmostEqual(mean_cosine_change(reference, orthogonal), 1.0)

    def test_topk_jaccard_ignores_order_and_padding(self) -> None:
        reference = torch.tensor([[[1, 2, -1], [4, 5, -1]]])
        changed = torch.tensor([[[2, 3, -1], [5, 4, -1]]])

        result = topk_jaccard(reference, changed)

        self.assertTrue(torch.allclose(result, torch.tensor([[1.0 / 3.0, 1.0]])))

    def test_key_geometry_reports_two_dimensional_centered_support(self) -> None:
        keys = np.asarray(
            [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
            dtype=np.float32,
        )

        result = key_geometry_summary(keys, pair_samples=6, seed=3)

        self.assertEqual(result["sample_count"], 4)
        self.assertAlmostEqual(result["effective_rank_participation"], 2.0, places=6)
        self.assertAlmostEqual(result["mean_l2_norm"], 1.0, places=6)

    def test_select_topk_event_ids_respects_valid_candidates(self) -> None:
        distances = torch.tensor([[[0.3, 0.1, 0.2, 0.0]]])
        event_ids = torch.tensor([[10, 11, 12, 13]])
        event_valid = torch.tensor([[True, True, True, False]])

        result = select_topk_event_ids(distances, event_ids, event_valid, k=2)

        self.assertTrue(torch.equal(result, torch.tensor([[[11, 12]]])))

    def test_weekly_donor_pairs_uses_same_calendar_position_from_another_week(self) -> None:
        context_ends = np.arange(100, 112, dtype=np.int64)

        result = weekly_donor_pairs(
            context_ends,
            weekly_steps=6,
            max_queries=4,
        )

        self.assertEqual(result.shape, (4, 2))
        self.assertTrue(np.all(np.abs(context_ends[result[:, 0]] - context_ends[result[:, 1]]) == 6))
        self.assertTrue(np.all(result[:, 0] != result[:, 1]))

    def test_collapse_cli_exposes_frozen_query_limit(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/diagnose_retrieval_collapse.py", "--help"],
            capture_output=True,
            check=False,
            text=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--max-queries", result.stdout)
        self.assertIn("--checkpoint", result.stdout)


if __name__ == "__main__":
    unittest.main()
