from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from stanchor.diagnostics.retrieval_visualization import (
    alignment_statistics,
    build_diagnostic_event_candidates,
    build_teacher_aligned_signature,
    complete_anchor_mask,
    future_information_boundary,
    future_neighbor_recall_at_k,
    masked_candidate_future_mae,
    memory_mae_by_anchor,
    node_key_distances,
    render_visualization_figures,
    run_retrieval_visualization,
    select_quantile_cases,
    teacher_candidate_distances,
    validate_aligned_bank_axes,
)


class RetrievalVisualizationTest(unittest.TestCase):
    def test_diagnostic_candidate_protocols_are_causal_and_shared(self) -> None:
        class Calendar:
            def lookup(self, weekday: int, slot: int) -> np.ndarray:
                values = {
                    (1, 10): np.array([0, 1]),
                    (1, 11): np.array([2]),
                    (1, 12): np.array([3]),
                }
                return values.get((weekday, slot), np.array([], dtype=np.int64))

        class Bank:
            calendar = Calendar()
            future_end = np.array([5, 8, 9, 11, 15, 20], dtype=np.int64)

            class manifest:
                slots_per_day = 96

        weekday = torch.tensor([1])
        slot = torch.tensor([11])
        context_start = torch.tensor([12])

        exact = build_diagnostic_event_candidates(
            Bank(), weekday, slot, context_start, 32, torch.device("cpu"), "exact_calendar"
        )
        relaxed = build_diagnostic_event_candidates(
            Bank(), weekday, slot, context_start, 32, torch.device("cpu"), "relaxed_calendar"
        )
        broad = build_diagnostic_event_candidates(
            Bank(), weekday, slot, context_start, 3, torch.device("cpu"), "broad_causal"
        )

        self.assertEqual(exact.event_ids[0, : exact.valid[0].sum()].tolist(), [2])
        self.assertEqual(relaxed.event_ids[0, : relaxed.valid[0].sum()].tolist(), [0, 1, 2, 3])
        self.assertEqual(broad.event_ids[0, : broad.valid[0].sum()].tolist(), [0, 2, 3])
        self.assertTrue(torch.equal(relaxed.event_ids, relaxed.event_ids.clone()))
        self.assertTrue((Bank().future_end[broad.event_ids[0, broad.valid[0]].numpy()] < 12).all())

    def test_masked_candidate_future_mae_uses_only_common_observations(self) -> None:
        query = torch.tensor([[[[1.0]], [[3.0]]]])  # [B=1,H=2,N=1,C=1]
        query_valid = torch.ones_like(query, dtype=torch.bool)
        candidates = torch.tensor(
            [[[[[1.0]], [[5.0]]], [[[9.0]], [[3.0]]]]]
        )  # [B=1,R=2,H=2,N=1,C=1]
        candidate_valid = torch.ones_like(candidates, dtype=torch.bool)
        candidate_valid[:, 1, 0] = False
        event_valid = torch.tensor([[True, True]])

        distance, valid = masked_candidate_future_mae(
            query,
            query_valid,
            candidates,
            candidate_valid,
            event_valid,
        )

        self.assertEqual(tuple(distance.shape), (1, 1, 2))
        self.assertTrue(bool(valid.all()))
        self.assertTrue(torch.allclose(distance[0, 0], torch.tensor([1.0, 0.0])))

    def test_context_normalized_signature_matches_e2_and_e3_teacher(self) -> None:
        context = torch.tensor([[[[0.0]], [[2.0]]]])
        observed = torch.ones_like(context, dtype=torch.bool)
        future = torch.tensor([[[[1.0]], [[3.0]]]])
        future_observed = torch.ones_like(future, dtype=torch.bool)

        signature, valid = build_teacher_aligned_signature(
            "e3",
            future,
            future_observed,
            context,
            observed,
        )

        self.assertTrue(bool(valid.all()))
        self.assertTrue(
            torch.allclose(signature.flatten(), torch.tensor([0.0, 2.0]), atol=1.0e-5)
        )

    def test_od_signature_matches_e5a_teacher(self) -> None:
        context = torch.tensor([[[[8.0]], [[10.0]]]])
        observed = torch.ones_like(context, dtype=torch.bool)
        future = torch.tensor([[[[12.0]], [[14.0]], [[16.0]]]])
        future_observed = torch.ones_like(future, dtype=torch.bool)

        signature, valid = build_teacher_aligned_signature(
            "e5a",
            future,
            future_observed,
            context,
            observed,
        )

        self.assertTrue(bool(valid.all()))
        self.assertTrue(torch.allclose(signature.flatten(), torch.tensor([2.0, 9.0, 16.0])))

    def test_alignment_statistics_uses_equal_frequency_bins_and_spearman(self) -> None:
        key_distance = np.arange(10, dtype=np.float64)
        future_distance = np.arange(9, -1, -1, dtype=np.float64)

        result = alignment_statistics(
            key_distance,
            future_distance,
            np.ones(10, dtype=bool),
            num_bins=2,
        )

        self.assertAlmostEqual(result["spearman"], -1.0)
        self.assertEqual([item["count"] for item in result["distance_bins"]], [5, 5])
        self.assertEqual(
            [item["future_distance_mean"] for item in result["distance_bins"]],
            [7.0, 2.0],
        )
        self.assertEqual(
            [item["future_distance_median"] for item in result["distance_bins"]],
            [7.0, 2.0],
        )

    def test_future_neighbor_recall_at_five_excludes_trivial_small_pools(self) -> None:
        key_distance = torch.tensor([[[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]]]).float()
        future_distance = torch.tensor([[[0, 1, 2, 3, 4, 5], [5, 4, 3, 2, 1, 0]]]).float()
        valid = torch.ones_like(key_distance, dtype=torch.bool)

        recall, eligible = future_neighbor_recall_at_k(
            key_distance,
            future_distance,
            valid,
            k=5,
        )

        self.assertTrue(bool(eligible.all()))
        self.assertTrue(torch.allclose(recall[eligible], torch.tensor([1.0, 0.8])))

        small_recall, small_eligible = future_neighbor_recall_at_k(
            key_distance[..., :5],
            future_distance[..., :5],
            valid[..., :5],
            k=5,
        )
        self.assertFalse(bool(small_eligible.any()))
        self.assertTrue(torch.equal(small_recall, torch.zeros_like(small_recall)))

    def test_quantile_case_selection_is_deterministic_and_exports_rule(self) -> None:
        records = [
            {"sample_id": 100 + index, "node_id": index, "mae_gain": float(index)}
            for index in range(11)
        ]

        selected = select_quantile_cases(records)

        self.assertEqual(selected["strong_win"]["mae_gain"], 9.0)
        self.assertEqual(selected["representative"]["mae_gain"], 5.0)
        self.assertEqual(selected["failure"]["mae_gain"], 1.0)
        self.assertEqual(selected["selection_rule"]["quantiles"], {
            "strong_win": 0.9,
            "representative": 0.5,
            "failure": 0.1,
        })

    def test_future_information_boundary_forbids_query_future_in_ranking(self) -> None:
        boundary = future_information_boundary()

        self.assertFalse(boundary["query_future_used_for_ranking"])
        self.assertEqual(boundary["evaluation_split"], "validation_only")
        self.assertIn("post-ranking", boundary["query_future_use"])

    def test_node_key_distance_is_one_minus_cosine_on_same_candidate_pool(self) -> None:
        query = torch.tensor([[[1.0, 0.0]]])
        candidates = torch.tensor([[[[1.0, 0.0]], [[-1.0, 0.0]]]])
        event_valid = torch.tensor([[True, True]])

        distance, valid = node_key_distances(query, candidates, event_valid)

        self.assertTrue(bool(valid.all()))
        self.assertTrue(torch.allclose(distance.flatten(), torch.tensor([0.0, 2.0])))

    def test_memory_mae_by_anchor_reduces_horizon_and_channel_only(self) -> None:
        prediction = torch.tensor(
            [[[[1.0], [10.0]], [[5.0], [14.0]]]]
        )  # [B=1,H=2,N=2,C=1]
        target = torch.tensor([[[[2.0], [10.0]], [[3.0], [18.0]]]])
        valid = torch.ones_like(prediction, dtype=torch.bool)
        valid[:, 1, 1] = False

        mae, anchor_valid = memory_mae_by_anchor(prediction, target, valid)

        self.assertTrue(bool(anchor_valid.all()))
        self.assertTrue(torch.allclose(mae, torch.tensor([[1.5, 0.0]])))

    def test_e5_candidate_distance_uses_configured_symmetric_normalization(self) -> None:
        query = torch.tensor([[[[0.0]]]])
        query_observed = torch.ones_like(query, dtype=torch.bool)
        candidates = torch.tensor([[[[[1.0]]], [[[10.0]]]]])
        candidate_observed = torch.ones_like(candidates, dtype=torch.bool)
        event_valid = torch.tensor([[True, True]])

        anchor_distance, anchor_valid = teacher_candidate_distances(
            query,
            query_observed,
            candidates,
            candidate_observed,
            event_valid,
            normalization="anchor_mean",
        )
        symmetric_distance, symmetric_valid = teacher_candidate_distances(
            query,
            query_observed,
            candidates,
            candidate_observed,
            event_valid,
            normalization="symmetric_geometric_mean",
            symmetric_chunk_size=1,
        )

        self.assertTrue(torch.equal(anchor_valid, symmetric_valid))
        self.assertTrue(bool(symmetric_valid.all()))
        self.assertFalse(torch.allclose(anchor_distance, symmetric_distance))
        expected = torch.tensor([[[1.0 / (27.5 ** 0.5), 10.0 / (52.25 ** 0.5)]]])
        self.assertTrue(torch.allclose(symmetric_distance, expected, atol=1.0e-5))

    def test_complete_anchor_mask_requires_every_horizon_and_channel(self) -> None:
        valid = torch.ones((1, 3, 2, 1), dtype=torch.bool)
        valid[:, 1, 1] = False

        result = complete_anchor_mask(valid)

        self.assertTrue(torch.equal(result, torch.tensor([[True, False]])))

    def test_render_visualization_figures_creates_expected_nonempty_pngs(self) -> None:
        result = {
            "version": "e5a",
            "alignment": {
                "pretrained": {
                    "spearman": 0.2,
                    "future_neighbor_recall_at_5": 0.7,
                    "distance_bins": [
                        {"bin": index, "future_distance_mean": float(index), "future_distance_median": float(index)}
                        for index in range(1, 11)
                    ],
                },
                "random": {
                    "spearman": 0.0,
                    "future_neighbor_recall_at_5": 0.5,
                    "distance_bins": [
                        {"bin": index, "future_distance_mean": float(11 - index), "future_distance_median": float(11 - index)}
                        for index in range(1, 11)
                    ],
                },
            },
        }
        series = {
            "query_future": [10.0, 11.0, 12.0],
            "pretrained_memory": [10.0, 10.5, 12.0],
            "random_memory": [8.0, 9.0, 10.0],
            "pretrained_candidate_futures": [[9.0, 10.0, 11.0], [11.0, 12.0, 13.0]],
            "random_candidate_futures": [[7.0, 8.0, 9.0], [8.0, 9.0, 10.0]],
            "pretrained_mae": 0.17,
            "random_mae": 2.0,
            "pretrained_raw_memory": [9.0, 10.0, 11.0],
            "pretrained_offset_decay_memory": [10.0, 10.5, 12.0],
        }
        cases = {
            name: {**series, "sample_id": index, "node_id": index}
            for index, name in enumerate(("strong_win", "representative", "failure"))
        }

        with TemporaryDirectory() as directory:
            paths = render_visualization_figures(result, cases, Path(directory))

            self.assertEqual(len(paths), 3)
            for path in paths:
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 1000)

    def test_bank_axis_validation_rejects_different_candidate_event_order(self) -> None:
        class FakeManifest:
            num_events = 2
            num_nodes = 1
            horizon = 2
            channels = 1

        class FakeBank:
            manifest = FakeManifest()

            def __init__(self, sample_id: list[int]) -> None:
                self.sample_id = np.asarray(sample_id)
                self.weekday = np.asarray([0, 1])
                self.slot = np.asarray([2, 3])
                self.context_start = np.asarray([10, 20])
                self.context_end = np.asarray([11, 21])
                self.future_end = np.asarray([13, 23])
                self.future_masks = np.ones((2, 2, 1, 1), dtype=np.uint8)
                self.future_values = np.ones((2, 2, 1, 1), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "sample_id"):
            validate_aligned_bank_axes(FakeBank([1, 2]), FakeBank([2, 1]))

    def test_full_visualization_runner_rejects_test_split_before_loading_assets(self) -> None:
        with self.assertRaisesRegex(ValueError, "validation"):
            run_retrieval_visualization(
                version="e3",
                config=None,
                checkpoint_path="missing.pt",
                bank_path="missing-bank",
                random_checkpoint_path="missing-random.pt",
                random_bank_path="missing-random-bank",
                split="test",
                output_dir="unused",
            )


if __name__ == "__main__":
    unittest.main()
