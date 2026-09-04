from __future__ import annotations

import unittest
import sys
import tempfile
import subprocess
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch

from stanchor.retrieval.retriever import EventCandidates
from stanchor.retrieval.strategies import raw_l1_node_candidates
from scripts.extract_spatiotemporal_mirages import (
    irregular_core_boundary,
    fit_key_umap,
    _population_cluster_plot,
    build_trend_signature,
    select_core_display_indices,
    select_cluster_display_indices,
    select_disjoint_key_cores,
    select_node_local_key_cores,
    summarize_overall_future_similarity,
    summarize_key_core_regions,
)
from scripts.plot_key_umap import _future_trend_signatures


class CaseStudyUpdateTest(unittest.TestCase):
    def test_raw_l1_node_candidates_rank_same_pool_and_use_raw_uniform_weights(self) -> None:
        values = np.asarray(
            [0.0, 0.0, 1.0, 1.0, 5.0, 5.0, 9.0, 9.0], dtype=np.float32
        ).reshape(8, 1, 1)
        series = SimpleNamespace(values=values, observed=np.ones_like(values, dtype=bool))
        scaler = SimpleNamespace(
            mean=np.zeros((1, 1), dtype=np.float32),
            std=np.ones((1, 1), dtype=np.float32),
            eps=1.0e-6,
        )
        bank = SimpleNamespace(context_end=np.asarray([1, 3, 5], dtype=np.int64))
        events = EventCandidates(
            event_ids=torch.tensor([[0, 1, 2, -1]]),
            scores=torch.zeros(1, 4),
            valid=torch.tensor([[True, True, True, False]]),
        )
        query = torch.tensor([[[[1.0]], [[1.0]]]])
        observed = torch.ones_like(query, dtype=torch.bool)

        candidates, distances, valid = raw_l1_node_candidates(
            query,
            observed,
            bank,
            events,
            series,
            scaler,
            context_length=2,
            top_k=2,
            device=torch.device("cpu"),
            candidate_chunk_size=2,
        )

        self.assertEqual(candidates.event_ids.tolist(), [[[1, 0]]])
        self.assertTrue(torch.allclose(candidates.weights, torch.tensor([[[0.5, 0.5]]])))
        self.assertEqual(tuple(distances.shape), (1, 1, 4))
        self.assertFalse(bool(valid[..., -1].any()))

    def test_cluster_display_indices_select_nearest_real_keys_in_original_space(self) -> None:
        keys = np.asarray(
            [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0], [9.0, 9.0], [20.0, 20.0]],
            dtype=np.float32,
        )
        labels = np.asarray([3, 3, 3, 3, 8], dtype=np.int64)

        selected = select_cluster_display_indices(keys, labels, cluster=3, max_points=3)

        self.assertEqual(selected.tolist(), [2, 1, 0])
        self.assertTrue(np.all(labels[selected] == 3))

    def test_disjoint_key_cores_are_real_compact_and_future_coherent(self) -> None:
        coordinates = np.asarray(
            [
                [0.00, 0.00], [0.03, 0.01], [0.01, -0.02], [-0.02, 0.01],
                [3.00, 3.00], [3.02, 3.01], [2.98, 3.03], [3.01, 2.97],
                [0.20, 3.00], [0.23, 3.01], [0.19, 3.03], [0.22, 2.98],
            ],
            dtype=np.float32,
        )
        signatures = np.asarray(
            [
                [0.0, 1.0, 2.0], [0.0, 1.1, 2.1], [0.0, 0.9, 1.9], [0.0, 1.0, 2.2],
                [0.0, -1.0, -2.0], [0.0, -1.1, -2.1], [0.0, -0.9, -1.9], [0.0, -1.0, -2.2],
                [0.0, 2.0, 1.0], [0.0, 2.1, 1.1], [0.0, 1.9, 0.9], [0.0, 2.2, 1.0],
            ],
            dtype=np.float32,
        )
        labels = np.repeat(np.arange(3, dtype=np.int64), 4)

        cores = select_disjoint_key_cores(
            coordinates,
            signatures,
            labels,
            points_per_core=4,
            max_cores=3,
            min_future_cosine=0.95,
        )

        self.assertEqual(len(cores), 3)
        self.assertEqual({int(core["trend_cluster"]) for core in cores}, {0, 1, 2})
        self.assertTrue(all(len(core["indices"]) == 4 for core in cores))
        self.assertTrue(all(float(core["within_future_cosine"]) >= 0.95 for core in cores))
        for left_index, left in enumerate(cores):
            for right in cores[left_index + 1:]:
                distance = np.linalg.norm(np.asarray(left["center"]) - np.asarray(right["center"]))
                self.assertGreater(distance, float(left["radius"]) + float(right["radius"]))

    def test_key_core_uses_natural_density_component_instead_of_fixed_knn_ball(self) -> None:
        dense = np.asarray(
            [[0.012 * index, 0.018 * (index % 4) + 0.004 * (index // 4)] for index in range(14)],
            dtype=np.float32,
        )
        outliers = np.asarray([[2.0 + index, 3.0 + index * 0.7] for index in range(5)], dtype=np.float32)
        coordinates = np.concatenate((dense, outliers), axis=0)
        signatures = np.tile(np.asarray([[0.0, 1.0, 2.0]], dtype=np.float32), (19, 1))
        labels = np.zeros(19, dtype=np.int64)

        cores = select_disjoint_key_cores(
            coordinates,
            signatures,
            labels,
            points_per_core=10,
            max_cores=1,
            min_future_cosine=0.95,
        )

        self.assertEqual(len(cores), 1)
        self.assertGreater(len(cores[0]["indices"]), 10)
        self.assertTrue(np.all(np.asarray(cores[0]["indices"]) < len(dense)))

    def test_node_local_key_cores_use_same_node_original_key_neighbors(self) -> None:
        keys = np.asarray(
            [
                [[1.00, 0.00], [0.00, 1.00]],
                [[0.99, 0.01], [0.01, 0.99]],
                [[0.98, 0.02], [0.02, 0.98]],
                [[0.97, 0.03], [0.03, 0.97]],
                [[0.00, 1.00], [1.00, 0.00]],
            ],
            dtype=np.float32,
        )
        signatures = np.asarray(
            [
                [[0.0, 1.0, 2.0], [0.0, -1.0, -2.0]],
                [[0.0, 1.1, 2.1], [0.0, -1.1, -2.1]],
                [[0.0, 0.9, 1.9], [0.0, -0.9, -1.9]],
                [[0.0, 1.0, 2.2], [0.0, -1.0, -2.2]],
                [[0.0, -1.0, -2.0], [0.0, 1.0, 2.0]],
            ],
            dtype=np.float32,
        )
        coordinates = keys.copy()

        cores = select_node_local_key_cores(
            keys,
            signatures,
            coordinates,
            points_per_core=4,
            max_cores=2,
            min_future_cosine=0.95,
            seed_candidates=5,
        )

        self.assertEqual(len(cores), 2)
        self.assertEqual({int(core["node"]) for core in cores}, {0, 1})
        for core in cores:
            node = int(core["node"])
            self.assertTrue(all(int(index) % 2 == node for index in core["indices"]))
            self.assertGreaterEqual(float(core["within_future_cosine"]), 0.95)

    def test_node_local_key_cores_can_rebuild_only_cached_candidate_nodes(self) -> None:
        keys = np.asarray(
            [
                [[1.00, 0.00], [0.00, 1.00]],
                [[0.99, 0.01], [0.01, 0.99]],
                [[0.98, 0.02], [0.02, 0.98]],
                [[0.97, 0.03], [0.03, 0.97]],
            ],
            dtype=np.float32,
        )
        signatures = np.asarray(
            [
                [[0.0, 1.0, 2.0], [0.0, -1.0, -2.0]],
                [[0.0, 1.1, 2.1], [0.0, -1.1, -2.1]],
                [[0.0, 0.9, 1.9], [0.0, -0.9, -1.9]],
                [[0.0, 1.0, 2.2], [0.0, -1.0, -2.2]],
            ],
            dtype=np.float32,
        )
        cores = select_node_local_key_cores(
            keys,
            signatures,
            keys,
            points_per_core=4,
            max_cores=2,
            min_future_cosine=0.95,
            candidate_nodes=[1],
            seed_candidates=4,
        )

        self.assertEqual(len(cores), 1)
        self.assertEqual(int(cores[0]["node"]), 1)

    def test_key_umap_embeds_only_fixed_core_points(self) -> None:
        rng = np.random.default_rng(17)
        keys = rng.normal(size=(30, 6)).astype(np.float32)
        cores = [
            {"indices": np.asarray([1, 3, 5], dtype=np.int64)},
            {"indices": np.asarray([20, 22, 24], dtype=np.int64)},
        ]

        class FakeUMAP:
            def __init__(self, **_: object) -> None:
                pass

            def fit_transform(self, values: np.ndarray) -> np.ndarray:
                return values[:, :2]

        with patch.dict(sys.modules, {"umap": SimpleNamespace(UMAP=FakeUMAP)}):
            coordinates = fit_key_umap(
                keys,
                cores,
                seed=11,
                n_neighbors=3,
                min_dist=0.05,
                n_epochs=20,
            )

        self.assertEqual(coordinates.shape, (30, 2))
        self.assertTrue(np.isfinite(coordinates[[1, 3, 5, 20, 22, 24]]).all())
        self.assertTrue(np.isnan(coordinates[[0, 2, 4, 6, 7]]).all())

    def test_core_display_sampling_is_seeded_bernoulli_and_does_not_change_pool(self) -> None:
        cores = [
            {"indices": np.arange(offset, offset + 80, dtype=np.int64)}
            for offset in (0, 100, 200, 300)
        ]

        first = select_core_display_indices(cores, keep_probability=0.75, seed=42)
        second = select_core_display_indices(cores, keep_probability=0.75, seed=42)

        self.assertEqual(len(first), 4)
        for core, left, right in zip(cores, first, second):
            self.assertEqual(left.tolist(), right.tolist())
            self.assertGreater(len(left), 0)
            self.assertLess(len(left), len(core["indices"]))
            self.assertTrue(set(left.tolist()).issubset(set(core["indices"].tolist())))
        self.assertGreater(len({len(indices) for indices in first}), 1)

    def test_key_core_summary_reports_original_centroid_separation(self) -> None:
        cores = [
            {
                "indices": np.asarray([0, 1]),
                "center": np.asarray([1.0, 0.0], dtype=np.float32),
                "radius": 0.1,
                "within_future_cosine": 0.9,
                "node": 2,
            },
            {
                "indices": np.asarray([2, 3]),
                "center": np.asarray([0.6, 0.8], dtype=np.float32),
                "radius": 0.2,
                "within_future_cosine": 0.85,
                "node": 4,
            },
        ]

        rows, overall = summarize_key_core_regions(cores)

        self.assertAlmostEqual(rows[0]["nearest_other_core_cosine_distance"], 0.4)
        self.assertAlmostEqual(rows[1]["nearest_other_core_cosine_distance"], 0.4)
        self.assertAlmostEqual(overall["minimum_inter_core_cosine_distance"], 0.4)

    def test_overall_future_similarity_uses_within_cluster_pair_weights(self) -> None:
        summary = [
            {"cluster": 0, "size": 3, "within_trend_cosine_mean": 1.0},
            {"cluster": 1, "size": 2, "within_trend_cosine_mean": 0.0},
        ]
        result = summarize_overall_future_similarity(summary)
        self.assertEqual(result["cluster_count"], 2)
        self.assertEqual(result["eligible_event_node_points"], 5)
        self.assertAlmostEqual(result["weighted_within_cluster_cosine"], 3.0 / 4.0)

    def test_irregular_core_boundary_uses_real_points_and_is_not_an_ellipse(self) -> None:
        points = np.asarray(
            [
                [0.0, 0.0], [1.0, 0.0], [1.2, 0.4], [0.7, 1.1],
                [0.0, 0.8], [0.35, 0.35],
            ],
            dtype=np.float32,
        )
        boundary = irregular_core_boundary(points)
        self.assertGreaterEqual(boundary.shape[0], 3)
        self.assertEqual(boundary.shape[1], 2)
        self.assertTrue(all(any(np.allclose(vertex, point) for point in points) for vertex in boundary))
        self.assertFalse(np.allclose(np.ptp(boundary[:, 0]), np.ptp(boundary[:, 1])))

    def test_key_umap_plot_uses_unconnected_point_only_style(self) -> None:
        coordinates = np.asarray(
            [[-1.0, -0.2], [-0.8, 0.1], [1.0, 0.2], [1.2, -0.1]],
            dtype=np.float32,
        )
        cores = [
            {
                "indices": np.asarray([0, 1]),
                "within_future_cosine": 0.81,
                "node": 10,
            },
            {
                "indices": np.asarray([2, 3]),
                "within_future_cosine": 0.77,
                "node": 11,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            summary = _population_cluster_plot(
                Path(directory), coordinates, cores, [core["indices"] for core in cores]
            )
            self.assertEqual(summary["visual_style"], "unconnected_filled_points")
            self.assertEqual(summary["figure_title"], "Local Key Neighborhoods")
            self.assertTrue((Path(directory) / "key_umap_local_regions.png").exists())

    def test_key_umap_entrypoint_runs_directly(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "scripts/plot_key_umap.py", "--help"],
            cwd=repository,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)

    def test_future_trend_signature_batch_matches_scalar_definition(self) -> None:
        futures = np.asarray(
            [
                [[2.0, 3.0, 5.0], [4.0, 4.0, 3.0]],
                [[1.0, 2.0, 3.0], [2.0, 0.0, 4.0]],
            ],
            dtype=np.float32,
        )
        masks = np.asarray(
            [
                [[True, True, True], [True, False, True]],
                [[True, True, True], [True, True, True]],
            ],
            dtype=bool,
        )
        actual = _future_trend_signatures(futures, masks)
        expected = np.asarray(
            [
                [
                    build_trend_signature(futures[0, 0], masks[0, 0]),
                    build_trend_signature(futures[0, 1], masks[0, 1]),
                ],
                [
                    build_trend_signature(futures[1, 0], masks[1, 0]),
                    build_trend_signature(futures[1, 1], masks[1, 1]),
                ],
            ],
            dtype=np.float32,
        )
        self.assertTrue(np.allclose(actual, expected, atol=1.0e-6))


if __name__ == "__main__":
    unittest.main()
