from __future__ import annotations

import unittest

import numpy as np

from stanchor.diagnostics.spatial_residual import (
    spatial_residual_metrics,
    target_degree_matched_nonedges,
)


class SpatialResidualDiagnosticTests(unittest.TestCase):
    def test_target_degree_matched_nonedges_are_legal(self) -> None:
        edges = np.asarray(
            [[0, 0, 1, 1, 2, 2, 3, 3], [1, 2, 0, 2, 0, 3, 1, 2]],
            dtype=np.int64,
        )
        nonedges = target_degree_matched_nonedges(edges, num_nodes=6, seed=42)
        edge_pairs = set(map(tuple, edges.T.tolist()))
        nonedge_pairs = set(map(tuple, nonedges.T.tolist()))
        self.assertEqual(nonedges.shape, edges.shape)
        self.assertTrue(edge_pairs.isdisjoint(nonedge_pairs))
        self.assertTrue(all(target != source for target, source in nonedge_pairs))
        for target in range(6):
            self.assertEqual(
                int((nonedges[0] == target).sum()),
                int((edges[0] == target).sum()),
            )

    def test_spatial_residual_metrics_detect_edge_specific_signal(self) -> None:
        rng = np.random.default_rng(7)
        samples, horizons, nodes = 600, 2, 8
        edges = np.asarray(
            [[0, 1, 2, 3, 4, 5], [1, 0, 3, 2, 5, 4]], dtype=np.int64
        )
        residual = rng.normal(size=(samples, horizons, nodes)).astype(np.float32)
        for left, right in ((0, 1), (2, 3), (4, 5)):
            shared = rng.normal(size=(samples, horizons)).astype(np.float32)
            residual[:, :, left] = shared + 0.1 * rng.normal(size=(samples, horizons))
            residual[:, :, right] = shared + 0.1 * rng.normal(size=(samples, horizons))
        valid = np.ones_like(residual, dtype=bool)
        helpful = residual > 0.0
        result = spatial_residual_metrics(
            residual, valid, helpful, valid, edges, seed=11
        )
        overall = result["overall"]
        self.assertGreater(
            overall["base_residual"]["centered_pearson_mean_excess"], 0.5
        )
        self.assertGreater(overall["candidate_helpfulness"]["phi_mean_excess"], 0.3)

    def test_common_temporal_factor_is_not_mistaken_for_edge_signal(self) -> None:
        rng = np.random.default_rng(13)
        samples, horizons, nodes = 800, 1, 10
        common = rng.normal(size=(samples, horizons, 1)).astype(np.float32)
        residual = common + 0.05 * rng.normal(size=(samples, horizons, nodes))
        valid = np.ones_like(residual, dtype=bool)
        helpful = residual > 0.0
        edges = np.asarray(
            [[0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 8]],
            dtype=np.int64,
        )
        result = spatial_residual_metrics(
            residual, valid, helpful, valid, edges, seed=17
        )
        excess = result["overall"]["base_residual"]
        self.assertLess(abs(excess["pearson_mean_excess"]), 0.02)
        self.assertLess(abs(excess["centered_pearson_mean_excess"]), 0.08)


if __name__ == "__main__":
    unittest.main()
