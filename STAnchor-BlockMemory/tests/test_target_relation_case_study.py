from __future__ import annotations

import unittest

import numpy as np

from stanchor.diagnostics.target_relation import (
    build_trend_signatures,
    future_similarity,
    summarize_rank_relation,
)


class TargetRelationCaseStudyTest(unittest.TestCase):
    def test_trend_signature_is_invariant_to_level_and_positive_scale(self) -> None:
        values = np.asarray(
            [[10.0, 12.0, 14.0, 16.0], [100.0, 104.0, 108.0, 112.0]],
            dtype=np.float32,
        )
        valid = np.ones_like(values, dtype=bool)
        signatures = build_trend_signatures(values, valid)
        self.assertTrue(np.allclose(signatures[0], signatures[1], atol=1.0e-5))

    def test_future_similarity_matches_query_against_candidate_trends(self) -> None:
        query = np.asarray([[[0.0, 1.0, 2.0, 3.0]]], dtype=np.float32)
        candidates = np.asarray(
            [[[[10.0, 12.0, 14.0, 16.0], [3.0, 2.0, 1.0, 0.0]]]],
            dtype=np.float32,
        )
        query_valid = np.ones_like(query, dtype=bool)
        candidate_valid = np.ones_like(candidates, dtype=bool)
        similarity, pair_valid = future_similarity(
            query, query_valid, candidates, candidate_valid
        )
        self.assertTrue(bool(pair_valid.all()))
        self.assertGreater(float(similarity[0, 0, 0]), 0.99)
        self.assertLess(float(similarity[0, 0, 1]), -0.99)

    def test_rank_summary_prefers_future_order_and_reports_recall(self) -> None:
        future = np.asarray([[[0.95, 0.80, 0.10]]], dtype=np.float32)
        learned = np.asarray([[[0.90, 0.70, 0.20]]], dtype=np.float32)
        valid = np.ones_like(future, dtype=bool)
        result = summarize_rank_relation(learned, future, valid, ks=(1, 2))
        self.assertGreater(result["spearman_mean"], 0.99)
        self.assertAlmostEqual(result["future_cosine_at_k"]["1"], 0.95, places=5)
        self.assertAlmostEqual(result["oracle_recall_at_k"]["2"], 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
