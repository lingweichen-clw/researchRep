from __future__ import annotations

import unittest

from stanchor.diagnostics.target_adaptation import (
    summarize_adaptation_deltas,
    summarize_hn_selector_deltas,
)


class TargetAdaptationCaseStudyTest(unittest.TestCase):
    def test_summary_reports_adapted_minus_source_for_all_relation_metrics(self) -> None:
        source = {
            "spearman_mean": 0.20,
            "future_cosine_mean": {"1": 0.40, "12": 0.70},
            "oracle_recall_mean": {"1": 0.30, "12": 0.60},
        }
        adapted = {
            "spearman_mean": 0.25,
            "future_cosine_mean": {"1": 0.46, "12": 0.68},
            "oracle_recall_mean": {"1": 0.32, "12": 0.65},
        }

        result = summarize_adaptation_deltas(source, adapted)

        self.assertAlmostEqual(result["spearman"], 0.05)
        self.assertAlmostEqual(result["future_cosine_at_k"]["1"], 0.06)
        self.assertAlmostEqual(result["future_cosine_at_k"]["12"], -0.02)
        self.assertAlmostEqual(result["oracle_recall_at_k"]["1"], 0.02)
        self.assertAlmostEqual(result["oracle_recall_at_k"]["12"], 0.05)

    def test_hn_selector_deltas_report_random_and_finetuned_comparisons(self) -> None:
        base = {
            "alignment": {
                "spearman": 0.10,
                "future_neighbor_recall_at_5": 0.20,
            },
            "ranking": {
                "spearman_mean": 0.11,
                "kendall_mean": 0.12,
                "recall_at_1_mean": 0.13,
                "ndcg_at_5_mean": 0.14,
                "recall_at_5_mean": 0.15,
            },
        }
        source = {
            "alignment": {
                "spearman": 0.30,
                "future_neighbor_recall_at_5": 0.40,
            },
            "ranking": {
                "spearman_mean": 0.31,
                "kendall_mean": 0.32,
                "recall_at_1_mean": 0.33,
                "ndcg_at_5_mean": 0.34,
                "recall_at_5_mean": 0.35,
            },
        }
        finetuned = {
            "alignment": {
                "spearman": 0.35,
                "future_neighbor_recall_at_5": 0.45,
            },
            "ranking": {
                "spearman_mean": 0.36,
                "kendall_mean": 0.37,
                "recall_at_1_mean": 0.38,
                "ndcg_at_5_mean": 0.39,
                "recall_at_5_mean": 0.40,
            },
        }

        result = summarize_hn_selector_deltas(base, source, finetuned)

        self.assertAlmostEqual(result["source_minus_random"]["alignment"]["spearman"], 0.20)
        self.assertAlmostEqual(result["finetuned_minus_source"]["ranking"]["recall_at_5_mean"], 0.05)
        self.assertEqual(
            set(result["source_minus_random"]["ranking"]),
            set(result["finetuned_minus_source"]["ranking"]),
        )


if __name__ == "__main__":
    unittest.main()
