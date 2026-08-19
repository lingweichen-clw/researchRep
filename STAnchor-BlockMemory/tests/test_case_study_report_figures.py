from __future__ import annotations

import unittest

from stanchor.diagnostics.case_study_report_figures import (
    horizon_gain_rows,
    legend_labels,
    ranking_gain_rows,
)


class CaseStudyReportFiguresTest(unittest.TestCase):
    def test_horizon_gain_rows_use_random_minus_trained_mae(self) -> None:
        metrics = {
            "memory_metrics": {
                "pretrained_memory": {"horizon_mae": [2.0, 3.0]},
                "random_memory": {"horizon_mae": [2.5, 4.25]},
            }
        }

        rows = horizon_gain_rows(metrics, frequency_minutes=5)

        self.assertEqual(rows[0], {
            "step": 1,
            "minutes": 5,
            "trained_mae": 2.0,
            "random_mae": 2.5,
            "gain": 0.5,
        })
        self.assertEqual(rows[1]["gain"], 1.25)

    def test_ranking_gain_rows_preserve_metric_semantics(self) -> None:
        metrics = {
            "ranking": {
                "pretrained": {
                    "spearman_mean": 0.5,
                    "kendall_mean": 0.4,
                    "recall_at_1_mean": 0.2,
                    "ndcg_at_5_mean": 0.6,
                },
                "random": {
                    "spearman_mean": 0.1,
                    "kendall_mean": 0.1,
                    "recall_at_1_mean": 0.05,
                    "ndcg_at_5_mean": 0.3,
                },
            }
        }

        rows = ranking_gain_rows(metrics)

        self.assertEqual([row["metric"] for row in rows], [
            "Spearman", "Kendall", "Recall@1", "NDCG@5"
        ])
        self.assertAlmostEqual(rows[0]["gain"], 0.4)
        self.assertAlmostEqual(rows[-1]["gain"], 0.3)

    def test_legend_labels_excludes_unlabeled_artists(self) -> None:
        class Axis:
            def get_legend_handles_labels(self):
                return [object(), object()], ["", "Validation relation"]

        self.assertEqual(legend_labels(Axis()), ["Validation relation"])


if __name__ == "__main__":
    unittest.main()
