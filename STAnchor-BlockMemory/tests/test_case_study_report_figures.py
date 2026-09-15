from __future__ import annotations

import unittest

from stanchor.diagnostics.case_study_report_figures import (
    context_quadrant_contrast_rows,
    horizon_comparison_rows,
    horizon_gain_rows,
    legend_labels,
    ranking_comparison_rows,
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

    def test_three_selector_exports_include_raw_l1(self) -> None:
        metrics = {
            "memory_metrics": {
                "pretrained_memory": {"horizon_mae": [2.0]},
                "raw_l1_offset_decay_memory": {"horizon_mae": [3.0]},
                "random_memory": {"horizon_mae": [2.5]},
            },
            "ranking": {
                "pretrained": {field: 0.6 for _, field in (
                    ("Spearman", "spearman_mean"),
                    ("Kendall", "kendall_mean"),
                    ("Recall@1", "recall_at_1_mean"),
                    ("Recall@5", "recall_at_5_mean"),
                    ("NDCG@5", "ndcg_at_5_mean"),
                )},
                "raw_l1": {field: 0.2 for field in (
                    "spearman_mean", "kendall_mean", "recall_at_1_mean",
                    "recall_at_5_mean", "ndcg_at_5_mean",
                )},
                "random": {field: 0.1 for field in (
                    "spearman_mean", "kendall_mean", "recall_at_1_mean",
                    "recall_at_5_mean", "ndcg_at_5_mean",
                )},
            },
        }

        horizon = horizon_comparison_rows(metrics, frequency_minutes=5)
        ranking = ranking_comparison_rows(metrics)

        self.assertEqual(horizon[0]["raw_l1_mae"], 3.0)
        self.assertEqual(horizon[0]["gain_vs_raw_l1"], 1.0)
        self.assertAlmostEqual(ranking[0]["gain_vs_raw_l1"], 0.4)
        self.assertAlmostEqual(ranking[0]["gain_vs_random"], 0.5)

    def test_context_contrast_uses_both_similar_as_within_model_baseline(self) -> None:
        rows = []
        for model in ("current", "reference", "random"):
            for quadrant, mean in (
                ("context_similar_future_similar", 0.95),
                ("context_similar_future_different", 0.90),
                ("context_different_future_similar", 0.85),
                ("context_different_future_different", 0.75),
            ):
                rows.append({
                    "model": model,
                    "model_label": model,
                    "quadrant": quadrant,
                    "mean": str(mean),
                    "ci_low": str(mean - 0.01),
                    "ci_high": str(mean + 0.01),
                })

        contrasts = context_quadrant_contrast_rows(rows)

        current = [row for row in contrasts if row["model"] == "current"]
        self.assertEqual(len(current), 3)
        self.assertAlmostEqual(current[0]["excess_key_distance"], 0.05)
        self.assertAlmostEqual(current[-1]["excess_key_distance"], 0.20)

    def test_legend_labels_excludes_unlabeled_artists(self) -> None:
        class Axis:
            def get_legend_handles_labels(self):
                return [object(), object()], ["", "Validation relation"]

        self.assertEqual(legend_labels(Axis()), ["Validation relation"])


if __name__ == "__main__":
    unittest.main()
