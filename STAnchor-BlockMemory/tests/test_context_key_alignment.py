from __future__ import annotations

import subprocess
import sys
import unittest

import numpy as np

from stanchor.diagnostics.context_key_alignment import (
    add_key_and_control_distances,
    build_quantile_relation_surface,
    partial_rank_correlation,
    select_context_future_quadrants,
    select_level_controlled_quadrants,
    summarize_quadrant_contrasts,
)


class ContextKeyAlignmentTest(unittest.TestCase):
    def test_level_controlled_quadrants_hold_level_similar(self) -> None:
        records = []
        for context_name, context in (("similar", 0.05), ("different", 0.95)):
            for future_name, future in (("similar", 0.05), ("different", 0.95)):
                for repeat in range(4):
                    records.append(
                        {
                            "pair": f"{context_name}_{future_name}_{repeat}",
                            "context_distance": context,
                            "future_distance": future,
                            "level_distance": 0.05,
                            "calendar_compatible": True,
                        }
                    )
        records.extend(
            {
                "pair": f"high_level_{repeat}",
                "context_distance": 0.05,
                "future_distance": 0.05,
                "level_distance": 1.0,
                "calendar_compatible": True,
            }
            for repeat in range(4)
        )

        quadrants, thresholds = select_level_controlled_quadrants(
            records,
            tail_quantile=0.25,
            level_quantile=0.80,
        )

        selected = [row for rows in quadrants.values() for row in rows]
        self.assertEqual(len(selected), 16)
        self.assertTrue(all(float(row["level_distance"]) <= 0.05 for row in selected))
        self.assertLess(thresholds["level_distance_max"], 1.0)

    def test_partial_rank_correlation_controls_future_and_level(self) -> None:
        rng = np.random.default_rng(7)
        context = rng.normal(size=200)
        future = rng.normal(size=200)
        level = rng.normal(size=200)
        rows = [
            {
                "context_distance": float(context[index]),
                "future_distance": float(future[index]),
                "level_distance": float(level[index]),
                "current_key_distance": float(2.0 * context[index] + 0.05 * rng.normal()),
            }
            for index in range(200)
        ]

        correlation = partial_rank_correlation(
            rows,
            outcome="current_key_distance",
            predictor="context_distance",
            controls=("future_distance", "level_distance"),
        )

        self.assertGreater(correlation, 0.95)

    def test_quantile_relation_surface_preserves_all_pairs(self) -> None:
        rows = [
            {
                "context_distance": float(index % 10),
                "future_distance": float(index // 10),
                "current_key_distance": float(index),
            }
            for index in range(100)
        ]

        surface = build_quantile_relation_surface(
            rows,
            value_name="current_key_distance",
            bins=5,
        )

        self.assertEqual(np.asarray(surface["mean"]).shape, (5, 5))
        self.assertEqual(int(np.asarray(surface["count"]).sum()), 100)

    def test_key_distances_are_measured_on_the_same_event_pair(self) -> None:
        records = [
            {
                "node": 0,
                "i": 0,
                "j": 1,
                "context_distance": 0.1,
                "future_distance": 0.2,
            }
        ]
        current = np.asarray([[[1.0, 0.0]], [[1.0, 0.0]]], dtype=np.float32)
        reference = np.asarray([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=np.float32)
        random = np.asarray([[[1.0, 0.0]], [[-1.0, 0.0]]], dtype=np.float32)
        levels = np.asarray([[[0.0, 1.0]], [[0.0, 1.0]]], dtype=np.float32)
        weekday = np.asarray([1, 2], dtype=np.int64)
        slot = np.asarray([7, 7], dtype=np.int64)

        enriched = add_key_and_control_distances(
            records,
            current_keys=current,
            reference_keys=reference,
            random_keys=random,
            level_features=levels,
            weekday=weekday,
            slot=slot,
        )

        self.assertEqual(len(enriched), 1)
        self.assertAlmostEqual(enriched[0]["current_key_distance"], 0.0)
        self.assertAlmostEqual(enriched[0]["reference_key_distance"], 1.0)
        self.assertAlmostEqual(enriched[0]["random_key_distance"], 2.0)
        self.assertAlmostEqual(enriched[0]["level_distance"], 0.0)
        self.assertTrue(enriched[0]["calendar_compatible"])

    def test_quadrants_retain_context_similar_future_similar_pairs(self) -> None:
        records = []
        fixtures = (
            ("ss", 0.05, 0.05, 0.05),
            ("sd", 0.05, 0.05, 0.95),
            ("ds", 0.95, 0.95, 0.05),
            ("dd", 0.95, 0.95, 0.95),
            ("middle", 0.50, 0.50, 0.50),
        )
        for index, (name, context, level, future) in enumerate(fixtures):
            records.append(
                {
                    "pair": name,
                    "node": 0,
                    "i": index,
                    "j": index + 10,
                    "context_distance": context,
                    "level_distance": level,
                    "future_distance": future,
                    "calendar_compatible": True,
                    "current_key_distance": future,
                    "reference_key_distance": future,
                    "random_key_distance": 0.5,
                }
            )

        quadrants, _ = select_context_future_quadrants(records, quantile=0.25)

        self.assertEqual(
            [row["pair"] for row in quadrants["context_similar_future_similar"]],
            ["ss"],
        )
        self.assertEqual(
            [row["pair"] for row in quadrants["context_similar_future_different"]],
            ["sd"],
        )
        self.assertEqual(
            [row["pair"] for row in quadrants["context_different_future_similar"]],
            ["ds"],
        )

    def test_contrasts_report_smaller_keys_for_jointly_similar_pairs(self) -> None:
        quadrants = {
            "context_similar_future_similar": [
                {
                    "current_key_distance": 0.1,
                    "reference_key_distance": 0.2,
                    "random_key_distance": 0.4,
                },
                {
                    "current_key_distance": 0.2,
                    "reference_key_distance": 0.3,
                    "random_key_distance": 0.5,
                },
            ],
            "context_similar_future_different": [
                {
                    "current_key_distance": 0.7,
                    "reference_key_distance": 0.6,
                    "random_key_distance": 0.5,
                }
            ],
            "context_different_future_similar": [
                {
                    "current_key_distance": 0.4,
                    "reference_key_distance": 0.5,
                    "random_key_distance": 0.6,
                }
            ],
            "context_different_future_different": [],
        }

        result = summarize_quadrant_contrasts(quadrants)

        self.assertAlmostEqual(
            result["current"]["joint_similar_minus_context_similar_future_different"],
            -0.55,
        )
        self.assertAlmostEqual(
            result["current"]["joint_similar_minus_context_different_future_similar"],
            -0.25,
        )
        self.assertAlmostEqual(
            result["joint_similar_current_minus_random"],
            -0.3,
        )

    def test_cli_exposes_current_reference_and_random_banks(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/analyze_context_key_alignment.py", "--help"],
            capture_output=True,
            check=False,
            text=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--bank", result.stdout)
        self.assertIn("--reference-bank", result.stdout)
        self.assertIn("--random-bank", result.stdout)


if __name__ == "__main__":
    unittest.main()
