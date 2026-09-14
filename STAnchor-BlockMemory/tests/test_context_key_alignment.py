from __future__ import annotations

import subprocess
import sys
import unittest

import numpy as np

from stanchor.diagnostics.context_key_alignment import (
    add_key_and_control_distances,
    select_context_future_quadrants,
    summarize_quadrant_contrasts,
)


class ContextKeyAlignmentTest(unittest.TestCase):
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
