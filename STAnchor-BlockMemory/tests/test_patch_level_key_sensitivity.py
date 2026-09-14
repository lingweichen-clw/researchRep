from __future__ import annotations

import subprocess
import sys
import unittest

import numpy as np

from stanchor.diagnostics.patch_level_key_sensitivity import (
    pairwise_cosine_key_distance,
    patch_component_spearman,
    patch_component_spearman_by_node,
    patch_level_statistics,
)


class PatchLevelKeySensitivityTest(unittest.TestCase):
    def test_patch_statistics_preserve_absolute_and_global_relative_levels(self) -> None:
        values = np.asarray([0.0, 2.0, 4.0, 6.0], dtype=np.float32).reshape(1, 4, 1, 1)
        observed = np.ones_like(values, dtype=bool)

        absolute, relative, valid = patch_level_statistics(
            values,
            observed,
            num_patches=2,
            eps=0.0,
        )

        self.assertEqual(absolute.shape, (1, 2, 1, 1, 4))
        np.testing.assert_allclose(absolute[0, :, 0, 0, 0], [1.0, 5.0])
        np.testing.assert_allclose(absolute[0, :, 0, 0, 1], [1.0, 1.0])
        np.testing.assert_allclose(absolute[0, :, 0, 0, 2], [2.0, 6.0])
        np.testing.assert_allclose(absolute[0, :, 0, 0, 3], [2.0, 2.0])
        global_std = np.sqrt(5.0)
        np.testing.assert_allclose(relative[0, :, 0, 0, 0], [-2.0 / global_std, 2.0 / global_std])
        np.testing.assert_allclose(relative[0, :, 0, 0, 3], [2.0 / global_std] * 2)
        self.assertTrue(valid.all())

    def test_missing_patch_is_marked_invalid(self) -> None:
        values = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape(1, 4, 1, 1)
        observed = np.asarray([False, False, True, True]).reshape(1, 4, 1, 1)

        absolute, relative, valid = patch_level_statistics(values, observed, num_patches=2)

        self.assertFalse(valid[0, 0, 0, 0])
        self.assertTrue(valid[0, 1, 0, 0])
        self.assertTrue(np.isfinite(absolute).all())
        self.assertTrue(np.isfinite(relative).all())

    def test_key_distance_and_patch_profile_use_the_same_event_pairs(self) -> None:
        keys = np.asarray(
            [
                [[1.0, 0.0]],
                [[1.0, 0.0]],
                [[0.0, 1.0]],
            ],
            dtype=np.float32,
        )
        pairs = np.asarray([[0, 1], [0, 2]], dtype=np.int64)
        key_distance = pairwise_cosine_key_distance(keys, pairs)
        self.assertEqual(key_distance.shape, (2, 1))
        np.testing.assert_allclose(key_distance[:, 0], [0.0, 1.0])

        levels = np.zeros((3, 2, 1, 1, 4), dtype=np.float32)
        levels[:, 0, 0, 0, 0] = [0.0, 0.0, 2.0]
        levels[:, 1, 0, 0, 0] = [0.0, 2.0, 0.0]
        valid = np.ones((3, 2, 1, 1), dtype=bool)

        profile = patch_component_spearman(key_distance, levels, valid, pairs)

        self.assertEqual(profile.shape, (2, 1, 4))
        self.assertAlmostEqual(profile[0, 0, 0], 1.0)
        self.assertAlmostEqual(profile[1, 0, 0], -1.0)

    def test_nodewise_profile_keeps_sensor_specific_correlations_separate(self) -> None:
        pairs = np.asarray([[0, 1], [0, 2]], dtype=np.int64)
        key_distance = np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        levels = np.zeros((3, 1, 2, 1, 4), dtype=np.float32)
        levels[:, 0, 0, 0, 0] = [0.0, 0.0, 2.0]
        levels[:, 0, 1, 0, 0] = [0.0, 2.0, 0.0]
        valid = np.ones((3, 1, 2, 1), dtype=bool)

        profile = patch_component_spearman_by_node(
            key_distance,
            levels,
            valid,
            pairs,
        )

        self.assertEqual(profile.shape, (1, 2, 1, 4))
        self.assertAlmostEqual(profile[0, 0, 0, 0], 1.0)
        self.assertAlmostEqual(profile[0, 1, 0, 0], -1.0)

    def test_cli_exposes_matched_bank_controls(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/analyze_patch_level_key_sensitivity.py", "--help"],
            capture_output=True,
            check=False,
            text=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--bank", result.stdout)
        self.assertIn("--reference-bank", result.stdout)
        self.assertIn("--random-bank", result.stdout)
        self.assertIn("--output", result.stdout)


if __name__ == "__main__":
    unittest.main()
