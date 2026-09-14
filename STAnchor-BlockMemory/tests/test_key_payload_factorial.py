from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from stanchor.diagnostics.key_payload_factorial import (
    NON_KEY_BANK_FILES,
    audit_non_key_bank_files,
    factorial_mae_effects,
    query_block_bootstrap_contrast,
    topk_jaccard_aligned,
)


class KeyPayloadFactorialTest(unittest.TestCase):
    def test_factorial_effects_separate_key_payload_and_interaction(self) -> None:
        cells = {
            "key_offset_only__payload_offset_only": {"tail": {"mae": 10.0}},
            "key_offset_only__payload_offset_decay": {"tail": {"mae": 7.0}},
            "key_offset_decay__payload_offset_only": {"tail": {"mae": 8.0}},
            "key_offset_decay__payload_offset_decay": {"tail": {"mae": 6.0}},
        }

        result = factorial_mae_effects(cells)

        self.assertEqual(result["tail"]["payload_effect_under_offset_only_key"], 3.0)
        self.assertEqual(result["tail"]["payload_effect_under_offset_decay_key"], 2.0)
        self.assertEqual(result["tail"]["key_effect_under_offset_only_payload"], 2.0)
        self.assertEqual(result["tail"]["key_effect_under_offset_decay_payload"], 1.0)
        self.assertEqual(result["tail"]["interaction_difference_in_differences"], 1.0)
        self.assertEqual(result["tail"]["average_payload_main_effect"], 2.5)
        self.assertEqual(result["tail"]["average_key_main_effect"], 1.5)
        self.assertEqual(result["tail"]["matched_diagonal_total_effect"], 4.0)

    def test_query_block_bootstrap_preserves_constant_paired_effect(self) -> None:
        delta = np.full((4, 3), 2.0, dtype=np.float64)
        strata = {
            "all": np.ones_like(delta, dtype=bool),
            "first_node": np.asarray(
                [[True, False, False]] * 4,
                dtype=bool,
            ),
        }

        result = query_block_bootstrap_contrast(
            delta,
            strata,
            resamples=200,
            seed=7,
        )

        for summary in result.values():
            self.assertEqual(summary["anchor_weighted_mean"], 2.0)
            self.assertEqual(summary["query_block_bootstrap_ci95"], [2.0, 2.0])
            self.assertEqual(summary["query_left_worse_fraction"], 1.0)

    def test_topk_jaccard_is_order_invariant_and_ignores_padding(self) -> None:
        left = np.asarray([[[1, 2, -1], [4, 5, -1]]])
        right = np.asarray([[[2, 3, -1], [5, 4, -1]]])
        left_valid = left >= 0
        right_valid = right >= 0

        result = topk_jaccard_aligned(left, left_valid, right, right_valid)

        self.assertTrue(np.allclose(result, [[1.0 / 3.0, 1.0]]))

    def test_non_key_bank_audit_hashes_every_required_payload_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left = root / "left"
            right = root / "right"
            left.mkdir()
            right.mkdir()
            for index, name in enumerate(NON_KEY_BANK_FILES):
                payload = f"{index}:{name}".encode("utf-8")
                (left / name).write_bytes(payload)
                (right / name).write_bytes(payload)

            result = audit_non_key_bank_files(left, right)

            self.assertTrue(result["all_identical"])
            self.assertEqual(set(result["files"]), set(NON_KEY_BANK_FILES))
            (right / "future_values.npy").write_bytes(b"different")
            with self.assertRaisesRegex(ValueError, "future_values.npy"):
                audit_non_key_bank_files(left, right)

    def test_cli_exposes_two_keys_and_two_payloads(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/diagnose_key_payload_factorial.py", "--help"],
            capture_output=True,
            check=False,
            text=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--offset-only-checkpoint", result.stdout)
        self.assertIn("--offset-decay-checkpoint", result.stdout)
        self.assertIn("--bootstrap-resamples", result.stdout)
        self.assertIn("--max-queries", result.stdout)


if __name__ == "__main__":
    unittest.main()
