from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

from stanchor.config import load_config


class PretrainingCliTest(unittest.TestCase):
    def test_pretrain_help_exposes_seed_override(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "scripts/pretrain.py", "--help"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("--seed", result.stdout)

    def test_e5_configs_freeze_teacher_contract(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        cases = (
            (
                "configs/metrla_e5_offset_decay_relation_v1.yaml",
                "offset_decay",
                0.0,
            ),
            (
                "configs/metrla_e5_offset_decay_increment_relation_v1.yaml",
                "offset_decay_increment",
                0.5,
            ),
            (
                "configs/pemsbay_e5_offset_decay_transfer_level0_v1.yaml",
                "offset_decay",
                0.0,
            ),
            (
                "configs/pemsbay_e5_offset_decay_increment_transfer_level0_v1.yaml",
                "offset_decay_increment",
                0.5,
            ),
        )
        for relative_path, expected_mode, expected_weight in cases:
            with self.subTest(config=relative_path):
                config = load_config(project_root / relative_path)
                self.assertEqual(config.pretrain.relation_teacher_mode, expected_mode)
                self.assertEqual(
                    config.pretrain.relation_distance_normalization,
                    "anchor_mean",
                )
                self.assertEqual(
                    config.pretrain.future_increment_weight,
                    expected_weight,
                )
                self.assertEqual(
                    config.target.downstream_mode,
                    "learned_topk_offset_decay_horizon",
                )


if __name__ == "__main__":
    unittest.main()
