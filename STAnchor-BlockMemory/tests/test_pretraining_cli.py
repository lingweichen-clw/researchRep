from __future__ import annotations

from dataclasses import asdict
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

    def test_visualization_help_exposes_level_weight_override(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "scripts/visualize_retrieval.py", "--help"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("--level-weight", result.stdout)

    def test_downstream_help_exposes_level_weight_override(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, "scripts/train_downstream.py", "--help"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("--level-weight", result.stdout)

    def test_e5_offset_configs_freeze_teacher_contract(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        cases = (
            (
                "configs/metrla_e5_offset_decay_relation_v1.yaml",
                "offset_decay",
                0.0,
            ),
            (
                "configs/pemsbay_e5_offset_decay_transfer_level0_v1.yaml",
                "offset_decay",
                0.0,
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

    def test_e5_final_r1_configs_differ_only_in_teacher_normalization_and_outputs(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        anchor = asdict(load_config(project_root / "configs/metrla_e5_final_anchor_mean_v1.yaml"))
        symnorm = asdict(load_config(project_root / "configs/metrla_e5_final_symnorm_v1.yaml"))

        allowed = {
            ("pretrain", "relation_distance_normalization"),
            ("bank", "output_dir"),
            ("runtime", "run_name"),
        }
        differences = {
            (section, key)
            for section, values in anchor.items()
            for key, value in values.items()
            if value != symnorm[section][key]
        }
        self.assertEqual(differences, allowed)
        self.assertEqual(anchor["pretrain"]["relation_distance_normalization"], "anchor_mean")
        self.assertEqual(
            symnorm["pretrain"]["relation_distance_normalization"],
            "symmetric_geometric_mean",
        )
        self.assertEqual(anchor["model"]["profile_dim"], 0)
        self.assertEqual(symnorm["model"]["profile_dim"], 0)

    def test_e5_final_local12_uses_one_token_per_five_minute_step(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        global288_config = load_config(
            project_root / "configs/metrla_e5_final_symnorm_v1.yaml"
        )
        local12_config = load_config(
            project_root / "configs/metrla_e5_final_symnorm_local12_v1.yaml"
        )
        global288 = asdict(global288_config)
        local12 = asdict(local12_config)

        allowed = {
            ("data", "retrieval_context_length"),
            ("model", "patch_size"),
            ("pretrain", "time_mask_block_size"),
            ("bank", "output_dir"),
            ("runtime", "run_name"),
        }
        differences = {
            (section, key)
            for section, values in global288.items()
            for key, value in values.items()
            if value != local12[section][key]
        }
        self.assertEqual(differences, allowed)

        self.assertEqual(local12_config.data.context_length, 12)
        self.assertEqual(local12_config.data.encoder_context_length, 12)
        self.assertEqual(local12_config.model.patch_size, 1)
        self.assertEqual(
            local12_config.data.encoder_context_length
            // local12_config.model.patch_size,
            12,
        )
        self.assertEqual(local12_config.pretrain.time_mask_block_size, 3)
        self.assertEqual(local12_config.model.profile_dim, 0)
        self.assertEqual(
            local12_config.pretrain.relation_teacher_mode,
            "offset_decay",
        )
        self.assertEqual(
            local12_config.pretrain.relation_distance_normalization,
            "symmetric_geometric_mean",
        )
        local12_config.validate()

    def test_e5_final_local12_profile_changes_only_cfdp_fields_and_outputs(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        no_profile = asdict(
            load_config(project_root / "configs/metrla_e5_final_symnorm_local12_v1.yaml")
        )
        profile = asdict(
            load_config(
                project_root / "configs/metrla_e5_final_sym_profile_local12_v1.yaml"
            )
        )

        allowed = {
            ("model", "profile_dim"),
            ("model", "latent_dim"),
            ("pretrain", "profile_loss_weight"),
            ("bank", "output_dir"),
            ("runtime", "run_name"),
        }
        differences = {
            (section, key)
            for section, values in no_profile.items()
            for key, value in values.items()
            if value != profile[section][key]
        }

        self.assertEqual(differences, allowed)
        self.assertEqual(profile["model"]["profile_dim"], 12)
        self.assertEqual(profile["model"]["latent_dim"], 36)
        self.assertEqual(profile["model"]["retrieval_dim"], 48)
        self.assertEqual(profile["model"]["profile_weight"], 0.25)
        self.assertEqual(profile["pretrain"]["profile_loss_weight"], 0.1)


if __name__ == "__main__":
    unittest.main()
