from __future__ import annotations

import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ExperimentQueueContractTest(unittest.TestCase):
    def test_cfdp_probe_queue_covers_both_context_lengths_and_oracle(self) -> None:
        path = PROJECT_ROOT / "scripts" / "run_cfdp_probe_queue.ps1"
        self.assertTrue(path.exists())
        text = path.read_text(encoding="utf-8")
        self.assertIn("metrla_e5_final_sym_profile_local12_v1.yaml", text)
        self.assertIn("metrla_e5_final_sym_profile_v1.yaml", text)
        self.assertIn("diagnose_cfdp_probes.py", text)
        self.assertIn("--epochs", text)
        self.assertIn("--output", text)
        self.assertIn("teacher_profile_oracle", text)

    def test_downstream_attribution_queue_controls_selector_variables(self) -> None:
        path = PROJECT_ROOT / "scripts" / "run_global288_downstream_attribution_queue.ps1"
        self.assertTrue(path.exists())
        text = path.read_text(encoding="utf-8")
        self.assertIn("base_only", text)
        self.assertIn("learned_topk_offset_decay_horizon", text)
        self.assertIn("--candidate-protocol", text)
        self.assertIn("exact_calendar", text)
        self.assertIn("--level-weight", text)
        self.assertIn('"0"', text)
        self.assertIn("init_random_checkpoint.py", text)
        self.assertIn("diagnose_downstream.py", text)

    def test_profile_weight_ablation_queue_is_zero_training_and_three_point_only(self) -> None:
        path = PROJECT_ROOT / "scripts" / "run_profile_weight_ablation_queue.ps1"
        self.assertTrue(path.exists())
        text = path.read_text(encoding="utf-8")
        self.assertIn("metrla_e5_final_sym_profile_v1.yaml", text)
        self.assertIn("pretrain_best_relation.pt", text)
        self.assertIn("visualize_retrieval.py", text)
        self.assertIn("--profile-weight-override", text)
        self.assertIn("exact_calendar", text)
        self.assertIn("--level-weight", text)
        self.assertIn('"0"', text)
        self.assertIn('Gamma = "0"', text)
        self.assertIn('Gamma = "0.25"', text)
        self.assertIn('Gamma = "1"', text)
        self.assertNotIn("build_bank.py", text)
        self.assertNotIn("train_pretrain.py", text)
        self.assertNotIn("train_downstream.py", text)

    def test_cc_fgda_queue_runs_pure_latent_and_conditioned_pairs(self) -> None:
        path = PROJECT_ROOT / "scripts" / "run_e5_latent48_cc_fgda_global288_queue.ps1"
        self.assertTrue(path.exists())
        text = path.read_text(encoding="utf-8")
        self.assertIn("metrla_e5_final_latent48_global288_v1.yaml", text)
        self.assertIn("metrla_e5_final_latent48_cc_fgda_global288_v1.yaml", text)
        self.assertIn('Stage = "pretrain"', text)
        self.assertIn('ValidateSet("pretrain")', text)
        self.assertIn("cc_fgda", text)
        self.assertNotIn("build_bank.py", text)
        self.assertNotIn("diagnose_retrieval.py", text)
        self.assertNotIn("train_downstream.py", text)

    def test_cc_fgda_local_queue_consumes_checkpoints_and_runs_posttrain(self) -> None:
        path = PROJECT_ROOT / "scripts" / "run_e5_latent48_cc_fgda_global288_local_queue.ps1"
        self.assertTrue(path.exists())
        text = path.read_text(encoding="utf-8")
        self.assertIn("Latent48Checkpoint", text)
        self.assertIn("CcFgdaCheckpoint", text)
        self.assertIn("build_bank.py", text)
        self.assertIn("diagnose_retrieval.py", text)
        self.assertIn("visualize_retrieval.py", text)
        self.assertIn("train_downstream.py", text)
        self.assertIn("learned_topk_offset_decay_horizon", text)
        self.assertIn("exact_calendar", text)
        self.assertIn("--level-weight", text)


if __name__ == "__main__":
    unittest.main()
