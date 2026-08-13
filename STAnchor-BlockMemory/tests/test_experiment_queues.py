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


if __name__ == "__main__":
    unittest.main()
