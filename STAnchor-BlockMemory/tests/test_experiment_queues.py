from __future__ import annotations

import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ExperimentQueueContractTest(unittest.TestCase):
    def test_posthoc_queue_uses_only_latent48_frozen_base_inputs(self) -> None:
        path = PROJECT_ROOT / "scripts" / "run_global288_posthoc_error_aware_queue.ps1"
        self.assertTrue(path.exists())
        text = path.read_text(encoding="utf-8")
        self.assertIn("--base-checkpoint", text)
        self.assertIn("latent48_posthoc_error_aware_basecap", text)
        self.assertIn("latent48_posthoc_error_aware_wide", text)
        self.assertIn("--candidate-protocol", text)
        self.assertIn("exact_calendar", text)
        self.assertIn("--level-weight", text)
        self.assertIn('"0"', text)
        self.assertIn("diagnose_downstream.py", text)
        self.assertNotIn("profile", text.lower())
        self.assertNotIn("fgda", text.lower())

    def test_posthoc_9feature_queue_is_basecap_only_and_isolated(self) -> None:
        path = PROJECT_ROOT / "scripts" / "run_global288_posthoc_error_aware_9feature_queue.ps1"
        self.assertTrue(path.exists())
        text = path.read_text(encoding="utf-8")
        self.assertIn("downstream_global288_posthoc_error_aware_9feature", text)
        self.assertIn("latent48_posthoc_error_aware_basecap", text)
        self.assertNotIn("latent48_posthoc_error_aware_wide", text)
        self.assertIn("--base-checkpoint", text)
        self.assertIn("--candidate-protocol", text)
        self.assertIn("exact_calendar", text)
        self.assertIn("diagnose_downstream.py", text)

if __name__ == "__main__":
    unittest.main()
