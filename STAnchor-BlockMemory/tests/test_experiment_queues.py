from __future__ import annotations

import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ExperimentQueueContractTest(unittest.TestCase):
    def test_matched_fulltrain_queue_contains_four_reruns_and_formal_protocol(self) -> None:
        path = PROJECT_ROOT / "scripts" / "run_tgge_v3_matched_fulltrain_queue.ps1"
        self.assertTrue(path.exists())
        text = path.read_text(encoding="utf-8")
        for name in (
            "downstream_tgge_v3_stgcn_base_only_fulltrain_seed42",
            "downstream_tgge_v3_stgcn_error_aware_fulltrain_seed42",
            "downstream_tgge_v3_graphwavenet_base_only_fulltrain_seed42",
            "downstream_tgge_v3_graphwavenet_error_aware_fulltrain_seed42",
        ):
            self.assertIn(name, text)
        self.assertIn('"--disable-early-stopping"', text)
        self.assertIn("pretrain_best_relation.pt", text)
        self.assertIn("StructuredErrorCorrector", text)

    def test_current_v3_error_aware_queue_has_three_latest_downstream_runs(self) -> None:
        path = PROJECT_ROOT / "scripts" / "run_tgge_v3_downstream_queue.ps1"
        self.assertTrue(path.exists())
        text = path.read_text(encoding="utf-8")
        for name in (
            "downstream_tgge_v3_stgcn_error_aware_seed42",
            "downstream_tgge_v3_graphwavenet_horizon_only_seed42",
            "downstream_tgge_v3_graphwavenet_error_aware_seed42",
        ):
            self.assertIn(name, text)
        self.assertIn("StructuredErrorCorrector", text)
        self.assertIn("--base-checkpoint", text)
        self.assertIn("--candidate-protocol", text)
        self.assertIn("exact_calendar", text)
        self.assertIn("--level-weight", text)

    def test_posthoc_queue_uses_only_latent48_frozen_base_inputs(self) -> None:
        path = PROJECT_ROOT / "scripts" / "run_global288_controlled_init_queue.ps1"
        self.assertTrue(path.exists())
        text = path.read_text(encoding="utf-8")
        self.assertIn("--pretrained-checkpoint", text)
        self.assertIn("latent48_pretrained_offset_decay", text)
        self.assertIn("--candidate-protocol", text)
        self.assertIn("exact_calendar", text)
        self.assertIn("--level-weight", text)
        self.assertIn('"0"', text)
        self.assertIn("diagnose_downstream.py", text)
        self.assertNotIn("profile", text.lower())
        self.assertNotIn("fgda", text.lower())

    def test_posthoc_9feature_queue_is_basecap_only_and_isolated(self) -> None:
        path = PROJECT_ROOT / "scripts" / "run_e5a_relaxed_downstream_queue.ps1"
        self.assertTrue(path.exists())
        text = path.read_text(encoding="utf-8")
        self.assertIn("downstream_candidate_protocol", text)
        self.assertIn("e5a_relaxed_pretrained_seed42", text)
        self.assertIn("--candidate-protocol", text)
        self.assertIn("relaxed_calendar", text)
        self.assertIn("diagnose_downstream.py", text)

if __name__ == "__main__":
    unittest.main()
