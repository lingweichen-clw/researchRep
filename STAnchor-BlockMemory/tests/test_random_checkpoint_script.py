from __future__ import annotations

import unittest

from stanchor.engine.random_checkpoint import build_random_checkpoint_payload


class RandomCheckpointPayloadTest(unittest.TestCase):
    def test_random_checkpoint_payload_has_zero_training_steps(self) -> None:
        from stanchor.config import load_config

        config = load_config("configs/cross_dataset_pems08_source_encoder_stage1.yaml")
        payload = build_random_checkpoint_payload(
            config,
            slots_per_day=288,
            normalizer={"mean": [0.0], "std": [1.0]},
            graph_fingerprint="graph",
            seed=123,
        )

        self.assertEqual(payload["checkpoint_kind"], "target_random_untrained")
        self.assertEqual(payload["trained_steps"], 0)
        self.assertEqual(payload["seed"], 123)
        self.assertTrue(payload["retrieval_fingerprint"])


if __name__ == "__main__":
    unittest.main()
