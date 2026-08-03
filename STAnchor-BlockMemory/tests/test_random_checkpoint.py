from __future__ import annotations

import unittest

import torch

from stanchor.config import DataConfig, ExperimentConfig, ModelConfig, PretrainConfig
from stanchor.engine.common import build_pretrain_model
from stanchor.engine.random_checkpoint import build_random_checkpoint_payload


class RandomCheckpointTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ExperimentConfig(
            data=DataConfig(
                raw_path="unused.h5",
                adjacency_path="unused.pkl",
                context_length=12,
                retrieval_context_length=12,
            ),
            model=ModelConfig(
                patch_size=3,
                hidden_dim=16,
                retrieval_dim=8,
                num_heads=4,
                encoder_layers=1,
                dropout=0.0,
            ),
            pretrain=PretrainConfig(time_mask_block_size=3),
        )
        self.normalizer = {"mean": [[1.0]], "std": [[2.0]]}

    def _payload(self, seed: int) -> dict:
        return build_random_checkpoint_payload(
            config=self.config,
            slots_per_day=288,
            normalizer=self.normalizer,
            graph_fingerprint="target-graph",
            seed=seed,
        )

    def test_same_seed_is_reproducible_and_strictly_loadable(self) -> None:
        first = self._payload(seed=42)
        second = self._payload(seed=42)

        self.assertEqual(first["checkpoint_kind"], "target_random_untrained")
        self.assertEqual(first["epoch"], 0)
        self.assertEqual(first["seed"], 42)
        self.assertEqual(first["trained_steps"], 0)
        self.assertEqual(first["graph_fingerprint"], "target-graph")
        self.assertEqual(first["normalizer"], self.normalizer)
        self.assertEqual(first["retrieval_fingerprint"], second["retrieval_fingerprint"])
        for name, tensor in first["model_state_dict"].items():
            self.assertTrue(torch.equal(tensor, second["model_state_dict"][name]), name)

        restored = build_pretrain_model(self.config, slots_per_day=288)
        restored.load_state_dict(first["model_state_dict"], strict=True)
        self.assertEqual(restored.retrieval_fingerprint(), first["retrieval_fingerprint"])

    def test_different_seed_changes_retrieval_fingerprint(self) -> None:
        first = self._payload(seed=42)
        second = self._payload(seed=2024)

        self.assertNotEqual(first["retrieval_fingerprint"], second["retrieval_fingerprint"])


if __name__ == "__main__":
    unittest.main()
