from __future__ import annotations

import importlib
import importlib.util
import unittest

import torch


class RetrievalTemperatureTest(unittest.TestCase):
    @staticmethod
    def _module():
        module_name = "stanchor.diagnostics.retrieval_temperature"
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            return None
        return importlib.import_module(module_name)

    def test_weight_settings_preserve_support_and_control_concentration(self) -> None:
        module = self._module()
        self.assertIsNotNone(
            module,
            "retrieval-temperature diagnostic module has not been implemented",
        )
        scores = torch.tensor([[[2.0, 1.0, -5.0]]])
        valid = torch.tensor([[[True, True, False]]])

        uniform = module.retrieval_weights(scores, valid, "uniform")
        temperature_020 = module.retrieval_weights(scores, valid, 0.20)
        temperature_005 = module.retrieval_weights(scores, valid, 0.05)
        hard_top1 = module.retrieval_weights(scores, valid, "hard_top1")

        self.assertTrue(torch.equal(uniform > 0, valid))
        self.assertTrue(torch.equal(temperature_020 > 0, valid))
        self.assertTrue(torch.equal(temperature_005 > 0, valid))
        self.assertTrue(torch.allclose(uniform, torch.tensor([[[0.5, 0.5, 0.0]]])))
        self.assertGreater(float(temperature_005[..., 0]), float(temperature_020[..., 0]))
        self.assertTrue(torch.allclose(temperature_020.sum(dim=-1), torch.ones(1, 1)))
        self.assertTrue(torch.allclose(temperature_005.sum(dim=-1), torch.ones(1, 1)))
        self.assertTrue(torch.equal(hard_top1, torch.tensor([[[1.0, 0.0, 0.0]]])))

    def test_weighted_payload_aggregation_renormalizes_each_missing_point(self) -> None:
        module = self._module()
        self.assertIsNotNone(
            module,
            "retrieval-temperature diagnostic module has not been implemented",
        )
        # candidate_futures: [B=1, H=2, N=1, K=3, C=1]
        candidate_futures = torch.tensor(
            [10.0, 20.0, 999.0, 10.0, 20.0, 999.0]
        ).reshape(1, 2, 1, 3, 1)
        candidate_masks = torch.tensor(
            [True, True, False, False, True, False]
        ).reshape_as(candidate_futures)
        weights = torch.tensor([[[0.75, 0.25, 0.0]]])

        result = module.aggregate_weighted_candidates(
            candidate_futures,
            candidate_masks,
            weights,
        )

        self.assertTrue(torch.allclose(result.prediction.flatten(), torch.tensor([12.5, 20.0])))
        self.assertTrue(torch.allclose(result.variance.flatten(), torch.tensor([18.75, 0.0])))
        self.assertTrue(bool(result.valid.all()))

    def test_common_mask_is_computed_within_each_setting(self) -> None:
        module = self._module()
        self.assertTrue(
            hasattr(module, "common_setting_masks"),
            "per-setting common evaluation masks have not been implemented",
        )
        target_valid = torch.tensor([[[[True]], [[True]]]])
        output_valid = {
            "pretrained": {
                "temperature_0.10": torch.tensor([[[[True]], [[True]]]]),
                "hard_top1": torch.tensor([[[[True]], [[False]]]]),
            },
            "random": {
                "temperature_0.10": torch.tensor([[[[True]], [[True]]]]),
                "hard_top1": torch.tensor([[[[True]], [[True]]]]),
            },
        }

        masks = module.common_setting_masks(target_valid, output_valid)

        self.assertTrue(torch.equal(masks["temperature_0.10"], target_valid))
        self.assertTrue(
            torch.equal(
                masks["hard_top1"],
                torch.tensor([[[[True]], [[False]]]]),
            )
        )

    def test_full_runner_rejects_test_split_before_loading_assets(self) -> None:
        module = self._module()
        self.assertIsNotNone(
            module,
            "retrieval-temperature diagnostic module has not been implemented",
        )
        with self.assertRaisesRegex(ValueError, "validation"):
            module.run_retrieval_temperature_sweep(
                config=None,
                checkpoint_path="missing.pt",
                bank_path="missing-bank",
                random_checkpoint_path="missing-random.pt",
                random_bank_path="missing-random-bank",
                output_dir="unused",
                split="test",
            )


if __name__ == "__main__":
    unittest.main()
