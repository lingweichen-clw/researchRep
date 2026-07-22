from __future__ import annotations

import math
import unittest

import torch

from stanchor.metrics import ForecastMetricAccumulator, select_common_horizon_metrics


class ForecastMetricsTest(unittest.TestCase):
    def test_common_horizons_select_15_30_60_minutes(self) -> None:
        metrics = {
            "horizon_mae": [float(index) for index in range(12)],
            "horizon_rmse": [float(index + 100) for index in range(12)],
            "horizon_mape": [float(index + 200) for index in range(12)],
        }

        selected = select_common_horizon_metrics(metrics, frequency_minutes=5)

        self.assertEqual(
            selected,
            {
                "15min": {"mae": 2.0, "rmse": 102.0, "mape": 202.0},
                "30min": {"mae": 5.0, "rmse": 105.0, "mape": 205.0},
                "60min": {"mae": 11.0, "rmse": 111.0, "mape": 211.0},
            },
        )

    def test_overall_and_horizon_metrics_use_the_same_observed_values(self) -> None:
        prediction = torch.tensor(
            [[[[2.0], [1.0]], [[4.0], [8.0]]]],
        )
        target = torch.tensor(
            [[[[1.0], [3.0]], [[2.0], [4.0]]]],
        )
        observed = torch.ones_like(target, dtype=torch.bool)
        accumulator = ForecastMetricAccumulator(horizon=2)

        accumulator.update(prediction, target, observed)
        result = accumulator.compute()

        self.assertAlmostEqual(result["mae"], 2.25)
        self.assertAlmostEqual(result["rmse"], 2.5)
        self.assertAlmostEqual(
            result["mape"],
            100.0 * (1.0 + 2.0 / 3.0 + 1.0 + 1.0) / 4.0,
            places=5,
        )
        self.assertEqual(len(result["horizon_mae"]), 2)
        self.assertEqual(len(result["horizon_rmse"]), 2)
        self.assertEqual(len(result["horizon_mape"]), 2)
        self.assertAlmostEqual(result["horizon_mae"][0], 1.5)
        self.assertAlmostEqual(result["horizon_mae"][1], 3.0)
        self.assertAlmostEqual(result["horizon_rmse"][0], math.sqrt(2.5))
        self.assertAlmostEqual(result["horizon_rmse"][1], math.sqrt(10.0))
        self.assertAlmostEqual(
            result["horizon_mape"][0],
            100.0 * (1.0 + 2.0 / 3.0) / 2.0,
            places=5,
        )
        self.assertAlmostEqual(result["horizon_mape"][1], 100.0)


if __name__ == "__main__":
    unittest.main()
