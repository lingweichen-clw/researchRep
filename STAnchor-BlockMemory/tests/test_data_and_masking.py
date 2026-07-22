from __future__ import annotations

import unittest

import numpy as np
import torch

from stanchor.data.dataset import TrafficSeries, TrafficWindowDataset
from stanchor.data.masking import StructuredMaskSampler
from stanchor.data.normalization import NodeStandardScaler, normalize_window


class DataAndMaskingTest(unittest.TestCase):
    def test_node_scaler_uses_observed_values(self) -> None:
        values = np.array([[[1.0]], [[3.0]], [[100.0]]], dtype=np.float32)
        observed = np.array([[[True]], [[True]], [[False]]])
        scaler = NodeStandardScaler.fit(values, observed)
        self.assertAlmostEqual(float(scaler.mean[0, 0]), 2.0, places=5)
        transformed = scaler.transform(values, observed)
        self.assertEqual(float(transformed[-1, 0, 0]), 0.0)
        restored = scaler.inverse_transform_torch(torch.from_numpy(transformed[:2]).unsqueeze(0))
        self.assertTrue(torch.allclose(restored.squeeze(0), torch.from_numpy(values[:2]), atol=1.0e-5))

    def test_window_dataset_contract(self) -> None:
        length, nodes = 40, 3
        values = np.arange(length * nodes, dtype=np.float32).reshape(length, nodes, 1) + 1
        observed = np.ones_like(values, dtype=bool)
        series = TrafficSeries(
            values=values,
            observed=observed,
            timestamps_ns=np.arange(length, dtype=np.int64),
            weekday=np.arange(length, dtype=np.int64) % 7,
            slot=np.arange(length, dtype=np.int64) % 288,
            slots_per_day=288,
        )
        scaler = NodeStandardScaler.fit(values[:24], observed[:24])
        dataset = TrafficWindowDataset(series, scaler, 0, 24, 12, 6)
        item = dataset[0]
        self.assertEqual(tuple(item["x"].shape), (12, nodes, 1))
        self.assertEqual(tuple(item["y"].shape), (6, nodes, 1))
        self.assertEqual(int(item["context_start"]), 0)
        self.assertEqual(int(item["future_end"]), 17)

    def test_mask_aware_window_statistics(self) -> None:
        values = torch.tensor([[[[1.0]], [[2.0]], [[100.0]], [[4.0]]]])
        observed = torch.tensor([[[[True]], [[True]], [[False]], [[True]]]])
        stats = normalize_window(values, observed)
        self.assertAlmostEqual(float(stats.mean), 7.0 / 3.0, places=5)
        self.assertEqual(tuple(stats.level_features.shape), (1, 1, 4))
        self.assertTrue(bool(stats.level_valid.item()))

    def test_time_mask_hides_exactly_one_patch_for_all_nodes(self) -> None:
        sampler = StructuredMaskSampler(12, 3, time_ratio=0.25)
        neighbors = torch.ones((5, 5), dtype=torch.bool)
        result = sampler.sample(3, 5, 1, neighbors, "cpu", task="time")
        self.assertEqual(tuple(result.patch_mask.shape), (3, 4, 5))
        self.assertTrue(torch.equal(result.patch_mask[:, :, 0], result.patch_mask[:, :, 4]))
        self.assertTrue(torch.equal(result.patch_mask[:, :, 0].sum(dim=1), torch.ones(3, dtype=torch.long)))
        self.assertTrue(torch.equal(result.value_mask.sum(dim=(1, 2, 3)), torch.full((3,), 15)))

    def test_time_mask_only_selects_observed_patch(self) -> None:
        sampler = StructuredMaskSampler(12, 3, time_ratio=0.25)
        neighbors = torch.ones((3, 3), dtype=torch.bool)
        observed = torch.zeros((1, 12, 3, 1), dtype=torch.bool)
        observed[:, 6:9] = True
        result = sampler.sample(
            1, 3, 1, neighbors, "cpu", observed=observed, task="time"
        )
        self.assertTrue(bool(result.patch_mask[0, 2].all()))
        self.assertEqual(int(result.patch_mask.sum()), 3)

    def test_time_mask_uses_contiguous_raw_step_block_with_patch_size_one(self) -> None:
        sampler = StructuredMaskSampler(
            12,
            1,
            time_ratio=0.25,
            time_block_size=3,
        )
        neighbors = torch.ones((5, 5), dtype=torch.bool)
        generator = torch.Generator().manual_seed(17)
        result = sampler.sample(
            4,
            5,
            1,
            neighbors,
            "cpu",
            task="time",
            generator=generator,
        )

        self.assertEqual(tuple(result.patch_mask.shape), (4, 12, 5))
        self.assertTrue(torch.equal(result.patch_mask[:, :, 0], result.patch_mask[:, :, -1]))
        for sample_mask in result.patch_mask[:, :, 0]:
            indices = torch.where(sample_mask)[0]
            self.assertEqual(indices.numel(), 3)
            self.assertTrue(torch.equal(indices, torch.arange(indices[0], indices[0] + 3)))
        self.assertTrue(
            torch.equal(result.value_mask.sum(dim=(1, 2, 3)), torch.full((4,), 15))
        )

    def test_time_mask_block_must_align_with_token_patches(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible by patch_size"):
            StructuredMaskSampler(12, 2, time_block_size=3)

    def test_space_mask_preserves_visible_neighbor(self) -> None:
        nodes = 8
        neighbors = torch.zeros((nodes, nodes), dtype=torch.bool)
        for node in range(nodes):
            neighbors[node, (node - 1) % nodes] = True
            neighbors[node, (node + 1) % nodes] = True
        sampler = StructuredMaskSampler(12, 3, space_ratio=0.25)
        result = sampler.sample(4, nodes, 1, neighbors, "cpu", task="space")
        node_mask = result.patch_mask[:, 0]
        self.assertTrue(torch.equal(node_mask.sum(dim=1), torch.full((4,), 2)))
        for mask in node_mask:
            visible_neighbor = (neighbors & (~mask).unsqueeze(0)).any(dim=1)
            self.assertTrue(bool(visible_neighbor[mask].all()))
            self.assertTrue(torch.equal(result.patch_mask[0, 0], result.patch_mask[0, -1]))


if __name__ == "__main__":
    unittest.main()
