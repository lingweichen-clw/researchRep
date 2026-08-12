from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from stanchor.bank.schema import BankManifest
from stanchor.bank.storage import BankWriter, MemoryBank
from stanchor.config import DataConfig, ExperimentConfig, PretrainConfig
from stanchor.data.normalization import WindowStatistics
from stanchor.losses.pretraining import build_future_relation_targets
from stanchor.losses.pretraining import compute_pretraining_loss
from stanchor.models.pretraining import STAnchorPretrainModel
from stanchor.models.retrieval_head import RetrievalHead
from stanchor.config import ModelConfig
from stanchor.data.graph import graph_from_dense
from stanchor.retrieval.semantic_profile import (
    build_cfdp_teacher,
    compose_profile_latent_key,
    resample_future_profile,
    symmetric_geometric_mean_normalize,
)


class E5FinalContractTest(unittest.TestCase):
    def test_horizon_12_resampling_is_identity_and_mask_aware(self) -> None:
        future = torch.arange(2 * 12, dtype=torch.float32).view(1, 12, 2, 1)
        observed = torch.ones_like(future, dtype=torch.bool)
        observed[:, 5, 1] = False
        values, valid = resample_future_profile(future, observed, profile_size=12)
        self.assertTrue(torch.equal(values, torch.where(observed, future, torch.zeros_like(future))))
        self.assertTrue(torch.equal(valid, observed))

    def test_cfdp_is_invariant_to_event_level_shift_and_positive_scale(self) -> None:
        context = torch.tensor([[[[10.0]], [[11.0]], [[12.0]], [[13.0]]]])
        future = torch.tensor([[[[14.0]], [[15.0]], [[16.0]], [[17.0]]]])
        context_mask = torch.ones_like(context, dtype=torch.bool)
        future_mask = torch.ones_like(future, dtype=torch.bool)
        base, base_valid = build_cfdp_teacher(future, future_mask, context, context_mask)
        shifted, shifted_valid = build_cfdp_teacher(
            future + 100.0, future_mask, context + 100.0, context_mask
        )
        scaled, scaled_valid = build_cfdp_teacher(
            future * 3.0, future_mask, context * 3.0, context_mask
        )
        self.assertTrue(torch.equal(base_valid, shifted_valid))
        self.assertTrue(torch.equal(base_valid, scaled_valid))
        self.assertTrue(torch.allclose(base, shifted, atol=1.0e-5))
        self.assertTrue(torch.allclose(base, scaled, atol=1.0e-5))

    def test_symnorm_is_symmetric_and_key_similarity_decomposes(self) -> None:
        distances = torch.tensor(
            [[[0.0], [2.0], [4.0]], [[2.0], [0.0], [1.0]], [[4.0], [1.0], [0.0]]]
        )
        valid = torch.ones_like(distances, dtype=torch.bool)
        normalized = symmetric_geometric_mean_normalize(distances, valid)
        self.assertTrue(torch.allclose(normalized, normalized.transpose(0, 1)))

        profile = torch.nn.functional.normalize(torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]]), dim=-1)
        latent = torch.nn.functional.normalize(torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]]]), dim=-1)
        total, profile_out, latent_out = compose_profile_latent_key(profile, latent, gamma=0.25)
        total_sim = torch.einsum("bid,bjd->bij", total, total)
        expected = 0.25 * torch.einsum("bid,bjd->bij", profile_out, profile_out)
        expected = expected + 0.75 * torch.einsum("bid,bjd->bij", latent_out, latent_out)
        self.assertTrue(torch.allclose(total_sim, expected, atol=1.0e-5))
        self.assertEqual(total.shape[-1], 4)

    def test_bank_expected_schema_rejects_old_manifest(self) -> None:
        manifest = BankManifest(
            schema_version=1,
            dataset_name="synthetic",
            num_events=1,
            num_nodes=1,
            context_length=4,
            horizon=2,
            channels=1,
            retrieval_dim=2,
            slots_per_day=4,
            key_dtype="float32",
            future_dtype="float32",
            encoder_fingerprint="encoder",
            graph_fingerprint="graph",
            scaler={"mean": [[0.0]], "std": [[1.0]], "eps": 1.0e-6},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            writer = BankWriter(path, manifest)
            writer.write(
                {
                    "event_keys": np.zeros((1, 2), dtype=np.float32),
                    "node_keys": np.zeros((1, 1, 2), dtype=np.float32),
                    "future_values": np.zeros((1, 2, 1, 1), dtype=np.float32),
                    "future_masks": np.ones((1, 2, 1, 1), dtype=np.uint8),
                    "level_features": np.zeros((1, 1, 4), dtype=np.float32),
                    "weekday": np.zeros(1, dtype=np.int16),
                    "slot": np.zeros(1, dtype=np.int16),
                    "context_start": np.zeros(1, dtype=np.int64),
                    "context_end": np.ones(1, dtype=np.int64),
                    "future_end": np.full(1, 3, dtype=np.int64),
                    "sample_id": np.zeros(1, dtype=np.int64),
                }
            )
            writer.finalize()
            with self.assertRaisesRegex(ValueError, "schema version"):
                MemoryBank(path, expected_schema_version=2)

    def test_bank_v2_round_trip_preserves_profile_latent_layout(self) -> None:
        manifest = BankManifest(
            schema_version=2,
            dataset_name="synthetic",
            num_events=1,
            num_nodes=1,
            context_length=12,
            horizon=2,
            channels=1,
            retrieval_dim=48,
            slots_per_day=288,
            key_dtype="float32",
            future_dtype="float32",
            encoder_fingerprint="encoder-v2",
            graph_fingerprint="graph",
            scaler={"mean": [[0.0]], "std": [[1.0]], "eps": 1.0e-6},
            key_layout="canonical_profile_latent",
            profile_dim=12,
            latent_dim=36,
            profile_weight=0.25,
            profile_grid_size=12,
            profile_scale_floor=0.1,
            relation_distance_normalization="symmetric_geometric_mean",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            writer = BankWriter(path, manifest)
            writer.write(
                {
                    "event_keys": np.zeros((1, 48), dtype=np.float32),
                    "node_keys": np.zeros((1, 1, 48), dtype=np.float32),
                    "future_values": np.zeros((1, 2, 1, 1), dtype=np.float32),
                    "future_masks": np.ones((1, 2, 1, 1), dtype=np.uint8),
                    "level_features": np.zeros((1, 1, 4), dtype=np.float32),
                    "weekday": np.zeros(1, dtype=np.int16),
                    "slot": np.zeros(1, dtype=np.int16),
                    "context_start": np.zeros(1, dtype=np.int64),
                    "context_end": np.ones(1, dtype=np.int64),
                    "future_end": np.full(1, 3, dtype=np.int64),
                    "sample_id": np.zeros(1, dtype=np.int64),
                }
            )
            writer.finalize()
            with MemoryBank(path, expected_schema_version=2) as bank:
                self.assertEqual(bank.manifest.profile_dim, 12)
                self.assertEqual(bank.manifest.latent_dim, 36)
                self.assertEqual(bank.manifest.key_layout, "canonical_profile_latent")

    def test_config_accepts_symmetric_geometric_mean_relation_teacher(self) -> None:
        config = ExperimentConfig(
            data=DataConfig(raw_path="data.h5", adjacency_path="adj.pkl"),
            pretrain=PretrainConfig(
                retrieval_loss_mode="relation",
                relation_teacher_mode="offset_decay",
                relation_distance_normalization="symmetric_geometric_mean",
            ),
        )
        config.validate()

    def test_symmetric_teacher_distance_is_pair_symmetric(self) -> None:
        future = torch.tensor(
            [
                [[1.0], [2.0]],
                [[1.2], [2.2]],
                [[4.0], [5.0]],
            ]
        ).unsqueeze(-1)
        observed = torch.ones_like(future, dtype=torch.bool)
        context = torch.tensor(
            [
                [[[0.0]], [[0.0]]],
                [[[0.2]], [[0.2]]],
                [[[3.0]], [[3.0]]],
            ]
        )
        context_observed = torch.ones_like(context, dtype=torch.bool)
        statistics = WindowStatistics(
            normalized=torch.zeros(3, 2, 1, 1),
            level_features=torch.zeros(3, 1, 4),
            level_valid=torch.ones(3, 1, 1, dtype=torch.bool),
            mean=torch.zeros(3, 1, 1),
            std=torch.ones(3, 1, 1),
        )
        targets = build_future_relation_targets(
            future,
            statistics,
            observed,
            torch.tensor([0, 20, 40]),
            torch.tensor([1, 21, 41]),
            relation_teacher_mode="offset_decay",
            forecast_context=context,
            forecast_context_observed=context_observed,
            relation_distance_normalization="symmetric_geometric_mean",
        )
        self.assertTrue(torch.allclose(targets.future_distance, targets.future_distance.transpose(0, 1)))

    def test_profile_latent_retrieval_head_keeps_total_key_at_48_dimensions(self) -> None:
        head = RetrievalHead(
            hidden_dim=96,
            retrieval_dim=48,
            profile_dim=12,
            latent_dim=36,
            profile_weight=0.25,
        )
        output = head(torch.randn(2, 4, 3, 96))
        self.assertEqual(tuple(output.profile_prediction.shape), (2, 3, 12))
        self.assertEqual(tuple(output.profile_keys.shape), (2, 3, 12))
        self.assertEqual(tuple(output.latent_keys.shape), (2, 3, 36))
        self.assertEqual(tuple(output.node_keys.shape), (2, 3, 48))
        self.assertEqual(tuple(output.event_keys.shape), (2, 48))
        self.assertEqual(tuple(output.event_profile_keys.shape), (2, 12))
        self.assertEqual(tuple(output.event_latent_keys.shape), (2, 36))
        total_similarity = torch.einsum("bnd,bnd->bn", output.node_keys[0:1], output.node_keys[1:2])
        split_similarity = 0.25 * torch.einsum(
            "bnd,bnd->bn", output.profile_keys[0:1], output.profile_keys[1:2]
        )
        split_similarity = split_similarity + 0.75 * torch.einsum(
            "bnd,bnd->bn", output.latent_keys[0:1], output.latent_keys[1:2]
        )
        self.assertTrue(torch.allclose(total_similarity, split_similarity, atol=1.0e-5))
        event_similarity = torch.einsum(
            "bd,bd->b", output.event_keys[0:1], output.event_keys[1:2]
        )
        event_split_similarity = 0.25 * torch.einsum(
            "bd,bd->b",
            output.event_profile_keys[0:1],
            output.event_profile_keys[1:2],
        )
        event_split_similarity = event_split_similarity + 0.75 * torch.einsum(
            "bd,bd->b",
            output.event_latent_keys[0:1],
            output.event_latent_keys[1:2],
        )
        self.assertTrue(
            torch.allclose(event_similarity, event_split_similarity, atol=1.0e-5)
        )

    def test_profile_loss_requires_visible_forecast_context(self) -> None:
        model = STAnchorPretrainModel(
            ModelConfig(
                patch_size=3,
                hidden_dim=16,
                retrieval_dim=8,
                num_heads=4,
                encoder_layers=1,
                dropout=0.0,
                profile_dim=3,
                latent_dim=5,
            ),
            PretrainConfig(time_mask_ratio=0.25, space_mask_ratio=0.25),
            context_length=12,
            slots_per_day=288,
        )
        graph = graph_from_dense(np.eye(2, dtype=np.float32))
        x = torch.randn(3, 12, 2, 1)
        observed = torch.ones_like(x, dtype=torch.bool)
        output = model.forward_pretrain(
            x,
            observed,
            torch.zeros(3, 12, dtype=torch.long),
            torch.arange(12).view(1, -1).expand(3, -1),
            graph,
            graph.dense_neighbors(include_self=False),
            mask_task="time",
        )
        with self.assertRaisesRegex(ValueError, "forecast_context"):
            compute_pretraining_loss(
                output=output,
                future_model=torch.randn(3, 3, 2, 1),
                observed_context=observed,
                observed_future=torch.ones(3, 3, 2, 1, dtype=torch.bool),
                context_start=torch.tensor([0, 20, 40]),
                future_end=torch.tensor([14, 34, 54]),
                retrieval_weight=0.1,
                retrieval_temperature=0.1,
                positive_quantile=0.2,
                context_quantile=0.3,
                negative_quantile=0.7,
                hard_negative_weight=2.0,
                profile_loss_weight=0.1,
            )


if __name__ == "__main__":
    unittest.main()
