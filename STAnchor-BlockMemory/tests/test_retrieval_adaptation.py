from __future__ import annotations

import copy
import unittest

import numpy as np
import torch

from stanchor.config import (
    AdaptationConfig,
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    PretrainConfig,
)
from stanchor.data.graph import graph_from_dense
from stanchor.engine.retrieval_adaptation import (
    build_t1_checkpoint_payload,
    configure_t1_parameters,
    run_retrieval_adaptation_epoch,
)
from stanchor.losses.adaptation import (
    compute_t1_adaptation_loss,
    cosine_key_distillation_loss,
)
from stanchor.losses.pretraining import compute_relation_only_loss
from stanchor.models.pretraining import STAnchorPretrainModel


class RetrievalAdaptationTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)
        self.batch = 6
        self.time = 12
        self.nodes = 5
        adjacency = np.eye(self.nodes, dtype=np.float32)
        for node in range(self.nodes):
            adjacency[node, (node + 1) % self.nodes] = 1.0
        self.graph = graph_from_dense(adjacency)
        self.model = STAnchorPretrainModel(
            ModelConfig(
                input_channels=1,
                output_channels=1,
                patch_size=3,
                hidden_dim=16,
                retrieval_dim=8,
                num_heads=4,
                encoder_layers=2,
                dropout=0.0,
                adapter_bottleneck_dim=8,
            ),
            PretrainConfig(
                time_mask_ratio=0.25,
                time_mask_block_size=3,
                space_mask_ratio=0.25,
            ),
            context_length=self.time,
            slots_per_day=288,
        )
        self.x = torch.randn(self.batch, self.time, self.nodes, 1)
        self.observed = torch.ones_like(self.x, dtype=torch.bool)
        self.weekday = torch.zeros(self.batch, self.time, dtype=torch.long)
        self.slot = torch.arange(self.time).unsqueeze(0).expand(self.batch, -1)
        self.future = torch.randn(self.batch, 12, self.nodes, 1)
        self.future_observed = torch.ones_like(self.future, dtype=torch.bool)
        self.context_start = torch.arange(self.batch) * 40
        self.future_end = self.context_start + self.time + 11

    def test_cosine_distillation_detaches_teacher_and_trains_student(self) -> None:
        teacher = torch.randn(4, 3, 8, requires_grad=True)
        student = torch.randn(4, 3, 8, requires_grad=True)

        loss = cosine_key_distillation_loss(student, teacher)
        loss.backward()

        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertIsNone(teacher.grad)
        self.assertIsNotNone(student.grad)
        self.assertGreater(float(student.grad.abs().sum()), 0.0)

    def test_t1_freezes_everything_except_retrieval_head_adapter_modules(self) -> None:
        trainable = configure_t1_parameters(self.model)
        trainable_names = {
            name for name, parameter in self.model.named_parameters() if parameter.requires_grad
        }

        expected_prefixes = (
            "retrieval_head.pool_projection.",
            "retrieval_head.pool_score.",
            "retrieval_head.key_mlp.",
            "retrieval_head.domain_adapter.",
        )
        self.assertTrue(trainable_names)
        self.assertTrue(
            all(name.startswith(expected_prefixes) for name in trainable_names)
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in trainable),
            sum(
                parameter.numel()
                for parameter in self.model.parameters()
                if parameter.requires_grad
            ),
        )
        self.assertFalse(any(p.requires_grad for p in self.model.embedding.parameters()))
        self.assertFalse(any(p.requires_grad for p in self.model.encoder.parameters()))
        self.assertFalse(
            any(p.requires_grad for p in self.model.reconstruction_head.parameters())
        )

    def test_t1_relation_loss_has_no_frozen_parameter_gradients(self) -> None:
        source = copy.deepcopy(self.model)
        for parameter in source.parameters():
            parameter.requires_grad_(False)
        configure_t1_parameters(self.model)
        self.model.eval()
        self.model.retrieval_head.train()

        clean = self.model.forward_relation(
            self.x,
            self.observed,
            self.weekday,
            self.slot,
            self.graph,
        )
        with torch.no_grad():
            source_keys = source.retrieval_head(clean.hidden.detach()).node_keys
        relation = compute_relation_only_loss(
            clean=clean,
            future_model=self.future,
            observed_future=self.future_observed,
            context_start=self.context_start,
            future_end=self.future_end,
            retrieval_weight=1.0,
            relation_teacher_temperature=0.1,
            relation_student_temperature=0.1,
            forecast_context=self.x,
            forecast_context_observed=self.observed,
            relation_teacher_mode="offset_decay",
            relation_distance_normalization="anchor_mean",
        )
        total = compute_t1_adaptation_loss(
            relation_loss=relation.retrieval,
            student_keys=clean.retrieval.node_keys,
            teacher_keys=source_keys,
            relation_weight=1.0,
            distill_weight=0.05,
        )
        total.total.backward()

        trainable_grad = sum(
            float(parameter.grad.abs().sum())
            for parameter in self.model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        )
        self.assertGreater(trainable_grad, 0.0)
        self.assertFalse(
            any(
                parameter.grad is not None
                for parameter in self.model.parameters()
                if not parameter.requires_grad
            )
        )
        self.assertTrue(bool(torch.isfinite(total.total)))

    def test_t1_epoch_updates_only_selected_parameters(self) -> None:
        source = copy.deepcopy(self.model)
        for parameter in source.parameters():
            parameter.requires_grad_(False)
        trainable = configure_t1_parameters(self.model)
        optimizer = torch.optim.AdamW(trainable, lr=3.0e-4)
        before = {
            name: parameter.detach().clone()
            for name, parameter in self.model.named_parameters()
        }
        batch = {
            "retrieval_x": self.x,
            "retrieval_observed": self.observed,
            "retrieval_weekday": self.weekday,
            "retrieval_slot": self.slot,
            "x": self.x,
            "x_observed": self.observed,
            "y": self.future,
            "y_observed": self.future_observed,
            "context_start": self.context_start,
            "future_end": self.future_end,
        }

        progress: list[tuple[int, int]] = []
        result = run_retrieval_adaptation_epoch(
            model=self.model,
            source_model=source,
            loader=[batch],
            graph=self.graph,
            config=AdaptationConfig(),
            pretrain_config=PretrainConfig(
                relation_teacher_mode="offset_decay",
                relation_distance_normalization="anchor_mean",
            ),
            device=torch.device("cpu"),
            optimizer=optimizer,
            progress_callback=lambda completed, total, _elapsed, _eta: progress.append(
                (completed, total)
            ),
        )

        self.assertEqual(result.batches, 1)
        self.assertGreater(result.valid_retrieval_anchors, 0)
        self.assertTrue(np.isfinite(result.total))
        changed = {
            name
            for name, parameter in self.model.named_parameters()
            if not torch.equal(before[name], parameter.detach())
        }
        self.assertTrue(changed)
        self.assertTrue(
            all(name.startswith(
                (
                    "retrieval_head.pool_projection.",
                    "retrieval_head.pool_score.",
                    "retrieval_head.key_mlp.",
                    "retrieval_head.domain_adapter.",
                )
            ) for name in changed)
        )
        self.assertFalse(any(parameter.grad is not None for parameter in source.parameters()))
        self.assertFalse(self.model.encoder.training)
        self.assertTrue(self.model.retrieval_head.training)
        self.assertEqual(progress, [(1, 1)])

    def test_t1_checkpoint_preserves_transfer_provenance(self) -> None:
        config = ExperimentConfig(
            data=DataConfig(raw_path="target.h5", adjacency_path="graph.pkl"),
            model=self.model.model_config,
        )
        metrics = {"epoch": 2, "val": {"relation": 0.5}}

        payload = build_t1_checkpoint_payload(
            model=self.model,
            config=config,
            normalizer={"mean": [1.0], "std": [2.0]},
            graph_fingerprint="target-graph",
            source_checkpoint="source.pt",
            source_retrieval_fingerprint="source-fingerprint",
            epoch=2,
            metrics=metrics,
        )

        self.assertEqual(payload["stage"], "t1_head_adapter")
        self.assertEqual(payload["source_checkpoint"], "source.pt")
        self.assertEqual(
            payload["source_retrieval_fingerprint"], "source-fingerprint"
        )
        self.assertEqual(payload["graph_fingerprint"], "target-graph")
        self.assertEqual(payload["normalizer"]["mean"], [1.0])
        self.assertEqual(payload["metrics"], metrics)
        self.assertEqual(payload["epoch"], 2)
        self.assertIn("model_state_dict", payload)
        self.assertIn("retrieval_state_dict", payload)
        self.assertEqual(
            payload["retrieval_fingerprint"], self.model.retrieval_fingerprint()
        )


if __name__ == "__main__":
    unittest.main()
