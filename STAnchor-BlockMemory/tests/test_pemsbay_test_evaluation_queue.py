from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch
import yaml

from scripts.run_pemsbay_test_evaluation_queue import (
    BACKBONES,
    BANK_DATASET_NAME,
    REQUIRED_BANK_FILES,
    _bank_complete,
    build_evaluation_specs,
    evaluation_complete,
    materialize_finetuned_evaluation_config,
    remove_temporary_bank,
    summarize_pair,
)


def _result(mae: float, rmse: float, mape: float, batches: int = 7) -> dict:
    return {
        "metrics": {
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "horizon_mae": [mae] * 12,
            "horizon_rmse": [rmse] * 12,
            "horizon_mape": [mape] * 12,
        },
        "batches": batches,
    }


class PemsBayTestEvaluationQueueTest(unittest.TestCase):
    def test_bank_identity_matches_the_manifest_saved_in_router_checkpoints(self) -> None:
        self.assertEqual(
            BANK_DATASET_NAME,
            "pemsbay_joint_context_offset_only_t1",
        )

    def test_builds_eight_base_and_eight_current_finetuned_evaluations(self) -> None:
        repo_root = Path("D:/repo")
        specs = build_evaluation_specs(repo_root)

        self.assertEqual(len(specs), 16)
        self.assertEqual({spec.backbone for spec in specs}, set(BACKBONES))
        self.assertEqual({spec.variant for spec in specs}, {"base_only", "finetuned"})
        self.assertEqual(len({spec.output for spec in specs}), 16)

        finetuned = [spec for spec in specs if spec.variant == "finetuned"]
        self.assertEqual(len(finetuned), 8)
        for spec in finetuned:
            self.assertEqual(spec.pretrained_checkpoint.name, "retrieval_t1_best.pt")
            self.assertEqual(spec.bank.name, "pemsbay_finetuned_test_eval")
            self.assertIn(
                f"main_table/pemsbay/ours/{spec.backbone}/downstream_best.pt",
                spec.downstream_checkpoint.as_posix(),
            )
            self.assertIn(
                f"main_table/pemsbay/ours/{spec.backbone}/test_metrics.json",
                spec.output.as_posix(),
            )

        base = [spec for spec in specs if spec.variant == "base_only"]
        self.assertEqual(len(base), 8)
        for spec in base:
            self.assertIsNone(spec.pretrained_checkpoint)
            self.assertIsNone(spec.bank)
            self.assertIn(
                f"main_table/pemsbay/base_only/{spec.backbone}/test_metrics.json",
                spec.output.as_posix(),
            )

    def test_completed_evaluation_requires_metrics_and_positive_batch_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "test_metrics.json"
            output.write_text(json.dumps(_result(1.5, 3.1, 3.0)), encoding="utf-8")
            self.assertTrue(evaluation_complete(output))

            output.write_text(
                json.dumps(_result(1.5, 3.1, 3.0, batches=0)), encoding="utf-8"
            )
            self.assertFalse(evaluation_complete(output))

    def test_bank_reuse_rejects_a_different_dataset_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bank = root / "bank"
            bank.mkdir()
            checkpoint = root / "retrieval.pt"
            torch.save({"retrieval_fingerprint": "fingerprint-1"}, checkpoint)
            for name in REQUIRED_BANK_FILES:
                (bank / name).write_bytes(b"test")
            (bank / "manifest.json").write_text(
                json.dumps(
                    {
                        "encoder_fingerprint": "fingerprint-1",
                        "dataset_name": BANK_DATASET_NAME + "_wrong",
                    }
                ),
                encoding="utf-8",
            )

            self.assertFalse(_bank_complete(bank, checkpoint))

    def test_finetuned_evaluation_config_restores_saved_bank_search_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.yaml"
            checkpoint = root / "router.pt"
            bank = root / "bank"
            output = root / "resolved.yaml"
            source_payload = {
                "data": {"raw_path": "../data/pems-bay.h5"},
                "model": {"hidden_dim": 128},
                "bank": {
                    "output_dir": "old",
                    "event_top_r": 32,
                    "node_top_k": 12,
                    "level_weight": 0.0,
                    "search_temperature": 0.1,
                },
                "target": {"backbone_name": "graph_wavenet"},
                "runtime": {"seed": 42},
            }
            source.write_text(
                yaml.safe_dump(source_payload, sort_keys=False), encoding="utf-8"
            )
            torch.save(
                {
                    "config": {
                        "bank": {
                            "output_dir": "training-bank",
                            "event_top_r": 96,
                            "node_top_k": 12,
                            "level_weight": 0.0,
                            "search_temperature": 0.1,
                        }
                    }
                },
                checkpoint,
            )
            spec = next(
                item
                for item in build_evaluation_specs(root)
                if item.backbone == "gwn" and item.variant == "finetuned"
            )
            spec = type(spec)(
                backbone=spec.backbone,
                variant=spec.variant,
                config=source,
                downstream_checkpoint=checkpoint,
                pretrained_checkpoint=spec.pretrained_checkpoint,
                bank=bank,
                output=spec.output,
            )

            materialize_finetuned_evaluation_config(spec, output)
            resolved = yaml.safe_load(output.read_text(encoding="utf-8"))

            self.assertEqual(resolved["bank"]["event_top_r"], 96)
            self.assertEqual(resolved["bank"]["node_top_k"], 12)
            self.assertEqual(resolved["bank"]["output_dir"], str(bank))
            self.assertEqual(resolved["data"], source_payload["data"])
            self.assertEqual(resolved["model"], source_payload["model"])
            self.assertEqual(resolved["target"], source_payload["target"])

    def test_summary_preserves_metrics_and_computes_mae_gain(self) -> None:
        row = summarize_pair(
            "gwn",
            _result(1.60, 3.20, 3.10),
            _result(1.52, 3.05, 2.95),
        )

        self.assertEqual(row["backbone"], "gwn")
        self.assertAlmostEqual(row["base_mae"], 1.60)
        self.assertAlmostEqual(row["finetuned_mae"], 1.52)
        self.assertAlmostEqual(row["mae_gain"], 0.08)
        self.assertAlmostEqual(row["mae_gain_percent"], 5.0)

    def test_bank_cleanup_rejects_paths_outside_temporary_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            temporary_root = root / "temporary_banks"
            temporary_root.mkdir()
            outside = root / "outside"
            outside.mkdir()
            with self.assertRaises(ValueError):
                remove_temporary_bank(outside, temporary_root)


if __name__ == "__main__":
    unittest.main()
