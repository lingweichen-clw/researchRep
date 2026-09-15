from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

import torch
import yaml

from scripts.run_paper_final_queue import (
    BACKBONES,
    DATASETS,
    MAIN_TABLE_BACKBONES,
    QueueRunner,
    _finish_bank_stage,
    build_parser,
    build_experiment_plan,
    discover_base_checkpoints,
    materialize_adaptation_config,
    materialize_downstream_config,
    safe_remove_temporary_bank,
)
from stanchor.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PaperFinalQueueTest(unittest.TestCase):
    def test_cli_exposes_machine_preflight_without_running_jobs(self) -> None:
        args = build_parser().parse_args(["--validate-only"])

        self.assertTrue(args.validate_only)
        self.assertFalse(args.dry_run)

    def test_main_table_reuses_only_the_twenty_four_available_bases(self) -> None:
        plan = build_experiment_plan()
        base = [job for job in plan if job["kind"] == "base_only"]
        base_validation = [
            job for job in plan if job["kind"] == "base_checkpoint_validation"
        ]
        full = [job for job in plan if job["kind"] == "final_router"]
        adaptation = [job for job in plan if job["kind"] == "retrieval_finetune"]

        self.assertEqual(len(DATASETS), 4)
        self.assertEqual(len(BACKBONES), 8)
        self.assertEqual(MAIN_TABLE_BACKBONES["metrla"], BACKBONES)
        self.assertEqual(MAIN_TABLE_BACKBONES["pemsbay"], BACKBONES)
        self.assertEqual(
            MAIN_TABLE_BACKBONES["pems04"],
            ("gwn", "stgcn", "staeformer", "argcn"),
        )
        self.assertEqual(MAIN_TABLE_BACKBONES["pems08"], MAIN_TABLE_BACKBONES["pems04"])
        self.assertEqual(len(base), 0)
        self.assertEqual(len(base_validation), 24)
        self.assertEqual(len(full), 24)
        self.assertEqual(len(adaptation), 3)
        self.assertNotIn("pems07", {job["dataset"] for job in plan})
        self.assertFalse(any(job["kind"] == "source_router" for job in plan))

    def test_target_order_builds_and_consumes_finetuned_bank_before_cleanup(self) -> None:
        plan = build_experiment_plan()
        labels = [job["label"] for job in plan]

        finetune = labels.index("pems04_retrieval_finetune")
        build = labels.index("pems04_finetuned_bank_build")
        first_router = labels.index("pems04_finetuned_router_gwn")
        cleanup = labels.index("pems04_finetuned_bank_cleanup")

        self.assertLess(finetune, build)
        self.assertLess(build, first_router)
        self.assertLess(first_router, cleanup)

    def test_existing_base_discovery_uses_checkpoint_metadata_not_directory_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            checkpoint = repo / "artifacts" / "arbitrary_name" / "downstream_best.pt"
            checkpoint.parent.mkdir(parents=True)
            torch.save(
                {
                    "downstream_mode": "base_only",
                    "seed": 42,
                    "epoch": 17,
                    "downstream_state_dict": {
                        "backbone.weight": torch.tensor([1.0, 2.0])
                    },
                    "config": {
                        "data": {"raw_path": "../data/METRLA_data/METR-LA.h5"},
                        "target": {
                            "downstream_mode": "base_only",
                            "backbone_name": "graph_wavenet",
                            "training_data_scope": "full_train",
                            "epochs": 50,
                            "early_stopping_enabled": False,
                        },
                        "runtime": {"seed": 42},
                    },
                },
                checkpoint,
            )
            registry = repo / "paper" / "base_checkpoint_registry.json"

            resolved = discover_base_checkpoints(
                repo, [("metrla", "gwn")], registry_path=registry
            )
            report = json.loads(registry.read_text(encoding="utf-8"))

        self.assertEqual(resolved[("metrla", "gwn")], checkpoint.resolve())
        self.assertEqual(report["selected_count"], 1)
        self.assertEqual(report["missing"], [])
        self.assertEqual(len(report["selected"][0]["backbone_state_fingerprint"]), 64)

    def test_existing_base_discovery_never_trains_or_substitutes_missing_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            (repo / "artifacts").mkdir()
            registry = repo / "paper" / "base_checkpoint_registry.json"

            with self.assertRaisesRegex(FileNotFoundError, "pems04/dcrnn"):
                discover_base_checkpoints(
                    repo, [("pems04", "dcrnn")], registry_path=registry
                )
            report = json.loads(registry.read_text(encoding="utf-8"))

        self.assertEqual(report["selected_count"], 0)
        self.assertEqual(report["missing"], ["pems04/dcrnn"])

    def test_materialized_missing_pems04_dcrnn_config_uses_final_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pems04_dcrnn.yaml"
            materialize_downstream_config(
                repo_root=PROJECT_ROOT,
                dataset="pems04",
                backbone="dcrnn",
                output_path=output,
                bank_output="artifacts/paper_final_joint_context_offset_only_seed42/temporary_banks/pems04_finetuned",
            )
            raw = yaml.safe_load(output.read_text(encoding="utf-8"))
            config = load_config(output)

        self.assertIn("pems04", raw["data"]["raw_path"].lower())
        self.assertEqual(raw["data"]["channel_index"], 2)
        self.assertEqual(raw["target"]["backbone_name"], "dcrnn")
        self.assertEqual(raw["target"]["candidate_payload"], "offset_only")
        self.assertEqual(
            raw["target"]["calibrator_arch"], "candidate_key_context_mha_router"
        )
        self.assertEqual(raw["target"]["candidate_key_bottleneck_dim"], 16)
        self.assertEqual(config.pretrain.relation_teacher_mode, "offset_only")
        config.validate()

    def test_cross_domain_calendar_pool_keeps_event_top_r_96(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pemsbay_gwn.yaml"
            materialize_downstream_config(
                repo_root=PROJECT_ROOT,
                dataset="pemsbay",
                backbone="gwn",
                output_path=output,
                bank_output="artifacts/temporary_banks/pemsbay_finetuned",
            )
            config = load_config(output)

        self.assertEqual(config.bank.event_top_r, 96)
        self.assertEqual(config.bank.node_top_k, 12)
        config.validate()

    def test_safe_bank_cleanup_rejects_root_and_outside_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            temporary_root = workspace / "temporary_banks"
            bank = temporary_root / "pems04_finetuned"
            bank.mkdir(parents=True)
            (bank / "manifest.json").write_text("{}", encoding="utf-8")
            outside = workspace / "unrelated"
            outside.mkdir()

            safe_remove_temporary_bank(bank, temporary_root)
            self.assertFalse(bank.exists())
            with self.assertRaises(ValueError):
                safe_remove_temporary_bank(temporary_root, temporary_root)
            with self.assertRaises(ValueError):
                safe_remove_temporary_bank(outside, temporary_root)

    def test_completed_cleanup_removes_a_recreated_temporary_bank(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            paper_root = workspace / "artifacts" / "paper"
            temporary_root = paper_root / "temporary_banks"
            bank = temporary_root / "metrla_source"
            bank.mkdir(parents=True)
            (bank / "manifest.json").write_text("{}", encoding="utf-8")
            inventory = paper_root / "bank_manifests" / "metrla_source" / "inventory.json"
            inventory.parent.mkdir(parents=True)
            inventory.write_text("{}", encoding="utf-8")
            consumer = paper_root / "results" / "metrics.json"
            consumer.parent.mkdir(parents=True)
            consumer.write_text("{}", encoding="utf-8")
            runner = QueueRunner(workspace, paper_root)
            runner.mark_complete(
                "metrla_cleanup", command=None, outputs=[inventory, consumer]
            )

            _finish_bank_stage(
                runner,
                label="metrla_cleanup",
                banks={"metrla_source": bank},
                temporary_root=temporary_root,
                consumer_outputs=[consumer],
            )
            self.assertFalse(bank.exists())

    def test_dataset_paths_resolve_to_repository_parent_data_directory(self) -> None:
        raw = yaml.safe_load(
            (PROJECT_ROOT / "configs/cross_dataset_pems08_source_encoder_stage1.yaml").read_text(
                encoding="utf-8"
            )
        )
        raw_path = (PROJECT_ROOT / raw["data"]["raw_path"]).resolve()
        adjacency_path = (PROJECT_ROOT / raw["data"]["adjacency_path"]).resolve()

        self.assertEqual(raw_path, PROJECT_ROOT.parent / "data" / "PEMS08" / "PEMS08.npz")
        self.assertEqual(adjacency_path, PROJECT_ROOT.parent / "data" / "PEMS08" / "PEMS08.csv")
        self.assertNotEqual(raw_path.parent, PROJECT_ROOT / "data" / "PEMS08")

    def test_all_resolved_configs_load_and_validate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            for dataset in DATASETS:
                for backbone in BACKBONES:
                    path = output_root / f"{dataset}_{backbone}.yaml"
                    materialize_downstream_config(
                        repo_root=PROJECT_ROOT,
                        dataset=dataset,
                        backbone=backbone,
                        output_path=path,
                        bank_output=f"artifacts/temporary_banks/{dataset}",
                    )
                    load_config(path).validate()
                if dataset != "metrla":
                    path = output_root / f"{dataset}_adaptation.yaml"
                    materialize_adaptation_config(
                        repo_root=PROJECT_ROOT,
                        dataset=dataset,
                        output_path=path,
                        bank_output=f"artifacts/temporary_banks/{dataset}",
                    )
                    config = load_config(path)
                    config.validate()
                    self.assertEqual(config.pretrain.context_relation_weight, 0.2)


if __name__ == "__main__":
    unittest.main()
