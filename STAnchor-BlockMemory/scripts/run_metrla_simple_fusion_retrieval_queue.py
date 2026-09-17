"""Run METR-LA retrieval baselines through the matched Simple Fusion path.

The queue reuses the three existing Learned+Simple checkpoints and only trains
the missing Raw-L1+Simple and Random+Simple controls for GWN, STGCN, and ARGCN.
Temporary Banks are retained on failure and removed only after all test metrics
and Bank inventories have been written.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml


PAPER_RUN = "paper_final_joint_context_offset_only_seed42"
COMPARISON_RUN = f"{PAPER_RUN}/thesis_only/simple_fusion_retrieval_comparison"
SOURCE_DATASET_NAME = "metrla_joint_context_offset_only_seed42_epoch34"
RANDOM_DATASET_NAME = "metrla_random_encoder_seed42_simple_fusion"
REQUIRED_BANK_FILES = (
    "manifest.json",
    "event_keys.npy",
    "node_keys.npy",
    "future_values.npy",
    "future_masks.npy",
    "level_features.npy",
    "sample_id.npy",
    "weekday.npy",
    "slot.npy",
    "context_start.npy",
    "context_end.npy",
    "future_end.npy",
)


@dataclass(frozen=True)
class RunSpec:
    backbone: str
    selector: str
    candidate_ranking: str
    source_config: Path
    pretrained_checkpoint: Path
    bank: Path
    base_checkpoint: Path
    run_name: str


def build_run_specs(repo_root: Path) -> list[RunSpec]:
    paper_root = repo_root / "artifacts" / PAPER_RUN
    source_checkpoint = (
        repo_root
        / "artifacts/metrla_e5_tgge_joint_context_offset_only_v1_transfer_"
        "hidden128_ffn2_b16_seed42/pretrain_best.pt"
    )
    random_checkpoint = paper_root / "controls/random_encoder_seed42.pt"
    temporary_root = paper_root / "temporary_banks"
    source_bank = temporary_root / "metrla_source_simple_fusion"
    random_bank = temporary_root / "metrla_random_simple_fusion"
    configs = {
        "gwn": repo_root / "configs/ablation_calibrator_simple_horizon_gwn.yaml",
        "stgcn": repo_root / "configs/ablation_calibrator_simple_horizon_stgcn.yaml",
        "argcn": repo_root / "configs/ablation_calibrator_simple_horizon_argcn.yaml",
    }
    base_checkpoints = {
        "gwn": (
            repo_root
            / "artifacts/convergence/downstream_tgge_v3_matched_fulltrain_queue/"
            "downstream_tgge_v3_graphwavenet_base_only_fulltrain_seed42/"
            "downstream_best.pt"
        ),
        "stgcn": (
            repo_root
            / "artifacts/convergence/downstream_tgge_v3_matched_fulltrain_queue/"
            "downstream_tgge_v3_stgcn_base_only_fulltrain_seed42/"
            "downstream_best.pt"
        ),
        "argcn": (
            repo_root
            / "artifacts/convergence/formal_20260826_argcn_base_only_v1/"
            "downstream_best.pt"
        ),
    }
    selectors = {
        "random": ("learned_key", random_checkpoint, random_bank),
        "raw_l1": ("raw_l1", source_checkpoint, source_bank),
    }
    return [
        RunSpec(
            backbone=backbone,
            selector=selector,
            candidate_ranking=ranking,
            source_config=configs[backbone],
            pretrained_checkpoint=checkpoint,
            bank=bank,
            base_checkpoint=base_checkpoints[backbone],
            run_name=f"{COMPARISON_RUN}/metrla/{backbone}/{selector}",
        )
        for selector, (ranking, checkpoint, bank) in selectors.items()
        for backbone in ("gwn", "stgcn", "argcn")
    ]


def materialize_config(spec: RunSpec, output: Path) -> None:
    raw = yaml.safe_load(spec.source_config.read_text(encoding="utf-8"))
    raw["bank"]["output_dir"] = str(spec.bank)
    raw["target"]["candidate_ranking"] = spec.candidate_ranking
    raw["target"]["candidate_payload"] = "offset_only"
    raw["target"]["downstream_mode"] = "learned_topk_offset_only_horizon"
    raw["target"]["epochs"] = 10
    raw["runtime"]["run_name"] = spec.run_name
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(raw, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def checkpoint_fingerprint(checkpoint: Path) -> str:
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    fingerprint = payload.get("retrieval_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise RuntimeError(f"checkpoint has no retrieval fingerprint: {checkpoint}")
    return fingerprint


def bank_complete(bank: Path, checkpoint: Path) -> bool:
    if not all((bank / name).is_file() for name in REQUIRED_BANK_FILES):
        return False
    manifest = json.loads((bank / "manifest.json").read_text(encoding="utf-8"))
    return manifest.get("encoder_fingerprint") == checkpoint_fingerprint(checkpoint)


def remove_temporary_bank(bank: Path, temporary_root: Path) -> None:
    root = temporary_root.resolve()
    target = bank.resolve()
    if target == root:
        raise ValueError("refusing to remove the temporary Bank root")
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ValueError(f"refusing to remove path outside {root}: {target}") from error
    if target.exists():
        shutil.rmtree(target)


def run(command: list[str], repo_root: Path) -> None:
    print("[RUN] " + subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=repo_root, check=True)


def ensure_bank(
    *,
    python: Path,
    repo_root: Path,
    config: Path,
    checkpoint: Path,
    bank: Path,
    dataset_name: str,
) -> None:
    if bank_complete(bank, checkpoint):
        print(f"[REUSE] matched Bank: {bank}", flush=True)
        return
    if bank.exists():
        remove_temporary_bank(bank, bank.parent)
    run(
        [
            str(python),
            "-u",
            "scripts/build_bank.py",
            "--config",
            str(config),
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(bank),
            "--dataset-name",
            dataset_name,
        ],
        repo_root,
    )
    if not bank_complete(bank, checkpoint):
        raise RuntimeError(f"Bank fingerprint/completeness check failed: {bank}")


def archive_bank(bank: Path, archive: Path) -> None:
    archive.mkdir(parents=True, exist_ok=True)
    shutil.copy2(bank / "manifest.json", archive / "manifest.json")
    inventory = {
        "temporary_bank": str(bank),
        "archived_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "files": {
            path.name: path.stat().st_size for path in bank.iterdir() if path.is_file()
        },
    }
    (archive / "inventory.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def training_complete(run_dir: Path) -> bool:
    checkpoint = run_dir / "downstream_best.pt"
    log = run_dir / "downstream.log"
    return (
        checkpoint.is_file()
        and log.is_file()
        and "Downstream training finished" in log.read_text(encoding="utf-8")
    )


def evaluate_to_json(
    *,
    python: Path,
    repo_root: Path,
    config: Path,
    pretrained_checkpoint: Path,
    downstream_checkpoint: Path,
    bank: Path,
    output: Path,
) -> dict:
    if output.is_file():
        print(f"[REUSE] test metrics: {output}", flush=True)
        return json.loads(output.read_text(encoding="utf-8"))
    command = [
        str(python),
        "-u",
        "scripts/evaluate.py",
        "--config",
        str(config),
        "--pretrained-checkpoint",
        str(pretrained_checkpoint),
        "--downstream-checkpoint",
        str(downstream_checkpoint),
        "--bank",
        str(bank),
        "--split",
        "test",
    ]
    print("[RUN] " + subprocess.list2cmdline(command), flush=True)
    completed = subprocess.run(
        command,
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    (output.parent / "evaluation.stderr.log").write_text(
        completed.stderr, encoding="utf-8"
    )
    result = json.loads(completed.stdout)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def metric_values(result: dict) -> dict[str, float]:
    metrics = result["metrics"]
    return {
        "mae": float(metrics["mae"]),
        "rmse": float(metrics["rmse"]),
        "mape": float(metrics["mape"]),
    }


def write_summary(repo_root: Path, comparison_root: Path) -> None:
    rows: list[dict] = []
    for backbone in ("gwn", "stgcn", "argcn"):
        main_path = (
            repo_root
            / f"artifacts/{PAPER_RUN}/main_table/metrla/ours/{backbone}/test_metrics.json"
        )
        main = json.loads(main_path.read_text(encoding="utf-8"))
        results = {
            "base_only": main["base_only"],
            "learned_full": main["ours"],
            "learned_simple": json.loads(
                (
                    repo_root
                    / f"artifacts/{PAPER_RUN}/ablations/calibrator_simple_horizon/"
                    f"metrla/{backbone}/test_metrics.json"
                ).read_text(encoding="utf-8")
            ),
            "random_simple": json.loads(
                (comparison_root / f"metrla/{backbone}/random/test_metrics.json").read_text(
                    encoding="utf-8"
                )
            ),
            "raw_l1_simple": json.loads(
                (comparison_root / f"metrla/{backbone}/raw_l1/test_metrics.json").read_text(
                    encoding="utf-8"
                )
            ),
        }
        base_mae = metric_values(results["base_only"])["mae"]
        for method, result in results.items():
            metrics = metric_values(result)
            rows.append(
                {
                    "backbone": backbone,
                    "method": method,
                    **metrics,
                    "mae_gain_vs_base": base_mae - metrics["mae"],
                    "mae_gain_percent": 100.0 * (base_mae - metrics["mae"]) / base_mae,
                }
            )
    (comparison_root / "summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (comparison_root / "summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    python = Path(sys.executable).resolve()
    paper_root = repo_root / "artifacts" / PAPER_RUN
    comparison_root = paper_root / "thesis_only/simple_fusion_retrieval_comparison"
    resolved_root = comparison_root / "resolved_configs"
    temporary_root = paper_root / "temporary_banks"
    specs = build_run_specs(repo_root)

    required = {spec.source_config for spec in specs}
    required.update(spec.pretrained_checkpoint for spec in specs)
    required.update(spec.base_checkpoint for spec in specs)
    required.update(
        repo_root
        / f"artifacts/{PAPER_RUN}/ablations/calibrator_simple_horizon/"
        f"metrla/{backbone}/downstream_best.pt"
        for backbone in ("gwn", "stgcn", "argcn")
    )
    missing = sorted(str(path) for path in required if not path.is_file())
    if missing:
        raise FileNotFoundError("missing required inputs:\n" + "\n".join(missing))

    from stanchor.config import load_config

    resolved_configs: dict[tuple[str, str], Path] = {}
    for spec in specs:
        resolved = resolved_root / f"{spec.backbone}_{spec.selector}.yaml"
        materialize_config(spec, resolved)
        load_config(resolved).validate()
        resolved_configs[(spec.backbone, spec.selector)] = resolved

    source_spec = next(spec for spec in specs if spec.selector == "raw_l1")
    random_spec = next(spec for spec in specs if spec.selector == "random")
    temporary_root.mkdir(parents=True, exist_ok=True)
    ensure_bank(
        python=python,
        repo_root=repo_root,
        config=source_spec.source_config,
        checkpoint=source_spec.pretrained_checkpoint,
        bank=source_spec.bank,
        dataset_name=SOURCE_DATASET_NAME,
    )
    ensure_bank(
        python=python,
        repo_root=repo_root,
        config=random_spec.source_config,
        checkpoint=random_spec.pretrained_checkpoint,
        bank=random_spec.bank,
        dataset_name=RANDOM_DATASET_NAME,
    )

    for spec in specs:
        run_dir = repo_root / "artifacts" / spec.run_name
        if training_complete(run_dir):
            print(f"[REUSE] completed training: {spec.backbone}/{spec.selector}", flush=True)
            continue
        run(
            [
                str(python),
                "-u",
                "scripts/train_downstream.py",
                "--config",
                str(resolved_configs[(spec.backbone, spec.selector)]),
                "--pretrained-checkpoint",
                str(spec.pretrained_checkpoint),
                "--bank",
                str(spec.bank),
                "--base-checkpoint",
                str(spec.base_checkpoint),
            ],
            repo_root,
        )
        if not training_complete(run_dir):
            raise RuntimeError(f"training returned without completion evidence: {run_dir}")

    for backbone in ("gwn", "stgcn", "argcn"):
        learned_dir = (
            paper_root / f"ablations/calibrator_simple_horizon/metrla/{backbone}"
        )
        source_config = next(
            spec.source_config for spec in specs if spec.backbone == backbone
        )
        evaluate_to_json(
            python=python,
            repo_root=repo_root,
            config=source_config,
            pretrained_checkpoint=source_spec.pretrained_checkpoint,
            downstream_checkpoint=learned_dir / "downstream_best.pt",
            bank=source_spec.bank,
            output=learned_dir / "test_metrics.json",
        )

    for spec in specs:
        run_dir = repo_root / "artifacts" / spec.run_name
        evaluate_to_json(
            python=python,
            repo_root=repo_root,
            config=resolved_configs[(spec.backbone, spec.selector)],
            pretrained_checkpoint=spec.pretrained_checkpoint,
            downstream_checkpoint=run_dir / "downstream_best.pt",
            bank=spec.bank,
            output=run_dir / "test_metrics.json",
        )

    write_summary(repo_root, comparison_root)
    archive_bank(
        source_spec.bank,
        paper_root / "bank_manifests/metrla_source_simple_fusion",
    )
    archive_bank(
        random_spec.bank,
        paper_root / "bank_manifests/metrla_random_simple_fusion",
    )
    remove_temporary_bank(source_spec.bank, temporary_root)
    remove_temporary_bank(random_spec.bank, temporary_root)
    completion = {
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset": "METR-LA",
        "backbones": ["gwn", "stgcn", "argcn"],
        "selectors": ["learned_key", "random_encoder", "raw_l1"],
        "candidate_payload": "offset_only",
        "fusion": "simple_horizon_12_parameters",
        "temporary_banks_removed": True,
    }
    (comparison_root / "COMPLETED.json").write_text(
        json.dumps(completion, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[DONE] comparison summary: {comparison_root / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
