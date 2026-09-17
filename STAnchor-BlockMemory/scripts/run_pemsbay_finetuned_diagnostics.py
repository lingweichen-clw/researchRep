"""Run the missing post-adaptation PEMS-BAY retrieval diagnostics.

The publication queue evaluates source zero-shot retrieval before target
adaptation, but historically did not repeat the same two diagnostics with the
adapted checkpoint.  This small resume-safe runner builds matched temporary
Banks, evaluates both protocols, archives Bank manifests, and removes the
large Banks only after every required output exists.
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
import subprocess
import sys
from pathlib import Path


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


def run(command: list[str], repo_root: Path) -> None:
    print("[RUN] " + subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, cwd=repo_root, check=True)


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
            path.name: path.stat().st_size
            for path in bank.iterdir()
            if path.is_file()
        },
    }
    (archive / "inventory.json").write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    python = Path(sys.executable).resolve()
    paper_root = repo_root / "artifacts/paper_final_joint_context_offset_only_seed42"
    config = paper_root / "resolved_configs/adaptation/pemsbay_joint_context_offset_only.yaml"
    finetuned_checkpoint = paper_root / "retrieval_adaptation/pemsbay/retrieval_t1_best.pt"
    random_checkpoint = paper_root / "controls/random_encoder_seed42.pt"
    temporary_root = paper_root / "temporary_banks"
    finetuned_bank = temporary_root / "pemsbay_finetuned_diagnostic"
    random_bank = temporary_root / "pemsbay_random_diagnostic"
    output_root = paper_root / "cross_domain/pemsbay/finetuned"
    completion = output_root / "COMPLETED.json"

    required = (config, finetuned_checkpoint, random_checkpoint)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required inputs:\n" + "\n".join(missing))

    outputs = [
        output_root / protocol / name
        for protocol in ("pretrain_broad_causal", "weekday_radius1_overlap")
        for name in ("metrics.json", "cases.json")
    ]
    if completion.is_file() and all(path.is_file() for path in outputs):
        print(f"[SKIP] post-adaptation diagnostics already complete: {completion}", flush=True)
        return

    temporary_root.mkdir(parents=True, exist_ok=True)
    ensure_bank(
        python=python,
        repo_root=repo_root,
        config=config,
        checkpoint=finetuned_checkpoint,
        bank=finetuned_bank,
        dataset_name="pemsbay_joint_context_offset_only_t1_diagnostic",
    )
    ensure_bank(
        python=python,
        repo_root=repo_root,
        config=config,
        checkpoint=random_checkpoint,
        bank=random_bank,
        dataset_name="pemsbay_random_encoder_diagnostic_control",
    )

    for protocol in ("pretrain_broad_causal", "weekday_radius1_overlap"):
        output_dir = output_root / protocol
        metrics = output_dir / "metrics.json"
        cases = output_dir / "cases.json"
        if metrics.is_file() and cases.is_file():
            print(f"[REUSE] {protocol}: {output_dir}", flush=True)
            continue
        run(
            [
                str(python),
                "-u",
                "scripts/visualize_retrieval.py",
                "--version",
                "hn_offset_only_v1",
                "--config",
                str(config),
                "--checkpoint",
                str(finetuned_checkpoint),
                "--bank",
                str(finetuned_bank),
                "--random-checkpoint",
                str(random_checkpoint),
                "--random-bank",
                str(random_bank),
                "--split",
                "val",
                "--output-dir",
                str(output_dir),
                "--candidate-protocol",
                protocol,
                "--event-top-r",
                "96",
                "--node-top-k",
                "12",
                "--level-weight",
                "0",
            ],
            repo_root,
        )

    if not all(path.is_file() for path in outputs):
        raise RuntimeError("diagnostic process returned without all formal outputs")

    archive_bank(
        finetuned_bank,
        paper_root / "bank_manifests/pemsbay_finetuned_diagnostic",
    )
    archive_bank(
        random_bank,
        paper_root / "bank_manifests/pemsbay_random_diagnostic",
    )
    remove_temporary_bank(finetuned_bank, temporary_root)
    remove_temporary_bank(random_bank, temporary_root)
    completion.parent.mkdir(parents=True, exist_ok=True)
    completion.write_text(
        json.dumps(
            {
                "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "checkpoint": str(finetuned_checkpoint),
                "checkpoint_epoch": 8,
                "retrieval_fingerprint": checkpoint_fingerprint(finetuned_checkpoint),
                "protocols": ["pretrain_broad_causal", "weekday_radius1_overlap"],
                "complete_validation": True,
                "temporary_banks_removed": True,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[DONE] post-adaptation diagnostics: {completion}", flush=True)


if __name__ == "__main__":
    main()
