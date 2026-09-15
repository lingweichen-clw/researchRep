"""Run the publication-final STAnchor experiment matrix on one Windows GPU."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Iterable

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stanchor.config import load_config
from stanchor.data import load_graph
from stanchor.engine.target import build_downstream_model


SEED = 42
PAPER_RUN = "paper_final_joint_context_offset_only_seed42"
PAPER_ROOT_REL = Path("artifacts") / PAPER_RUN
SOURCE_CHECKPOINT_REL = Path(
    "artifacts/metrla_e5_tgge_joint_context_offset_only_v1_transfer_hidden128_ffn2_b16_seed42/pretrain_best.pt"
)
REFERENCE_CHECKPOINT_REL = Path(
    "artifacts/metrla_e5_tgge_hn_offset_decay_v2_transfer_hidden128_ffn2_b16_seed42/pretrain_best.pt"
)
FINAL_CONFIG_REL = Path(
    "configs/metrla_e5_tgge_joint_context_offset_only_v1_transfer_hidden128_ffn2_b16.yaml"
)
REFERENCE_CONFIG_REL = Path(
    "configs/metrla_e5_tgge_hn_offset_decay_v2_transfer_hidden128_ffn2_b16.yaml"
)

BACKBONES = (
    "gwn",
    "stgcn",
    "staeformer",
    "argcn",
    "dcrnn",
    "dlinear",
    "st_norm",
    "st_ssdl",
)

# Only combinations with an already completed Base-only checkpoint enter this
# queue. PEMS04/PEMS08 currently have the four established baselines; the
# remaining four combinations are deliberately deferred instead of silently
# retraining a new Base-only model here.
MAIN_TABLE_BACKBONES = {
    "metrla": BACKBONES,
    "pemsbay": BACKBONES,
    "pems04": ("gwn", "stgcn", "staeformer", "argcn"),
    "pems08": ("gwn", "stgcn", "staeformer", "argcn"),
}

DATASETS = {
    "metrla": {
        "data_template": FINAL_CONFIG_REL,
        "router_pattern": "configs/formal_base_as_candidate_{backbone}.yaml",
        "canonical_router": "configs/formal_base_as_candidate_gwn.yaml",
        "finetune_template": None,
    },
    "pemsbay": {
        "data_template": Path("configs/cross_dataset_pemsbay_source_encoder_stage1.yaml"),
        "router_pattern": "configs/cross_dataset_pemsbay_router_{backbone}.yaml",
        "canonical_router": "configs/cross_dataset_pemsbay_router_gwn.yaml",
        "finetune_template": "configs/cross_dataset_pemsbay_t1_head_adapter.yaml",
    },
    "pems04": {
        "data_template": Path("configs/cross_dataset_pems04_source_encoder_stage1.yaml"),
        "router_pattern": "configs/cross_dataset_pems04_router_{backbone}.yaml",
        "canonical_router": "configs/cross_dataset_pems04_router_gwn.yaml",
        "finetune_template": "configs/cross_dataset_pems04_t1_head_adapter.yaml",
    },
    "pems08": {
        "data_template": Path("configs/cross_dataset_pems08_source_encoder_stage1.yaml"),
        "router_pattern": "configs/cross_dataset_pems08_router_{backbone}.yaml",
        "canonical_router": "configs/cross_dataset_pems08_router_gwn.yaml",
        "finetune_template": "configs/cross_dataset_pems08_t1_head_adapter.yaml",
    },
}

BACKBONE_PREFIXES = {
    "gwn": ("graph_wavenet_",),
    "stgcn": ("stgcn_",),
    "staeformer": ("staeformer_",),
    "argcn": ("argcn_",),
    "dcrnn": ("dcrnn_",),
    "dlinear": ("dlinear_",),
    "st_norm": ("st_norm_",),
    "st_ssdl": ("st_ssdl_",),
}

FINAL_ROUTER_KEYS = (
    "downstream_mode",
    "training_protocol",
    "training_data_scope",
    "candidate_protocol",
    "candidate_payload",
    "confidence_hidden_dim",
    "confidence_weight",
    "confidence_level_temperature",
    "help_margin",
    "help_temperature",
    "risk_hidden_dim",
    "fusion_feature_hidden_dim",
    "horizon_aggregation_hidden_dim",
    "risk_weight",
    "blend_weight",
    "candidate_quality_weight",
    "candidate_quality_temperature",
    "validation_loss_variant",
    "forecast_loss_space",
    "validation_correction_variant",
    "calibrator_arch",
    "candidate_token_dim",
    "calibrator_state_dim",
    "candidate_attention_heads",
    "candidate_trajectory_hidden_dim",
    "candidate_key_bottleneck_dim",
    "routing_hidden_dim",
    "mha_dropout",
    "use_horizon_embedding",
    "base_logit_init_bias",
    "frozen_path_cache",
    "blend_minimum_direction_norm",
    "base_warmup_epochs",
    "calibrator_warmup_epochs",
    "backbone_learning_rate_scale",
)

BANK_REQUIRED_FILES = (
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


def _read_yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return payload


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _exact_router_path(repo_root: Path, dataset: str, backbone: str) -> Path:
    pattern = str(DATASETS[dataset]["router_pattern"])
    return repo_root / pattern.format(backbone=backbone)


def _backbone_target(repo_root: Path, dataset: str, backbone: str) -> dict:
    exact = _exact_router_path(repo_root, dataset, backbone)
    if exact.exists():
        return copy.deepcopy(_read_yaml(exact)["target"])

    canonical = repo_root / str(DATASETS[dataset]["canonical_router"])
    donor = _exact_router_path(repo_root, "pemsbay", backbone)
    if not donor.exists():
        raise FileNotFoundError(f"missing backbone template: {donor}")
    target = copy.deepcopy(_read_yaml(canonical)["target"])
    donor_target = _read_yaml(donor)["target"]
    target["backbone_name"] = donor_target["backbone_name"]
    prefixes = BACKBONE_PREFIXES[backbone]
    for name, value in donor_target.items():
        if name.startswith(prefixes):
            target[name] = copy.deepcopy(value)
    return target


def _compose_config(
    *,
    repo_root: Path,
    dataset: str,
    backbone: str,
    bank_output: str,
) -> dict:
    if dataset not in DATASETS:
        raise ValueError(f"unsupported dataset: {dataset}")
    if backbone not in BACKBONES:
        raise ValueError(f"unsupported backbone: {backbone}")
    final = _read_yaml(repo_root / FINAL_CONFIG_REL)
    data_template = _read_yaml(repo_root / Path(DATASETS[dataset]["data_template"]))
    target = _backbone_target(repo_root, dataset, backbone)
    final_target = final["target"]
    for name in FINAL_ROUTER_KEYS:
        if name in final_target:
            target[name] = copy.deepcopy(final_target[name])
    target.update(
        {
            "downstream_mode": "learned_topk_error_aware",
            "training_protocol": "posthoc_frozen_base",
            "training_data_scope": "full_train",
            "candidate_protocol": "weekday_radius1_overlap",
            "candidate_payload": "offset_only",
            "candidate_ranking": "learned_key",
            "calibrator_arch": "candidate_key_context_mha_router",
            "candidate_key_bottleneck_dim": 16,
            "epochs": 50,
            "early_stopping_enabled": False,
            "frozen_path_cache": True,
        }
    )
    bank = copy.deepcopy(final["bank"])
    bank.update(
        {
            "output_dir": bank_output,
            # Cross-domain weekday_radius1_overlap pools can contain more
            # than 32 legal events (PEMS-BAY reaches 37).  Keep the source
            # METR-LA protocol at 32, but preserve the established 96-event
            # cross-domain candidate pool so the fair calendar set is not
            # truncated before retrieval ranking.
            "event_top_r": 32 if dataset == "metrla" else 96,
            "node_top_k": 12,
            "level_weight": 0.0,
        }
    )
    runtime = copy.deepcopy(final["runtime"])
    runtime.update(
        {
            "seed": SEED,
            "device": "cuda:0",
            "output_dir": "artifacts",
            "run_name": f"{PAPER_RUN}/resolved/{dataset}_{backbone}",
        }
    )
    return {
        "data": copy.deepcopy(data_template["data"]),
        "model": copy.deepcopy(final["model"]),
        "pretrain": copy.deepcopy(final["pretrain"]),
        "bank": bank,
        "target": target,
        "runtime": runtime,
    }


def materialize_downstream_config(
    *,
    repo_root: Path,
    dataset: str,
    backbone: str,
    output_path: Path,
    bank_output: str,
) -> Path:
    payload = _compose_config(
        repo_root=repo_root,
        dataset=dataset,
        backbone=backbone,
        bank_output=bank_output,
    )
    _write_yaml(output_path, payload)
    return output_path


def materialize_adaptation_config(
    *,
    repo_root: Path,
    dataset: str,
    output_path: Path,
    bank_output: str,
) -> Path:
    if dataset == "metrla":
        raise ValueError("METR-LA is the source domain and has no T1 adaptation config")
    payload = _compose_config(
        repo_root=repo_root,
        dataset=dataset,
        backbone="gwn",
        bank_output=bank_output,
    )
    template_path = repo_root / str(DATASETS[dataset]["finetune_template"])
    payload["adaptation"] = copy.deepcopy(_read_yaml(template_path)["adaptation"])
    payload["pretrain"].update(
        {
            "objective": "relation_only",
            "reconstruction_weight": 0.0,
            "retrieval_loss_mode": "relation",
            "relation_teacher_mode": "offset_only",
            "relation_distance_normalization": "symmetric_geometric_mean",
            "context_relation_weight": 0.2,
            "rank_loss_weight": 0.0,
        }
    )
    payload["runtime"]["run_name"] = f"{PAPER_RUN}/retrieval_adaptation/{dataset}"
    _write_yaml(output_path, payload)
    return output_path


def build_experiment_plan() -> list[dict[str, str]]:
    """Return the ordered publication queue without touching the filesystem."""
    jobs: list[dict[str, str]] = []
    for dataset in DATASETS:
        for backbone in MAIN_TABLE_BACKBONES[dataset]:
            jobs.append(
                {
                    "kind": "base_checkpoint_validation",
                    "dataset": dataset,
                    "backbone": backbone,
                    "label": f"{dataset}_existing_base_{backbone}",
                }
            )
    jobs.append(
        {"kind": "random_checkpoint", "dataset": "metrla", "label": "random_checkpoint"}
    )
    for dataset in DATASETS:
        if dataset == "metrla":
            for variant in ("source", "random", "reference"):
                jobs.append(
                    {
                        "kind": "bank_build",
                        "dataset": dataset,
                        "variant": variant,
                        "label": f"{dataset}_{variant}_bank_build",
                    }
                )
            for protocol in ("pretrain_broad_causal", "weekday_radius1_overlap"):
                jobs.append(
                    {
                        "kind": "case_study",
                        "dataset": dataset,
                        "protocol": protocol,
                        "label": f"{dataset}_case_study_{protocol}",
                    }
                )
            jobs.extend(
                (
                    {"kind": "mirage", "dataset": dataset, "label": "metrla_mirage_ab"},
                    {
                        "kind": "context_key",
                        "dataset": dataset,
                        "label": "metrla_context_future_key",
                    },
                )
            )
        else:
            for variant in ("source", "random"):
                jobs.append(
                    {
                        "kind": "bank_build",
                        "dataset": dataset,
                        "variant": variant,
                        "label": f"{dataset}_{variant}_bank_build",
                    }
                )
            for protocol in ("pretrain_broad_causal", "weekday_radius1_overlap"):
                jobs.append(
                    {
                        "kind": "zero_shot_diagnostic",
                        "dataset": dataset,
                        "protocol": protocol,
                        "label": f"{dataset}_zero_shot_{protocol}",
                    }
                )
            jobs.append(
                {
                    "kind": "bank_cleanup",
                    "dataset": dataset,
                    "variant": "source_zero_shot_and_random",
                    "label": f"{dataset}_zero_shot_bank_cleanup",
                }
            )
            jobs.append(
                {
                    "kind": "retrieval_finetune",
                    "dataset": dataset,
                    "label": f"{dataset}_retrieval_finetune",
                }
            )
            jobs.append(
                {
                    "kind": "bank_build",
                    "dataset": dataset,
                    "variant": "finetuned",
                    "label": f"{dataset}_finetuned_bank_build",
                }
            )
        for backbone in MAIN_TABLE_BACKBONES[dataset]:
            encoder_variant = "source" if dataset == "metrla" else "finetuned"
            jobs.append(
                {
                    "kind": "final_router",
                    "dataset": dataset,
                    "backbone": backbone,
                    "label": f"{dataset}_{encoder_variant}_router_{backbone}",
                }
            )
        jobs.append(
            {
                "kind": "bank_cleanup",
                "dataset": dataset,
                "variant": "source" if dataset == "metrla" else "finetuned",
                "label": f"{dataset}_{'source' if dataset == 'metrla' else 'finetuned'}_bank_cleanup",
            }
        )
    return jobs


def safe_remove_temporary_bank(path: Path, temporary_root: Path) -> None:
    """Delete only a concrete child of the queue-owned temporary Bank root."""
    root = temporary_root.resolve()
    target = path.resolve()
    if target == root:
        raise ValueError("refusing to remove the temporary Bank root itself")
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ValueError(f"refusing to remove path outside temporary Bank root: {target}") from error
    if target.exists():
        shutil.rmtree(target)


class QueueRunner:
    def __init__(self, repo_root: Path, paper_root: Path) -> None:
        self.repo_root = repo_root.resolve()
        self.paper_root = paper_root.resolve()
        self.logs = self.paper_root / "logs"
        self.markers = self.paper_root / "queue_state" / "completed"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.markers.mkdir(parents=True, exist_ok=True)

    def marker(self, label: str) -> Path:
        return self.markers / f"{label}.json"

    def is_complete(self, label: str, outputs: Iterable[Path]) -> bool:
        return self.marker(label).exists() and all(path.exists() for path in outputs)

    def mark_complete(
        self,
        label: str,
        *,
        command: list[str] | None,
        outputs: Iterable[Path],
    ) -> None:
        payload = {
            "label": label,
            "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "command": command,
            "outputs": [str(path) for path in outputs],
        }
        target = self.marker(label)
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(target)

    def run(self, label: str, arguments: list[str], outputs: Iterable[Path]) -> None:
        required = [Path(path) for path in outputs]
        if self.is_complete(label, required):
            print(f"[SKIP] {label}", flush=True)
            return
        marker = self.marker(label)
        if marker.exists():
            marker.unlink()
        command = [sys.executable, "-u", *arguments]
        child_environment = os.environ.copy()
        child_environment["PYTHONUTF8"] = "1"
        child_environment["PYTHONUNBUFFERED"] = "1"
        child_environment.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
        log_path = self.logs / f"{label}.log"
        print(f"[START] {label}", flush=True)
        print("        " + subprocess.list2cmdline(command), flush=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n===== {dt.datetime.now().isoformat()} {label} =====\n")
            log.write(subprocess.list2cmdline(command) + "\n")
            process = subprocess.Popen(
                command,
                cwd=self.repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=child_environment,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"job failed: {label} (exit code {return_code})")
        missing = [path for path in required if not path.exists()]
        if missing:
            raise RuntimeError(f"job {label} returned success but missed outputs: {missing}")
        self.mark_complete(label, command=command, outputs=required)
        print(f"[DONE] {label}", flush=True)


def _torch_load(path: Path) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint payload is not a mapping: {path}")
    return payload


def _checkpoint_fingerprint(path: Path) -> str:
    fingerprint = _torch_load(path).get("retrieval_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError(f"checkpoint lacks retrieval_fingerprint: {path}")
    return fingerprint


def _canonical_dataset_name(raw_path: object) -> str | None:
    normalized = str(raw_path or "").lower().replace("-", "").replace("_", "")
    for dataset in DATASETS:
        if dataset in normalized:
            return dataset
    return None


def _canonical_backbone_name(name: object) -> str | None:
    normalized = str(name or "").lower()
    aliases = {"graph_wavenet": "gwn", "agcrn": "argcn"}
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in BACKBONES else None


def _backbone_state_fingerprint(state: dict) -> str:
    """Hash exact frozen backbone weights without depending on private APIs."""
    digest = hashlib.sha256()
    for name in sorted(state):
        if not name.startswith("backbone."):
            continue
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _base_checkpoint_record(path: Path) -> tuple[tuple[str, str], dict] | None:
    checkpoint = _torch_load(path)
    config = checkpoint.get("config", {})
    target = config.get("target", {}) if isinstance(config, dict) else {}
    data = config.get("data", {}) if isinstance(config, dict) else {}
    runtime = config.get("runtime", {}) if isinstance(config, dict) else {}
    mode = checkpoint.get("downstream_mode", target.get("downstream_mode"))
    if mode != "base_only":
        return None
    dataset = _canonical_dataset_name(data.get("raw_path"))
    backbone = _canonical_backbone_name(target.get("backbone_name"))
    if dataset is None or backbone is None:
        return None
    seed = checkpoint.get("seed", runtime.get("seed"))
    scope = target.get("training_data_scope")
    state = checkpoint.get("downstream_state_dict")
    if int(seed) != SEED or scope != "full_train" or not isinstance(state, dict):
        return None
    if not any(name.startswith("backbone.") for name in state):
        return None
    record = {
        "path": str(path.resolve()),
        "dataset": dataset,
        "backbone": backbone,
        "downstream_mode": mode,
        "training_data_scope": scope,
        "training_protocol": checkpoint.get(
            "training_protocol", target.get("training_protocol")
        ),
        "seed": int(seed),
        "epochs_configured": target.get("epochs"),
        "early_stopping_enabled": target.get("early_stopping_enabled"),
        "selected_epoch": checkpoint.get("epoch"),
        "raw_path": data.get("raw_path"),
        "channel_index": data.get("channel_index", 0),
        "context_length": data.get("context_length", 12),
        "horizon": data.get("horizon", 12),
        "frequency_minutes": data.get("frequency_minutes", 5),
        "backbone_state_fingerprint": _backbone_state_fingerprint(state),
        "metrics": checkpoint.get("metrics"),
    }
    return (dataset, backbone), record


def discover_base_checkpoints(
    repo_root: Path,
    required: Iterable[tuple[str, str]],
    *,
    registry_path: Path | None = None,
    compatibility_check: Callable[[tuple[str, str], Path], None] | None = None,
) -> dict[tuple[str, str], Path]:
    """Resolve existing publication Base-only checkpoints without training any."""
    required_keys = set(required)
    candidates: dict[tuple[str, str], list[dict]] = {
        key: [] for key in required_keys
    }
    artifact_root = repo_root / "artifacts"
    for path in sorted(artifact_root.rglob("downstream_best.pt")):
        try:
            inspected = _base_checkpoint_record(path)
        except (OSError, RuntimeError, TypeError, ValueError):
            continue
        if inspected is None:
            continue
        key, record = inspected
        if key in candidates:
            candidates[key].append(record)

    selected: dict[tuple[str, str], Path] = {}
    selected_records: list[dict] = []
    ambiguous: dict[str, list[str]] = {}
    rejected: dict[str, list[dict[str, str]]] = {}
    for key in sorted(required_keys):
        records = sorted(candidates[key], key=lambda item: item["path"].lower())
        if not records:
            continue
        # Selection is path-deterministic and never based on validation score.
        for record in records:
            path = Path(record["path"])
            if compatibility_check is not None:
                try:
                    compatibility_check(key, path)
                except (OSError, RuntimeError, TypeError, ValueError) as error:
                    rejected.setdefault(f"{key[0]}/{key[1]}", []).append(
                        {"path": str(path), "reason": str(error)}
                    )
                    continue
            selected[key] = path
            selected_records.append(record)
            break
        if len(records) > 1:
            ambiguous[f"{key[0]}/{key[1]}"] = [item["path"] for item in records]

    missing = sorted(required_keys - set(selected))
    registry = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "selection_rule": "base_only + full_train + seed42; lexicographically first path; never selected by metric",
        "required_count": len(required_keys),
        "selected_count": len(selected),
        "selected": selected_records,
        "multiple_compatible_candidates": ambiguous,
        "rejected_incompatible_candidates": rejected,
        "missing": [f"{dataset}/{backbone}" for dataset, backbone in missing],
    }
    if registry_path is not None:
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry_path.write_text(
            json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if missing:
        formatted = "\n".join(f"- {dataset}/{backbone}" for dataset, backbone in missing)
        raise FileNotFoundError(
            "missing compatible existing Base-only checkpoints; this queue will not "
            f"train replacements:\n{formatted}"
        )
    return selected


def build_base_compatibility_check(
    repo_root: Path,
    downstream_configs: dict[tuple[str, str], Path],
) -> Callable[[tuple[str, str], Path], None]:
    """Build a strict data-contract and backbone-state compatibility gate."""
    expected_states: dict[tuple[str, str], dict[str, tuple[tuple[int, ...], str]]] = {}
    expected_configs: dict[tuple[str, str], object] = {}

    def check(key: tuple[str, str], checkpoint_path: Path) -> None:
        if key not in expected_states:
            config = load_config(downstream_configs[key])
            graph_path = (repo_root / config.data.adjacency_path).resolve()
            graph = load_graph(graph_path)
            model = build_downstream_model(config, graph)
            expected_states[key] = {
                name: (tuple(value.shape), str(value.dtype))
                for name, value in model.backbone.state_dict().items()
            }
            expected_configs[key] = config
            del model

        checkpoint = _torch_load(checkpoint_path)
        payload_config = checkpoint.get("config", {})
        actual_data = payload_config.get("data", {})
        actual_model = payload_config.get("model", {})
        expected_config = expected_configs[key]
        for name in (
            "channel_index",
            "context_length",
            "horizon",
            "frequency_minutes",
            "train_ratio",
            "val_ratio",
            "zero_is_missing",
        ):
            actual_value = actual_data.get(name, getattr(expected_config.data, name))
            expected_value = getattr(expected_config.data, name)
            if actual_value != expected_value:
                raise ValueError(
                    f"data contract mismatch for {name}: {actual_value!r} != {expected_value!r}"
                )
        for name in ("input_channels", "output_channels"):
            actual_value = actual_model.get(name, getattr(expected_config.model, name))
            expected_value = getattr(expected_config.model, name)
            if actual_value != expected_value:
                raise ValueError(
                    f"model contract mismatch for {name}: {actual_value!r} != {expected_value!r}"
                )

        state = checkpoint.get("downstream_state_dict", {})
        prefix = "backbone."
        actual_state = {
            name[len(prefix) :]: (tuple(value.shape), str(value.dtype))
            for name, value in state.items()
            if name.startswith(prefix)
        }
        expected_state = expected_states[key]
        if actual_state != expected_state:
            missing = sorted(set(expected_state) - set(actual_state))[:5]
            unexpected = sorted(set(actual_state) - set(expected_state))[:5]
            shape_mismatch = [
                name
                for name in sorted(set(actual_state) & set(expected_state))
                if actual_state[name] != expected_state[name]
            ][:5]
            raise ValueError(
                "incompatible backbone state: "
                f"missing={missing}, unexpected={unexpected}, shape_or_dtype={shape_mismatch}"
            )

    return check


def _bank_is_complete(bank: Path, checkpoint: Path) -> bool:
    if not all((bank / name).is_file() for name in BANK_REQUIRED_FILES):
        return False
    manifest = json.loads((bank / "manifest.json").read_text(encoding="utf-8"))
    return manifest.get("encoder_fingerprint") == _checkpoint_fingerprint(checkpoint)


def _ensure_bank(
    runner: QueueRunner,
    *,
    label: str,
    config: Path,
    checkpoint: Path,
    bank: Path,
    dataset_name: str,
    temporary_root: Path,
) -> None:
    if _bank_is_complete(bank, checkpoint):
        outputs = [bank / name for name in BANK_REQUIRED_FILES]
        if not runner.is_complete(label, outputs):
            runner.mark_complete(label, command=None, outputs=outputs)
        print(f"[REUSE] {label}: {bank}", flush=True)
        return
    if bank.exists():
        safe_remove_temporary_bank(bank, temporary_root)
    outputs = [bank / name for name in BANK_REQUIRED_FILES]
    runner.run(
        label,
        [
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
        outputs,
    )
    if not _bank_is_complete(bank, checkpoint):
        runner.marker(label).unlink(missing_ok=True)
        raise RuntimeError(f"Bank fingerprint/completeness check failed: {bank}")


def _archive_bank(paper_root: Path, label: str, bank: Path) -> Path:
    archive = paper_root / "bank_manifests" / label
    archive.mkdir(parents=True, exist_ok=True)
    shutil.copy2(bank / "manifest.json", archive / "manifest.json")
    inventory = {
        "bank": str(bank),
        "archived_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "files": {
            path.name: path.stat().st_size for path in bank.iterdir() if path.is_file()
        },
    }
    inventory_path = archive / "inventory.json"
    inventory_path.write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return inventory_path


def _run_router(
    runner: QueueRunner,
    *,
    dataset: str,
    backbone: str,
    config: Path,
    encoder: Path,
    bank: Path,
    base_checkpoint: Path,
    paper_root: Path,
) -> Path:
    run_name = f"{PAPER_RUN}/main_table/{dataset}/ours/{backbone}"
    checkpoint = paper_root / "main_table" / dataset / "ours" / backbone / "downstream_best.pt"
    encoder_variant = "source" if dataset == "metrla" else "finetuned"
    runner.run(
        f"{dataset}_{encoder_variant}_router_{backbone}",
        [
            "scripts/train_downstream.py",
            "--config",
            str(config),
            "--pretrained-checkpoint",
            str(encoder),
            "--bank",
            str(bank),
            "--base-checkpoint",
            str(base_checkpoint),
            "--mode",
            "learned_topk_error_aware",
            "--candidate-protocol",
            "weekday_radius1_overlap",
            "--level-weight",
            "0",
            "--candidate-quality-weight",
            "0",
            "--epochs",
            "50",
            "--disable-early-stopping",
            "--frozen-path-cache",
            "--seed",
            str(SEED),
            "--run-name",
            run_name,
        ],
        [checkpoint],
    )
    return checkpoint


def _run_visualization(
    runner: QueueRunner,
    *,
    label: str,
    config: Path,
    checkpoint: Path,
    bank: Path,
    random_checkpoint: Path,
    random_bank: Path,
    output_dir: Path,
    protocol: str,
) -> None:
    runner.run(
        label,
        [
            "scripts/visualize_retrieval.py",
            "--version",
            "hn_offset_only_v1",
            "--config",
            str(config),
            "--checkpoint",
            str(checkpoint),
            "--bank",
            str(bank),
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
        [output_dir / "metrics.json", output_dir / "cases.json"],
    )


def _finish_bank_stage(
    runner: QueueRunner,
    *,
    label: str,
    banks: dict[str, Path],
    temporary_root: Path,
    consumer_outputs: Iterable[Path] = (),
) -> None:
    outputs = [
        runner.paper_root / "bank_manifests" / name / "inventory.json" for name in banks
    ] + [Path(path) for path in consumer_outputs]
    if runner.is_complete(label, outputs):
        for bank in banks.values():
            safe_remove_temporary_bank(bank, temporary_root)
        print(f"[SKIP] {label}; no completed-stage temporary Bank retained", flush=True)
        return
    for name, bank in banks.items():
        inventory = runner.paper_root / "bank_manifests" / name / "inventory.json"
        if bank.exists():
            _archive_bank(runner.paper_root, name, bank)
        elif not inventory.exists():
            raise RuntimeError(
                f"Bank and its archived inventory are both missing before stage completion: {bank}"
            )
    for bank in banks.values():
        safe_remove_temporary_bank(bank, temporary_root)
    runner.mark_complete(label, command=None, outputs=outputs)
    print(f"[DONE] {label}; temporary Banks removed", flush=True)


def _materialize_configs(repo_root: Path, paper_root: Path) -> tuple[dict[tuple[str, str], Path], dict[str, Path]]:
    config_root = paper_root / "resolved_configs"
    temporary = paper_root / "temporary_banks"
    downstream: dict[tuple[str, str], Path] = {}
    adaptation: dict[str, Path] = {}
    for dataset in DATASETS:
        bank = temporary / ("metrla_source" if dataset == "metrla" else f"{dataset}_finetuned")
        bank_relative = bank.relative_to(repo_root).as_posix()
        for backbone in MAIN_TABLE_BACKBONES[dataset]:
            path = config_root / "downstream" / f"{dataset}_{backbone}.yaml"
            materialize_downstream_config(
                repo_root=repo_root,
                dataset=dataset,
                backbone=backbone,
                output_path=path,
                bank_output=bank_relative,
            )
            load_config(path).validate()
            downstream[(dataset, backbone)] = path
        if dataset != "metrla":
            path = config_root / "adaptation" / f"{dataset}_joint_context_offset_only.yaml"
            materialize_adaptation_config(
                repo_root=repo_root,
                dataset=dataset,
                output_path=path,
                bank_output=bank_relative,
            )
            load_config(path).validate()
            adaptation[dataset] = path
    return downstream, adaptation


def _preflight(repo_root: Path, downstream: dict[tuple[str, str], Path]) -> None:
    required = [repo_root / SOURCE_CHECKPOINT_REL, repo_root / REFERENCE_CHECKPOINT_REL]
    for dataset in DATASETS:
        config = _read_yaml(downstream[(dataset, "gwn")])
        for name in ("raw_path", "adjacency_path"):
            required.append((repo_root / config["data"][name]).resolve())
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("preflight missing paths:\n" + "\n".join(map(str, missing)))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the active Python environment")
    properties = torch.cuda.get_device_properties(0)
    print(
        f"GPU: {properties.name}, memory={properties.total_memory / 1024**3:.1f} GiB",
        flush=True,
    )


def run_queue(repo_root: Path) -> None:
    paper_root = repo_root / PAPER_ROOT_REL
    temporary_root = paper_root / "temporary_banks"
    temporary_root.mkdir(parents=True, exist_ok=True)
    downstream, adaptation = _materialize_configs(repo_root, paper_root)
    _preflight(repo_root, downstream)
    required_bases = [
        (dataset, backbone)
        for dataset in DATASETS
        for backbone in MAIN_TABLE_BACKBONES[dataset]
    ]
    base_checkpoints = discover_base_checkpoints(
        repo_root,
        required_bases,
        registry_path=paper_root / "base_checkpoint_registry.json",
        compatibility_check=build_base_compatibility_check(repo_root, downstream),
    )
    print(
        f"Existing Base-only checkpoints validated: {len(base_checkpoints)}; "
        "no Base-only training will run.",
        flush=True,
    )
    runner = QueueRunner(repo_root, paper_root)
    source_checkpoint = repo_root / SOURCE_CHECKPOINT_REL
    reference_checkpoint = repo_root / REFERENCE_CHECKPOINT_REL
    random_checkpoint = paper_root / "controls" / "random_encoder_seed42.pt"
    random_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    runner.run(
        "random_checkpoint",
        [
            "scripts/build_random_checkpoint.py",
            "--config",
            str(downstream[("metrla", "gwn")]),
            "--output",
            str(random_checkpoint),
            "--seed",
            str(SEED),
        ],
        [random_checkpoint],
    )

    for dataset in DATASETS:
        selected_backbones = MAIN_TABLE_BACKBONES[dataset]
        print(
            f"\n######## {dataset.upper()} FINAL ROUTER ({len(selected_backbones)}) ########",
            flush=True,
        )

        if dataset == "metrla":
            metrla_stage_outputs = [
                paper_root / "bank_manifests" / name / "inventory.json"
                for name in (
                    "metrla_source",
                    "metrla_random",
                    "metrla_reference_offset_decay",
                )
            ] + [
                paper_root / "source_case_study" / protocol / "metrics.json"
                for protocol in ("pretrain_broad_causal", "weekday_radius1_overlap")
            ] + [
                paper_root / "source_case_study" / "mirage_ab" / "mirage_cases.json",
                paper_root
                / "source_case_study"
                / "context_future_key"
                / "context_key_alignment.json",
            ] + [
                paper_root / "main_table" / dataset / "ours" / backbone / "downstream_best.pt"
                for backbone in selected_backbones
            ]
            if runner.is_complete("metrla_source_bank_cleanup", metrla_stage_outputs):
                print("[SKIP] metrla_source_bank_cleanup", flush=True)
                continue
            source_bank = temporary_root / "metrla_source"
            random_bank = temporary_root / "metrla_random"
            reference_bank = temporary_root / "metrla_reference_offset_decay"
            _ensure_bank(
                runner,
                label="metrla_source_bank_build",
                config=downstream[(dataset, "gwn")],
                checkpoint=source_checkpoint,
                bank=source_bank,
                dataset_name="metrla_final_joint_context_offset_only",
                temporary_root=temporary_root,
            )
            _ensure_bank(
                runner,
                label="metrla_random_bank_build",
                config=downstream[(dataset, "gwn")],
                checkpoint=random_checkpoint,
                bank=random_bank,
                dataset_name="metrla_random_encoder_control",
                temporary_root=temporary_root,
            )
            _ensure_bank(
                runner,
                label="metrla_reference_bank_build",
                config=repo_root / REFERENCE_CONFIG_REL,
                checkpoint=reference_checkpoint,
                bank=reference_bank,
                dataset_name="metrla_offset_decay_reference",
                temporary_root=temporary_root,
            )
            for protocol in ("pretrain_broad_causal", "weekday_radius1_overlap"):
                _run_visualization(
                    runner,
                    label=f"metrla_case_study_{protocol}",
                    config=downstream[(dataset, "gwn")],
                    checkpoint=source_checkpoint,
                    bank=source_bank,
                    random_checkpoint=random_checkpoint,
                    random_bank=random_bank,
                    output_dir=paper_root / "source_case_study" / protocol,
                    protocol=protocol,
                )
            mirage_dir = paper_root / "source_case_study" / "mirage_ab"
            runner.run(
                "metrla_mirage_ab",
                [
                    "scripts/extract_spatiotemporal_mirages.py",
                    "--data",
                    str((repo_root / _read_yaml(downstream[(dataset, "gwn")])["data"]["raw_path"]).resolve()),
                    "--bank",
                    str(source_bank),
                    "--output-dir",
                    str(mirage_dir),
                    "--num-events",
                    "5000",
                    "--seed",
                    str(SEED),
                    "--skip-population-clustering",
                ],
                [
                    mirage_dir / "mirage_cases.json",
                    mirage_dir / "context_similar_future_different_cluster.png",
                    mirage_dir / "context_different_future_similar_cluster.png",
                ],
            )
            context_dir = paper_root / "source_case_study" / "context_future_key"
            runner.run(
                "metrla_context_future_key",
                [
                    "scripts/analyze_context_key_alignment.py",
                    "--data",
                    str((repo_root / _read_yaml(downstream[(dataset, "gwn")])["data"]["raw_path"]).resolve()),
                    "--bank",
                    str(source_bank),
                    "--reference-bank",
                    str(reference_bank),
                    "--random-bank",
                    str(random_bank),
                    "--output",
                    str(context_dir / "context_key_alignment.json"),
                    "--num-events",
                    "2500",
                    "--num-nodes",
                    "64",
                    "--max-pairs-per-node",
                    "1500",
                    "--quantile",
                    "0.20",
                    "--level-control-quantile",
                    "0.30",
                    "--surface-bins",
                    "8",
                    "--bootstrap-samples",
                    "1000",
                    "--seed",
                    str(SEED),
                ],
                [
                    context_dir / "context_key_alignment.json",
                    context_dir / "context_future_key_relation.png",
                    context_dir / "context_future_key_relation.pdf",
                ],
            )
            for backbone in selected_backbones:
                _run_router(
                    runner,
                    dataset=dataset,
                    backbone=backbone,
                    config=downstream[(dataset, backbone)],
                    encoder=source_checkpoint,
                    bank=source_bank,
                    base_checkpoint=base_checkpoints[(dataset, backbone)],
                    paper_root=paper_root,
                )
            _finish_bank_stage(
                runner,
                label="metrla_source_bank_cleanup",
                banks={
                    "metrla_source": source_bank,
                    "metrla_random": random_bank,
                    "metrla_reference_offset_decay": reference_bank,
                },
                temporary_root=temporary_root,
                consumer_outputs=metrla_stage_outputs[3:],
            )
            continue

        source_bank = temporary_root / f"{dataset}_source_zero_shot"
        random_bank = temporary_root / f"{dataset}_random"
        zero_shot_stage = f"{dataset}_zero_shot_bank_cleanup"
        zero_shot_archives = [
            paper_root / "bank_manifests" / f"{dataset}_source_zero_shot" / "inventory.json",
            paper_root / "bank_manifests" / f"{dataset}_random" / "inventory.json",
            paper_root / "cross_domain" / dataset / "pretrain_broad_causal" / "metrics.json",
            paper_root / "cross_domain" / dataset / "weekday_radius1_overlap" / "metrics.json",
        ]
        if not runner.is_complete(zero_shot_stage, zero_shot_archives):
            _ensure_bank(
                runner,
                label=f"{dataset}_source_bank_build",
                config=adaptation[dataset],
                checkpoint=source_checkpoint,
                bank=source_bank,
                dataset_name=f"{dataset}_source_zero_shot",
                temporary_root=temporary_root,
            )
            _ensure_bank(
                runner,
                label=f"{dataset}_random_bank_build",
                config=adaptation[dataset],
                checkpoint=random_checkpoint,
                bank=random_bank,
                dataset_name=f"{dataset}_random_encoder_control",
                temporary_root=temporary_root,
            )
            for protocol in ("pretrain_broad_causal", "weekday_radius1_overlap"):
                _run_visualization(
                    runner,
                    label=f"{dataset}_zero_shot_{protocol}",
                    config=adaptation[dataset],
                    checkpoint=source_checkpoint,
                    bank=source_bank,
                    random_checkpoint=random_checkpoint,
                    random_bank=random_bank,
                    output_dir=paper_root / "cross_domain" / dataset / protocol,
                    protocol=protocol,
                )
            _finish_bank_stage(
                runner,
                label=zero_shot_stage,
                banks={
                    f"{dataset}_source_zero_shot": source_bank,
                    f"{dataset}_random": random_bank,
                },
                temporary_root=temporary_root,
                consumer_outputs=zero_shot_archives[2:],
            )
        else:
            print(f"[SKIP] {zero_shot_stage}", flush=True)

        finetune_dir = paper_root / "retrieval_adaptation" / dataset
        finetuned_checkpoint = finetune_dir / "retrieval_t1_best.pt"
        runner.run(
            f"{dataset}_retrieval_finetune",
            [
                "scripts/finetune_retrieval.py",
                "--config",
                str(adaptation[dataset]),
                "--source-checkpoint",
                str(source_checkpoint),
                "--epochs",
                "8",
                "--run-name",
                f"{PAPER_RUN}/retrieval_adaptation/{dataset}",
                "--seed",
                str(SEED),
            ],
            [finetuned_checkpoint],
        )
        finetuned_bank = temporary_root / f"{dataset}_finetuned"
        final_stage = f"{dataset}_finetuned_bank_cleanup"
        final_archives = [
            paper_root / "bank_manifests" / f"{dataset}_finetuned" / "inventory.json"
        ] + [
            paper_root / "main_table" / dataset / "ours" / backbone / "downstream_best.pt"
            for backbone in selected_backbones
        ]
        if not runner.is_complete(final_stage, final_archives):
            _ensure_bank(
                runner,
                label=f"{dataset}_finetuned_bank_build",
                config=adaptation[dataset],
                checkpoint=finetuned_checkpoint,
                bank=finetuned_bank,
                dataset_name=f"{dataset}_joint_context_offset_only_t1",
                temporary_root=temporary_root,
            )
            for backbone in selected_backbones:
                _run_router(
                    runner,
                    dataset=dataset,
                    backbone=backbone,
                    config=downstream[(dataset, backbone)],
                    encoder=finetuned_checkpoint,
                    bank=finetuned_bank,
                    base_checkpoint=base_checkpoints[(dataset, backbone)],
                    paper_root=paper_root,
                )
            _finish_bank_stage(
                runner,
                label=final_stage,
                banks={f"{dataset}_finetuned": finetuned_bank},
                temporary_root=temporary_root,
                consumer_outputs=final_archives[1:],
            )
        else:
            print(f"[SKIP] {final_stage}", flush=True)

    completion = paper_root / "queue_state" / "ALL_EXPERIMENTS_COMPLETED.json"
    completion.write_text(
        json.dumps(
            {
                "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "datasets": list(DATASETS),
                "backbones": list(BACKBONES),
                "main_table_backbones": {
                    dataset: list(backbones)
                    for dataset, backbones in MAIN_TABLE_BACKBONES.items()
                },
                "base_only_runs": 0,
                "reused_base_only_checkpoints": 24,
                "final_method_runs": 24,
                "pems07_included": False,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nALL PAPER EXPERIMENTS COMPLETED: {completion}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the publication-final single-GPU experiment queue."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the ordered task matrix without writing artifacts or running jobs.",
    )
    mode.add_argument(
        "--validate-only",
        action="store_true",
        help="Materialize configs and validate paths, checkpoints, and CUDA without running jobs.",
    )
    parser.add_argument(
        "--allow-env-mismatch",
        action="store_true",
        help="Allow execution outside the experiment conda environment (engineering only).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)
    if args.dry_run:
        plan = build_experiment_plan()
        for index, job in enumerate(plan, start=1):
            print(f"{index:03d} {job['kind']:<24} {job['label']}")
        counts: dict[str, int] = {}
        for job in plan:
            counts[job["kind"]] = counts.get(job["kind"], 0) + 1
        print(json.dumps({"total": len(plan), "counts": counts}, indent=2))
        return
    active_environment = os.environ.get("CONDA_DEFAULT_ENV")
    if not args.allow_env_mismatch and active_environment != "experiment":
        raise RuntimeError(
            "queue must run in conda environment 'experiment'; "
            f"observed CONDA_DEFAULT_ENV={active_environment!r}"
        )
    if args.validate_only:
        paper_root = repo_root / PAPER_ROOT_REL
        downstream, _ = _materialize_configs(repo_root, paper_root)
        _preflight(repo_root, downstream)
        required_bases = [
            (dataset, backbone)
            for dataset in DATASETS
            for backbone in MAIN_TABLE_BACKBONES[dataset]
        ]
        base_checkpoints = discover_base_checkpoints(
            repo_root,
            required_bases,
            registry_path=paper_root / "base_checkpoint_registry.json",
            compatibility_check=build_base_compatibility_check(repo_root, downstream),
        )
        print(f"Existing Base-only checkpoints validated: {len(base_checkpoints)}")
        print(
            "Source retrieval fingerprint: "
            + _checkpoint_fingerprint(repo_root / SOURCE_CHECKPOINT_REL)
        )
        print(
            "Reference retrieval fingerprint: "
            + _checkpoint_fingerprint(repo_root / REFERENCE_CHECKPOINT_REL)
        )
        print("VALIDATION PASSED; no training or Bank build was started.")
        return
    run_queue(repo_root)


if __name__ == "__main__":
    main()
