"""Compare random, source, and T1 keys with source-domain HN metrics."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stanchor.bank.storage import MemoryBank
from stanchor.config import load_config, resolve_project_path
from stanchor.diagnostics.retrieval_visualization import (
    CURRENT_VISUALIZATION_VERSION,
    _candidate_teacher_signatures,
    alignment_statistics,
    anchor_wise_ranking_metrics,
    build_diagnostic_event_candidates,
    build_teacher_aligned_signature,
    future_information_boundary,
    future_neighbor_recall_at_k,
    node_key_distances,
    teacher_candidate_distances,
    validate_aligned_bank_axes,
)
from stanchor.diagnostics.target_adaptation import summarize_hn_selector_deltas
from stanchor.engine.common import build_data_and_graph, load_pretrained_model
from stanchor.engine.target import _validate_bank
from stanchor.retrieval.strategies import event_candidate_futures
from stanchor.utils import resolve_device, save_json


SELECTORS = ("random", "source", "finetuned")


def _candidate_node_keys(
    bank: MemoryBank,
    event_ids: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    safe_ids = event_ids.clamp_min(0).cpu().numpy()
    values = np.asarray(bank.node_keys[safe_ids], dtype=np.float32)
    return torch.from_numpy(values).to(device)


def _ranking_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if not isinstance(value, np.ndarray)
    }


def _pool_summary(chunks: list[np.ndarray]) -> dict[str, float | int]:
    values = np.concatenate(chunks).astype(np.float64, copy=False)
    positive = values[values > 0]
    if positive.size == 0:
        raise ValueError("candidate protocol produced no causal candidates")
    return {
        "count": int(values.size),
        "mean": float(positive.mean()),
        "std": float(positive.std()),
        "min": int(positive.min()),
        "median": float(np.median(positive)),
        "max": int(positive.max()),
        "coverage": float(positive.size / values.size),
    }


@torch.no_grad()
def run_case_study(
    config_path: str | Path,
    random_checkpoint: str | Path,
    random_bank: str | Path,
    source_checkpoint: str | Path,
    source_bank: str | Path,
    adapted_checkpoint: str | Path,
    adapted_bank: str | Path,
    output: str | Path,
    *,
    candidate_protocol: str = "weekday_radius1_overlap",
    event_top_r: int = 96,
    node_top_k: int = 12,
    batch_size: int | None = None,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Run one target-domain protocol with a shared HN teacher and event axis."""
    if event_top_r <= 5:
        raise ValueError("event_top_r must exceed 5 for non-trivial Recall@5")
    if node_top_k <= 0:
        raise ValueError("node_top_k must be positive")

    started = time.perf_counter()
    config = load_config(config_path)
    device = resolve_device(config.runtime.device)
    data, graph_cpu = build_data_and_graph(config)
    graph = graph_cpu.to(device)
    evaluation_batch_size = int(batch_size or config.target.batch_size)
    if evaluation_batch_size <= 0:
        raise ValueError("batch_size must be positive")
    loader = DataLoader(
        data.val,
        batch_size=evaluation_batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )

    checkpoint_paths = {
        "random": resolve_project_path(random_checkpoint),
        "source": resolve_project_path(source_checkpoint),
        "finetuned": resolve_project_path(adapted_checkpoint),
    }
    bank_paths = {
        "random": resolve_project_path(random_bank),
        "source": resolve_project_path(source_bank),
        "finetuned": resolve_project_path(adapted_bank),
    }
    loaded = {
        name: load_pretrained_model(
            config,
            checkpoint_paths[name],
            data.series.slots_per_day,
            device,
        )
        for name in SELECTORS
    }
    models = {name: loaded[name][0].eval() for name in SELECTORS}
    states = {name: loaded[name][1] for name in SELECTORS}

    key_chunks: dict[str, list[np.ndarray]] = {name: [] for name in SELECTORS}
    future_chunks: list[np.ndarray] = []
    valid_chunks: list[np.ndarray] = []
    recall_chunks: dict[str, list[np.ndarray]] = {name: [] for name in SELECTORS}
    pool_chunks: list[np.ndarray] = []
    query_count = 0
    batch_count = 0

    with (
        MemoryBank(bank_paths["random"]) as random_memory,
        MemoryBank(bank_paths["source"]) as source_memory,
        MemoryBank(bank_paths["finetuned"]) as finetuned_memory,
    ):
        source_random_alignment = validate_aligned_bank_axes(source_memory, random_memory)
        source_finetuned_alignment = validate_aligned_bank_axes(source_memory, finetuned_memory)
        memories = {
            "random": random_memory,
            "source": source_memory,
            "finetuned": finetuned_memory,
        }
        for name in SELECTORS:
            _validate_bank(
                memories[name],
                models[name],
                graph_cpu,
                data.scaler.state_dict(),
            )

        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            query_keys = {}
            for name in SELECTORS:
                encoding = models[name].encode_clean(
                    batch["retrieval_x"].to(device),
                    batch["retrieval_observed"].to(device).bool(),
                    batch["retrieval_weekday"].to(device),
                    batch["retrieval_slot"].to(device),
                    graph,
                )
                query_keys[name] = encoding.retrieval.node_keys

            events = build_diagnostic_event_candidates(
                source_memory,
                batch["query_weekday"].to(device),
                batch["query_slot"].to(device),
                batch["context_start"].to(device),
                event_top_r,
                device,
                candidate_protocol,
            )
            pool_chunks.append(events.valid.sum(dim=1).cpu().numpy())

            key_distances = {}
            key_valid = {}
            for name in SELECTORS:
                candidate_keys = _candidate_node_keys(
                    memories[name], events.event_ids, device
                )
                key_distances[name], key_valid[name] = node_key_distances(
                    query_keys[name], candidate_keys, events.valid
                )

            event_future, event_future_valid = event_candidate_futures(
                source_memory, events.event_ids, events.valid, device
            )
            candidate_future = event_future.permute(0, 3, 1, 2, 4).contiguous()
            candidate_future_valid = event_future_valid.permute(0, 3, 1, 2, 4).contiguous()
            query_signature, query_signature_valid = build_teacher_aligned_signature(
                CURRENT_VISUALIZATION_VERSION,
                batch["y"].to(device),
                batch["y_observed"].to(device).bool(),
                batch["x"].to(device),
                batch["x_observed"].to(device).bool(),
            )
            candidate_signature, candidate_signature_valid = _candidate_teacher_signatures(
                source_memory,
                events,
                candidate_future,
                candidate_future_valid,
                data,
                config.data.context_length,
                device,
            )
            future_distance, future_valid = teacher_candidate_distances(
                query_signature,
                query_signature_valid,
                candidate_signature,
                candidate_signature_valid,
                events.valid,
                config.pretrain.relation_distance_normalization,
            )
            common_valid = future_valid
            for name in SELECTORS:
                common_valid = common_valid & key_valid[name]

            for name in SELECTORS:
                key_chunks[name].append(key_distances[name].cpu().numpy())
                recall, eligible = future_neighbor_recall_at_k(
                    key_distances[name], future_distance, common_valid, k=5
                )
                recall_chunks[name].append(recall.masked_select(eligible).cpu().numpy())
            future_chunks.append(future_distance.cpu().numpy())
            valid_chunks.append(common_valid.cpu().numpy())
            query_count += int(batch["y"].shape[0])
            batch_count += 1
            print(
                f"[{candidate_protocol}] batch={batch_count}/{len(loader)} "
                f"queries={query_count}",
                flush=True,
            )

    if query_count == 0:
        raise ValueError("no validation queries were processed")
    future = np.concatenate(future_chunks, axis=0)
    valid = np.concatenate(valid_chunks, axis=0)
    del future_chunks, valid_chunks
    methods = {}
    for name in SELECTORS:
        keys = np.concatenate(key_chunks[name], axis=0)
        ranking = anchor_wise_ranking_metrics(
            keys,
            future,
            valid,
            ndcg_k=5,
            teacher_temperature=config.pretrain.relation_teacher_temperature,
        )
        alignment = alignment_statistics(keys, future, valid)
        recalls = np.concatenate(recall_chunks[name])
        alignment["future_neighbor_recall_at_5"] = float(recalls.mean())
        alignment["recall_at_5_eligible_anchors"] = int(recalls.size)
        methods[name] = {
            "alignment": alignment,
            "ranking": _ranking_summary(ranking),
        }
        del keys

    result = {
        "schema_version": 2,
        "study": "target_source_random_t1_hn_offset_decay_v2",
        "dataset": source_memory.manifest.dataset_name,
        "split": "val",
        "complete_validation": max_batches is None,
        "candidate_protocol": candidate_protocol,
        "event_top_r": event_top_r,
        "node_top_k": node_top_k,
        "evaluation_batch_size": evaluation_batch_size,
        "queries": query_count,
        "batches": batch_count,
        "candidate_pool": _pool_summary(pool_chunks),
        "teacher_signature": {
            "name": "HN-OffsetDecay-v2",
            "distance_normalization": config.pretrain.relation_distance_normalization,
            "context_steps": config.data.context_length,
        },
        "methods": methods,
        "deltas": summarize_hn_selector_deltas(
            methods["random"], methods["source"], methods["finetuned"]
        ),
        "checkpoints": {
            name: {
                "path": str(checkpoint_paths[name].resolve()),
                "epoch": states[name].get("epoch"),
            }
            for name in SELECTORS
        },
        "banks": {name: str(bank_paths[name].resolve()) for name in SELECTORS},
        "bank_alignment": {
            "source_random": source_random_alignment,
            "source_finetuned": source_finetuned_alignment,
            "shared_event_axis_for_all_selectors": True,
        },
        "future_information_boundary": future_information_boundary(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    output_path = resolve_project_path(output)
    save_json(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--random-checkpoint", required=True)
    parser.add_argument("--random-bank", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--source-bank", required=True)
    parser.add_argument("--adapted-checkpoint", required=True)
    parser.add_argument("--adapted-bank", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidate-protocol", default="weekday_radius1_overlap")
    parser.add_argument("--event-top-r", type=int, default=96)
    parser.add_argument("--node-top-k", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()
    options = vars(args)
    options["config_path"] = options.pop("config")
    result = run_case_study(**options)
    compact = {
        "dataset": result["dataset"],
        "candidate_protocol": result["candidate_protocol"],
        "queries": result["queries"],
        "candidate_pool": result["candidate_pool"],
        "deltas": result["deltas"],
        "elapsed_seconds": result["elapsed_seconds"],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
