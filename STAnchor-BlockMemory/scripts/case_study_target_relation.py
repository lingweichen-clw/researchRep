"""Measure whether a frozen source retrieval key preserves future relations on a target domain.

The script is validation-only.  Candidate pools are built without query futures;
future trajectories are used only after ranking to compute offline relation metrics.
"""

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
    build_diagnostic_event_candidates,
    validate_aligned_bank_axes,
)
from stanchor.diagnostics.target_relation import (
    future_similarity,
    relation_metric_arrays,
    summarize_query_metrics,
)
from stanchor.engine.common import build_data_and_graph, load_pretrained_model
from stanchor.engine.target import _validate_bank
from stanchor.retrieval.strategies import candidate_contexts, event_candidate_futures, raw_l1_candidate_scores
from stanchor.utils import resolve_device, save_json


def _key_cosine_scores(query_keys: torch.Tensor, candidate_keys: torch.Tensor) -> torch.Tensor:
    """Return node-level cosine scores as [B,N,R], larger is better."""
    query = torch.nn.functional.normalize(query_keys.float(), dim=-1)
    candidate = torch.nn.functional.normalize(candidate_keys.float(), dim=-1)
    return torch.einsum("bnd,brnd->bnr", query, candidate).clamp(-1.0, 1.0)


def _candidate_node_keys(bank: MemoryBank, event_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
    safe_ids = event_ids.clamp_min(0).cpu().numpy()
    return torch.from_numpy(np.asarray(bank.node_keys[safe_ids], dtype=np.float32)).to(device)


def _future_arrays(
    batch: dict[str, torch.Tensor],
    event_future: torch.Tensor,
    event_future_valid: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert [B,H,N,R,C] payloads to target_relation's [B,N,R,H] arrays."""
    query = batch["y"].detach().cpu().numpy().astype(np.float32)
    query_valid = batch["y_observed"].detach().cpu().numpy().astype(bool)
    candidates = event_future.detach().cpu().numpy().astype(np.float32)
    candidates_valid = event_future_valid.detach().cpu().numpy().astype(bool)
    # The target configs select one physical speed channel.  Averaging is only
    # a defensive fallback for a future multi-channel config.
    query = query.mean(axis=-1).transpose(0, 2, 1)
    query_valid = query_valid.all(axis=-1).transpose(0, 2, 1)
    candidates = candidates.mean(axis=-1).transpose(0, 2, 3, 1)
    candidates_valid = candidates_valid.all(axis=-1).transpose(0, 2, 3, 1)
    return (query, query_valid), (candidates, candidates_valid)


def _pair_density(
    method_scores: list[np.ndarray],
    future_scores: list[np.ndarray],
    valid: list[np.ndarray],
    *,
    seed: int,
    max_pairs: int,
) -> dict[str, list[float]]:
    key_values: list[np.ndarray] = []
    future_values: list[np.ndarray] = []
    for keys, futures, masks in zip(method_scores, future_scores, valid):
        key_values.append(np.asarray(keys)[np.asarray(masks)])
        future_values.append(np.asarray(futures)[np.asarray(masks)])
    if not key_values:
        return {"key_cosine": [], "future_cosine": []}
    keys = np.concatenate(key_values)
    futures = np.concatenate(future_values)
    if keys.size > max_pairs:
        rng = np.random.default_rng(seed)
        selected = rng.choice(keys.size, size=max_pairs, replace=False)
        keys, futures = keys[selected], futures[selected]
    return {"key_cosine": keys.astype(float).tolist(), "future_cosine": futures.astype(float).tolist()}


@torch.no_grad()
def run_case_study(
    config_path: str | Path,
    source_checkpoint: str | Path,
    source_bank: str | Path,
    random_checkpoint: str | Path,
    random_bank: str | Path,
    output: str | Path,
    *,
    candidate_protocol: str = "weekday_radius1_overlap",
    event_top_r: int = 96,
    node_top_k: int = 12,
    max_batches: int | None = None,
    max_density_pairs: int = 60000,
    seed: int = 42,
) -> dict[str, Any]:
    started = time.perf_counter()
    config = load_config(config_path)
    device = resolve_device(config.runtime.device)
    data, graph_cpu = build_data_and_graph(config)
    graph = graph_cpu.to(device)
    loader = DataLoader(
        data.val,
        batch_size=config.target.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )
    source_checkpoint = resolve_project_path(source_checkpoint)
    random_checkpoint = resolve_project_path(random_checkpoint)
    source_bank = resolve_project_path(source_bank)
    random_bank = resolve_project_path(random_bank)

    source_model, source_state = load_pretrained_model(
        config, source_checkpoint, data.series.slots_per_day, device
    )
    random_model, random_state = load_pretrained_model(
        config, random_checkpoint, data.series.slots_per_day, device
    )
    source_model.eval()
    random_model.eval()
    method_arrays: dict[str, list[dict[str, Any]]] = {
        "source_key": [],
        "random_key": [],
        "raw_l1": [],
        "oracle_future": [],
    }
    density_scores: dict[str, list[np.ndarray]] = {
        "source_key": [],
        "random_key": [],
    }
    density_future: list[np.ndarray] = []
    density_valid: dict[str, list[np.ndarray]] = {"source_key": [], "random_key": []}
    pool_counts: list[np.ndarray] = []
    pair_count = 0

    with MemoryBank(source_bank) as source_memory, MemoryBank(random_bank) as random_memory:
        alignment = validate_aligned_bank_axes(source_memory, random_memory)
        _validate_bank(source_memory, source_model, graph_cpu, data.scaler.state_dict())
        _validate_bank(random_memory, random_model, graph_cpu, data.scaler.state_dict())
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            retrieval_x = batch["retrieval_x"].to(device)
            retrieval_observed = batch["retrieval_observed"].to(device).bool()
            source_encoding = source_model.encode_clean(
                retrieval_x,
                retrieval_observed,
                batch["retrieval_weekday"].to(device),
                batch["retrieval_slot"].to(device),
                graph,
            )
            random_encoding = random_model.encode_clean(
                retrieval_x,
                retrieval_observed,
                batch["retrieval_weekday"].to(device),
                batch["retrieval_slot"].to(device),
                graph,
            )
            events = build_diagnostic_event_candidates(
                source_memory,
                batch["query_weekday"].to(device),
                batch["query_slot"].to(device),
                batch["context_start"].to(device),
                event_top_r,
                device,
                candidate_protocol,
            )
            pool_counts.append(events.valid.sum(dim=1).cpu().numpy())
            source_candidates = _candidate_node_keys(source_memory, events.event_ids, device)
            random_candidates = _candidate_node_keys(random_memory, events.event_ids, device)
            source_scores = _key_cosine_scores(source_encoding.retrieval.node_keys, source_candidates)
            random_scores = _key_cosine_scores(random_encoding.retrieval.node_keys, random_candidates)
            query_x = batch["x"].to(device)
            query_x_valid = batch["x_observed"].to(device).bool()
            contexts, contexts_valid = candidate_contexts(
                source_memory,
                events.event_ids,
                data.series,
                data.scaler,
                config.data.context_length,
                device,
            )
            raw_distance, raw_valid = raw_l1_candidate_scores(
                query_x, query_x_valid, contexts, contexts_valid, events.valid
            )
            raw_scores = -raw_distance
            event_future, event_future_valid = event_candidate_futures(
                source_memory, events.event_ids, events.valid, device
            )
            (query_future, query_future_valid), (candidate_future, candidate_future_valid) = _future_arrays(
                batch, event_future, event_future_valid
            )
            future_scores, future_valid = future_similarity(
                query_future, query_future_valid, candidate_future, candidate_future_valid
            )
            event_valid = events.valid.detach().cpu().numpy()[:, None, :]
            source_valid = future_valid & np.broadcast_to(event_valid, future_valid.shape)
            random_valid = source_valid
            raw_valid_np = raw_valid.detach().cpu().numpy() & future_valid
            source_np = source_scores.detach().cpu().numpy()
            random_np = random_scores.detach().cpu().numpy()
            raw_np = raw_scores.detach().cpu().numpy()
            for name, scores, valid in (
                ("source_key", source_np, source_valid),
                ("random_key", random_np, random_valid),
                ("raw_l1", raw_np, raw_valid_np),
                ("oracle_future", future_scores, future_valid),
            ):
                method_arrays[name].append(relation_metric_arrays(scores, future_scores, valid))
            for name, scores in (("source_key", source_np), ("random_key", random_np)):
                density_scores[name].append(scores)
                density_valid[name].append(source_valid if name == "source_key" else random_valid)
            density_future.append(future_scores)
            pair_count += int(future_valid.sum())

    summaries = {
        name: summarize_query_metrics(values, seed=seed, bootstrap_samples=200)
        for name, values in method_arrays.items()
    }
    pools = np.concatenate(pool_counts) if pool_counts else np.zeros(0, dtype=np.float32)
    density = {
        name: _pair_density(
            density_scores[name], density_future, density_valid[name],
            seed=seed, max_pairs=max_density_pairs,
        )
        for name in ("source_key", "random_key")
    }
    result = {
        "schema_version": 1,
        "study": "target_future_relation_transfer",
        "dataset": config.data.raw_path,
        "split": "val",
        "candidate_protocol": candidate_protocol,
        "event_top_r": event_top_r,
        "node_top_k": node_top_k,
        "queries": int(sum(len(v["query_spearman"]) for v in method_arrays["source_key"])),
        "candidate_pool": {
            "mean": float(pools.mean()) if pools.size else 0.0,
            "median": float(np.median(pools)) if pools.size else 0.0,
            "min": int(pools.min()) if pools.size else 0,
            "max": int(pools.max()) if pools.size else 0,
        },
        "methods": summaries,
        "density_pairs": density,
        "pair_count_before_density_sampling": pair_count,
        "source_checkpoint": str(source_checkpoint.resolve()),
        "random_checkpoint": str(random_checkpoint.resolve()),
        "source_bank": str(source_bank.resolve()),
        "random_bank": str(random_bank.resolve()),
        "bank_alignment": alignment,
        "future_information_boundary": {
            "ranking_inputs": ["query_history", "calendar", "causal_bank_metadata", "historical_keys"],
            "query_future_used_for_ranking": False,
            "future_use": "validation-only relation metrics and plots",
        },
        "elapsed_seconds": time.perf_counter() - started,
        "source_checkpoint_epoch": source_state.get("epoch"),
        "random_checkpoint_epoch": random_state.get("epoch"),
    }
    output = resolve_project_path(output)
    save_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--source-bank", required=True)
    parser.add_argument("--random-checkpoint", required=True)
    parser.add_argument("--random-bank", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidate-protocol", default="weekday_radius1_overlap")
    parser.add_argument("--event-top-r", type=int, default=96)
    parser.add_argument("--node-top-k", type=int, default=12)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--max-density-pairs", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    options = vars(args)
    options["config_path"] = options.pop("config")
    result = run_case_study(**options)
    print(json.dumps({
        "dataset": result["dataset"],
        "queries": result["queries"],
        "candidate_pool": result["candidate_pool"],
        "methods": {
            name: {
                "spearman_mean": values["spearman_mean"],
                "future_cosine_mean": values["future_cosine_mean"],
                "oracle_recall_mean": values["oracle_recall_mean"],
            }
            for name, values in result["methods"].items()
        },
        "elapsed_seconds": result["elapsed_seconds"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
