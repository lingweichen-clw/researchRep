"""Test whether trained retrieval keys preserve joint context/future similarity."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from stanchor.diagnostics.context_key_alignment import (
    add_key_and_control_distances,
    select_context_future_quadrants,
    summarize_quadrant_contrasts,
)


AXIS_ARRAYS = (
    "sample_id.npy",
    "weekday.npy",
    "slot.npy",
    "context_start.npy",
    "context_end.npy",
    "future_end.npy",
)


def _load_manifest(bank: Path) -> dict:
    return json.loads((bank / "manifest.json").read_text(encoding="utf-8"))


def _validate_aligned_banks(banks: list[Path], manifests: list[dict]) -> None:
    fields = ("num_events", "num_nodes", "context_length", "horizon", "retrieval_dim")
    for field in fields:
        values = [manifest[field] for manifest in manifests]
        if len(set(values)) != 1:
            raise ValueError(f"banks disagree on manifest field {field}: {values}")
    for filename in AXIS_ARRAYS:
        reference = np.load(banks[0] / filename, mmap_mode="r")
        for bank in banks[1:]:
            candidate = np.load(bank / filename, mmap_mode="r")
            if not np.array_equal(reference, candidate):
                raise ValueError(f"bank event axes differ for {filename}: {bank}")


def _calendar_pairs(weekday: np.ndarray, slot: np.ndarray) -> np.ndarray:
    pairs: list[tuple[int, int]] = []
    for slot_value in np.unique(slot):
        event_ids = np.flatnonzero(slot == slot_value)
        for left, right in combinations(event_ids.tolist(), 2):
            gap = abs(int(weekday[left]) - int(weekday[right])) % 7
            if min(gap, 7 - gap) <= 1:
                pairs.append((left, right))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def _masked_rms(
    values: np.ndarray,
    valid: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    overlap = valid[left] & valid[right]
    counts = overlap.sum(axis=-1)
    delta = values[left] - values[right]
    squared = np.where(overlap, np.square(delta, dtype=np.float32), 0.0).sum(axis=-1)
    distance = np.sqrt(squared / np.maximum(counts, 1))
    return distance, counts / values.shape[-1]


def _build_context_and_future_features(
    *,
    raw_values: np.ndarray,
    raw_observed: np.ndarray,
    sample_ids: np.ndarray,
    node_ids: np.ndarray,
    level_features: np.ndarray,
    future_values: np.ndarray,
    future_masks: np.ndarray,
    manifest: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    history_steps = int(manifest["context_length"])
    if history_steps % 24:
        raise ValueError("context length must be divisible into 24 patches")
    contexts = np.stack(
        [
            raw_values[sample - history_steps + 1 : sample + 1, node_ids].T
            for sample in sample_ids
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    context_valid = np.stack(
        [
            raw_observed[sample - history_steps + 1 : sample + 1, node_ids].T
            for sample in sample_ids
        ],
        axis=0,
    )
    scaler_mean = np.asarray(manifest["scaler"]["mean"], dtype=np.float32).reshape(-1)[node_ids]
    scaler_std = np.asarray(manifest["scaler"]["std"], dtype=np.float32).reshape(-1)[node_ids]
    scaler_eps = float(manifest["scaler"].get("eps", 1.0e-6))
    model_context = (
        contexts - scaler_mean[None, :, None]
    ) / (scaler_std[None, :, None] + scaler_eps)

    window_mean = level_features[..., 0]
    window_std = level_features[..., 1]
    normalized = (model_context - window_mean[..., None]) / (
        window_std[..., None] + 1.0e-6
    )
    normalized = np.where(context_valid, normalized, 0.0)
    event_count, node_count, _ = normalized.shape
    patch_size = history_steps // 24
    patch_values = normalized.reshape(event_count, node_count, 24, patch_size)
    patch_mask = context_valid.reshape(event_count, node_count, 24, patch_size)
    patch_count = patch_mask.sum(axis=-1)
    patch_means = np.where(
        patch_count > 0,
        np.where(patch_mask, patch_values, 0.0).sum(axis=-1)
        / np.maximum(patch_count, 1),
        0.0,
    ).astype(np.float32)
    patch_valid = patch_count > 0

    forecast_context = model_context[..., -int(manifest["horizon"]) :]
    forecast_valid = context_valid[..., -int(manifest["horizon"]) :]
    visible_count = forecast_valid.sum(axis=-1)
    visible_mean = np.where(
        visible_count > 0,
        np.where(forecast_valid, forecast_context, 0.0).sum(axis=-1)
        / np.maximum(visible_count, 1),
        0.0,
    )
    endpoint = np.where(forecast_valid[..., -1], forecast_context[..., -1], visible_mean)
    endpoint_valid = visible_count > 0
    future_model = np.asarray(future_values, dtype=np.float32).transpose(0, 2, 1)
    future_valid = np.asarray(future_masks, dtype=bool).transpose(0, 2, 1)
    signature_valid = future_valid & endpoint_valid[..., None]
    signature = np.where(
        signature_valid,
        future_model - endpoint[..., None],
        0.0,
    ).astype(np.float32)
    return patch_means, patch_valid, signature, signature_valid


def _pair_records(
    context: np.ndarray,
    context_valid: np.ndarray,
    future: np.ndarray,
    future_valid: np.ndarray,
    event_pairs: np.ndarray,
    *,
    max_pairs_per_node: int,
    seed: int,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    node_count = context.shape[1]
    for node in range(node_count):
        pairs = event_pairs
        if len(pairs) > max_pairs_per_node:
            chosen = np.sort(rng.choice(len(pairs), max_pairs_per_node, replace=False))
            pairs = pairs[chosen]
        left, right = pairs[:, 0], pairs[:, 1]
        context_distance, context_overlap = _masked_rms(
            context[:, node], context_valid[:, node], left, right
        )
        future_distance, future_overlap = _masked_rms(
            future[:, node], future_valid[:, node], left, right
        )
        keep = (
            (context_overlap >= 0.80)
            & (future_overlap >= 0.80)
            & np.isfinite(context_distance)
            & np.isfinite(future_distance)
        )
        for pair_index in np.flatnonzero(keep):
            records.append(
                {
                    "node": int(node),
                    "i": int(left[pair_index]),
                    "j": int(right[pair_index]),
                    "context_distance": float(context_distance[pair_index]),
                    "future_distance": float(future_distance[pair_index]),
                    "context_overlap_fraction": float(context_overlap[pair_index]),
                    "future_overlap_fraction": float(future_overlap[pair_index]),
                }
            )
    return records


def _spearman(rows: list[dict[str, object]], left_name: str, right_name: str) -> float:
    if len(rows) < 2:
        return float("nan")
    left = pd.Series([float(row[left_name]) for row in rows])
    right = pd.Series([float(row[right_name]) for row in rows])
    return float(left.corr(right, method="spearman"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Controlled context-key alignment test for an Offset-only retrieval Bank."
    )
    parser.add_argument("--data", default="../data/METRLA_data/METR-LA.h5")
    parser.add_argument(
        "--bank",
        default="artifacts/case_bank_hn_offset_only_v1_transfer_hidden128_ffn2_b16_seed42_epoch32",
        help="Current Offset-only Bank whose key behavior is being diagnosed.",
    )
    parser.add_argument(
        "--reference-bank",
        default="artifacts/case_bank_hn_offset_decay_v2_transfer_hidden128_ffn2_b16_seed42",
        help="Trained OffsetDecay reference Bank.",
    )
    parser.add_argument(
        "--random-bank",
        default="artifacts/case_bank_hn_offset_decay_v2_transfer_hidden128_ffn2_b16_random_seed42",
        help="Matched-architecture random-encoder Bank.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/diagnostics/context_key_alignment_offset_only_epoch32/context_key_alignment.json",
    )
    parser.add_argument("--num-events", type=int, default=2500)
    parser.add_argument("--num-nodes", type=int, default=64)
    parser.add_argument("--max-pairs-per-node", type=int, default=1500)
    parser.add_argument("--quantile", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    banks = [Path(args.bank), Path(args.reference_bank), Path(args.random_bank)]
    manifests = [_load_manifest(bank) for bank in banks]
    _validate_aligned_banks(banks, manifests)
    manifest = manifests[0]
    event_count = int(manifest["num_events"])
    node_count = int(manifest["num_nodes"])
    rng = np.random.default_rng(args.seed)
    selected_events = np.sort(
        rng.choice(event_count, min(event_count, args.num_events), replace=False)
    )
    selected_nodes = np.sort(
        rng.choice(node_count, min(node_count, args.num_nodes), replace=False)
    )
    sample_ids_all = np.load(banks[0] / "sample_id.npy", mmap_mode="r")
    sample_ids = np.asarray(sample_ids_all[selected_events], dtype=np.int64)
    weekday = np.asarray(
        np.load(banks[0] / "weekday.npy", mmap_mode="r")[selected_events],
        dtype=np.int64,
    )
    slot = np.asarray(
        np.load(banks[0] / "slot.npy", mmap_mode="r")[selected_events],
        dtype=np.int64,
    )
    event_pairs = _calendar_pairs(weekday, slot)
    if event_pairs.size == 0:
        raise ValueError("selected events contain no calendar-compatible pairs")

    with pd.HDFStore(Path(args.data), "r") as store:
        raw_values = store.get("/df").to_numpy(dtype=np.float32)
    raw_observed = np.isfinite(raw_values) & (raw_values != 0)
    current_levels = np.asarray(
        np.load(banks[0] / "level_features.npy", mmap_mode="r")[selected_events],
        dtype=np.float32,
    )[:, selected_nodes]
    current_future = np.asarray(
        np.load(banks[0] / "future_values.npy", mmap_mode="r")[selected_events],
        dtype=np.float32,
    )[:, :, selected_nodes, 0]
    current_future_mask = np.asarray(
        np.load(banks[0] / "future_masks.npy", mmap_mode="r")[selected_events],
        dtype=bool,
    )[:, :, selected_nodes, 0]
    for other in banks[1:]:
        other_future = np.asarray(
            np.load(other / "future_values.npy", mmap_mode="r")[selected_events],
            dtype=np.float32,
        )[:, :, selected_nodes, 0]
        other_mask = np.asarray(
            np.load(other / "future_masks.npy", mmap_mode="r")[selected_events],
            dtype=bool,
        )[:, :, selected_nodes, 0]
        if not np.array_equal(current_future_mask, other_mask) or not np.allclose(
            current_future, other_future, rtol=0.0, atol=0.0
        ):
            raise ValueError(f"future payload is not aligned across Banks: {other}")

    context, context_valid, future, future_valid = _build_context_and_future_features(
        raw_values=raw_values,
        raw_observed=raw_observed,
        sample_ids=sample_ids,
        node_ids=selected_nodes,
        level_features=current_levels,
        future_values=current_future,
        future_masks=current_future_mask,
        manifest=manifest,
    )
    pair_records = _pair_records(
        context,
        context_valid,
        future,
        future_valid,
        event_pairs,
        max_pairs_per_node=args.max_pairs_per_node,
        seed=args.seed,
    )
    key_arrays = [
        np.asarray(
            np.load(bank / "node_keys.npy", mmap_mode="r")[selected_events],
            dtype=np.float32,
        )[:, selected_nodes]
        for bank in banks
    ]
    records = add_key_and_control_distances(
        pair_records,
        current_keys=key_arrays[0],
        reference_keys=key_arrays[1],
        random_keys=key_arrays[2],
        level_features=current_levels,
        weekday=weekday,
        slot=slot,
    )
    quadrants, thresholds = select_context_future_quadrants(
        records,
        quantile=args.quantile,
        require_calendar_compatible=True,
    )
    result = {
        "protocol": {
            "current_bank": str(banks[0]),
            "reference_bank": str(banks[1]),
            "random_bank": str(banks[2]),
            "encoder_fingerprints": [manifest_["encoder_fingerprint"] for manifest_ in manifests],
            "selected_events": int(len(selected_events)),
            "selected_nodes": int(len(selected_nodes)),
            "calendar_event_pairs": int(len(event_pairs)),
            "evaluated_node_pairs": int(len(records)),
            "seed": int(args.seed),
            "tail_quantile": float(args.quantile),
            "calendar_control": "same slot and cyclic weekday distance <= 1",
            "context_shape": "masked RMS over 24 patch means after the encoder's per-window normalization",
            "context_level": "mean absolute distance over Bank [mean,std,last,slope] features",
            "future_signature": "exact Offset-only teacher Y - 12-step forecast-context endpoint in scaler/model units",
            "key_distance": "1 - cosine similarity on the same event-node pair",
        },
        "thresholds": thresholds,
        "counts": {name: len(rows) for name, rows in quadrants.items()},
        "summary": summarize_quadrant_contrasts(quadrants),
        "rank_correlations": {
            prefix: {
                "context_shape": _spearman(records, f"{prefix}_key_distance", "context_distance"),
                "context_level": _spearman(records, f"{prefix}_key_distance", "level_distance"),
                "offset_only_future": _spearman(records, f"{prefix}_key_distance", "future_distance"),
            }
            for prefix in ("current", "reference", "random")
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
