"""Aggregate context/future mirage pair distances without key-based selection bias."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.extract_spatiotemporal_mirages import (
        _candidate_pair_records,
        build_trend_signature,
    )
except ModuleNotFoundError:
    from extract_spatiotemporal_mirages import (  # type: ignore[no-redef]
        _candidate_pair_records,
        build_trend_signature,
    )


def _summary(rows: list[dict]) -> dict:
    result: dict[str, object] = {"count": len(rows)}
    for name in ("context_distance", "future_distance", "key_distance"):
        values = np.asarray([row[name] for row in rows], dtype=np.float64)
        result[name] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "p05": float(np.quantile(values, 0.05)),
            "p25": float(np.quantile(values, 0.25)),
            "p50": float(np.quantile(values, 0.50)),
            "p75": float(np.quantile(values, 0.75)),
            "p95": float(np.quantile(values, 0.95)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../data/METRLA_data/METR-LA.h5")
    parser.add_argument(
        "--bank",
        default="artifacts/case_bank_hn_offset_decay_v2_transfer_hidden128_ffn2_b16_seed42",
    )
    parser.add_argument(
        "--output",
        default="artifacts/diagnostics/mirage_pair_stratification_20260913.json",
    )
    parser.add_argument("--num-events", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--future-overlap-threshold", type=float, default=0.8)
    parser.add_argument("--context-overlap-threshold", type=float, default=0.8)
    parser.add_argument("--max-pairs-per-node", type=int, default=2000)
    args = parser.parse_args()

    bank = Path(args.bank)
    data_path = Path(args.data)
    manifest = json.loads((bank / "manifest.json").read_text(encoding="utf-8"))
    history_steps = int(manifest["context_length"])
    with pd.HDFStore(data_path, "r") as store:
        values = store.get("/df").to_numpy(dtype=np.float32)
    observed = np.isfinite(values) & (values != 0)
    sample_ids_all = np.load(bank / "sample_id.npy").astype(np.int64)
    future_values = np.load(bank / "future_values.npy", mmap_mode="r")
    future_masks_all = np.load(bank / "future_masks.npy", mmap_mode="r")
    node_keys = np.load(bank / "node_keys.npy", mmap_mode="r")
    events, nodes, dimension = node_keys.shape
    rng = np.random.default_rng(args.seed)
    selected = (
        np.arange(events)
        if events <= args.num_events
        else np.sort(rng.choice(events, args.num_events, replace=False))
    )
    sample_ids = sample_ids_all[selected]
    eligible = (sample_ids >= history_steps - 1) & (sample_ids < len(values))
    selected, sample_ids = selected[eligible], sample_ids[eligible]
    contexts = np.stack(
        [values[sample - history_steps + 1 : sample + 1].T for sample in sample_ids],
        axis=0,
    )
    context_masks = np.stack(
        [observed[sample - history_steps + 1 : sample + 1].T for sample in sample_ids],
        axis=0,
    )

    standardized_future = np.asarray(future_values[selected, :, :, 0], dtype=np.float32)
    future_masks = np.asarray(
        future_masks_all[selected, :, :, 0], dtype=bool
    ).transpose(0, 2, 1)
    scaler_mean = np.asarray(manifest["scaler"]["mean"], dtype=np.float32).reshape(nodes)
    scaler_std = np.asarray(manifest["scaler"]["std"], dtype=np.float32).reshape(nodes)
    futures = (
        standardized_future * scaler_std[None, None, :] + scaler_mean[None, None, :]
    ).transpose(0, 2, 1)
    keys = np.asarray(node_keys[selected], dtype=np.float32)
    event_count = len(selected)
    signatures = np.zeros((event_count, nodes, futures.shape[-1]), dtype=np.float32)
    for event in range(event_count):
        for node in range(nodes):
            signatures[event, node] = build_trend_signature(
                futures[event, node], future_masks[event, node]
            )

    context_records, future_records = _candidate_pair_records(
        contexts,
        context_masks,
        signatures,
        future_masks,
        keys,
        args.future_overlap_threshold,
        args.context_overlap_threshold,
        max_pairs_per_node=args.max_pairs_per_node,
    )
    context_distances = np.asarray(
        [row["context_distance"] for row in context_records], dtype=np.float64
    )
    context_future_distances = np.asarray(
        [row["future_distance"] for row in context_records], dtype=np.float64
    )
    context_key_distances = np.asarray(
        [row["key_distance"] for row in context_records], dtype=np.float64
    )
    future_context_distances = np.asarray(
        [row["context_distance"] for row in future_records], dtype=np.float64
    )
    future_distances = np.asarray(
        [row["future_distance"] for row in future_records], dtype=np.float64
    )
    future_key_distances = np.asarray(
        [row["key_distance"] for row in future_records], dtype=np.float64
    )
    thresholds = {
        "a_context_p08": float(np.quantile(context_distances, 0.08)),
        "a_future_p92": float(np.quantile(context_future_distances, 0.92)),
        "a_key_p92": float(np.quantile(context_key_distances, 0.92)),
        "b_context_p92": float(np.quantile(future_context_distances, 0.92)),
        "b_future_p08": float(np.quantile(future_distances, 0.08)),
        "b_key_p08": float(np.quantile(future_key_distances, 0.08)),
    }
    context_similar_future_different = [
        row
        for row in context_records
        if row["context_distance"] <= thresholds["a_context_p08"]
        and row["future_distance"] >= thresholds["a_future_p92"]
    ]
    context_different_future_similar = [
        row
        for row in future_records
        if row["context_distance"] >= thresholds["b_context_p92"]
        and row["future_distance"] <= thresholds["b_future_p08"]
    ]
    result = {
        "protocol": {
            "events": int(event_count),
            "nodes": int(nodes),
            "retrieval_dim": int(dimension),
            "candidate_context_records": len(context_records),
            "candidate_future_records": len(future_records),
            "future_overlap_threshold": args.future_overlap_threshold,
            "context_overlap_threshold": args.context_overlap_threshold,
            "context_distance": "masked RMS distance between robust history curves",
            "future_distance": "masked RMS distance between level-invariant future trend signatures",
            "key_distance": "L2 distance of 64-D keys divided by sqrt(64)",
        },
        "thresholds": thresholds,
        "context_future_only": {
            "context_similar_future_different": _summary(
                context_similar_future_different
            ),
            "context_different_future_similar": _summary(
                context_different_future_similar
            ),
        },
        "with_existing_key_filter": {
            "context_similar_future_different": _summary(
                [
                    row
                    for row in context_similar_future_different
                    if row["key_distance"] >= thresholds["a_key_p92"]
                ]
            ),
            "context_different_future_similar": _summary(
                [
                    row
                    for row in context_different_future_similar
                    if row["key_distance"] <= thresholds["b_key_p08"]
                ]
            ),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
