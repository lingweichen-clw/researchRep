from __future__ import annotations

"""Generate the paper-facing local UMAP figure without case-pair proposals."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    # Package import is used by tests and by callers importing ``scripts``.
    from scripts.extract_spatiotemporal_mirages import (
        _population_cluster_plot,
        build_trend_signature,
        fit_key_umap,
        select_core_display_indices,
        select_node_local_key_cores,
        summarize_key_core_regions,
    )
except ModuleNotFoundError:
    # Direct ``python scripts/plot_key_umap.py`` puts this directory on sys.path.
    from extract_spatiotemporal_mirages import (
        _population_cluster_plot,
        build_trend_signature,
        fit_key_umap,
        select_core_display_indices,
        select_node_local_key_cores,
        summarize_key_core_regions,
    )


def _future_trend_signatures(
    futures: np.ndarray,
    masks: np.ndarray,
) -> np.ndarray:
    """Build level-invariant future trends for offline local-region validation."""
    futures = np.asarray(futures, dtype=np.float32)
    masks = np.asarray(masks, dtype=bool)
    if futures.ndim != 3 or masks.shape != futures.shape:
        raise ValueError("futures and masks must be aligned [E,N,H] arrays")
    rows = futures.reshape(-1, futures.shape[-1])
    valid = masks.reshape(-1, masks.shape[-1]) & np.isfinite(rows)
    horizon = rows.shape[-1]
    positions = np.arange(horizon, dtype=np.int64)[None, :]
    previous = np.where(valid, positions, 0)
    previous = np.maximum.accumulate(previous, axis=1)
    following = np.where(valid, positions, horizon - 1)
    following = np.minimum.accumulate(following[:, ::-1], axis=1)[:, ::-1]
    row_ids = np.arange(rows.shape[0], dtype=np.int64)[:, None]
    left_values = rows[row_ids, previous]
    right_values = rows[row_ids, following]
    span = following - previous
    fraction = np.divide(
        positions - previous,
        span,
        out=np.zeros_like(rows, dtype=np.float32),
        where=span > 0,
    )
    filled = left_values + (right_values - left_values) * fraction
    no_valid = ~valid.any(axis=1)
    filled[no_valid] = 0.0
    centered = filled - filled[:, :1]
    scale = np.maximum(centered.std(axis=-1), 1.0e-6)
    return (centered / scale[:, None]).reshape(futures.shape).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render only the local learned-key UMAP case-study figure."
    )
    parser.add_argument("--bank", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-events", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-regions", type=int, default=12)
    parser.add_argument(
        "--candidate-nodes",
        default="91,84,64,22,117,12,74",
        help=(
            "comma-separated node shortlist retained by the prior full rule; "
            "use an empty value only when supplying a different shortlist"
        ),
    )
    parser.add_argument("--region-pool-points", type=int, default=80)
    parser.add_argument("--min-future-cosine", type=float, default=0.70)
    parser.add_argument("--min-centroid-distance", type=float, default=0.12)
    parser.add_argument("--display-probability", type=float, default=0.75)
    parser.add_argument("--umap-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.25)
    args = parser.parse_args()

    if args.num_events <= 0 or args.max_regions <= 0:
        raise ValueError("num-events and max-regions must be positive")
    bank_dir = Path(args.bank)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((bank_dir / "manifest.json").read_text(encoding="utf-8"))
    node_keys = np.load(bank_dir / "node_keys.npy", mmap_mode="r")
    future_values = np.load(bank_dir / "future_values.npy", mmap_mode="r")
    future_masks = np.load(bank_dir / "future_masks.npy", mmap_mode="r")
    event_count, node_count, key_dim = node_keys.shape
    rng = np.random.default_rng(args.seed)
    selected = (
        np.arange(event_count, dtype=np.int64)
        if event_count <= args.num_events
        else np.sort(rng.choice(event_count, args.num_events, replace=False))
    )
    keys = np.asarray(node_keys[selected], dtype=np.float32)
    standardized_futures = np.asarray(future_values[selected, :, :, 0], dtype=np.float32)
    masks = np.asarray(future_masks[selected, :, :, 0], dtype=bool).transpose(0, 2, 1)
    scaler_mean = np.asarray(manifest["scaler"]["mean"], dtype=np.float32).reshape(node_count)
    scaler_std = np.asarray(manifest["scaler"]["std"], dtype=np.float32).reshape(node_count)
    futures = (
        standardized_futures * scaler_std[None, None, :] + scaler_mean[None, None, :]
    ).transpose(0, 2, 1)
    signatures = _future_trend_signatures(futures, masks)
    key_norms = np.linalg.norm(keys, axis=-1, keepdims=True)
    normalized_keys = keys / np.maximum(key_norms, 1.0e-8)
    candidate_nodes = [
        int(token.strip())
        for token in str(args.candidate_nodes).split(",")
        if token.strip()
    ]
    if not candidate_nodes:
        raise ValueError("candidate-nodes must contain at least one node id")
    regions = select_node_local_key_cores(
        keys,
        signatures,
        normalized_keys,
        points_per_core=args.region_pool_points,
        max_cores=args.max_regions,
        min_future_cosine=args.min_future_cosine,
        min_centroid_cosine_distance=args.min_centroid_distance,
        candidate_nodes=candidate_nodes,
    )
    if not regions:
        raise RuntimeError("no local key regions met the requested validation rule")
    flat_keys = keys.reshape(-1, key_dim)
    coordinates = fit_key_umap(
        flat_keys,
        regions,
        seed=args.seed,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
    )
    display_indices = select_core_display_indices(
        regions,
        keep_probability=args.display_probability,
        seed=args.seed,
    )
    region_summary, overall = summarize_key_core_regions(regions, display_indices)
    plot_summary = _population_cluster_plot(
        output_dir,
        coordinates,
        regions,
        display_indices,
    )
    pd.DataFrame(region_summary).to_csv(
        output_dir / "key_umap_local_regions.csv", index=False
    )
    payload = {
        "selection_rule": {
            "space": "same-node original learned key cosine neighborhoods",
            "pool_points_per_region": int(args.region_pool_points),
            "minimum_within_future_trend_cosine": float(args.min_future_cosine),
            "minimum_inter_region_centroid_cosine_distance": float(
                args.min_centroid_distance
            ),
            "maximum_requested_regions": int(args.max_regions),
            "candidate_nodes": candidate_nodes,
        },
        "umap": {
            "metric": "cosine",
            "n_neighbors": int(args.umap_neighbors),
            "min_dist": float(args.umap_min_dist),
            "seed": int(args.seed),
            "visualization_only": True,
        },
        "display": {
            "keep_probability": float(args.display_probability),
            "sampling": "seeded independent Bernoulli sample within each fixed region",
            **plot_summary,
        },
        "data": {
            "events_sampled": int(selected.size),
            "nodes": int(node_count),
            "key_dimension": int(key_dim),
        },
        "regions": region_summary,
        "overall_future_similarity": overall,
    }
    (output_dir / "key_umap_local_regions.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
