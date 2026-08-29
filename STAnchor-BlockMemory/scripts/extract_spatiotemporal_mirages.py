from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def quant(values: np.ndarray, percentile: float) -> float:
    return float(np.nanquantile(values, percentile))


def masked_rms_distance(
    left: np.ndarray,
    right: np.ndarray,
    valid: np.ndarray,
) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    overlap = np.asarray(valid, dtype=bool) & np.isfinite(left) & np.isfinite(right)
    if not bool(overlap.any()):
        return float("nan")
    delta = left[overlap] - right[overlap]
    return float(np.sqrt(np.mean(np.square(delta))))


def interpolate_valid_1d(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(values)
    if values.ndim != 1 or valid.shape != values.shape:
        raise ValueError("values and valid must be aligned one-dimensional arrays")
    indices = np.flatnonzero(valid)
    if indices.size == 0:
        return np.zeros_like(values)
    if indices.size == 1:
        return np.full_like(values, values[indices[0]])
    return np.interp(np.arange(values.size), indices, values[indices]).astype(np.float32)


def build_trend_signature(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Return a level-invariant future trajectory while preserving direction."""
    filled = interpolate_valid_1d(values, valid)
    centered = filled - filled[0]
    scale = max(float(np.std(centered)), 1.0e-6)
    return (centered / scale).astype(np.float32)


def select_min_size_clusters(labels: np.ndarray, min_cluster_size: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    if min_cluster_size <= 0:
        raise ValueError("min_cluster_size must be positive")
    nonnegative = labels[labels >= 0]
    if nonnegative.size == 0:
        return np.zeros(labels.shape, dtype=bool)
    counts = np.bincount(nonnegative)
    return np.asarray(
        [label >= 0 and counts[label] >= min_cluster_size for label in labels],
        dtype=bool,
    )


def _unit_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1.0e-8)


def summarize_trend_clusters(
    signatures: np.ndarray,
    labels: np.ndarray,
    *,
    valid_fractions: np.ndarray | None = None,
    context_embeddings: np.ndarray | None = None,
    keys: np.ndarray | None = None,
) -> list[dict]:
    """Summarize cluster coherence without overlaying every future curve."""
    signatures = np.asarray(signatures, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    if signatures.ndim != 2 or labels.shape != (signatures.shape[0],):
        raise ValueError("signatures must be [M,D] and labels must be [M]")
    units = _unit_rows(signatures)
    cluster_ids = sorted(int(value) for value in np.unique(labels) if value >= 0)
    means = {
        cluster: _unit_rows(units[labels == cluster].mean(axis=0, keepdims=True))[0]
        for cluster in cluster_ids
    }
    summary: list[dict] = []
    for cluster in cluster_ids:
        selected = labels == cluster
        cluster_units = units[selected]
        size = int(cluster_units.shape[0])
        if size > 1:
            resultant = cluster_units.sum(axis=0)
            within = (float(resultant @ resultant) - size) / (size * (size - 1))
        else:
            within = 1.0
        other_means = [
            float(means[cluster] @ means[other])
            for other in cluster_ids
            if other != cluster
        ]
        row: dict[str, float | int] = {
            "cluster": cluster,
            "size": size,
            "within_trend_cosine_mean": float(within),
            "between_trend_cosine_mean": (
                float(np.mean(other_means)) if other_means else float("nan")
            ),
        }
        if valid_fractions is not None:
            row["future_valid_fraction_median"] = float(
                np.median(np.asarray(valid_fractions)[selected])
            )
        for name, values in (
            ("context_centroid_rms_median", context_embeddings),
            ("key_centroid_rms_median", keys),
        ):
            if values is None:
                continue
            cluster_values = np.asarray(values, dtype=np.float32)[selected]
            centroid = cluster_values.mean(axis=0)
            distance = np.sqrt(np.mean(np.square(cluster_values - centroid), axis=1))
            row[name] = float(np.median(distance))
        summary.append(row)
    return summary


def select_diverse_pairs(
    rows: list[dict],
    count: int,
    max_pairs_per_node: int = 2,
) -> list[dict]:
    """Select auditable pairs with distinct events and limited node repetition."""
    if not rows or count <= 0:
        return []
    selected: list[dict] = []
    used_events: set[int] = set()
    node_counts: dict[int, int] = {}
    for row in rows:
        left, right = int(row["i"]), int(row["j"])
        node = int(row["node"])
        if left in used_events or right in used_events:
            continue
        if node_counts.get(node, 0) >= max_pairs_per_node:
            continue
        selected.append({**row, "i": left, "j": right})
        used_events.update((left, right))
        node_counts[node] = node_counts.get(node, 0) + 1
        if len(selected) >= count:
            break
    if len(selected) < count:
        for row in rows:
            left, right = int(row["i"]), int(row["j"])
            if left in used_events or right in used_events:
                continue
            selected.append({**row, "i": left, "j": right})
            used_events.update((left, right))
            if len(selected) >= count:
                break
    return selected


def select_compact_pairs(
    rows: list[dict],
    count: int,
    max_pairs_per_node: int = 2,
) -> list[dict]:
    """Select representative rows near the centre of one case class.

    The quantile filters define the scientific case class first.  Within that
    fixed class, standardised (context, future-trend, key) distances are
    compared with the class medoid and the closest rows are considered first.
    This keeps the displayed examples visually compact without changing the
    full-population thresholds or metrics.
    """
    if not rows or count <= 0:
        return []
    values = np.asarray(
        [
            [row["context_distance"], row["future_distance"], row["key_distance"]]
            for row in rows
        ],
        dtype=np.float64,
    )
    centre = np.median(values, axis=0)
    q75, q25 = np.percentile(values, [75.0, 25.0], axis=0)
    scale = np.maximum(q75 - q25, 1.0e-8)
    compactness = np.linalg.norm((values - centre) / scale, axis=1)
    order = sorted(
        range(len(rows)),
        key=lambda index: (
            float(compactness[index]),
            int(rows[index]["node"]),
            int(rows[index]["i"]),
            int(rows[index]["j"]),
        ),
    )
    return select_diverse_pairs(
        [rows[index] for index in order], count, max_pairs_per_node=max_pairs_per_node
    )


def _robust_curve(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    filled = interpolate_valid_1d(values, valid)
    location = float(np.median(filled))
    # Speeds are measured in mph.  A physical-unit floor prevents nearly
    # constant/mostly-missing curves from producing artificial huge distances.
    scale = max(float(np.percentile(filled, 75) - np.percentile(filled, 25)), 1.0)
    return ((filled - location) / scale).astype(np.float32)


def _context_embedding(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Compress 288 steps into 24 hourly means for search and cluster diagnostics."""
    filled = interpolate_valid_1d(values, valid)
    robust = _robust_curve(filled, np.ones_like(valid, dtype=bool))
    if robust.size % 24 != 0:
        return robust
    return robust.reshape(24, robust.size // 24).mean(axis=1).astype(np.float32)


def _case_plot(
    output_dir: Path,
    kind: str,
    rows: list[dict],
    contexts: np.ndarray,
    context_masks: np.ndarray,
    futures: np.ndarray,
    future_masks: np.ndarray,
    key_pca: PCA,
    keys: np.ndarray,
    pairs_per_cluster: int,
) -> dict:
    rows = rows[:pairs_per_cluster]
    if not rows:
        return {"pairs_plotted": 0, "curves_plotted": 0, "key_points_plotted": 0}
    colors = plt.cm.viridis(np.linspace(0.18, 0.88, max(1, len(rows))))
    figure = plt.figure(figsize=(14.5, 3.65 * len(rows)))
    grid = figure.add_gridspec(
        len(rows), 3, width_ratios=(1.2, 1.0, 1.05), hspace=0.58, wspace=0.27
    )
    key_axis = figure.add_subplot(grid[:, 2])
    context_hours = np.linspace(-24.0, 0.0, contexts.shape[-1])
    future_minutes = np.arange(1, futures.shape[-1] + 1) * 5
    key_points: list[np.ndarray] = []
    for index, row in enumerate(rows):
        node = int(row["node"])
        left, right = int(row["i"]), int(row["j"])
        context_axis = figure.add_subplot(grid[index, 0])
        future_axis = figure.add_subplot(grid[index, 1])
        for event, color, style, label in (
            (left, colors[index], "-", f"event {row['sample_i']}"),
            (right, "#555555", "--", f"event {row['sample_j']}"),
        ):
            context_curve = np.where(
                context_masks[event, node], contexts[event, node], np.nan
            )
            future_curve = np.where(
                future_masks[event, node], futures[event, node], np.nan
            )
            context_axis.plot(
                context_hours, context_curve, color=color, linestyle=style,
                linewidth=1.35, label=label,
            )
            future_axis.plot(
                future_minutes, future_curve, color=color, linestyle=style,
                linewidth=2.15, marker="o", markersize=2.8, label=label,
            )
        context_axis.set_title(
            f"Pair {index + 1} | node {node} | context d={row['context_distance']:.3f}"
        )
        future_axis.set_title(
            f"Pair {index + 1} | trend d={row['future_distance']:.3f}"
        )
        context_axis.set_xlim(-24, 0)
        context_axis.set_xticks([-24, -12, 0])
        future_axis.set_xlim(5, 60)
        future_axis.set_xticks([5, 30, 60])
        for axis in (context_axis, future_axis):
            axis.grid(alpha=0.18)
            axis.legend(frameon=False, fontsize=7, loc="best")
            axis.set_ylabel("Traffic speed (mph)")
        if index == len(rows) - 1:
            context_axis.set_xlabel("Hours before query")
            future_axis.set_xlabel("Minutes after query")
        key_points.extend((keys[left, node], keys[right, node]))
    transformed = key_pca.transform(np.asarray(key_points, dtype=np.float32))
    for index in range(len(rows)):
        pair_points = transformed[index * 2 : index * 2 + 2]
        key_axis.plot(
            pair_points[:, 0], pair_points[:, 1], color=colors[index],
            linewidth=1.2, alpha=0.75,
        )
        key_axis.scatter(
            pair_points[0, 0], pair_points[0, 1], s=78, color=colors[index],
            marker="X", edgecolors="#333333", linewidths=0.7,
            label=f"pair {index + 1} anchor",
        )
        key_axis.scatter(
            pair_points[1, 0], pair_points[1, 1], s=48, color=colors[index],
            marker="o", edgecolors="white", linewidths=0.7,
            label=f"pair {index + 1} comparison",
        )
    key_axis.set_title(
        "A: separated keys" if kind.startswith("context_similar") else "B: nearby keys"
    )
    key_axis.set_xlabel("Key PC1")
    key_axis.set_ylabel("Key PC2")
    key_axis.grid(alpha=0.18)
    key_axis.legend(frameon=False, fontsize=7, ncol=2)
    figure.suptitle(
        "A. Similar 24-hour context, divergent future trend"
        if kind.startswith("context_similar")
        else "B. Different 24-hour context, similar future trend",
        fontsize=14,
    )
    figure.savefig(output_dir / f"{kind}_cluster.png", dpi=230, facecolor="white")
    plt.close(figure)
    return {
        "pairs_plotted": len(rows),
        "curves_plotted": len(rows) * 2,
        "key_points_plotted": len(transformed),
    }


def _fit_population_pca(keys: np.ndarray, sample_count: int, seed: int) -> tuple[PCA, np.ndarray]:
    rng = np.random.default_rng(seed)
    flat = keys.reshape(-1, keys.shape[-1])
    flat = flat[np.isfinite(flat).all(axis=1)]
    if flat.shape[0] > sample_count:
        flat = flat[rng.choice(flat.shape[0], sample_count, replace=False)]
    if flat.shape[0] < 2:
        raise ValueError("at least two finite key vectors are required for PCA")
    pca = PCA(n_components=2, random_state=seed).fit(flat)
    return pca, flat


def _population_cluster_plot(
    output_dir: Path,
    pca: PCA,
    background_keys: np.ndarray,
    keys_flat: np.ndarray,
    labels: np.ndarray,
    cluster_summary: list[dict],
    *,
    seed: int,
    points_per_cluster: int = 80,
) -> dict:
    background = pca.transform(background_keys)
    figure, axis = plt.subplots(figsize=(9.2, 6.6), constrained_layout=True)
    axis.scatter(
        background[:, 0], background[:, 1], s=3, color="#E4E8EB", alpha=0.55,
        linewidths=0, rasterized=True, label="Bank key population",
    )
    palette = plt.cm.tab10(np.linspace(0.0, 0.85, max(1, len(cluster_summary))))
    displayed = 0
    for color, row in zip(palette, cluster_summary):
        cluster = int(row["cluster"])
        indices = np.flatnonzero(labels == cluster)
        cluster_points = pca.transform(keys_flat[indices])
        centroid = cluster_points.mean(axis=0)
        if indices.size > points_per_cluster:
            # Keep the displayed representatives close to the cluster centre;
            # the full cluster remains represented by its reported size and
            # the all-population PCA background.
            order = np.argsort(np.sum(np.square(cluster_points - centroid), axis=1))
            indices = indices[order[:points_per_cluster]]
            points = pca.transform(keys_flat[indices])
        else:
            points = cluster_points
        axis.scatter(
            points[:, 0], points[:, 1], s=9, color=color, alpha=0.38,
            linewidths=0, rasterized=True,
            label=f"cluster {cluster + 1} (n={int(row['size']):,})",
        )
        axis.scatter(
            centroid[0], centroid[1], s=120, color=color, marker="X",
            edgecolors="white", linewidths=1.0,
        )
        axis.annotate(
            f"C{cluster + 1}", centroid, xytext=(5, 5), textcoords="offset points",
            fontsize=9, weight="bold", color=color,
        )
        displayed += int(indices.size)
    axis.set_title("Future-trend clusters in the learned key space")
    axis.set_xlabel("Key PC1")
    axis.set_ylabel("Key PC2")
    axis.grid(alpha=0.14)
    axis.legend(frameon=False, fontsize=8, ncol=2, loc="best")
    figure.savefig(output_dir / "key_pca_population_clusters.png", dpi=250, facecolor="white")
    plt.close(figure)
    return {
        "background_points": int(background.shape[0]),
        "cluster_points_displayed": displayed,
        "display_sampling": f"at most {points_per_cluster} deterministic points per cluster",
        "pca_fit_sample_count": int(background_keys.shape[0]),
    }


def _cluster_evidence_plot(output_dir: Path, summary: list[dict]) -> None:
    labels = [f"C{int(row['cluster']) + 1}\n(n={int(row['size']):,})" for row in summary]
    within = [float(row["within_trend_cosine_mean"]) for row in summary]
    between = [float(row["between_trend_cosine_mean"]) for row in summary]
    valid = [float(row["future_valid_fraction_median"]) for row in summary]
    x = np.arange(len(summary))
    width = 0.34
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 4.7), constrained_layout=True)
    axes[0].bar(x - width / 2, within, width, color="#0072B2", label="Within cluster")
    axes[0].bar(x + width / 2, between, width, color="#D55E00", label="Between clusters")
    axes[0].axhline(0.0, color="#777777", linewidth=0.8)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Mean trend cosine similarity")
    axes[0].set_title("Trend coherence: within vs. between clusters")
    axes[0].grid(axis="y", alpha=0.18)
    axes[0].legend(frameon=False)
    axes[1].bar(x, valid, color="#009E73", width=0.58)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Median observed future fraction")
    axes[1].set_title("Observation support of each cluster")
    axes[1].grid(axis="y", alpha=0.18)
    figure.savefig(output_dir / "trend_cluster_evidence.png", dpi=250, facecolor="white")
    plt.close(figure)


def _pair_record(
    node: int,
    left: int,
    right: int,
    contexts: np.ndarray,
    context_masks: np.ndarray,
    signatures: np.ndarray,
    future_masks: np.ndarray,
    keys: np.ndarray,
    future_overlap_threshold: float,
    context_overlap_threshold: float,
    normalized_context: np.ndarray | None = None,
) -> dict | None:
    overlap = future_masks[left, node] & future_masks[right, node]
    overlap_fraction = float(overlap.mean())
    if overlap_fraction < future_overlap_threshold:
        return None
    context_overlap = context_masks[left, node] & context_masks[right, node]
    context_overlap_fraction = float(context_overlap.mean())
    if context_overlap_fraction < context_overlap_threshold:
        return None
    if normalized_context is None:
        left_context = _robust_curve(contexts[left, node], context_masks[left, node])
        right_context = _robust_curve(contexts[right, node], context_masks[right, node])
    else:
        left_context = normalized_context[left]
        right_context = normalized_context[right]
    context_distance = masked_rms_distance(left_context, right_context, context_overlap)
    future_distance = masked_rms_distance(
        signatures[left, node], signatures[right, node], overlap
    )
    key_distance = float(
        np.linalg.norm(keys[left, node] - keys[right, node]) / np.sqrt(keys.shape[-1])
    )
    if not np.isfinite(context_distance + future_distance + key_distance):
        return None
    return {
        "node": int(node),
        "i": int(left),
        "j": int(right),
        "context_distance": context_distance,
        "future_distance": future_distance,
        "key_distance": key_distance,
        "future_overlap_fraction": overlap_fraction,
        "context_overlap_fraction": context_overlap_fraction,
    }


def _candidate_pair_records(
    contexts: np.ndarray,
    context_masks: np.ndarray,
    signatures: np.ndarray,
    future_masks: np.ndarray,
    keys: np.ndarray,
    future_overlap_threshold: float,
    context_overlap_threshold: float = 0.8,
    neighbors: int = 8,
    max_pairs_per_node: int = 2000,
) -> tuple[list[dict], list[dict]]:
    context_records: list[dict] = []
    future_records: list[dict] = []
    event_count, node_count = contexts.shape[:2]
    for node in range(node_count):
        context_search = np.stack(
            [_context_embedding(contexts[event, node], context_masks[event, node]) for event in range(event_count)]
        )
        normalized_context = np.stack(
            [_robust_curve(contexts[event, node], context_masks[event, node]) for event in range(event_count)]
        )
        future_search = signatures[:, node]
        valid_future = future_masks[:, node].mean(axis=1) >= future_overlap_threshold
        context_neighbors = NearestNeighbors(
            n_neighbors=min(neighbors, event_count), algorithm="auto"
        ).fit(context_search)
        _, context_indices = context_neighbors.kneighbors(context_search)
        future_indices = np.flatnonzero(valid_future)
        if future_indices.size >= 2:
            future_neighbors = NearestNeighbors(
                n_neighbors=min(neighbors, future_indices.size), algorithm="auto"
            ).fit(future_search[future_indices])
            _, local_future_indices = future_neighbors.kneighbors(future_search[future_indices])
        else:
            local_future_indices = np.empty((0, 0), dtype=np.int64)
        seen_context: set[tuple[int, int]] = set()
        for left in range(event_count):
            for right in context_indices[left, 1:]:
                pair = tuple(sorted((left, int(right))))
                seen_context.add(pair)
        context_pairs = sorted(seen_context)
        if len(context_pairs) > max_pairs_per_node:
            keep = np.linspace(0, len(context_pairs) - 1, max_pairs_per_node, dtype=np.int64)
            context_pairs = [context_pairs[int(index)] for index in np.unique(keep)]
        for pair in context_pairs:
            record = _pair_record(
                node, pair[0], pair[1], contexts, context_masks, signatures,
                future_masks, keys, future_overlap_threshold,
                context_overlap_threshold, normalized_context,
            )
            if record is not None:
                context_records.append(record)
        seen_future: set[tuple[int, int]] = set()
        for local_left, event_left in enumerate(future_indices):
            for local_right in local_future_indices[local_left, 1:]:
                event_right = int(future_indices[int(local_right)])
                pair = tuple(sorted((int(event_left), event_right)))
                seen_future.add(pair)
        future_pairs = sorted(seen_future)
        if len(future_pairs) > max_pairs_per_node:
            keep = np.linspace(0, len(future_pairs) - 1, max_pairs_per_node, dtype=np.int64)
            future_pairs = [future_pairs[int(index)] for index in np.unique(keep)]
        for pair in future_pairs:
            record = _pair_record(
                node, pair[0], pair[1], contexts, context_masks, signatures,
                future_masks, keys, future_overlap_threshold,
                context_overlap_threshold, normalized_context,
            )
            if record is not None:
                future_records.append(record)
    return context_records, future_records


def _select_mirage_cases(
    context_records: list[dict],
    future_records: list[dict],
    count: int,
) -> tuple[dict[str, list[dict]], dict]:
    context_values = np.asarray([row["context_distance"] for row in context_records])
    context_future_values = np.asarray([row["future_distance"] for row in context_records])
    context_key_values = np.asarray([row["key_distance"] for row in context_records])
    future_context_values = np.asarray([row["context_distance"] for row in future_records])
    future_values = np.asarray([row["future_distance"] for row in future_records])
    future_key_values = np.asarray([row["key_distance"] for row in future_records])
    thresholds = {
        "a_context_low": quant(context_values, 0.08),
        "a_future_high": quant(context_future_values, 0.92),
        "a_key_high": quant(context_key_values, 0.92),
        "b_context_high": quant(future_context_values, 0.92),
        "b_future_low": quant(future_values, 0.08),
        "b_key_low": quant(future_key_values, 0.08),
    }
    group_a = [
        row for row in context_records
        if row["context_distance"] <= thresholds["a_context_low"]
        and row["future_distance"] >= thresholds["a_future_high"]
        and row["key_distance"] >= thresholds["a_key_high"]
    ]
    group_b = [
        row for row in future_records
        if row["context_distance"] >= thresholds["b_context_high"]
        and row["future_distance"] <= thresholds["b_future_low"]
        and row["key_distance"] <= thresholds["b_key_low"]
    ]
    return {
        "context_similar_future_different": select_compact_pairs(group_a, count),
        "context_different_future_similar": select_compact_pairs(group_b, count),
    }, thresholds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-events", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pairs-per-cluster", type=int, default=3)
    parser.add_argument("--min-cluster-size", type=int, default=60)
    parser.add_argument("--max-clusters", type=int, default=6)
    parser.add_argument("--future-overlap-threshold", type=float, default=0.8)
    parser.add_argument("--context-overlap-threshold", type=float, default=0.8)
    parser.add_argument("--max-pairs-per-node", type=int, default=2000)
    parser.add_argument("--cluster-display-points", type=int, default=80)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bank = Path(args.bank)
    manifest = json.loads((bank / "manifest.json").read_text(encoding="utf-8"))
    history_steps = int(manifest["context_length"])
    with pd.HDFStore(args.data, "r") as store:
        values = store.get("/df").to_numpy(dtype=np.float32)
    observed = np.isfinite(values) & (values != 0)
    sample_ids_all = np.load(bank / "sample_id.npy").astype(np.int64)
    future_values = np.load(bank / "future_values.npy", mmap_mode="r")
    future_masks_all = np.load(bank / "future_masks.npy", mmap_mode="r")
    node_keys = np.load(bank / "node_keys.npy", mmap_mode="r")
    events, nodes, dimension = node_keys.shape
    selected = (
        np.arange(events)
        if events <= args.num_events
        else np.sort(rng.choice(events, args.num_events, replace=False))
    )
    sample_ids = sample_ids_all[selected]
    eligible = (sample_ids >= history_steps - 1) & (sample_ids < len(values))
    selected, sample_ids = selected[eligible], sample_ids[eligible]
    contexts = np.stack(
        [values[sample - history_steps + 1 : sample + 1].T for sample in sample_ids], axis=0
    )
    context_masks = np.stack(
        [observed[sample - history_steps + 1 : sample + 1].T for sample in sample_ids], axis=0
    )
    standardized_future = np.asarray(future_values[selected, :, :, 0], dtype=np.float32)
    future_masks = np.asarray(future_masks_all[selected, :, :, 0], dtype=bool).transpose(0, 2, 1)
    scaler_mean = np.asarray(manifest["scaler"]["mean"], dtype=np.float32).reshape(nodes)
    scaler_std = np.asarray(manifest["scaler"]["std"], dtype=np.float32).reshape(nodes)
    futures = (
        standardized_future * scaler_std[None, None, :] + scaler_mean[None, None, :]
    ).transpose(0, 2, 1)
    keys = np.asarray(node_keys[selected], dtype=np.float32)
    event_count = len(selected)
    signatures = np.zeros((event_count, nodes, futures.shape[-1]), dtype=np.float32)
    context_embeddings = np.zeros((event_count, nodes, 24), dtype=np.float32)
    for event in range(event_count):
        for node in range(nodes):
            signatures[event, node] = build_trend_signature(
                futures[event, node], future_masks[event, node]
            )
            context_embeddings[event, node] = _context_embedding(
                contexts[event, node], context_masks[event, node]
            )

    flat_signatures = signatures.reshape(-1, signatures.shape[-1])
    flat_future_valid = future_masks.reshape(-1, future_masks.shape[-1]).mean(axis=1)
    flat_context = context_embeddings.reshape(-1, context_embeddings.shape[-1])
    flat_keys = keys.reshape(-1, dimension)
    cluster_eligible = flat_future_valid >= args.future_overlap_threshold
    cluster_count = min(args.max_clusters, max(2, int(cluster_eligible.sum() // args.min_cluster_size)))
    cluster_model = MiniBatchKMeans(
        n_clusters=cluster_count,
        random_state=args.seed,
        batch_size=8192,
        n_init=5,
        max_iter=200,
    )
    eligible_labels = cluster_model.fit_predict(flat_signatures[cluster_eligible])
    labels = np.full(flat_signatures.shape[0], -1, dtype=np.int64)
    labels[cluster_eligible] = eligible_labels
    retained = select_min_size_clusters(labels, args.min_cluster_size)
    labels[~retained] = -1
    cluster_summary = summarize_trend_clusters(
        flat_signatures[labels >= 0],
        labels[labels >= 0],
        valid_fractions=flat_future_valid[labels >= 0],
        context_embeddings=flat_context[labels >= 0],
        keys=flat_keys[labels >= 0],
    )
    context_records, future_records = _candidate_pair_records(
        contexts, context_masks, signatures, future_masks, keys,
        args.future_overlap_threshold,
        args.context_overlap_threshold,
        max_pairs_per_node=args.max_pairs_per_node,
    )
    chosen, thresholds = _select_mirage_cases(
        context_records, future_records, args.pairs_per_cluster
    )
    for rows in chosen.values():
        for row in rows:
            row.update(
                {
                    "sample_i": int(sample_ids[row["i"]]),
                    "sample_j": int(sample_ids[row["j"]]),
                    "node_id": int(row["node"]),
                }
            )
    pca, pca_fit_keys = _fit_population_pca(keys, sample_count=24000, seed=args.seed)
    plot_summary = {
        kind: _case_plot(
            output_dir, kind, rows, contexts, context_masks, futures, future_masks,
            pca, keys, args.pairs_per_cluster,
        )
        for kind, rows in chosen.items()
    }
    plot_summary["population_cluster_pca"] = _population_cluster_plot(
        output_dir, pca, pca_fit_keys, flat_keys, labels, cluster_summary,
        seed=args.seed, points_per_cluster=args.cluster_display_points,
    )
    _cluster_evidence_plot(output_dir, cluster_summary)
    pd.DataFrame(cluster_summary).to_csv(output_dir / "trend_cluster_summary.csv", index=False)
    payload = {
        "schema_version": 4,
        "selection": {
            "context_similar_future_different": (
                "context-neighbor proposals: context <= P8, future trend >= P92, key >= P92"
            ),
            "context_different_future_similar": (
                "future-neighbor proposals: context >= P92, future trend <= P8, key <= P8"
            ),
            "manual_selection": False,
            "pairs_per_cluster": args.pairs_per_cluster,
            "future_overlap_threshold": args.future_overlap_threshold,
            "context_overlap_threshold": args.context_overlap_threshold,
            "max_pairs_per_node": args.max_pairs_per_node,
            "cluster_display_points": args.cluster_display_points,
            "diversity_constraint": (
                "no repeated event across selected pairs; at most 2 pairs per node when possible"
            ),
            "case_display_sampling": (
                "within each fixed quantile-defined class, rank rows by standardized distance "
                "to the class median of (context, future-trend, key) distances"
            ),
        },
        "data_contract": {
            "num_events_used": int(event_count),
            "nodes": int(nodes),
            "retrieval_dim": int(dimension),
            "history_steps": int(history_steps),
            "context_hours": float(history_steps * 5 / 60),
            "horizon_steps": int(futures.shape[-1]),
            "horizon_minutes": int(futures.shape[-1] * 5),
            "future_unit": "traffic speed (mph)",
            "missing_value_policy": "zero/unobserved values are masked, never treated as observations",
        },
        "trend_clustering": {
            "definition": (
                "future trajectory minus its first valid level, divided by its temporal standard deviation"
            ),
            "algorithm": "MiniBatchKMeans",
            "requested_max_clusters": int(args.max_clusters),
            "retained_clusters": len(cluster_summary),
            "minimum_cluster_size": int(args.min_cluster_size),
            "eligible_event_node_points": int(cluster_eligible.sum()),
            "cluster_summary": cluster_summary,
        },
        "pair_proposal_counts": {
            "context_neighbor_pairs": len(context_records),
            "future_neighbor_pairs": len(future_records),
        },
        "thresholds": thresholds,
        "plot_summary": plot_summary,
        "cases": chosen,
    }
    (output_dir / "mirage_cases.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        [{"type": kind, **row} for kind, rows in chosen.items() for row in rows]
    ).to_csv(output_dir / "mirage_cases.csv", index=False)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
