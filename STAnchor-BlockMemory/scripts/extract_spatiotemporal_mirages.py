from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, Delaunay, QhullError
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


def summarize_overall_future_similarity(cluster_summary: list[dict]) -> dict:
    """Aggregate all-cluster future coherence with within-cluster pair weights."""
    if not cluster_summary:
        return {
            "cluster_count": 0,
            "eligible_event_node_points": 0,
            "weighted_within_cluster_cosine": float("nan"),
        }
    pair_weights = np.asarray(
        [
            max(0, int(row["size"])) * max(0, int(row["size"]) - 1)
            for row in cluster_summary
        ],
        dtype=np.float64,
    )
    values = np.asarray(
        [float(row["within_trend_cosine_mean"]) for row in cluster_summary],
        dtype=np.float64,
    )
    denominator = float(pair_weights.sum())
    return {
        "cluster_count": int(len(cluster_summary)),
        "eligible_event_node_points": int(sum(int(row["size"]) for row in cluster_summary)),
        "weighted_within_cluster_cosine": (
            float(np.sum(values * pair_weights) / denominator)
            if denominator > 0.0 else float("nan")
        ),
    }


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


def select_key_compact_pairs(
    rows: list[dict],
    count: int,
    max_pairs_per_node: int = 1,
) -> list[dict]:
    """Prefer the tightest key pairs within an already filtered case class.

    The scientific P8/P92 filters are applied before this function.  Ordering
    by key distance only changes which valid examples are displayed; it does
    not alter the class definition or any population statistic.
    """
    if not rows or count <= 0:
        return []
    order = sorted(
        range(len(rows)),
        key=lambda index: (
            float(rows[index]["key_distance"]),
            float(rows[index]["future_distance"]),
            -float(rows[index]["context_distance"]),
            int(rows[index]["node"]),
            int(rows[index]["i"]),
            int(rows[index]["j"]),
        ),
    )
    return select_diverse_pairs(
        [rows[index] for index in order],
        count,
        max_pairs_per_node=max_pairs_per_node,
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


def select_cluster_display_indices(
    keys: np.ndarray,
    labels: np.ndarray,
    *,
    cluster: int,
    max_points: int,
    projection_model: PCA | None = None,
) -> np.ndarray:
    """Select real cluster members nearest its displayed-space centroid."""
    if keys.ndim != 2 or labels.ndim != 1 or keys.shape[0] != labels.shape[0]:
        raise ValueError("keys must be [M,D] and labels must be [M]")
    if max_points <= 0:
        raise ValueError("max_points must be positive")
    indices = np.flatnonzero(labels == int(cluster))
    if indices.size <= max_points:
        return indices
    cluster_keys = keys[indices]
    if projection_model is None:
        projected = cluster_keys
    else:
        projected = projection_model.transform(cluster_keys)
    centroid = projected.mean(axis=0)
    distances = np.sum(np.square(projected - centroid), axis=1)
    order = np.lexsort((indices, distances))
    return indices[order[:max_points]]


def select_disjoint_key_cores(
    coordinates: np.ndarray,
    signatures: np.ndarray,
    labels: np.ndarray,
    *,
    points_per_core: int,
    max_cores: int,
    min_future_cosine: float,
    seed_candidates: int = 96,
    max_points_per_core: int | None = None,
    separation_margin: float = 1.06,
) -> list[dict]:
    """Select real, future-coherent PCA cores whose radial supports do not overlap.

    The selection only subsamples original PCA coordinates. It never moves a key or
    changes its cluster label for presentation. A core is a compact visual subset of
    one future-trend cluster, rather than a claim that an entire KMeans cluster is
    globally separable in two dimensions.
    """
    coordinates = np.asarray(coordinates, dtype=np.float32)
    signatures = np.asarray(signatures, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must be [M,2]")
    if signatures.ndim != 2 or signatures.shape[0] != coordinates.shape[0]:
        raise ValueError("signatures must be [M,T] aligned with coordinates")
    if labels.shape != (coordinates.shape[0],):
        raise ValueError("labels must be [M] aligned with coordinates")
    if points_per_core < 2 or max_cores <= 0 or seed_candidates <= 0:
        raise ValueError("points_per_core, max_cores, and seed_candidates must be positive")
    if max_points_per_core is None:
        max_points_per_core = max(points_per_core + 1, int(round(points_per_core * 1.75)))
    if max_points_per_core < points_per_core:
        raise ValueError("max_points_per_core must be at least points_per_core")
    if not -1.0 <= min_future_cosine <= 1.0:
        raise ValueError("min_future_cosine must be in [-1, 1]")
    if separation_margin < 1.0:
        raise ValueError("separation_margin must be at least 1")

    units = _unit_rows(signatures)
    best_per_cluster: list[dict] = []
    for cluster in sorted(int(value) for value in np.unique(labels) if value >= 0):
        member_indices = np.flatnonzero(labels == cluster)
        if member_indices.size < points_per_core:
            continue
        seed_positions = np.linspace(
            0,
            member_indices.size - 1,
            num=min(seed_candidates, member_indices.size),
            dtype=np.int64,
        )
        candidates: list[dict] = []
        member_coordinates = coordinates[member_indices]
        for seed_position in np.unique(seed_positions):
            squared_distance = np.sum(
                np.square(member_coordinates - member_coordinates[seed_position]), axis=1
            )
            order = np.lexsort((member_indices, squared_distance))
            # Let the local point density determine the displayed count. The kth
            # distance is used only to estimate a local support radius; this avoids
            # drawing every region as an identical fixed-size ball.
            kth_distance = float(np.sqrt(squared_distance[order[min(points_per_core - 1, order.size - 1)]]))
            nearest_count = min(8, order.size - 1)
            local_spacing = (
                float(np.sqrt(squared_distance[order[1:nearest_count + 1]]).mean())
                if nearest_count > 0 else kth_distance
            )
            support_distance = max(kth_distance * 1.35, local_spacing * 3.0, 1.0e-6)
            support_order = order[squared_distance[order] <= support_distance**2]
            if support_order.size < points_per_core:
                support_order = order[:points_per_core]
            indices = member_indices[support_order[:max_points_per_core]]
            core_units = units[indices]
            resultant = core_units.sum(axis=0)
            count = core_units.shape[0]
            within_cosine = (float(resultant @ resultant) - count) / (count * (count - 1))
            if within_cosine < min_future_cosine:
                continue
            core_coordinates = coordinates[indices]
            center = core_coordinates.mean(axis=0)
            radius = float(np.linalg.norm(core_coordinates - center, axis=1).max())
            candidates.append(
                {
                    "trend_cluster": cluster,
                    "indices": indices,
                    "center": center,
                    "radius": radius,
                    "within_future_cosine": float(within_cosine),
                }
            )
        if candidates:
            candidates.sort(
                key=lambda row: (
                    float(row["radius"]),
                    -float(row["within_future_cosine"]),
                    int(row["trend_cluster"]),
                    int(np.min(row["indices"])),
                )
            )
            best_per_cluster.append(candidates[0])

    best_per_cluster.sort(
        key=lambda row: (
            float(row["radius"]),
            -float(row["within_future_cosine"]),
            int(row["trend_cluster"]),
        )
    )
    selected: list[dict] = []
    for candidate in best_per_cluster:
        center = np.asarray(candidate["center"], dtype=np.float32)
        radius = float(candidate["radius"])
        separated = all(
            float(np.linalg.norm(center - np.asarray(existing["center"])))
            > separation_margin * (radius + float(existing["radius"]))
            for existing in selected
        )
        if separated:
            selected.append(candidate)
        if len(selected) >= max_cores:
            break
    return selected


def select_node_local_key_cores(
    keys: np.ndarray,
    signatures: np.ndarray,
    coordinates: np.ndarray,
    *,
    points_per_core: int,
    max_cores: int,
    min_future_cosine: float,
    seed_candidates: int = 96,
    min_centroid_cosine_distance: float = 0.20,
    candidate_nodes: list[int] | np.ndarray | None = None,
) -> list[dict]:
    """Select future-coherent natural neighbours from one node's 64-D key space."""
    keys = np.asarray(keys, dtype=np.float32)
    signatures = np.asarray(signatures, dtype=np.float32)
    coordinates = np.asarray(coordinates, dtype=np.float32)
    if keys.ndim != 3:
        raise ValueError("keys must be [E,N,D]")
    if signatures.shape[:2] != keys.shape[:2] or signatures.ndim != 3:
        raise ValueError("signatures must be [E,N,H] aligned with keys")
    if coordinates.ndim != 3 or coordinates.shape[:2] != keys.shape[:2]:
        raise ValueError("coordinates must be [E,N,P] aligned with keys")
    if points_per_core < 2 or max_cores <= 0 or seed_candidates <= 0:
        raise ValueError("points_per_core, max_cores, and seed_candidates must be positive")

    event_count, node_count = keys.shape[:2]
    if candidate_nodes is None:
        nodes_to_scan = range(node_count)
    else:
        nodes_to_scan = sorted(
            {
                int(node)
                for node in np.asarray(candidate_nodes, dtype=np.int64).reshape(-1)
                if 0 <= int(node) < node_count
            }
        )
        if not nodes_to_scan:
            raise ValueError("candidate_nodes must contain at least one valid node")
    units = _unit_rows(signatures.reshape(-1, signatures.shape[-1])).reshape(signatures.shape)
    candidates: list[dict] = []
    for node in nodes_to_scan:
        node_keys = keys[:, node]
        finite = np.isfinite(node_keys).all(axis=1)
        valid_events = np.flatnonzero(finite)
        if valid_events.size < points_per_core:
            continue
        normalized_keys = _unit_rows(node_keys[valid_events]).astype(
            np.float32, copy=False
        )
        seed_positions = np.unique(
            np.linspace(
                0,
                valid_events.size - 1,
                num=min(seed_candidates, valid_events.size),
                dtype=np.int64,
            )
        )
        # Batch cosine search keeps the same exact local-neighbour definition while
        # avoiding the per-query sklearn dispatch that dominates 5,000-event banks.
        similarities = normalized_keys[seed_positions] @ normalized_keys.T
        neighbor_positions = np.argpartition(
            -similarities,
            kth=points_per_core - 1,
            axis=1,
        )[:, :points_per_core]
        node_candidates: list[dict] = []
        for positions in neighbor_positions:
            events = valid_events[positions]
            core_units = units[events, node]
            count = core_units.shape[0]
            resultant = core_units.sum(axis=0)
            within_cosine = (float(resultant @ resultant) - count) / (count * (count - 1))
            if within_cosine < min_future_cosine:
                continue
            core_coordinates = coordinates[events, node]
            center = core_coordinates.mean(axis=0)
            center = center / max(float(np.linalg.norm(center)), 1.0e-8)
            radius = float(np.linalg.norm(core_coordinates - center, axis=1).max())
            node_candidates.append(
                {
                    "node": int(node),
                    "events": events,
                    "indices": events * node_count + node,
                    "center": center,
                    "radius": radius,
                    "within_future_cosine": float(within_cosine),
                }
            )
        if node_candidates:
            node_candidates.sort(
                key=lambda row: (
                    -float(row["within_future_cosine"]),
                    float(row["radius"]),
                    int(np.min(row["events"])),
                )
            )
            candidates.append(node_candidates[0])

    candidates.sort(
        key=lambda row: (
            -float(row["within_future_cosine"]),
            float(row["radius"]),
            int(row["node"]),
        )
    )
    selected: list[dict] = []
    for candidate in candidates:
        center = np.asarray(candidate["center"], dtype=np.float32)
        if all(
            1.0 - float(center @ np.asarray(existing["center"], dtype=np.float32))
            >= min_centroid_cosine_distance
            for existing in selected
        ):
            selected.append(candidate)
        if len(selected) >= max_cores:
            break
    return selected


def select_core_display_indices(
    cores: list[dict],
    *,
    keep_probability: float,
    seed: int,
) -> list[np.ndarray]:
    """Draw a fixed random display sample without changing each core's support."""
    if not 0.0 < keep_probability <= 1.0:
        raise ValueError("keep_probability must be in (0, 1]")
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for core in cores:
        indices = np.asarray(core["indices"], dtype=np.int64)
        keep = rng.random(indices.size) < keep_probability
        if indices.size and not bool(keep.any()):
            keep[int(rng.integers(indices.size))] = True
        selected.append(indices[keep].copy())
    return selected


def summarize_key_core_regions(
    cores: list[dict],
    display_indices: list[np.ndarray] | None = None,
) -> tuple[list[dict], dict]:
    """Build JSON/CSV-safe summaries for the selected visual key cores."""
    if display_indices is None:
        display_indices = [np.asarray(core["indices"], dtype=np.int64) for core in cores]
    if len(display_indices) != len(cores):
        raise ValueError("display_indices must align with cores")
    centers = [
        np.asarray(core["center"], dtype=np.float64)
        / max(float(np.linalg.norm(core["center"])), 1.0e-8)
        for core in cores
    ]
    rows: list[dict] = []
    for core_index, (core, shown) in enumerate(zip(cores, display_indices)):
        other_distances = [
            1.0 - float(centers[core_index] @ centers[other_index])
            for other_index in range(len(cores))
            if other_index != core_index
        ]
        row = {
            "region": f"C{core_index + 1}",
            "pool_points": int(len(core["indices"])),
            "shown_points": int(len(shown)),
            "within_future_cosine": float(core["within_future_cosine"]),
            "original_key_support_radius_l2": float(core["radius"]),
            "nearest_other_core_cosine_distance": (
                float(min(other_distances)) if other_distances else float("nan")
            ),
        }
        if "node" in core:
            row["node"] = int(core["node"])
        if "trend_cluster" in core:
            row["trend_cluster"] = int(core["trend_cluster"])
        rows.append(row)
    weights = np.asarray([row["pool_points"] for row in rows], dtype=np.float64)
    similarities = np.asarray([row["within_future_cosine"] for row in rows], dtype=np.float64)
    pairwise_distances = [
        1.0 - float(centers[left] @ centers[right])
        for left in range(len(centers))
        for right in range(left + 1, len(centers))
    ]
    overall = {
        "regions": int(len(rows)),
        "pool_event_node_points": int(weights.sum()),
        "displayed_event_node_points": int(sum(row["shown_points"] for row in rows)),
        "weighted_within_future_cosine": (
            float(np.average(similarities, weights=weights)) if weights.size else float("nan")
        ),
        "minimum_within_future_cosine": (
            float(similarities.min()) if similarities.size else float("nan")
        ),
        "minimum_inter_core_cosine_distance": (
            float(min(pairwise_distances)) if pairwise_distances else float("nan")
        ),
    }
    return rows, overall


def irregular_core_boundary(points: np.ndarray) -> np.ndarray:
    """Return a density-aware irregular boundary made from real PCA points.

    A Delaunay alpha-style hull retains only locally supported outer triangles,
    which gives a natural, mildly concave envelope rather than a template ellipse.
    """
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must be [M,2]")
    unique = np.unique(points, axis=0)
    if unique.shape[0] <= 2:
        return unique
    try:
        triangulation = Delaunay(unique)
    except QhullError:
        # Collinear cores have no two-dimensional hull; retain their endpoint chain.
        order = np.lexsort((unique[:, 1], unique[:, 0]))
        return unique[order[[0, -1]]]

    triangles: list[tuple[float, tuple[int, int, int]]] = []
    for simplex in triangulation.simplices:
        a, b, c = unique[simplex]
        side_a = float(np.linalg.norm(b - c))
        side_b = float(np.linalg.norm(a - c))
        side_c = float(np.linalg.norm(a - b))
        cross_value = (float(b[0] - a[0]) * float(c[1] - a[1])) - (
            float(b[1] - a[1]) * float(c[0] - a[0])
        )
        area = abs(cross_value) * 0.5
        if area <= 1.0e-10:
            continue
        circumradius = side_a * side_b * side_c / (4.0 * area)
        triangles.append((circumradius, tuple(int(value) for value in simplex)))
    if len(triangles) < 2:
        return unique[ConvexHull(unique).vertices]

    radii = np.asarray([row[0] for row in triangles], dtype=np.float64)
    # Keep a dense majority of local triangles but reject long bridges between lobes.
    cutoff = max(float(np.quantile(radii, 0.78)), float(np.median(radii) * 1.35))
    boundary_edges: dict[tuple[int, int], int] = {}
    for radius, simplex in triangles:
        if radius > cutoff:
            continue
        for left, right in ((simplex[0], simplex[1]), (simplex[1], simplex[2]), (simplex[2], simplex[0])):
            edge = tuple(sorted((left, right)))
            boundary_edges[edge] = boundary_edges.get(edge, 0) + 1
    edges = [edge for edge, count in boundary_edges.items() if count == 1]
    if len(edges) < 3:
        return unique[ConvexHull(unique).vertices]

    adjacency: dict[int, list[int]] = {}
    for left, right in edges:
        adjacency.setdefault(left, []).append(right)
        adjacency.setdefault(right, []).append(left)
    loops: list[list[int]] = []
    remaining = set(edges)
    while remaining:
        first_edge = min(remaining)
        start, current = first_edge
        previous: int | None = None
        loop = [start]
        while True:
            loop.append(current)
            remaining.discard(tuple(sorted((loop[-2], current))))
            candidates = [node for node in adjacency.get(current, []) if node != previous]
            if not candidates or candidates[0] == start:
                break
            previous, current = current, min(candidates)
            if len(loop) > len(edges) + 2:
                break
        if len(loop) >= 3 and loop[-1] == start:
            loop.pop()
        if len(loop) >= 3:
            loops.append(loop)
    if not loops:
        return unique[ConvexHull(unique).vertices]
    # A core can have tiny holes; display the largest outer boundary only.
    def polygon_area(loop: list[int]) -> float:
        polygon = unique[np.asarray(loop, dtype=np.int64)]
        return abs(float(np.sum(polygon[:, 0] * np.roll(polygon[:, 1], -1) - polygon[:, 1] * np.roll(polygon[:, 0], -1))) * 0.5)

    selected_loop = max(loops, key=polygon_area)
    return unique[np.asarray(selected_loop, dtype=np.int64)]


def fit_key_umap(
    keys: np.ndarray,
    cores: list[dict],
    *,
    seed: int,
    n_neighbors: int = 15,
    min_dist: float = 0.25,
    n_epochs: int = 800,
) -> np.ndarray:
    """Embed fixed original-space core members; UMAP never selects a member."""
    keys = np.asarray(keys, dtype=np.float32)
    if keys.ndim != 2 or keys.shape[0] < 2:
        raise ValueError("keys must be [M,D] with at least two rows")
    if n_neighbors < 2 or not 0.0 <= min_dist <= 1.0 or n_epochs <= 0:
        raise ValueError("invalid UMAP parameters")
    core_indices = np.unique(
        np.concatenate(
            [np.asarray(core["indices"], dtype=np.int64) for core in cores], axis=0
        ) if cores else np.empty(0, dtype=np.int64)
    )
    if core_indices.size < 3:
        raise ValueError("at least three fixed core keys are required")
    try:
        from umap import UMAP
    except ImportError as error:
        raise RuntimeError(
            "UMAP visualization requires 'umap-learn'; install the diagnostics extra"
        ) from error
    normalized = _unit_rows(keys[core_indices]).astype(np.float32, copy=False)
    embedded = UMAP(
        n_components=2,
        n_neighbors=min(int(n_neighbors), core_indices.size - 1),
        min_dist=float(min_dist),
        metric="cosine",
        random_state=seed,
        n_epochs=int(n_epochs),
        low_memory=True,
        n_jobs=1,
    ).fit_transform(normalized)
    coordinates = np.full((keys.shape[0], 2), np.nan, dtype=np.float32)
    coordinates[core_indices] = embedded.astype(np.float32, copy=False)
    return coordinates


def _population_cluster_plot(
    output_dir: Path,
    coordinates: np.ndarray,
    cores: list[dict],
    display_indices: list[np.ndarray],
) -> dict:
    """Plot fixed random samples as unconnected points from local key regions."""
    coordinates = np.asarray(coordinates, dtype=np.float32)
    visible_points = [
        coordinates[np.asarray(shown, dtype=np.int64)]
        for shown in display_indices
    ]
    all_points = np.concatenate(visible_points, axis=0)
    if all_points.ndim != 2 or all_points.shape[0] == 0 or not np.isfinite(all_points).all():
        raise ValueError("display_indices must select finite UMAP coordinates")
    lower = all_points.min(axis=0)
    upper = all_points.max(axis=0)
    span = np.maximum(upper - lower, 1.0e-3)
    padding = span * 0.055
    aspect_ratio = float(span[0] / span[1])
    figure_width = float(np.clip(6.2 * aspect_ratio, 6.6, 10.0))
    figure, axis = plt.subplots(
        figsize=(figure_width, 6.2),
        constrained_layout=True,
    )
    palette = plt.cm.tab10(np.linspace(0.0, 0.85, max(1, len(cores))))
    displayed = 0
    for color, shown in zip(
        palette, display_indices,
    ):
        points = coordinates[np.asarray(shown, dtype=np.int64)]
        axis.scatter(
            points[:, 0],
            points[:, 1],
            marker="o",
            s=24,
            color=color,
            alpha=0.82,
            edgecolors="none",
            linewidths=0.0,
            rasterized=True,
            zorder=2,
        )
        displayed += int(points.shape[0])
    axis.set_xlim(float(lower[0] - padding[0]), float(upper[0] + padding[0]))
    axis.set_ylim(float(lower[1] - padding[1]), float(upper[1] + padding[1]))
    axis.set_aspect("equal", adjustable="box")
    axis.set_title("Local Key Neighborhoods", fontsize=14, pad=10)
    axis.set_axis_off()
    figure.savefig(
        output_dir / "key_umap_local_regions.png",
        dpi=300,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.035,
    )
    plt.close(figure)
    return {
        "region_pool_points": int(sum(len(core["indices"]) for core in cores)),
        "cluster_points_displayed": displayed,
        "visual_style": "unconnected_filled_points",
        "figure_title": "Local Key Neighborhoods",
        "display_sampling": (
            "fixed random sample (seeded) from each original-space same-node key region; "
            "UMAP is visualization-only and does not select, move, or relabel points"
        ),
    }


def _cluster_evidence_plot(
    output_dir: Path,
    core_summary: list[dict],
    *,
    min_future_cosine: float,
) -> None:
    labels = [str(row["region"]) for row in core_summary]
    within = [float(row["within_future_cosine"]) for row in core_summary]
    radii = [float(row["original_key_support_radius_l2"]) for row in core_summary]
    shown = [int(row["shown_points"]) for row in core_summary]
    x = np.arange(len(core_summary))
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 4.7), constrained_layout=True)
    axes[0].bar(x, within, color="#0072B2", width=0.58, label="Within core")
    axes[0].axhline(
        min_future_cosine, color="#D55E00", linewidth=1.2, linestyle="--",
        label=f"Selection threshold ({min_future_cosine:.2f})",
    )
    axes[0].set_ylim(max(-0.05, min(min_future_cosine - 0.08, 0.70)), 1.02)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Mean pairwise future-trend cosine")
    axes[0].set_title("Future coherence of displayed cores")
    axes[0].grid(axis="y", alpha=0.18)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].bar(x - 0.17, radii, 0.34, color="#009E73", label="64-D key support radius")
    axes[1].bar(x + 0.17, shown, 0.34, color="#CC79A7", label="Displayed points")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Support radius / point count")
    axes[1].set_title("Compactness and display support")
    axes[1].grid(axis="y", alpha=0.18)
    axes[1].legend(frameon=False, fontsize=8)
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
        "context_different_future_similar": select_key_compact_pairs(group_b, count),
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
    parser.add_argument("--max-clusters", type=int, default=12)
    parser.add_argument("--future-trend-clusters", type=int, default=32)
    parser.add_argument("--future-overlap-threshold", type=float, default=0.8)
    parser.add_argument("--context-overlap-threshold", type=float, default=0.8)
    parser.add_argument("--max-pairs-per-node", type=int, default=2000)
    parser.add_argument("--cluster-pool-points", type=int, default=80)
    parser.add_argument("--cluster-display-probability", type=float, default=0.75)
    parser.add_argument("--core-min-future-cosine", type=float, default=0.70)
    parser.add_argument("--core-min-centroid-distance", type=float, default=0.12)
    parser.add_argument("--umap-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.25)
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
    cluster_count = min(
        args.future_trend_clusters,
        max(2, int(cluster_eligible.sum() // args.min_cluster_size)),
    )
    cluster_model = MiniBatchKMeans(
        n_clusters=cluster_count,
        random_state=args.seed,
        batch_size=16384,
        n_init=5,
        max_iter=100,
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
    normalized_key_space = _unit_rows(flat_keys).astype(np.float32, copy=False)
    display_cores = select_node_local_key_cores(
        keys,
        signatures,
        normalized_key_space.reshape(event_count, nodes, dimension),
        points_per_core=args.cluster_pool_points,
        max_cores=args.max_clusters,
        min_future_cosine=args.core_min_future_cosine,
        min_centroid_cosine_distance=args.core_min_centroid_distance,
    )
    umap_coordinates = fit_key_umap(
        flat_keys,
        display_cores,
        seed=args.seed,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
    )
    core_display_indices = select_core_display_indices(
        display_cores,
        keep_probability=args.cluster_display_probability,
        seed=args.seed,
    )
    core_summary, overall_future_similarity = summarize_key_core_regions(
        display_cores,
        core_display_indices,
    )
    overall_future_similarity["all_cluster_statistics"] = summarize_overall_future_similarity(
        cluster_summary
    )
    plot_summary = {
        kind: _case_plot(
            output_dir, kind, rows, contexts, context_masks, futures, future_masks,
            pca, keys, args.pairs_per_cluster,
        )
        for kind, rows in chosen.items()
    }
    plot_summary["local_region_umap"] = _population_cluster_plot(
        output_dir,
        umap_coordinates,
        display_cores,
        core_display_indices,
    )
    _cluster_evidence_plot(
        output_dir,
        core_summary,
        min_future_cosine=args.core_min_future_cosine,
    )
    pd.DataFrame(cluster_summary).to_csv(output_dir / "trend_cluster_summary.csv", index=False)
    pd.DataFrame(core_summary).to_csv(output_dir / "key_umap_local_regions.csv", index=False)
    payload = {
        "schema_version": 7,
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
            "cluster_pool_points": args.cluster_pool_points,
            "cluster_display_probability": args.cluster_display_probability,
            "core_min_future_cosine": args.core_min_future_cosine,
            "core_min_centroid_distance": args.core_min_centroid_distance,
            "diversity_constraint": (
                "no repeated event across selected pairs; at most 2 pairs per node when possible"
            ),
            "case_display_sampling": (
                "within each fixed quantile-defined class, rank rows by standardized distance "
                "to the class median of (context, future-trend, key) distances"
            ),
            "b_case_display_sampling": (
                "within the fixed B class, prioritize the smallest key distance, then future "
                "distance, with at most one displayed pair per node when possible"
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
            "requested_future_trend_clusters": int(args.future_trend_clusters),
            "retained_clusters": len(cluster_summary),
            "minimum_cluster_size": int(args.min_cluster_size),
            "eligible_event_node_points": int(cluster_eligible.sum()),
            "cluster_summary": cluster_summary,
        },
        "key_umap_local_display": {
            "definition": (
                "same-node local regions formed from fixed-size cosine nearest neighbours "
                "in the original 64-D learned-key space. Query future is used only for "
                "offline region validation and ranking. Distinct regions satisfy a fixed "
                "centroid cosine-distance threshold in the original key space"
            ),
            "coordinate_system": (
                "2-D UMAP of all retained region members using cosine distance; "
                "visualization only, not used for region selection or quantitative claims"
            ),
            "umap_parameters": {
                "n_neighbors": int(args.umap_neighbors),
                "min_dist": float(args.umap_min_dist),
                "metric": "cosine",
                "seed": int(args.seed),
            },
            "requested_max_regions": int(args.max_clusters),
            "pool_points_per_region": int(args.cluster_pool_points),
            "display_keep_probability": float(args.cluster_display_probability),
            "minimum_within_future_cosine": float(args.core_min_future_cosine),
            "minimum_inter_region_centroid_cosine_distance": float(
                args.core_min_centroid_distance
            ),
            "display_sampling": (
                "independent Bernoulli sampling with one shared probability and the fixed "
                "global seed; resulting display counts vary naturally by region"
            ),
            "regions": core_summary,
            "overall_future_similarity": overall_future_similarity,
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
