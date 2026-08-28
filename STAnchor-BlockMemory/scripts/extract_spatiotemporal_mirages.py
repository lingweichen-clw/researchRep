from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def quant(values: np.ndarray, percentile: float) -> float:
    return float(np.nanquantile(values, percentile))


def select_anchor_cluster(rows: list[dict], count: int) -> list[dict]:
    """Choose one anchor and several alternatives on the same node."""
    if not rows:
        return []
    best_anchor = None
    best_group = []
    for anchor_key in ("i", "j"):
        groups = {}
        for row in rows:
            anchor = int(row[anchor_key])
            groups.setdefault((int(row["node"]), anchor), []).append(row)
        for key, group in groups.items():
            if len(group) > len(best_group):
                best_anchor, best_group = key, group
    node, anchor = best_anchor
    selected = []
    used = {anchor}
    for row in best_group:
        left, right = int(row["i"]), int(row["j"])
        other = right if left == anchor else left if right == anchor else None
        if other is None or other in used:
            continue
        selected.append({**row, "i": anchor, "j": other})
        used.add(other)
        if len(selected) >= count:
            break
    return selected
def cluster_plot(output_dir: Path, kind: str, rows: list[dict], contexts: np.ndarray,
                 futures: np.ndarray, keys: np.ndarray, pairs_per_cluster: int) -> dict:
    rows = rows[:pairs_per_cluster]
    if not rows:
        return {"pairs_plotted": 0, "curves_plotted": 0, "key_points_plotted": 0}
    node = int(rows[0]["node"])
    anchor = int(rows[0]["i"])
    colors = plt.cm.viridis(np.linspace(0.18, 0.88, max(1, len(rows))))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    node_context = contexts[:, node, :]
    node_future = futures[:, node, :]
    context_location = float(np.nanmedian(node_context))
    context_scale = max(float(np.nanpercentile(node_context, 75) - np.nanpercentile(node_context, 25)), 1e-4)
    future_location = float(np.nanmedian(node_future))
    future_scale = max(float(np.nanpercentile(node_future, 75) - np.nanpercentile(node_future, 25)), 1e-4)
    anchor_context = (contexts[anchor, node] - context_location) / context_scale
    anchor_future = (futures[anchor, node] - future_location) / future_scale
    axes[0].plot(anchor_context, color="#222222", linewidth=2.8, label="anchor")
    axes[1].plot(anchor_future, color="#222222", linewidth=2.8, label="anchor")
    key_points = [keys[anchor, node]]
    for index, row in enumerate(rows):
        other = int(row["j"])
        axes[0].plot((contexts[other, node] - context_location) / context_scale, color=colors[index], alpha=0.82, linewidth=1.6, linestyle="--", label=f"sample {index + 1}")
        axes[1].plot((futures[other, node] - future_location) / future_scale, color=colors[index], alpha=0.82, linewidth=1.6, linestyle="--", label=f"sample {index + 1}")
        key_points.append(keys[other, node])
    axes[0].set_title(f"{kind}: context, node {node}")
    axes[1].set_title(f"{kind}: future, node {node}")
    for axis, length in ((axes[0], contexts.shape[-1]), (axes[1], futures.shape[-1])):
        axis.set_xticks([0, length // 2, length - 1])
        axis.grid(alpha=0.22)
        axis.legend(frameon=False, fontsize=8)
    axes[0].set_xlabel("Context step (5 min)"); axes[1].set_xlabel("Future step (5 min)")
    axes[0].set_ylabel("Node-wise robust value"); axes[1].set_ylabel("Node-wise robust value")
    points = np.asarray(key_points)
    points = PCA(n_components=2, random_state=42).fit_transform(points) if len(points) >= 2 else np.zeros((1, 2))
    axes[2].scatter(points[0, 0], points[0, 1], s=100, color="#222222", marker="X", label="anchor")
    if len(points) > 1:
        axes[2].scatter(points[1:, 0], points[1:, 1], c=colors[:len(points)-1], s=56, alpha=0.9, edgecolors="none", label="comparison")
    axes[2].set_title(f"{kind}: local key samples"); axes[2].set_xlabel("Key PC1"); axes[2].set_ylabel("Key PC2"); axes[2].grid(alpha=0.22); axes[2].legend(frameon=False, fontsize=8)
    fig.savefig(output_dir / f"{kind}_cluster.png", dpi=220); plt.close(fig)
    return {"pairs_plotted": len(rows), "curves_plotted": len(rows) + 1, "key_points_plotted": len(points), "node": node, "anchor_index": anchor}
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-events", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pairs-per-cluster", type=int, default=4)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with pd.HDFStore(args.data, "r") as store:
        values = store.get("/df").to_numpy(dtype=np.float32)
    bank = Path(args.bank)
    sample_ids = np.load(bank / "sample_id.npy").astype(np.int64)
    future = np.load(bank / "future_values.npy", mmap_mode="r").astype(np.float32)
    node_keys = np.load(bank / "node_keys.npy", mmap_mode="r").astype(np.float32)
    events, nodes, dimension = node_keys.shape
    horizon = future.shape[1]
    selected = np.arange(events) if events <= args.num_events else np.sort(rng.choice(events, args.num_events, replace=False))
    sample_ids = sample_ids[selected]
    valid = (sample_ids >= 12) & (sample_ids < len(values))
    selected, sample_ids = selected[valid], sample_ids[valid]
    contexts = np.stack([values[sample - 11:sample + 1].T for sample in sample_ids], axis=0)
    fill = np.nanmedian(values, axis=0)
    contexts = np.where(np.isfinite(contexts), contexts, fill[None, :, None])
    futures = np.asarray(future[selected, :, :, 0]).transpose(0, 2, 1)
    keys = np.asarray(node_keys[selected])
    records: list[dict] = []
    for node in range(nodes):
        context = contexts[:, node, :]
        target = futures[:, node, :]
        key = keys[:, node, :]
        context_z = (context - np.nanmedian(context, axis=0)) / (np.nanstd(context, axis=0) + 1e-4)
        future_z = (target - np.nanmedian(target, axis=0)) / (np.nanstd(target, axis=0) + 1e-4)
        neighbors = NearestNeighbors(n_neighbors=min(25, len(context_z))).fit(context_z)
        _, indices = neighbors.kneighbors(context_z)
        for left in range(len(context_z)):
            for rank in range(1, indices.shape[1]):
                right = int(indices[left, rank])
                if left >= right:
                    continue
                records.append({"node": node, "i": left, "j": right, "context_distance": float(np.linalg.norm(context_z[left] - context_z[right]) / np.sqrt(context_z.shape[-1])), "future_distance": float(np.linalg.norm(future_z[left] - future_z[right]) / np.sqrt(horizon)), "key_distance": float(np.linalg.norm(key[left] - key[right]) / np.sqrt(dimension))})
    context_distance = np.asarray([item["context_distance"] for item in records])
    future_distance = np.asarray([item["future_distance"] for item in records])
    key_distance = np.asarray([item["key_distance"] for item in records])
    thresholds = {"context_low": quant(context_distance, .08), "context_high": quant(context_distance, .92), "future_low": quant(future_distance, .08), "future_high": quant(future_distance, .92), "key_low": quant(key_distance, .08), "key_high": quant(key_distance, .92)}
    group_a = [item for item in records if item["context_distance"] <= thresholds["context_low"] and item["future_distance"] >= thresholds["future_high"] and item["key_distance"] >= thresholds["key_high"]]
    group_b = [item for item in records if item["context_distance"] >= thresholds["context_high"] and item["future_distance"] <= thresholds["future_low"] and item["key_distance"] <= thresholds["key_low"]]
    group_a.sort(key=lambda item: (-item["future_distance"], -item["key_distance"], item["node"], item["i"], item["j"]))
    group_b.sort(key=lambda item: (-item["context_distance"], -item["key_distance"], item["node"], item["i"], item["j"]))
    chosen = {
        "context_similar_future_different": select_anchor_cluster(group_a, args.pairs_per_cluster),
        "context_different_future_similar": select_anchor_cluster(group_b, args.pairs_per_cluster),
    }
    for rows in chosen.values():
        for item in rows:
            item.update({"sample_i": int(sample_ids[item["i"]]), "sample_j": int(sample_ids[item["j"]]), "node_id": int(item["node"])})
    summary = {kind: cluster_plot(output_dir, kind, rows, contexts, futures, keys, args.pairs_per_cluster) for kind, rows in chosen.items()}
    payload = {"schema_version": 2, "selection": {"context_similar_future_different": "context <= P8, future >= P92, key >= P92", "context_different_future_similar": "context >= P92, future <= P8, key <= P8", "manual_selection": False, "pairs_per_cluster": args.pairs_per_cluster, "diversity_constraint": "at most 2 pairs per node and no repeated event within each cluster"}, "num_events_used": int(len(selected)), "nodes": nodes, "retrieval_dim": dimension, "history_steps": 12, "horizon": horizon, "thresholds": thresholds, "cluster_plot_summary": summary, "cases": chosen}
    (output_dir / "mirage_cases.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([{"type": kind, **item} for kind, rows in chosen.items() for item in rows]).to_csv(output_dir / "mirage_cases.csv", index=False)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()




