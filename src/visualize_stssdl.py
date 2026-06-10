"""Visualization pipeline for ST-SSDL baseline experiments."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, TensorDataset

from .data import load_npz_splits, normalize_splits, prepare_x_y
from .metrics import horizon_metrics
from .models import STSSDLBaseline
from .preprocessing import _weekday_slot, build_history_anchor
from .train import build_model
from .utils import load_adj, project_root, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize ST-SSDL baseline behaviors.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing config.json and best_model.pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--missing-ratio", type=float, default=0.2)
    parser.add_argument("--pca-query-samples", type=int, default=400)
    parser.add_argument("--top-k-prototypes", type=int, default=7)
    parser.add_argument("--pattern-min-count", type=int, default=100)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["all"],
        choices=["all", "fig4", "fig5", "fig6", "fig7to9", "fig10", "usage"],
        help="Which visualization experiments to run.",
    )
    return parser.parse_args()


def _load_run_config(run_dir: Path) -> dict:
    with open(run_dir / "config.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_path(root: Path, path_like: str) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else root / path


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])


def _make_dirs(run_dir: Path) -> Dict[str, Path]:
    vis_root = run_dir / "visualization"
    cache_dir = vis_root / "cache"
    fig_dir = vis_root / "figures"
    table_dir = vis_root / "tables"
    for path in (cache_dir, fig_dir, table_dir):
        path.mkdir(parents=True, exist_ok=True)
    return {"root": vis_root, "cache": cache_dir, "fig": fig_dir, "table": table_dir}


def _load_raw_dataframe(data_dir: Path):
    import pandas as pd

    traffic_h5 = data_dir.parent / "METR-LA.h5"
    if not traffic_h5.exists():
        raise FileNotFoundError(f"Missing raw METR-LA file for anchor reconstruction: {traffic_h5}")
    return pd.read_hdf(traffic_h5)


def _compute_history_full(data_dir: Path, train_ratio: float) -> np.ndarray:
    df = _load_raw_dataframe(data_dir)
    return build_history_anchor(df, train_ratio=train_ratio, slots_per_day=288)


def _split_sample_positions(data: Dict[str, np.ndarray], split: str) -> np.ndarray:
    split_sizes = {
        "train": data["x_train"].shape[0],
        "val": data["x_val"].shape[0],
        "test": data["x_test"].shape[0],
    }
    offset = 0
    if split == "val":
        offset = split_sizes["train"]
    elif split == "test":
        offset = split_sizes["train"] + split_sizes["val"]
    return np.arange(offset, offset + split_sizes[split], dtype=np.int64)


def _build_eval_loader(data: Dict[str, np.ndarray], split: str, batch_size: int) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(data[f"x_{split}"]).float(),
        torch.from_numpy(data[f"y_{split}"]).float(),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)


def _future_anchor_from_positions(history_full: np.ndarray, sample_positions: np.ndarray, seq_len: int, horizon: int) -> np.ndarray:
    min_t = seq_len - 1
    anchors = []
    for sample_pos in sample_positions:
        current_t = min_t + int(sample_pos)
        anchors.append(history_full[current_t + 1 : current_t + 1 + horizon])
    return np.stack(anchors, axis=0).astype(np.float32)


def _compute_history_table(data_dir: Path, train_ratio: float) -> np.ndarray:
    df = _load_raw_dataframe(data_dir)
    values = df.values.astype(np.float32)
    train_end = int(values.shape[0] * train_ratio)
    slot_ids = _weekday_slot(df.index, 288)
    history = np.zeros((7 * 288, values.shape[1]), dtype=np.float32)
    counts = np.zeros_like(history)
    train_slots = slot_ids[:train_end]
    train_values = values[:train_end]
    for slot in range(history.shape[0]):
        mask = train_slots == slot
        if not np.any(mask):
            continue
        slot_values = train_values[mask]
        nonzero = slot_values != 0
        counts[slot] = nonzero.sum(axis=0)
        summed = np.where(nonzero, slot_values, 0.0).sum(axis=0)
        history[slot] = np.divide(summed, counts[slot], out=np.zeros_like(summed), where=counts[slot] > 0)
    sensor_mean = np.divide(train_values.sum(axis=0), np.maximum((train_values != 0).sum(axis=0), 1)).astype(np.float32)
    empty = counts == 0
    history[empty] = np.take(sensor_mean, np.where(empty)[1])
    return history


def collect_intermediates(
    model: STSSDLBaseline,
    loader,
    scaler,
    device: torch.device,
    history_future: np.ndarray,
    max_samples: int,
) -> Dict[str, np.ndarray]:
    model.eval()
    storage: Dict[str, List[np.ndarray]] = {
        "prediction": [],
        "target": [],
        "x_current": [],
        "x_current_norm": [],
        "x_anchor": [],
        "x_anchor_norm": [],
        "x_anchor_future": [],
        "x_cov": [],
        "y_cov": [],
        "q_c": [],
        "q_a": [],
        "p_c": [],
        "p_a": [],
        "mask_c": [],
        "mask_a": [],
        "latent_dis": [],
        "prototype_dis": [],
        "attention_c": [],
        "attention_a": [],
        "support": [],
    }
    total = 0
    cursor = 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            x, x_cov, x_his, y, y_cov = prepare_x_y(x_batch, y_batch)
            output = model(x, x_cov, x_his, y_cov, return_intermediates=True)
            batch_size = x.shape[0]
            keep = min(batch_size, max_samples - total)
            if keep <= 0:
                break

            prediction = scaler.inverse_transform(output["prediction"][:keep]).cpu().numpy()
            x_current = scaler.inverse_transform(x[:keep]).cpu().numpy()
            x_anchor = scaler.inverse_transform(x_his[:keep]).cpu().numpy()
            target = y[:keep].cpu().numpy()
            x_cov_np = x_cov[:keep].cpu().numpy()
            y_cov_np = y_cov[:keep].cpu().numpy()
            future_anchor = history_future[cursor : cursor + keep][..., None]

            storage["prediction"].append(prediction)
            storage["target"].append(target)
            storage["x_current"].append(x_current)
            storage["x_current_norm"].append(x[:keep].cpu().numpy())
            storage["x_anchor"].append(x_anchor)
            storage["x_anchor_norm"].append(x_his[:keep].cpu().numpy())
            storage["x_anchor_future"].append(future_anchor.astype(np.float32))
            storage["x_cov"].append(x_cov_np)
            storage["y_cov"].append(y_cov_np)
            storage["q_c"].append(output["query"][0, :keep].cpu().numpy())
            storage["q_a"].append(output["query"][1, :keep].cpu().numpy())
            storage["p_c"].append(output["pos"][0, :keep].cpu().numpy())
            storage["p_a"].append(output["pos"][1, :keep].cpu().numpy())
            storage["mask_c"].append(output["mask"][0, :keep, :, 0].cpu().numpy())
            storage["mask_a"].append(output["mask"][1, :keep, :, 0].cpu().numpy())
            storage["latent_dis"].append(output["latent_dis"][:keep].cpu().numpy())
            storage["prototype_dis"].append(output["prototype_dis"][:keep].cpu().numpy())
            storage["attention_c"].append(output["attention_c"][:keep].cpu().numpy())
            storage["attention_a"].append(output["attention_a"][:keep].cpu().numpy())
            storage["support"].append(output["clean_support"][:keep].cpu().numpy())
            total += keep
            cursor += keep
    result = {key: np.concatenate(value, axis=0) for key, value in storage.items()}
    result["prototypes"] = model.prototypes["prototypes"].detach().cpu().numpy()
    return result


def _save_cache(cache_dir: Path, arrays: Dict[str, np.ndarray]) -> Path:
    cache_path = cache_dir / "test_intermediates.npz"
    np.savez_compressed(cache_path, **arrays)
    return cache_path


def _sample_deviation(arrays: Dict[str, np.ndarray]) -> np.ndarray:
    return arrays["latent_dis"].mean(axis=1)


def _case_indices(sample_deviation: np.ndarray) -> Dict[str, int]:
    q33, q66 = np.quantile(sample_deviation, [0.33, 0.66])
    groups = {
        "low": np.where(sample_deviation <= q33)[0],
        "medium": np.where((sample_deviation > q33) & (sample_deviation <= q66))[0],
        "high": np.where(sample_deviation > q66)[0],
    }
    selected = {}
    for name, indices in groups.items():
        values = sample_deviation[indices]
        median = np.median(values)
        selected[name] = int(indices[np.argmin(np.abs(values - median))])
    return selected


def _node_index(arrays: Dict[str, np.ndarray], sample_idx: int) -> int:
    return int(np.argmax(arrays["prototype_dis"][sample_idx]))


def _plot_figure4(arrays: Dict[str, np.ndarray], fig_dir: Path, dpi: int) -> None:
    sample_deviation = _sample_deviation(arrays)
    selected = _case_indices(sample_deviation)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for col, (name, sample_idx) in enumerate(selected.items()):
        node_idx = _node_index(arrays, sample_idx)
        ax = axes[0, col]
        gt_24 = np.concatenate(
            [
                arrays["x_current"][sample_idx, :, node_idx, 0],
                arrays["target"][sample_idx, :, node_idx, 0],
            ]
        )
        anchor_24 = np.concatenate(
            [
                arrays["x_anchor"][sample_idx, :, node_idx, 0],
                arrays["x_anchor_future"][sample_idx, :, node_idx, 0],
            ]
        )
        pred_12 = arrays["prediction"][sample_idx, :, node_idx, 0]
        steps = np.arange(24)
        ax.axvspan(0, 11, color="0.9", alpha=0.8)
        ax.axvspan(12, 23, color="#d9edf7", alpha=0.8)
        ax.plot(steps, gt_24, color="#1f77b4", lw=2, label="Ground Truth")
        ax.plot(steps, anchor_24, color="#2ca02c", lw=2, ls="--", label="History Anchor")
        ax.plot(np.arange(12, 24), pred_12, color="#d62728", lw=2, label="Prediction")
        ax.set_title(f"({chr(97 + col)}) {name.title()} Deviation")
        ax.set_ylabel("Traffic Speed")
        ax.set_xlim(0, 23)

        ax2 = axes[1, col]
        q_c = arrays["q_c"][sample_idx, node_idx]
        q_a = arrays["q_a"][sample_idx, node_idx]
        p_c = arrays["p_c"][sample_idx, node_idx]
        p_a = arrays["p_a"][sample_idx, node_idx]
        points = np.stack([q_c, q_a, p_c, p_a], axis=0)
        pca = PCA(n_components=2).fit(points)
        proj = pca.transform(points)
        ax2.scatter(proj[0, 0], proj[0, 1], c="#1f77b4", marker="o", s=60, label="q_c")
        ax2.scatter(proj[1, 0], proj[1, 1], c="#17becf", marker="o", s=60, label="q_a")
        ax2.scatter(proj[2, 0], proj[2, 1], c="#d62728", marker="*", s=180, label="p_c")
        ax2.scatter(proj[3, 0], proj[3, 1], c="#ff9896", marker="*", s=180, label="p_a")
        ax2.plot([proj[0, 0], proj[2, 0]], [proj[0, 1], proj[2, 1]], color="#d62728", alpha=0.7)
        ax2.plot([proj[1, 0], proj[3, 0]], [proj[1, 1], proj[3, 1]], color="#17becf", alpha=0.7)
        ax2.set_title(
            f"mask_c={arrays['mask_c'][sample_idx, node_idx]}  "
            f"mask_a={arrays['mask_a'][sample_idx, node_idx]}\n"
            f"|qc-qa|={arrays['latent_dis'][sample_idx, node_idx]:.2f}  "
            f"|pc-pa|={arrays['prototype_dis'][sample_idx, node_idx]:.2f}"
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.savefig(fig_dir / "fig4_like_deviation_cases.png", dpi=dpi)
    plt.close(fig)


def _metrics_by_deviation(arrays: Dict[str, np.ndarray]) -> Tuple[Dict[str, Dict[str, float]], np.ndarray]:
    sample_deviation = _sample_deviation(arrays)
    q33, q66 = np.quantile(sample_deviation, [0.33, 0.66])
    groups = {
        "low": sample_deviation <= q33,
        "medium": (sample_deviation > q33) & (sample_deviation <= q66),
        "high": sample_deviation > q66,
    }
    metrics = {}
    for name, mask in groups.items():
        metrics[name] = horizon_metrics(
            torch.from_numpy(arrays["prediction"][mask]),
            torch.from_numpy(arrays["target"][mask]),
        )
    return metrics, sample_deviation


def _write_metrics_table(metrics: Dict[str, Dict[str, float]], table_dir: Path) -> None:
    header = [
        "group",
        "mae",
        "rmse",
        "mape",
        "mae_15min",
        "rmse_15min",
        "mape_15min",
        "mae_30min",
        "rmse_30min",
        "mape_30min",
        "mae_60min",
        "rmse_60min",
        "mape_60min",
    ]
    lines = [",".join(header)]
    for group, values in metrics.items():
        row = [group] + [f"{values.get(key, float('nan')):.6f}" for key in header[1:]]
        lines.append(",".join(row))
    with open(table_dir / "metrics_by_deviation.csv", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _plot_figure5(arrays: Dict[str, np.ndarray], fig_dir: Path, table_dir: Path, dpi: int, pca_query_samples: int, top_k: int) -> None:
    mask_c = arrays["mask_c"].reshape(-1)
    unique, counts = np.unique(mask_c, return_counts=True)
    order = np.argsort(counts)[::-1][:top_k]
    selected_proto_ids = unique[order]
    valid_mask = np.isin(mask_c, selected_proto_ids)
    q_c = arrays["q_c"].reshape(-1, arrays["q_c"].shape[-1])[valid_mask]
    q_labels = mask_c[valid_mask]
    if q_c.shape[0] > pca_query_samples:
        rng = np.random.default_rng(999)
        sample_idx = rng.choice(q_c.shape[0], size=pca_query_samples, replace=False)
        q_c = q_c[sample_idx]
        q_labels = q_labels[sample_idx]
    prototypes = arrays["prototypes"][selected_proto_ids]
    stacked = np.concatenate([q_c, prototypes], axis=0)
    coords = PCA(n_components=2).fit_transform(stacked)
    q_coords = coords[: q_c.shape[0]]
    p_coords = coords[q_c.shape[0] :]
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    cmap = plt.get_cmap("tab10")
    for color_idx, proto_id in enumerate(selected_proto_ids):
        q_mask = q_labels == proto_id
        ax.scatter(
            q_coords[q_mask, 0],
            q_coords[q_mask, 1],
            s=18,
            color=cmap(color_idx % 10),
            alpha=0.75,
            label=f"Queries -> P{int(proto_id)}",
        )
        ax.scatter(
            p_coords[color_idx, 0],
            p_coords[color_idx, 1],
            s=240,
            marker="*",
            edgecolors="black",
            linewidths=0.8,
            color=cmap(color_idx % 10),
            label=f"Prototype P{int(proto_id)}",
        )
    ax.set_title("Figure 5-like PCA of queries and prototypes")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best", fontsize=8, frameon=False)
    fig.savefig(fig_dir / "fig5_like_pca_queries_prototypes.png", dpi=dpi)
    plt.close(fig)

    same_proto_rate = float(np.mean(arrays["mask_c"] == arrays["mask_a"]))
    switch_rate = 1.0 - same_proto_rate
    mean_query_distance = float(arrays["latent_dis"].mean())
    mean_proto_distance = float(arrays["prototype_dis"].mean())
    lines = [
        "same_proto_rate,prototype_switch_rate,mean_query_distance,mean_proto_distance",
        f"{same_proto_rate:.6f},{switch_rate:.6f},{mean_query_distance:.6f},{mean_proto_distance:.6f}",
    ]
    with open(table_dir / "prototype_alignment_by_deviation.csv", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _prototype_usage(mask_c: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    unique, counts = np.unique(mask_c.reshape(-1), return_counts=True)
    probs = counts / counts.sum()
    entropy = float(-(probs * np.log(np.clip(probs, 1e-12, None))).sum())
    effective_proto_num = float(np.exp(entropy))
    return unique, counts, effective_proto_num


def _plot_prototype_usage(mask_c: np.ndarray, fig_dir: Path, table_dir: Path, dpi: int) -> None:
    unique, counts, effective_proto_num = _prototype_usage(mask_c)
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.bar(unique.astype(int), counts, color="#4c72b0")
    ax.set_title(f"Prototype usage (effective={effective_proto_num:.2f})")
    ax.set_xlabel("Prototype ID")
    ax.set_ylabel("Assignments")
    fig.savefig(fig_dir / "fig_prototype_usage_hist.png", dpi=dpi)
    plt.close(fig)
    lines = ["prototype_id,count", *[f"{int(pid)},{int(cnt)}" for pid, cnt in zip(unique, counts)]]
    lines.append(f"effective_proto_num,{effective_proto_num:.6f}")
    with open(table_dir / "prototype_usage.csv", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _classify_pattern(curve: np.ndarray) -> str:
    slope = curve[-1] - curve[0]
    amplitude = float(np.std(curve))
    if slope < -8:
        return "rapidly decreasing"
    if slope < -2:
        return "gradually decreasing"
    if slope > 2:
        return "increasing"
    if amplitude < 3:
        return "flat / small fluctuation"
    return "mixed"


def _plot_figure6(arrays: Dict[str, np.ndarray], fig_dir: Path, table_dir: Path, dpi: int, min_count: int) -> None:
    mask_c = arrays["mask_c"].reshape(-1)
    x_current = arrays["x_current"][..., 0]
    unique, counts = np.unique(mask_c, return_counts=True)
    selected = [(int(pid), int(cnt)) for pid, cnt in zip(unique, counts) if cnt >= min_count]
    selected.sort(key=lambda item: item[1], reverse=True)
    if not selected:
        return
    rows = math.ceil(len(selected) / 2)
    fig, axes = plt.subplots(rows, 2, figsize=(12, 3.2 * rows), constrained_layout=True)
    axes = np.atleast_1d(axes).reshape(rows, 2)
    summary_lines = ["prototype_id,count,pattern_label,start_value,end_value,std_mean"]
    flat_nodes = x_current.transpose(0, 2, 1).reshape(-1, x_current.shape[1])
    for ax in axes.flat:
        ax.axis("off")
    for idx, (proto_id, count) in enumerate(selected):
        ax = axes.flat[idx]
        proto_mask = mask_c == proto_id
        curves = flat_nodes[proto_mask]
        mean_curve = curves.mean(axis=0)
        std_curve = curves.std(axis=0)
        label = _classify_pattern(mean_curve)
        t = np.arange(mean_curve.shape[0])
        ax.axis("on")
        ax.plot(t, mean_curve, color="#2ca02c", lw=2)
        ax.fill_between(t, mean_curve - std_curve, mean_curve + std_curve, color="#2ca02c", alpha=0.2)
        ax.set_title(f"P{proto_id}: {label} (n={count})")
        ax.set_xlabel("Input Step")
        ax.set_ylabel("Traffic Speed")
        summary_lines.append(
            f"{proto_id},{count},{label},{mean_curve[0]:.6f},{mean_curve[-1]:.6f},{std_curve.mean():.6f}"
        )
    fig.savefig(fig_dir / "fig6_like_physical_prototype_patterns.png", dpi=dpi)
    plt.close(fig)
    with open(table_dir / "prototype_pattern_summary.csv", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")


def _plot_single_prediction_case(
    x_current: np.ndarray,
    anchor_24: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    output_path: Path,
    title: str,
    dpi: int,
) -> None:
    gt_24 = np.concatenate([x_current, target])
    fig, ax = plt.subplots(figsize=(7, 3.5), constrained_layout=True)
    ax.axvspan(0, 11, color="0.9", alpha=0.8)
    ax.axvspan(12, 23, color="#d9edf7", alpha=0.8)
    ax.plot(np.arange(24), anchor_24, color="#2ca02c", lw=2, ls="--", label="History Anchor")
    ax.plot(np.arange(24), gt_24, color="#1f77b4", lw=2, label="Ground Truth")
    ax.plot(np.arange(12, 24), prediction, color="#d62728", lw=2, label="Prediction")
    ax.set_title(title)
    ax.set_ylabel("Traffic Speed")
    ax.set_xlim(0, 23)
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _plot_figure7_9(arrays: Dict[str, np.ndarray], fig_dir: Path, dpi: int) -> None:
    selected = _case_indices(_sample_deviation(arrays))
    names = {
        "low": "fig7_like_low_deviation_stssdl.png",
        "medium": "fig8_like_medium_deviation_stssdl.png",
        "high": "fig9_like_high_deviation_stssdl.png",
    }
    for group, sample_idx in selected.items():
        node_idx = _node_index(arrays, sample_idx)
        anchor_24 = np.concatenate(
            [
                arrays["x_anchor"][sample_idx, :, node_idx, 0],
                arrays["x_anchor_future"][sample_idx, :, node_idx, 0],
            ]
        )
        _plot_single_prediction_case(
            arrays["x_current"][sample_idx, :, node_idx, 0],
            anchor_24,
            arrays["target"][sample_idx, :, node_idx, 0],
            arrays["prediction"][sample_idx, :, node_idx, 0],
            fig_dir / names[group],
            f"{group.title()} Deviation Prediction Comparison",
            dpi,
        )


def _plot_figure10(arrays: Dict[str, np.ndarray], model: STSSDLBaseline, device: torch.device, fig_dir: Path, dpi: int, missing_ratio: float) -> None:
    sample_idx = _case_indices(_sample_deviation(arrays))["medium"]
    x_current = torch.from_numpy(arrays["x_current_norm"][sample_idx : sample_idx + 1]).to(device)
    x_anchor = torch.from_numpy(arrays["x_anchor_norm"][sample_idx : sample_idx + 1]).to(device)
    x_cov = torch.from_numpy(arrays["x_cov"][sample_idx : sample_idx + 1]).to(device)
    y_cov = torch.from_numpy(arrays["y_cov"][sample_idx : sample_idx + 1]).to(device)
    target = arrays["target"][sample_idx, :, :, 0]
    missing_input = x_current.clone()
    rng = np.random.default_rng(999)
    total = missing_input.numel()
    miss_count = int(total * missing_ratio)
    flat_indices = rng.choice(total, size=miss_count, replace=False)
    missing_input.view(-1)[flat_indices] = 0.0
    with torch.no_grad():
        output = model(missing_input, x_cov, x_anchor, y_cov)
    pred_missing = output["prediction"][0, :, :, 0].cpu().numpy()
    node_idx = int(np.argmax(np.abs(pred_missing - target).mean(axis=0)))
    gt_24 = np.concatenate([arrays["x_current"][sample_idx, :, node_idx, 0], target[:, node_idx]])
    anchor_24 = np.concatenate(
        [
            arrays["x_anchor"][sample_idx, :, node_idx, 0],
            arrays["x_anchor_future"][sample_idx, :, node_idx, 0],
        ]
    )
    fig, ax = plt.subplots(figsize=(7, 3.5), constrained_layout=True)
    ax.axvspan(0, 11, color="0.9", alpha=0.8)
    ax.axvspan(12, 23, color="#d9edf7", alpha=0.8)
    ax.plot(np.arange(24), anchor_24, color="#2ca02c", lw=2, ls="--", label="History Anchor")
    ax.plot(np.arange(24), gt_24, color="#1f77b4", lw=2, label="Ground Truth")
    ax.plot(np.arange(12, 24), arrays["prediction"][sample_idx, :, node_idx, 0], color="#ff9896", lw=2, label="Prediction (clean)")
    ax.plot(np.arange(12, 24), pred_missing[:, node_idx], color="#d62728", lw=2, label="Prediction (missing)")
    ax.set_title("Partially Missing Values Prediction Comparison")
    ax.set_ylabel("Traffic Speed")
    ax.legend(frameon=False)
    fig.savefig(fig_dir / "fig10_like_missing_values_stssdl.png", dpi=dpi)
    plt.close(fig)


def _enabled(experiments: List[str], name: str) -> bool:
    return "all" in experiments or name in experiments


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    run_dir = Path(args.run_dir)
    root = project_root()
    config = _load_run_config(run_dir)
    config["model"] = "baseline"
    config["use_ssdl"] = config.get("use_ssdl", True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    data_dir = _resolve_path(root, config["processed_dir"])
    adj_path = _resolve_path(root, config["adj_path"])
    data, scaler = normalize_splits(load_npz_splits(data_dir))
    loader = _build_eval_loader(data, args.split, int(config["batch_size"]))
    supports_np, raw_adj_np = load_adj(adj_path, config["adj_type"])
    model_args = argparse.Namespace(**config)
    model = build_model(model_args, device, raw_adj_np.shape[0], supports_np, raw_adj_np)
    _load_checkpoint(model, run_dir / "best_model.pt", device)
    model.eval()

    sample_positions = _split_sample_positions(data, args.split)
    history_full = _compute_history_full(data_dir, float(config["train_ratio"]))
    history_future = _future_anchor_from_positions(
        history_full,
        sample_positions[: min(sample_positions.shape[0], args.max_samples)],
        int(config["seq_len"]),
        int(config["horizon"]),
    )
    dirs = _make_dirs(run_dir)
    arrays = collect_intermediates(
        model,
        loader,
        scaler,
        device,
        history_future,
        args.max_samples,
    )
    _save_cache(dirs["cache"], arrays)
    metrics, _ = _metrics_by_deviation(arrays)
    _write_metrics_table(metrics, dirs["table"])
    if _enabled(args.experiments, "fig4"):
        _plot_figure4(arrays, dirs["fig"], args.dpi)
    if _enabled(args.experiments, "fig5"):
        _plot_figure5(arrays, dirs["fig"], dirs["table"], args.dpi, args.pca_query_samples, args.top_k_prototypes)
    if _enabled(args.experiments, "fig6"):
        _plot_figure6(arrays, dirs["fig"], dirs["table"], args.dpi, args.pattern_min_count)
    if _enabled(args.experiments, "usage"):
        _plot_prototype_usage(arrays["mask_c"], dirs["fig"], dirs["table"], args.dpi)
    if _enabled(args.experiments, "fig7to9"):
        _plot_figure7_9(arrays, dirs["fig"], args.dpi)
    if _enabled(args.experiments, "fig10"):
        _plot_figure10(arrays, model, device, dirs["fig"], args.dpi, args.missing_ratio)


if __name__ == "__main__":
    main()
