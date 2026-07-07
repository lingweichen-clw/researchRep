"""Diagnostics for checking whether DCD-ST modules learned useful signals."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import build_loaders, load_npz_splits, normalize_splits, prepare_x_y
from src.metrics import horizon_metrics
from src.train import build_model
from src.utils import load_adj, set_seed


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _quantiles(values: np.ndarray, prefix: str) -> Dict[str, float]:
    if values.size == 0:
        return {f"{prefix}_q{q}": float("nan") for q in (1, 10, 25, 50, 75, 90, 99)}
    qs = np.percentile(values, [1, 10, 25, 50, 75, 90, 99])
    return {f"{prefix}_q{name}": float(value) for name, value in zip((1, 10, 25, 50, 75, 90, 99), qs)}


def _safe_corr(left: Iterable[float], right: Iterable[float]) -> float:
    x = np.asarray(list(left), dtype=np.float64)
    y = np.asarray(list(right), dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2 or float(x.std()) < 1e-12 or float(y.std()) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _tensor_norm_ratio(numerator: torch.Tensor, denominator: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    num = torch.linalg.vector_norm(numerator, dim=-1)
    den = torch.linalg.vector_norm(denominator, dim=-1).clamp_min(eps)
    return num / den


def _support_stats(support: torch.Tensor, static_support: torch.Tensor) -> Dict[str, float]:
    num_nodes = support.shape[-1]
    entropy = -(support * torch.log(support.clamp_min(1e-12))).sum(dim=-1) / math.log(num_nodes)
    row_max = support.max(dim=-1).values
    dyn_flat = support.reshape(support.shape[0], -1)
    static_flat = static_support.reshape(1, -1).expand_as(dyn_flat)
    cosine = torch.nn.functional.cosine_similarity(dyn_flat, static_flat, dim=-1)
    return {
        "support_entropy_mean": float(entropy.mean().detach().cpu()),
        "support_entropy_std": float(entropy.std(unbiased=False).detach().cpu()),
        "support_row_max_mean": float(row_max.mean().detach().cpu()),
        "support_static_cosine_mean": float(cosine.mean().detach().cpu()),
        "support_static_cosine_std": float(cosine.std(unbiased=False).detach().cpu()),
    }


def _update_node_sums(sums: dict, node_values: Dict[str, torch.Tensor]) -> None:
    for key, value in node_values.items():
        arr = value.detach().cpu().numpy()
        sums[f"{key}_sum"] += arr.sum(axis=0)
        sums[f"{key}_sq_sum"] += np.square(arr).sum(axis=0)
    sums["count"] += node_values["gate"].shape[0]


def _finalize_node_rows(sums: dict) -> list[dict]:
    count = max(int(sums["count"]), 1)
    rows = []
    num_nodes = sums["gate_sum"].shape[0]
    for node_idx in range(num_nodes):
        row = {"node": node_idx}
        for key in ("gate", "deviation", "error"):
            mean = sums[f"{key}_sum"][node_idx] / count
            var = max(sums[f"{key}_sq_sum"][node_idx] / count - mean * mean, 0.0)
            row[f"{key}_mean"] = float(mean)
            row[f"{key}_std"] = float(math.sqrt(var))
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def diagnose(args) -> dict:
    run_dir = _resolve(args.run_dir)
    config = _load_json(run_dir / "config.json")
    config["model"] = "dcd"
    if args.device is not None:
        config["device"] = args.device
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size

    set_seed(int(config.get("seed", 999)))
    device = torch.device(config["device"] if torch.cuda.is_available() or config["device"] == "cpu" else "cpu")
    data_dir = _resolve(args.processed_dir or config["processed_dir"])
    adj_path = _resolve(args.adj_path or config["adj_path"])
    output_dir = _resolve(args.output_dir) if args.output_dir else run_dir / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    data, scaler = normalize_splits(load_npz_splits(data_dir))
    loaders = build_loaders(data, batch_size=int(config["batch_size"]), num_workers=0)
    supports_np, raw_adj_np = load_adj(adj_path, config.get("adj_type", "symadj"))
    model = build_model(SimpleNamespace(**config), device, raw_adj_np.shape[0], supports_np, raw_adj_np)
    checkpoint = _load_checkpoint(run_dir / args.checkpoint, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    static_support = torch.as_tensor(supports_np[0], dtype=torch.float32, device=device)
    gates = []
    gate_logits = []
    deviations = []
    errors = []
    correction_ratios = []
    delta_ratios = []
    anchor_gap_ratios = []
    support_rows = []
    predictions = []
    labels = []
    cf_predictions: dict[float, list[torch.Tensor]] = {value: [] for value in args.gate_overrides}
    node_sums = {
        "gate_sum": np.zeros(raw_adj_np.shape[0], dtype=np.float64),
        "gate_sq_sum": np.zeros(raw_adj_np.shape[0], dtype=np.float64),
        "deviation_sum": np.zeros(raw_adj_np.shape[0], dtype=np.float64),
        "deviation_sq_sum": np.zeros(raw_adj_np.shape[0], dtype=np.float64),
        "error_sum": np.zeros(raw_adj_np.shape[0], dtype=np.float64),
        "error_sq_sum": np.zeros(raw_adj_np.shape[0], dtype=np.float64),
        "count": 0,
    }

    loader = loaders[args.split]
    with torch.no_grad():
        for batch_idx, (x_batch, y_batch) in enumerate(loader, start=1):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            x, x_cov, x_his, target, y_cov = prepare_x_y(x_batch, y_batch)
            output = model(x, x_cov, x_his, y_cov, return_intermediates=True)
            pred = scaler.inverse_transform(output["prediction"])
            predictions.append(pred.cpu())
            labels.append(target.cpu())

            gate = output["g_dev"]
            gate_node = gate.mean(dim=-1)
            gate_logit = torch.logit(gate.clamp(1e-6, 1 - 1e-6))
            deviation_node = output["z_dev"].abs().mean(dim=-1)
            error_node = torch.abs(pred - target).mean(dim=(1, 3))
            correction = gate * output["delta_h"]
            correction_ratios.append(_tensor_norm_ratio(correction, output["h_c"]).detach().cpu())
            delta_ratios.append(_tensor_norm_ratio(output["delta_h"], output["h_c"]).detach().cpu())
            anchor_gap_ratios.append(_tensor_norm_ratio(output["h_c"] - output["h_a"], output["h_c"]).detach().cpu())
            support_rows.append(_support_stats(output["clean_support"], static_support))

            gates.append(gate.detach().cpu())
            gate_logits.append(gate_logit.detach().cpu())
            deviations.append(deviation_node.detach().cpu())
            errors.append(error_node.detach().cpu())
            _update_node_sums(
                node_sums,
                {
                    "gate": gate_node,
                    "deviation": deviation_node,
                    "error": error_node,
                },
            )

            for value in args.gate_overrides:
                cf_output = model(x, x_cov, x_his, y_cov, gate_override=value)
                cf_predictions[value].append(scaler.inverse_transform(cf_output["prediction"]).cpu())

            if args.max_batches is not None and batch_idx >= args.max_batches:
                break

    pred_tensor = torch.cat(predictions, dim=0)
    label_tensor = torch.cat(labels, dim=0)
    gate_tensor = torch.cat(gates, dim=0)
    gate_logit_tensor = torch.cat(gate_logits, dim=0)
    deviation_tensor = torch.cat(deviations, dim=0)
    error_tensor = torch.cat(errors, dim=0)
    correction_ratio_tensor = torch.cat(correction_ratios, dim=0)
    delta_ratio_tensor = torch.cat(delta_ratios, dim=0)
    anchor_gap_ratio_tensor = torch.cat(anchor_gap_ratios, dim=0)

    gate_flat = gate_tensor.numpy().reshape(-1)
    gate_node_flat = gate_tensor.mean(dim=-1).numpy().reshape(-1)
    gate_logit_flat = gate_logit_tensor.numpy().reshape(-1)
    deviation_flat = deviation_tensor.numpy().reshape(-1)
    error_flat = error_tensor.numpy().reshape(-1)
    support_summary = {
        key: float(np.mean([row[key] for row in support_rows]))
        for key in support_rows[0]
    }
    summary = {
        "run_dir": str(run_dir),
        "split": args.split,
        "checkpoint": args.checkpoint,
        "num_samples": int(pred_tensor.shape[0]),
        "metrics": horizon_metrics(pred_tensor, label_tensor),
        "gate": {
            "mean": float(gate_flat.mean()),
            "std": float(gate_flat.std()),
            "min": float(gate_flat.min()),
            "max": float(gate_flat.max()),
            "frac_lt_0_1": float((gate_flat < 0.1).mean()),
            "frac_lt_0_2": float((gate_flat < 0.2).mean()),
            "frac_gt_0_8": float((gate_flat > 0.8).mean()),
            "frac_gt_0_9": float((gate_flat > 0.9).mean()),
            **_quantiles(gate_flat, "gate"),
        },
        "gate_logit": {
            "mean": float(gate_logit_flat.mean()),
            "std": float(gate_logit_flat.std()),
            **_quantiles(gate_logit_flat, "gate_logit"),
        },
        "deviation": {
            "mean_abs": float(deviation_flat.mean()),
            "std_abs": float(deviation_flat.std()),
            **_quantiles(deviation_flat, "deviation_abs"),
        },
        "representation": {
            "correction_to_hc_mean": float(correction_ratio_tensor.mean()),
            "correction_to_hc_std": float(correction_ratio_tensor.std(unbiased=False)),
            "delta_to_hc_mean": float(delta_ratio_tensor.mean()),
            "anchor_gap_to_hc_mean": float(anchor_gap_ratio_tensor.mean()),
        },
        "correlation": {
            "gate_vs_deviation_abs": _safe_corr(gate_node_flat, deviation_flat),
            "gate_vs_prediction_error": _safe_corr(gate_node_flat, error_flat),
            "deviation_abs_vs_prediction_error": _safe_corr(deviation_flat, error_flat),
        },
        "support": support_summary,
        "counterfactual": {},
    }
    for value, cf_parts in cf_predictions.items():
        cf_pred = torch.cat(cf_parts, dim=0)
        summary["counterfactual"][f"gate_{value:g}"] = horizon_metrics(cf_pred, label_tensor)

    node_rows = _finalize_node_rows(node_sums)
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    _write_csv(output_dir / "node_gate_stats.csv", node_rows)
    _write_csv(
        output_dir / "summary_metrics.csv",
        [
            {"group": group, "metric": metric, "value": value}
            for group, values in summary.items()
            if isinstance(values, dict)
            for metric, value in values.items()
            if not isinstance(value, dict)
        ],
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved diagnostics to {output_dir}")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose whether DCD-ST modules learned useful signals.")
    parser.add_argument("--run-dir", default="log/metrla_dcd_v1")
    parser.add_argument("--checkpoint", default="best_model.pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--adj-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--gate-overrides", type=float, nargs="*", default=[0.0, 0.5, 1.0])
    return parser.parse_args()


if __name__ == "__main__":
    diagnose(parse_args())
