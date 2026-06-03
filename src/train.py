"""Training and smoke-test entry point for ST-SSDL baselines and variants."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from .data import build_loaders, load_npz_splits, normalize_splits, prepare_x_y
from .losses import LossWeights, compute_training_loss
from .metrics import horizon_metrics, masked_mae
from .models import RegionAwareSTSSDL, STSSDLBaseline
from .preprocessing import generate_metrla_splits
from .utils import count_parameters, load_adj, project_root, set_seed, to_torch_supports


def build_model(args, device: torch.device, num_nodes: int, supports_np, raw_adj_np):
    supports = to_torch_supports(supports_np, device)
    common_kwargs = dict(
        num_nodes=num_nodes,
        supports=supports,
        horizon=args.horizon,
        rnn_units=args.rnn_units,
        rnn_layers=args.rnn_layers,
        cheb_k=args.cheb_k,
        prototype_num=args.prototype_num,
        prototype_dim=args.prototype_dim,
        input_embedding_dim=args.input_embedding_dim,
        tod_embed_dim=args.tod_embed_dim,
        node_embedding_dim=args.node_embedding_dim,
        adaptive_embedding_dim=args.adaptive_embedding_dim,
        use_curriculum_learning=args.use_curriculum_learning,
        use_ssdl=args.use_ssdl,
    )
    if args.model == "baseline":
        return STSSDLBaseline(**common_kwargs).to(device)
    if args.model == "region":
        return RegionAwareSTSSDL(
            raw_adj=raw_adj_np,
            dataset_name=args.dataset_name,
            bcc_edge_threshold=args.bcc_edge_threshold,
            graph_static_weight=args.graph_static_weight,
            use_region_loss=args.use_region_loss,
            use_graph_denoise=args.use_graph_denoise,
            **common_kwargs,
        ).to(device)
    raise ValueError(f"Unsupported model: {args.model}")


def evaluate(model, loader, scaler, device: torch.device) -> Dict[str, float]:
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            x, x_cov, x_his, y, y_cov = prepare_x_y(x_batch, y_batch)
            output = model(x, x_cov, x_his, y_cov)
            preds.append(scaler.inverse_transform(output["prediction"]).cpu())
            labels.append(y.cpu())
    pred_tensor = torch.cat(preds, dim=0)
    label_tensor = torch.cat(labels, dim=0)
    return horizon_metrics(pred_tensor, label_tensor)


def _format_metrics(prefix: str, metrics: Dict[str, float]) -> str:
    items = [
        f"{prefix}_mae={metrics['mae']:.4f}",
        f"{prefix}_rmse={metrics['rmse']:.4f}",
        f"{prefix}_mape={metrics['mape'] * 100:.2f}%",
    ]
    for name in ("15min", "30min", "60min"):
        mae_key = f"mae_{name}"
        rmse_key = f"rmse_{name}"
        mape_key = f"mape_{name}"
        if mae_key in metrics:
            items.append(f"{prefix}_{name}_mae={metrics[mae_key]:.4f}")
            items.append(f"{prefix}_{name}_rmse={metrics[rmse_key]:.4f}")
            items.append(f"{prefix}_{name}_mape={metrics[mape_key] * 100:.2f}%")
    return " ".join(items)


def _make_experiment_dir(root: Path, args) -> Path:
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    run_prefix = "metrla_stssdl" if args.model == "baseline" else "metrla_region"
    run_name = args.run_name or datetime.now().strftime(f"{run_prefix}_%Y%m%d_%H%M%S")
    exp_dir = output_dir / run_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def _log(message: str, log_file: Path | None = None) -> None:
    print(message, flush=True)
    if log_file is not None:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")


def _parse_int_list(value: str) -> list[int]:
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _build_scheduler(args, optimizer):
    if args.lr_scheduler == "none":
        return None
    if args.lr_scheduler == "multistep":
        milestones = _parse_int_list(args.lr_milestones)
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=args.lr_decay_ratio,
        )
    raise ValueError(f"Unsupported lr_scheduler: {args.lr_scheduler}")


def train(args) -> None:
    root = project_root()
    exp_dir = _make_experiment_dir(root, args)
    log_file = exp_dir / "train.log"
    config_file = exp_dir / "config.json"
    best_model_file = exp_dir / "best_model.pt"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    data_dir = Path(args.processed_dir)
    if not data_dir.is_absolute():
        data_dir = root / data_dir
    traffic_h5 = Path(args.traffic_h5)
    if not traffic_h5.is_absolute():
        traffic_h5 = root / traffic_h5
    adj_path = Path(args.adj_path)
    if not adj_path.is_absolute():
        adj_path = root / adj_path

    if args.generate_data:
        summary = generate_metrla_splits(
            traffic_h5,
            data_dir,
            seq_len=args.seq_len,
            horizon=args.horizon,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            max_windows=args.max_windows,
        )
        _log(f"Generated splits: {summary}", log_file)

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    _log(f"Experiment directory: {exp_dir}", log_file)
    _log(f"Using device: {device}", log_file)
    data, scaler = normalize_splits(load_npz_splits(data_dir))
    loaders = build_loaders(data, batch_size=args.batch_size, num_workers=args.num_workers)
    supports_np, raw_adj_np = load_adj(adj_path, args.adj_type)
    model = build_model(args, device, raw_adj_np.shape[0], supports_np, raw_adj_np)
    _log(f"Trainable parameters: {count_parameters(model)}", log_file)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-3)
    scheduler = _build_scheduler(args, optimizer)
    weights = LossWeights(
        contrastive=args.lamb_c,
        deviation=args.lamb_d,
        region=args.lamb_region,
        graph_reg=args.lamb_graph,
        use_contrastive=args.use_contrastive_loss,
        use_deviation=args.use_deviation_loss,
    )
    batches_seen = 0
    best_val_mae = float("inf")
    wait = 0
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        running = {
            "total": [],
            "mae": [],
            "contrastive": [],
            "deviation": [],
            "region": [],
            "graph_reg": [],
        }
        for batch_idx, (x_batch, y_batch) in enumerate(loaders["train"], start=1):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            x, x_cov, x_his, y, y_cov = prepare_x_y(x_batch, y_batch)
            labels = scaler.transform(y)
            output = model(x, x_cov, x_his, y_cov, labels=labels, batches_seen=batches_seen)
            losses = compute_training_loss(output, y, scaler, weights)
            optimizer.zero_grad()
            losses["total"].backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            for loss_name in running:
                running[loss_name].append(float(losses[loss_name].detach().cpu()))
            batches_seen += 1
            if args.max_batches and batch_idx >= args.max_batches:
                break

        train_time = time.perf_counter() - epoch_start
        val_start = time.perf_counter()
        val_metrics = evaluate(model, loaders["val"], scaler, device)
        val_time = time.perf_counter() - val_start
        lr = optimizer.param_groups[0]["lr"]
        improved = val_metrics["mae"] < best_val_mae
        if improved:
            best_val_mae = val_metrics["mae"]
            wait = 0
            if args.save_best:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_val_mae": best_val_mae,
                        "args": vars(args),
                    },
                    best_model_file,
                )
        else:
            wait += 1

        _log(
            f"epoch={epoch}/{args.epochs} batches={len(running['total'])} "
            f"lr={lr:.6g} "
            f"train_loss={np.mean(running['total']):.4f} "
            f"train_mae={np.mean(running['mae']):.4f} "
            f"contrastive={np.mean(running['contrastive']):.4f} "
            f"deviation={np.mean(running['deviation']):.4f} "
            f"region={np.mean(running['region']):.4f} "
            f"graph_reg={np.mean(running['graph_reg']):.4f} "
            f"{_format_metrics('val', val_metrics)} "
            f"best_val_mae={best_val_mae:.4f} "
            f"improved={improved} "
            f"train_time={train_time:.2f}s val_time={val_time:.2f}s",
            log_file,
        )

        if args.patience > 0 and wait >= args.patience:
            _log(f"Early stopping at epoch={epoch}; best_val_mae={best_val_mae:.4f}", log_file)
            break
        if scheduler is not None:
            scheduler.step()

    if args.save_best and best_model_file.exists():
        try:
            checkpoint = torch.load(best_model_file, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(best_model_file, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        _log(
            f"Loaded best checkpoint from epoch={checkpoint['epoch']} "
            f"with val_mae={checkpoint['best_val_mae']:.4f}",
            log_file,
        )

    test_start = time.perf_counter()
    test_metrics = evaluate(model, loaders["test"], scaler, device)
    test_time = time.perf_counter() - test_start
    _log(f"{_format_metrics('test', test_metrics)} test_time={test_time:.2f}s", log_file)


def smoke_test(args) -> None:
    set_seed(args.seed)
    device = torch.device("cpu")
    batch_size, seq_len, horizon, num_nodes = 2, args.seq_len, args.horizon, 8
    x = torch.rand(batch_size, seq_len, num_nodes, 3)
    y = torch.rand(batch_size, horizon, num_nodes, 3)
    x[..., 1] = torch.linspace(0, 0.9, seq_len).view(1, seq_len, 1).expand(batch_size, -1, num_nodes)
    y[..., 1] = torch.linspace(0.1, 0.95, horizon).view(1, horizon, 1).expand(batch_size, -1, num_nodes)
    adj = np.eye(num_nodes, dtype=np.float32)
    for i in range(num_nodes - 1):
        adj[i, i + 1] = 1.0
        adj[i + 1, i] = 1.0
    supports_np = [adj / np.maximum(adj.sum(axis=1, keepdims=True), 1.0)]

    class _Scaler:
        def transform(self, value):
            return value

        def inverse_transform(self, value):
            return value

    model = build_model(args, device, num_nodes, supports_np, adj)
    x0, x_cov, x_his, target, y_cov = prepare_x_y(x, y)
    output = model(x0, x_cov, x_his, y_cov, labels=target, batches_seen=0)
    losses = compute_training_loss(
        output,
        target,
        _Scaler(),
        LossWeights(
            args.lamb_c,
            args.lamb_d,
            args.lamb_region,
            args.lamb_graph,
            args.use_contrastive_loss,
            args.use_deviation_loss,
        ),
    )
    losses["total"].backward()
    mae = masked_mae(output["prediction"], target)
    print(
        "smoke ok: "
        f"prediction={tuple(output['prediction'].shape)} "
        f"clean_support={tuple(output['clean_support'].shape)} "
        f"loss={float(losses['total'].detach()):.4f} "
        f"mae={float(mae.detach()):.4f}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train ST-SSDL baseline or region-aware variant.")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--model", default="baseline", choices=["baseline", "region"])
    parser.add_argument("--generate-data", action="store_true")
    parser.add_argument("--traffic-h5", default="data/METR-LA.h5")
    parser.add_argument("--processed-dir", default="data/METRLA")
    parser.add_argument("--adj-path", default="data/adj_mx.pkl")
    parser.add_argument("--dataset-name", default="METR-LA")
    parser.add_argument("--adj-type", default="symadj", choices=["symadj", "transition", "doubletransition", "identity"])
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--output-dir", default="log")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr-scheduler", default="none", choices=["none", "multistep"])
    parser.add_argument("--lr-milestones", default="40,70")
    parser.add_argument("--lr-decay-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--rnn-units", type=int, default=128)
    parser.add_argument("--rnn-layers", type=int, default=1)
    parser.add_argument("--cheb-k", type=int, default=3)
    parser.add_argument("--prototype-num", type=int, default=20)
    parser.add_argument("--prototype-dim", type=int, default=64)
    parser.add_argument("--input-embedding-dim", type=int, default=3)
    parser.add_argument("--tod-embed-dim", type=int, default=20)
    parser.add_argument("--node-embedding-dim", type=int, default=25)
    parser.add_argument("--adaptive-embedding-dim", type=int, default=0)
    parser.add_argument("--lamb-c", type=float, default=0.01)
    parser.add_argument("--lamb-d", type=float, default=1.0)
    parser.add_argument("--lamb-region", type=float, default=0.05)
    parser.add_argument("--lamb-graph", type=float, default=0.001)
    parser.add_argument("--graph-static-weight", type=float, default=0.15)
    parser.add_argument("--bcc-edge-threshold", type=float, default=None)
    parser.set_defaults(use_contrastive_loss=True)
    parser.add_argument("--use-contrastive-loss", dest="use_contrastive_loss", action="store_true")
    parser.add_argument("--no-contrastive-loss", dest="use_contrastive_loss", action="store_false")
    parser.set_defaults(use_deviation_loss=True)
    parser.add_argument("--use-deviation-loss", dest="use_deviation_loss", action="store_true")
    parser.add_argument("--no-deviation-loss", dest="use_deviation_loss", action="store_false")
    parser.set_defaults(use_ssdl=True)
    parser.add_argument("--use-ssdl", dest="use_ssdl", action="store_true")
    parser.add_argument("--no-ssdl", dest="use_ssdl", action="store_false")
    parser.set_defaults(use_region_loss=True)
    parser.add_argument("--use-region-loss", dest="use_region_loss", action="store_true")
    parser.add_argument("--no-region-loss", dest="use_region_loss", action="store_false")
    parser.set_defaults(use_graph_denoise=True)
    parser.add_argument("--use-graph-denoise", dest="use_graph_denoise", action="store_true")
    parser.add_argument("--no-graph-denoise", dest="use_graph_denoise", action="store_false")
    parser.set_defaults(use_curriculum_learning=True)
    parser.add_argument("--use-curriculum-learning", dest="use_curriculum_learning", action="store_true")
    parser.add_argument("--no-curriculum-learning", dest="use_curriculum_learning", action="store_false")
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--device", default="cuda:0")
    parser.set_defaults(save_best=True)
    parser.add_argument("--save-best", dest="save_best", action="store_true")
    parser.add_argument("--no-save-best", dest="save_best", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        smoke_test(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
