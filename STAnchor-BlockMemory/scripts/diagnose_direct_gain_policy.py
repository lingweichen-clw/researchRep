from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, mean_absolute_error, r2_score, roc_auc_score
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stanchor.bank.storage import MemoryBank
from stanchor.config import load_config, resolve_project_path
from stanchor.diagnostics.direct_gain import build_direct_gain_features
from stanchor.engine.common import build_data_and_graph, load_checkpoint, load_pretrained_model
from stanchor.engine.target import (
    _validate_bank,
    build_downstream_model,
    checkpoint_bank_level_weight,
    checkpoint_candidate_protocol,
    checkpoint_downstream_mode,
    retrieve_for_downstream_mode,
)
from stanchor.losses.downstream import build_blend_target
from stanchor.retrieval.retriever import TwoStageRetriever
from stanchor.utils import resolve_device, save_json


def _collect(config, pretrained_path, downstream_path, bank_path, split, max_samples, seed):
    device = resolve_device(config.runtime.device)
    data, graph_cpu = build_data_and_graph(config)
    graph = graph_cpu.to(device)
    pretrained, _ = load_pretrained_model(config, pretrained_path, data.series.slots_per_day, device)
    checkpoint = load_checkpoint(downstream_path, device)
    mode = checkpoint_downstream_mode(checkpoint)
    config = config.__class__(
        **{
            **config.__dict__,
            "bank": config.bank.__class__(**{**config.bank.__dict__, "level_weight": checkpoint_bank_level_weight(checkpoint, config.bank.level_weight)}),
            "target": config.target.__class__(**{**config.target.__dict__, "downstream_mode": mode, "candidate_protocol": checkpoint_candidate_protocol(checkpoint)}),
        }
    )
    downstream = build_downstream_model(config, graph).to(device)
    downstream.load_state_dict(checkpoint["downstream_state_dict"], strict=True)
    pretrained.eval(); downstream.eval()
    loader = DataLoader(getattr(data, split), batch_size=config.target.batch_size, shuffle=False, num_workers=config.data.num_workers)
    rng = np.random.default_rng(seed + (0 if split == "train" else 1))
    fs, als, hs, bases, memories, targets = [], [], [], [], [], []
    seen = 0
    with MemoryBank(bank_path, expected_schema_version=(2 if pretrained.model_config.profile_dim > 0 else 1)) as bank:
        _validate_bank(bank, pretrained, graph_cpu, data.scaler.state_dict())
        if checkpoint.get("bank_manifest") != bank.manifest.to_dict():
            raise ValueError("policy collection Bank differs from downstream checkpoint Bank")
        retriever = TwoStageRetriever(bank, config.bank.event_top_r, config.bank.node_top_k, config.bank.level_weight, config.bank.level_temperature, config.bank.search_temperature, device)
        with torch.no_grad():
            for batch in loader:
                x = batch["x"].to(device); y_model = batch["y"].to(device); observed = batch["y_observed"].to(device)
                candidates, aggregation = retrieve_for_downstream_mode(mode, pretrained, retriever, bank, data, graph, batch, x, batch["x_observed"].to(device), device, candidate_protocol=config.target.candidate_protocol)
                output = downstream(x, candidates, aggregation)
                valid = observed.bool() & output.memory_valid.expand_as(observed)
                if not bool(valid.any()):
                    continue
                risk, alpha_valid = build_blend_target(output.base_prediction.detach(), output.memory_prediction.detach(), y_model, observed, output.memory_valid, config.target.blend_minimum_direction_norm)
                valid = valid & alpha_valid.expand_as(valid)
                features = build_direct_gain_features(output.confidence_features, candidates, aggregation, output.base_prediction.detach())
                base = data.scaler.inverse_transform_torch(output.base_prediction)
                memory = data.scaler.inverse_transform_torch(output.memory_prediction)
                target = data.scaler.inverse_transform_torch(y_model)
                flat_valid = valid.squeeze(-1)
                f = features[flat_valid].cpu().numpy(); a = risk.squeeze(-1)[flat_valid].cpu().numpy()
                helpful = ((memory - target).abs().mean(-1) < (base - target).abs().mean(-1))[flat_valid].cpu().numpy().astype(np.float32)
                take = min(max_samples - seen, f.shape[0])
                if take <= 0:
                    break
                idx = rng.choice(f.shape[0], size=take, replace=False) if take < f.shape[0] else np.arange(f.shape[0])
                fs.append(f[idx]); als.append(a[idx]); hs.append(helpful[idx])
                if split != "train":
                    bases.append(base[flat_valid].cpu().numpy()[idx]); memories.append(memory[flat_valid].cpu().numpy()[idx]); targets.append(target[flat_valid].cpu().numpy()[idx])
                seen += take
                if seen >= max_samples:
                    break
    result = {"features": np.concatenate(fs), "alpha": np.concatenate(als), "helpful": np.concatenate(hs)}
    if split != "train":
        result.update({"base": np.concatenate(bases), "memory": np.concatenate(memories), "target": np.concatenate(targets)})
    return result


def _mae(pred, target):
    return float(np.abs(pred - target).mean())


def main():
    parser = argparse.ArgumentParser(description="Leakage-safe direct gain/helpfulness policy validation")
    parser.add_argument("--config", required=True); parser.add_argument("--pretrained-checkpoint", required=True); parser.add_argument("--downstream-checkpoint", required=True); parser.add_argument("--bank", required=True); parser.add_argument("--output", required=True); parser.add_argument("--max-train-samples", type=int, default=200000); parser.add_argument("--max-val-samples", type=int, default=200000); parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(); config = load_config(args.config)
    train = _collect(config, args.pretrained_checkpoint, args.downstream_checkpoint, args.bank, "train", args.max_train_samples, args.seed)
    val = _collect(config, args.pretrained_checkpoint, args.downstream_checkpoint, args.bank, "val", args.max_val_samples, args.seed)
    clf = HistGradientBoostingClassifier(max_iter=160, max_leaf_nodes=31, learning_rate=0.06, random_state=args.seed).fit(train["features"], train["helpful"].astype(np.int8))
    reg = HistGradientBoostingRegressor(max_iter=160, max_leaf_nodes=31, learning_rate=0.06, loss="absolute_error", random_state=args.seed).fit(train["features"], train["alpha"])
    p = clf.predict_proba(val["features"])[:, 1]; alpha = np.clip(reg.predict(val["features"]), 0.0, 1.0)
    base, memory, target = val["base"], val["memory"], val["target"]
    pred_alpha = base + alpha[:, None] * (memory - base)
    pred_help = base + (p * alpha)[:, None] * (memory - base)
    output = {"schema_version": 1, "diagnostic": "line_b_direct_gain_policy", "train_positions": int(train["features"].shape[0]), "val_positions": int(val["features"].shape[0]), "future_information_boundary": "train future labels fit the policy; validation future is evaluation-only", "methods": {"base": {"mae": _mae(base, target)}, "memory": {"mae": _mae(memory, target)}, "direct_alpha": {"mae": _mae(pred_alpha, target)}, "direct_helpfulness_times_alpha": {"mae": _mae(pred_help, target)}}, "helpfulness": {"prevalence": float(val["helpful"].mean()), "auroc": float(roc_auc_score(val["helpful"], p)), "auprc": float(average_precision_score(val["helpful"], p)), "alpha_spearman": float(spearmanr(val["alpha"], alpha).statistic), "alpha_r2": float(r2_score(val["alpha"], alpha)), "alpha_mae": float(mean_absolute_error(val["alpha"], alpha))}}
    save_json(resolve_project_path(args.output), output); print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
