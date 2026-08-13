from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stanchor.config import load_config, resolve_project_path
from stanchor.diagnostics.cfdp_probe import (
    HorizonSpecificPoolingProbe,
    SharedPooledLinearProbe,
    SharedPooledMLPProbe,
    masked_profile_loss,
    profile_relation_metrics,
    weighted_metric_average,
)
from stanchor.engine.common import build_data_and_graph, load_pretrained_model
from stanchor.engine.pretrainer import build_validation_loader
from stanchor.losses.pretraining import build_future_relation_targets
from stanchor.retrieval.semantic_profile import build_cfdp_teacher
from stanchor.utils import resolve_device, save_json, set_seed


def _teacher_and_relations(config, batch, device, encoding):
    teacher, teacher_valid = build_cfdp_teacher(
        batch["y"].to(device),
        batch["y_observed"].to(device),
        batch["x"].to(device),
        batch["x_observed"].to(device),
        profile_size=config.model.profile_dim,
        scale_floor=config.pretrain.profile_scale_floor,
    )
    teacher = teacher.squeeze(-1).permute(0, 2, 1).contiguous()
    teacher_valid = teacher_valid.squeeze(-1).permute(0, 2, 1).contiguous()
    relation = build_future_relation_targets(
        future_model=batch["y"].to(device),
        context_statistics=encoding.statistics,
        future_observed=batch["y_observed"].to(device),
        context_start=batch["context_start"].to(device),
        future_end=batch["future_end"].to(device),
        teacher_temperature=config.pretrain.relation_teacher_temperature,
        relation_teacher_mode=config.pretrain.relation_teacher_mode,
        forecast_context=batch["x"].to(device),
        forecast_context_observed=batch["x_observed"].to(device),
        relation_distance_normalization=config.pretrain.relation_distance_normalization,
        future_increment_weight=config.pretrain.future_increment_weight,
    )
    return teacher, teacher_valid, relation.future_distance, relation.candidate_mask


def _pooled_hidden(encoding) -> torch.Tensor:
    hidden = encoding.hidden
    weights = encoding.retrieval.pooling_weights
    return (weights.unsqueeze(-1) * hidden).sum(dim=1)


def _step_heads(model, batch, config, device, optimizer=None):
    with torch.no_grad():
        encoding = model.encode_clean(
            batch["retrieval_x"].to(device),
            batch["retrieval_observed"].to(device),
            batch["retrieval_weekday"].to(device),
            batch["retrieval_slot"].to(device),
            model._probe_graph,
        )
        teacher, teacher_valid, od_distance, candidate_mask = _teacher_and_relations(
            config, batch, device, encoding
        )
        pooled = _pooled_hidden(encoding)
        hidden = encoding.hidden
        original = encoding.retrieval.profile_prediction
    predictions = {
        "original": original,
        "linear_pooled": model._probe_heads["linear_pooled"](pooled),
        "mlp_pooled": model._probe_heads["mlp_pooled"](pooled),
        "horizon_specific": model._probe_heads["horizon_specific"](hidden),
    }
    losses = {
        name: masked_profile_loss(prediction, teacher, teacher_valid)
        for name, prediction in predictions.items()
        if name != "original"
    }
    if optimizer is not None:
        loss = sum(losses.values()) / len(losses)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for head in model._probe_heads.values() for parameter in head.parameters()],
            max_norm=5.0,
        )
        optimizer.step()
    return predictions, teacher, teacher_valid, od_distance, candidate_mask, losses


@torch.no_grad()
def _evaluate(model, loader, config, device, max_batches):
    records = {
        name: []
        for name in (
            "original",
            "linear_pooled",
            "mlp_pooled",
            "horizon_specific",
            "teacher_profile_oracle",
        )
    }
    batches = 0
    points = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        predictions, teacher, valid, od_distance, candidate_mask, _ = _step_heads(
            model, batch, config, device, optimizer=None
        )
        predictions["teacher_profile_oracle"] = teacher
        for name, prediction in predictions.items():
            records[name].append(
                profile_relation_metrics(
                    prediction,
                    teacher,
                    valid,
                    od_distance,
                    candidate_mask,
                    top_k=min(5, max(1, od_distance.shape[1] - 1)),
                )
            )
        batches += 1
        points += int(valid.sum().item())
    if batches == 0:
        raise ValueError("probe evaluation processed no batches")
    return {
        name: weighted_metric_average(batch_records)
        for name, batch_records in records.items()
    }, {"batches": batches, "profile_points": points}


def run_probe_diagnostic(
    config,
    checkpoint_path: str | Path,
    output_path: str | Path,
    split: str = "val",
    epochs: int = 5,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
) -> dict:
    if split != "val":
        raise ValueError("CFDP probe diagnostic currently uses val evaluation")
    if config.model.profile_dim <= 0:
        raise ValueError("CFDP probe diagnostic requires a profile-enabled checkpoint")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    started = time.perf_counter()
    set_seed(config.runtime.seed)
    device = resolve_device(config.runtime.device)
    data, graph_cpu = build_data_and_graph(config)
    graph = graph_cpu.to(device)
    model, checkpoint = load_pretrained_model(
        config,
        resolve_project_path(checkpoint_path),
        data.series.slots_per_day,
        device,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model._probe_graph = graph
    model._probe_heads = torch.nn.ModuleDict(
        {
            "linear_pooled": SharedPooledLinearProbe(
                config.model.hidden_dim, config.model.profile_dim
            ).to(device),
            "mlp_pooled": SharedPooledMLPProbe(
                config.model.hidden_dim, config.model.profile_dim
            ).to(device),
            "horizon_specific": HorizonSpecificPoolingProbe(
                config.model.hidden_dim, config.model.profile_dim
            ).to(device),
        }
    )
    optimizer = torch.optim.AdamW(
        model._probe_heads.parameters(),
        lr=config.pretrain.learning_rate,
        weight_decay=config.pretrain.weight_decay,
    )
    parameter_counts = {
        name: sum(parameter.numel() for parameter in head.parameters())
        for name, head in model._probe_heads.items()
    }
    train_loader = DataLoader(
        data.train,
        batch_size=config.pretrain.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        drop_last=False,
    )
    val_loader = build_validation_loader(
        data.val,
        config.pretrain.batch_size,
        config.data.num_workers,
        config.runtime.seed,
    )
    train_history = []
    for epoch in range(1, epochs + 1):
        totals = {name: 0.0 for name in ("linear_pooled", "mlp_pooled", "horizon_specific")}
        count = 0
        for batch_index, batch in enumerate(train_loader):
            if max_train_batches is not None and batch_index >= max_train_batches:
                break
            _, _, _, _, _, losses = _step_heads(
                model, batch, config, device, optimizer=optimizer
            )
            for name, value in losses.items():
                totals[name] += float(value.detach())
            count += 1
        if count == 0:
            raise ValueError("probe training processed no batches")
        train_history.append(
            {"epoch": epoch, **{name: value / count for name, value in totals.items()}}
        )
    validation, counts = _evaluate(
        model, val_loader, config, device, max_val_batches
    )
    result = {
        "schema_version": 1,
        "diagnostic": "frozen_cfdp_probes",
        "split": split,
        "checkpoint": str(resolve_project_path(checkpoint_path)),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "context_length": config.data.encoder_context_length,
        "forecast_horizon": config.data.horizon,
        "profile_dim": config.model.profile_dim,
        "probe_epochs": epochs,
        "max_train_batches": max_train_batches,
        "max_val_batches": max_val_batches,
        "probe_parameter_counts": parameter_counts,
        "probe_contract": {
            "original": "checkpoint profile head without retraining",
            "linear_pooled": "A: checkpoint shared pooling followed by a newly trained linear head",
            "mlp_pooled": "B: checkpoint shared pooling followed by a newly trained two-layer MLP",
            "horizon_specific": "C: one learned history-token attention distribution per future horizon",
            "teacher_profile_oracle": (
                "true CFDP used directly as an offline profile key; this is a geometry upper-bound diagnostic, "
                "not a deployable method"
            ),
        },
        "train_history": train_history,
        "validation": validation,
        "counts": counts,
        "future_information_boundary": (
            "future is used only to construct source-train probe targets and offline validation diagnostics; "
            "the frozen encoder and all probe inputs use history only"
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    save_json(resolve_project_path(output_path), result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose CFDP pooling with frozen encoder probes.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="val", choices=("val",))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    args = parser.parse_args()
    output = resolve_project_path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite probe output: {output}")
    config = load_config(args.config)
    result = run_probe_diagnostic(
        config,
        args.checkpoint,
        args.output,
        split=args.split,
        epochs=args.epochs,
        max_train_batches=args.max_train_batches,
        max_val_batches=args.max_val_batches,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
