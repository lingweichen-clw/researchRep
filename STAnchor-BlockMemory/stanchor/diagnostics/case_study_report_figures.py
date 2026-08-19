"""Publication-facing figures built from completed retrieval diagnostics."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


RANKING_FIELDS = (
    ("Spearman", "spearman_mean"),
    ("Kendall", "kendall_mean"),
    ("Recall@1", "recall_at_1_mean"),
    ("NDCG@5", "ndcg_at_5_mean"),
)


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def horizon_gain_rows(
    metrics: dict[str, Any], frequency_minutes: int
) -> list[dict[str, float | int]]:
    trained = metrics["memory_metrics"]["pretrained_memory"]["horizon_mae"]
    random = metrics["memory_metrics"]["random_memory"]["horizon_mae"]
    if len(trained) != len(random):
        raise ValueError("trained/random horizon arrays must have equal length")
    return [
        {
            "step": index + 1,
            "minutes": (index + 1) * frequency_minutes,
            "trained_mae": float(trained[index]),
            "random_mae": float(random[index]),
            "gain": float(random[index]) - float(trained[index]),
        }
        for index in range(len(trained))
    ]


def ranking_gain_rows(metrics: dict[str, Any]) -> list[dict[str, float | str]]:
    ranking = metrics["ranking"]
    return [
        {
            "metric": label,
            "trained": float(ranking["pretrained"][field]),
            "random": float(ranking["random"][field]),
            "gain": float(ranking["pretrained"][field])
            - float(ranking["random"][field]),
        }
        for label, field in RANKING_FIELDS
    ]


def _read_history(path: str | Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("pretraining history is empty")
    return records


def _tight_axis(axis: Any, values: list[float]) -> None:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return
    span = max(finite) - min(finite)
    margin = max(0.12 * span, 1.0e-3)
    axis.set_ylim(min(finite) - margin, max(finite) + margin)


def legend_labels(axis: Any) -> list[str]:
    """Return non-empty labels so callers avoid creating empty legends."""
    _, labels = axis.get_legend_handles_labels()
    return [label for label in labels if label and not label.startswith("_")]


def _plot_training(
    history_path: str | Path,
    model_label: str,
    objective: str,
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    records = _read_history(history_path)
    epochs = [int(record["epoch"]) for record in records]
    train_relation = [float(record["train"]["retrieval"]) for record in records]
    val_epochs = [int(record["epoch"]) for record in records if record.get("val")]
    val_relation = [
        float(record["val"]["retrieval"]) for record in records if record.get("val")
    ]
    teacher_support = [
        float(record["val"]["teacher_effective_support"])
        for record in records
        if record.get("val")
    ]
    student_support = [
        float(record["val"]["student_effective_support"])
        for record in records
        if record.get("val")
    ]

    is_joint = objective != "relation_only"
    columns = 4 if is_joint else 3
    figure, axes = plt.subplots(1, columns, figsize=(4.2 * columns, 4.1), constrained_layout=True)

    axes[0].plot(epochs, train_relation, color="#4C78A8", label="Train relation")
    axes[0].plot(val_epochs, val_relation, color="#C43C39", marker="o", markersize=3, label="Validation relation")
    axes[0].set_title("Future-relation loss")
    _tight_axis(axes[0], train_relation + val_relation)

    best_index = int(np.argmin(val_relation))
    axes[1].plot(val_epochs, val_relation, color="#C43C39", marker="o", markersize=3)
    axes[1].scatter([val_epochs[best_index]], [val_relation[best_index]], color="#111111", zorder=3)
    axes[1].annotate(
        f"best={val_relation[best_index]:.4f}\nepoch={val_epochs[best_index]}",
        (val_epochs[best_index], val_relation[best_index]),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=8,
    )
    axes[1].set_title("Validation relation (zoomed)")
    _tight_axis(axes[1], val_relation)

    support_axis = axes[-1]
    support_axis.plot(val_epochs, teacher_support, color="#111111", label="Teacher support")
    support_axis.plot(val_epochs, student_support, color="#B279A2", label="Student support")
    support_axis.set_title("Effective candidate support")
    _tight_axis(support_axis, teacher_support + student_support)

    if is_joint:
        val_reconstruction = [
            float(record["val"]["reconstruction"])
            for record in records
            if record.get("val")
        ]
        axes[2].plot(val_epochs, val_reconstruction, color="#59A14F", marker="o", markersize=3)
        axes[2].set_title("Validation reconstruction (zoomed)")
        _tight_axis(axes[2], val_reconstruction)

    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(axis="y", alpha=0.25)
        if legend_labels(axis):
            axis.legend(frameon=False, fontsize=8)
    figure.suptitle(f"{model_label}: training convergence", fontsize=14)
    figure.savefig(output, dpi=240, facecolor="white")
    plt.close(figure)


def _plot_rank_profile(
    protocol_metrics: list[tuple[str, dict[str, Any]]],
    model_label: str,
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, len(protocol_metrics), figsize=(6.2 * len(protocol_metrics), 4.4), constrained_layout=True)
    if len(protocol_metrics) == 1:
        axes = [axes]
    for axis, (protocol_label, metrics) in zip(axes, protocol_metrics):
        for selector, color, label in (
            ("pretrained", "#C43C39", model_label),
            ("random", "#4C78A8", "Matched random"),
        ):
            bins = metrics["alignment"][selector]["distance_bins"]
            axis.plot(
                [item["bin"] for item in bins],
                [item["future_distance_mean"] for item in bins],
                color=color,
                marker="o",
                linewidth=2.2,
                label=label,
            )
        axis.set_title(protocol_label)
        axis.set_xlabel("Key-distance decile (near to far)")
        axis.set_ylabel("Mean teacher future distance")
        axis.set_xticks(range(1, 11))
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    figure.suptitle(f"{model_label}: aggregate key-future rank profile", fontsize=14)
    figure.savefig(output, dpi=240, facecolor="white")
    plt.close(figure)


def _plot_ranking_gain(
    protocol_metrics: list[tuple[str, dict[str, Any]]],
    model_label: str,
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, len(protocol_metrics), figsize=(6.4 * len(protocol_metrics), 7.4), constrained_layout=True)
    if len(protocol_metrics) == 1:
        axes = np.asarray(axes).reshape(2, 1)
    for column, (protocol_label, metrics) in enumerate(protocol_metrics):
        rows = ranking_gain_rows(metrics)
        labels = [str(row["metric"]) for row in rows]
        trained = [float(row["trained"]) for row in rows]
        random = [float(row["random"]) for row in rows]
        gains = [float(row["gain"]) for row in rows]
        positions = np.arange(len(rows))
        width = 0.34
        top = axes[0, column]
        top.bar(positions - width / 2, trained, width, color="#C43C39", label=model_label)
        top.bar(positions + width / 2, random, width, color="#4C78A8", label="Matched random")
        top.set_xticks(positions, labels)
        top.set_title(f"{protocol_label}: full-validation scores")
        top.set_ylabel("Anchor-wise score")
        top.grid(axis="y", alpha=0.25)
        top.legend(frameon=False, fontsize=8)

        bottom = axes[1, column]
        bars = bottom.bar(positions, gains, color="#59A14F")
        bottom.axhline(0.0, color="#333333", linewidth=0.8)
        bottom.set_xticks(positions, labels)
        bottom.set_title("Gain = trained - matched random")
        bottom.set_ylabel("Absolute score gain")
        bottom.grid(axis="y", alpha=0.25)
        bottom.bar_label(bars, fmt="%+.3f", padding=3, fontsize=8)
        _tight_axis(bottom, gains + [0.0])
    figure.suptitle(f"{model_label}: full-validation ranking gain", fontsize=14)
    figure.savefig(output, dpi=240, facecolor="white")
    plt.close(figure)


def _plot_horizon_gain(
    protocol_metrics: list[tuple[str, dict[str, Any]]],
    model_label: str,
    frequency_minutes: int,
    output: Path,
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, len(protocol_metrics), figsize=(6.4 * len(protocol_metrics), 7.4), constrained_layout=True)
    if len(protocol_metrics) == 1:
        axes = np.asarray(axes).reshape(2, 1)
    for column, (protocol_label, metrics) in enumerate(protocol_metrics):
        rows = horizon_gain_rows(metrics, frequency_minutes)
        minutes = [int(row["minutes"]) for row in rows]
        trained = [float(row["trained_mae"]) for row in rows]
        random = [float(row["random_mae"]) for row in rows]
        gains = [float(row["gain"]) for row in rows]
        top = axes[0, column]
        top.plot(minutes, trained, color="#C43C39", marker="o", label=model_label)
        top.plot(minutes, random, color="#4C78A8", marker="o", label="Matched random")
        top.set_title(f"{protocol_label}: Memory MAE")
        top.set_ylabel("MAE (traffic-speed unit)")
        top.grid(axis="y", alpha=0.25)
        top.legend(frameon=False, fontsize=8)

        bottom = axes[1, column]
        bottom.plot(minutes, gains, color="#59A14F", marker="o", linewidth=2.2)
        bottom.fill_between(minutes, 0.0, gains, color="#59A14F", alpha=0.18)
        bottom.axhline(0.0, color="#333333", linewidth=0.8)
        bottom.set_title("Gain = random MAE - trained MAE")
        bottom.set_ylabel("MAE reduction")
        bottom.grid(axis="y", alpha=0.25)
        _tight_axis(bottom, gains + [0.0])
        for axis in (top, bottom):
            axis.set_xlabel("Forecast horizon (minutes)")
            axis.set_xticks(minutes[2::3] + ([minutes[-1]] if minutes[-1] not in minutes[2::3] else []))
    figure.suptitle(f"{model_label}: horizon-wise retrieval gain", fontsize=14)
    figure.savefig(output, dpi=240, facecolor="white")
    plt.close(figure)


def render_case_study_report_figures(
    *,
    history_path: str | Path,
    broad_metrics_path: str | Path,
    exact_metrics_path: str | Path,
    model_label: str,
    objective: str,
    frequency_minutes: int,
    output_dir: str | Path,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    protocols = [
        ("Broad-causal (pretraining-aligned)", load_json(broad_metrics_path)),
        ("Exact-calendar (deployment-side)", load_json(exact_metrics_path)),
    ]
    paths = [
        output / "training_convergence.png",
        output / "aggregate_rank_profile.png",
        output / "full_validation_ranking_gain.png",
        output / "horizon_wise_gain.png",
    ]
    _plot_training(history_path, model_label, objective, paths[0])
    _plot_rank_profile(protocols, model_label, paths[1])
    _plot_ranking_gain(protocols, model_label, paths[2])
    _plot_horizon_gain(protocols, model_label, frequency_minutes, paths[3])

    with (output / "ranking_gain.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("protocol", "metric", "trained", "random", "gain"))
        writer.writeheader()
        for protocol_label, metrics in protocols:
            for row in ranking_gain_rows(metrics):
                writer.writerow({"protocol": protocol_label, **row})
    with (output / "horizon_gain.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("protocol", "step", "minutes", "trained_mae", "random_mae", "gain"))
        writer.writeheader()
        for protocol_label, metrics in protocols:
            for row in horizon_gain_rows(metrics, frequency_minutes):
                writer.writerow({"protocol": protocol_label, **row})
    return paths
