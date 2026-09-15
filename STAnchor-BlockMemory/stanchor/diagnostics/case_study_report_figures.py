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

REPORT_RANKING_FIELDS = (
    ("Spearman", "spearman_mean"),
    ("Kendall", "kendall_mean"),
    ("Recall@1", "recall_at_1_mean"),
    ("Recall@5", "recall_at_5_mean"),
    ("NDCG@5", "ndcg_at_5_mean"),
)

SELECTOR_SPECS = (
    ("pretrained", "Learned", "#C43C39"),
    ("raw_l1", "Raw-L1", "#E69F00"),
    ("random", "Matched random", "#4C78A8"),
)


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _raw_memory(metrics: dict[str, Any]) -> dict[str, Any]:
    memory = metrics["memory_metrics"]
    if "raw_l1_memory" in memory:
        return memory["raw_l1_memory"]
    # Historical artifact field name. The completed final experiment records that
    # every selector uses the same Offset-only payload, so the suffix is legacy only.
    return memory["raw_l1_offset_decay_memory"]


def horizon_gain_rows(
    metrics: dict[str, Any], frequency_minutes: int
) -> list[dict[str, float | int]]:
    """Keep the original learned-versus-random export contract."""
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


def horizon_comparison_rows(
    metrics: dict[str, Any], frequency_minutes: int
) -> list[dict[str, float | int]]:
    learned = metrics["memory_metrics"]["pretrained_memory"]["horizon_mae"]
    raw_l1 = _raw_memory(metrics)["horizon_mae"]
    random = metrics["memory_metrics"]["random_memory"]["horizon_mae"]
    if len({len(learned), len(raw_l1), len(random)}) != 1:
        raise ValueError("learned/raw/random horizon arrays must have equal length")
    return [
        {
            "step": index + 1,
            "minutes": (index + 1) * frequency_minutes,
            "learned_mae": float(learned[index]),
            "raw_l1_mae": float(raw_l1[index]),
            "random_mae": float(random[index]),
            "gain_vs_raw_l1": float(raw_l1[index]) - float(learned[index]),
            "gain_vs_random": float(random[index]) - float(learned[index]),
        }
        for index in range(len(learned))
    ]


def ranking_gain_rows(metrics: dict[str, Any]) -> list[dict[str, float | str]]:
    """Keep the original learned-versus-random export contract."""
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


def ranking_comparison_rows(
    metrics: dict[str, Any],
) -> list[dict[str, float | str]]:
    ranking = metrics["ranking"]
    rows: list[dict[str, float | str]] = []
    for label, field in REPORT_RANKING_FIELDS:
        learned = float(ranking["pretrained"][field])
        raw_l1 = float(ranking["raw_l1"][field])
        random = float(ranking["random"][field])
        rows.append(
            {
                "metric": label,
                "learned": learned,
                "raw_l1": raw_l1,
                "random": random,
                "gain_vs_raw_l1": learned - raw_l1,
                "gain_vs_random": learned - random,
            }
        )
    return rows


def context_quadrant_contrast_rows(
    rows: list[dict[str, str]],
) -> list[dict[str, float | str]]:
    """Compute within-encoder excess Key distance over the both-similar group."""
    result: list[dict[str, float | str]] = []
    for model in ("current", "reference", "random"):
        model_rows = {row["quadrant"]: row for row in rows if row["model"] == model}
        baseline = model_rows["context_similar_future_similar"]
        baseline_mean = float(baseline["mean"])
        baseline_se = (
            float(baseline["ci_high"]) - float(baseline["ci_low"])
        ) / (2.0 * 1.96)
        for quadrant in (
            "context_similar_future_different",
            "context_different_future_similar",
            "context_different_future_different",
        ):
            row = model_rows[quadrant]
            mean = baseline_mean - float(row["mean"])
            row_se = (float(row["ci_high"]) - float(row["ci_low"])) / (2.0 * 1.96)
            half_width = 1.96 * math.sqrt(baseline_se**2 + row_se**2)
            result.append(
                {
                    "model": model,
                    "model_label": row["model_label"],
                    "quadrant": quadrant,
                    "excess_key_distance": mean,
                    "ci_low": mean - half_width,
                    "ci_high": mean + half_width,
                }
            )
    return result


def _read_history(path: str | Path) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("pretraining history is empty")
    return records


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def _save_figure(figure: Any, output: Path) -> list[Path]:
    pdf = output.with_suffix(".pdf")
    figure.savefig(output, dpi=320, facecolor="white", bbox_inches="tight")
    figure.savefig(pdf, facecolor="white", bbox_inches="tight")
    return [output, pdf]


def _plot_training(
    history_path: str | Path,
    model_label: str,
    objective: str,
    output: Path,
) -> list[Path]:
    import matplotlib.pyplot as plt

    records = _read_history(history_path)
    epochs = [int(record["epoch"]) for record in records]
    train_relation = [float(record["train"]["retrieval"]) for record in records]
    val_epochs = [int(record["epoch"]) for record in records if record.get("val")]
    val_relation = [
        float(record["val"]["retrieval"]) for record in records if record.get("val")
    ]
    val_total = [
        float(record["val"]["total"]) for record in records if record.get("val")
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
    figure, axes = plt.subplots(
        1, columns, figsize=(4.15 * columns, 4.05), constrained_layout=True
    )

    axes[0].plot(epochs, train_relation, color="#4C78A8", label="Train relation")
    axes[0].plot(
        val_epochs,
        val_relation,
        color="#C43C39",
        marker="o",
        markersize=3,
        label="Validation relation",
    )
    axes[0].set_title("(a) Future-relation loss")
    _tight_axis(axes[0], train_relation + val_relation)

    best_total_index = int(np.argmin(val_total))
    axes[1].plot(val_epochs, val_total, color="#7A5195", marker="o", markersize=3)
    axes[1].scatter(
        [val_epochs[best_total_index]],
        [val_total[best_total_index]],
        color="#111111",
        zorder=3,
    )
    axes[1].annotate(
        f"checkpoint\n{val_total[best_total_index]:.4f} @ E{val_epochs[best_total_index]}",
        (val_epochs[best_total_index], val_total[best_total_index]),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=8,
    )
    axes[1].set_title("(b) Validation objective")
    _tight_axis(axes[1], val_total)

    support_axis = axes[-1]
    support_axis.plot(val_epochs, teacher_support, color="#111111", label="Teacher")
    support_axis.plot(val_epochs, student_support, color="#B279A2", label="Student")
    support_axis.set_title("(d) Effective candidate support" if is_joint else "(c) Effective candidate support")
    _tight_axis(support_axis, teacher_support + student_support)

    if is_joint:
        val_reconstruction = [
            float(record["val"]["reconstruction"])
            for record in records
            if record.get("val")
        ]
        axes[2].plot(
            val_epochs,
            val_reconstruction,
            color="#59A14F",
            marker="o",
            markersize=3,
        )
        axes[2].set_title("(c) Validation reconstruction")
        _tight_axis(axes[2], val_reconstruction)

    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(axis="y", alpha=0.25)
        if legend_labels(axis):
            axis.legend(frameon=False, fontsize=8)
    figure.suptitle(f"{model_label}: pretraining convergence", fontsize=14)
    paths = _save_figure(figure, output)
    plt.close(figure)
    return paths


def _plot_rank_profile(
    protocol_metrics: list[tuple[str, dict[str, Any]]],
    model_label: str,
    output: Path,
) -> list[Path]:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        1,
        len(protocol_metrics),
        figsize=(6.25 * len(protocol_metrics), 4.45),
        constrained_layout=True,
    )
    if len(protocol_metrics) == 1:
        axes = [axes]
    for axis, (protocol_label, metrics) in zip(axes, protocol_metrics):
        for selector, default_label, color in SELECTOR_SPECS:
            bins = metrics["alignment"][selector]["distance_bins"]
            axis.plot(
                [item["bin"] for item in bins],
                [item["future_distance_mean"] for item in bins],
                color=color,
                marker="o",
                linewidth=2.3 if selector == "pretrained" else 1.7,
                label=model_label if selector == "pretrained" else default_label,
            )
        axis.set_title(protocol_label)
        axis.set_xlabel("Key-distance decile (near → far)")
        axis.set_ylabel("Mean Offset-only future distance")
        axis.set_xticks(range(1, 11))
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    figure.suptitle("Aggregate Key–future monotonicity on the full validation set", fontsize=14)
    paths = _save_figure(figure, output)
    plt.close(figure)
    return paths


def _plot_ranking_comparison(
    protocol_metrics: list[tuple[str, dict[str, Any]]],
    model_label: str,
    output: Path,
) -> list[Path]:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        2,
        len(protocol_metrics),
        figsize=(6.55 * len(protocol_metrics), 7.8),
        constrained_layout=True,
    )
    if len(protocol_metrics) == 1:
        axes = np.asarray(axes).reshape(2, 1)
    for column, (protocol_label, metrics) in enumerate(protocol_metrics):
        rows = ranking_comparison_rows(metrics)
        labels = [str(row["metric"]) for row in rows]
        positions = np.arange(len(rows), dtype=np.float64)
        width = 0.24
        top = axes[0, column]
        for selector_index, (selector, default_label, color) in enumerate(SELECTOR_SPECS):
            values = [float(row["learned" if selector == "pretrained" else selector]) for row in rows]
            bars = top.bar(
                positions + (selector_index - 1) * width,
                values,
                width,
                color=color,
                label=model_label if selector == "pretrained" else default_label,
            )
            top.bar_label(bars, fmt="%.3f", padding=2, fontsize=7, rotation=90)
        pair_values = [
            float(metrics["alignment"][selector]["spearman"])
            for selector, _, _ in SELECTOR_SPECS
        ]
        top.text(
            0.02,
            0.97,
            "Pair ρ (Learned / Raw-L1 / random): "
            + " / ".join(f"{value:.3f}" for value in pair_values),
            transform=top.transAxes,
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "#DDDDDD", "alpha": 0.9},
        )
        top.set_xticks(positions, labels)
        top.set_title(f"{protocol_label}: anchor-wise metrics")
        top.set_ylabel("Score (higher is better)")
        top.set_ylim(0.0, max(0.72, max(float(row["learned"]) for row in rows) + 0.10))
        top.grid(axis="y", alpha=0.25)
        top.legend(frameon=False, fontsize=8, loc="upper right")

        bottom = axes[1, column]
        raw_gain = [float(row["gain_vs_raw_l1"]) for row in rows]
        random_gain = [float(row["gain_vs_random"]) for row in rows]
        bars_raw = bottom.bar(
            positions - width / 2,
            raw_gain,
            width,
            color="#E69F00",
            label="Learned − Raw-L1",
        )
        bars_random = bottom.bar(
            positions + width / 2,
            random_gain,
            width,
            color="#4C78A8",
            label="Learned − random",
        )
        bottom.axhline(0.0, color="#333333", linewidth=0.8)
        bottom.set_xticks(positions, labels)
        bottom.set_title("Absolute gain over matched selectors")
        bottom.set_ylabel("Score gain")
        bottom.grid(axis="y", alpha=0.25)
        bottom.legend(frameon=False, fontsize=8)
        bottom.bar_label(bars_raw, fmt="%+.3f", padding=2, fontsize=7, rotation=90)
        bottom.bar_label(bars_random, fmt="%+.3f", padding=2, fontsize=7, rotation=90)
        _tight_axis(bottom, raw_gain + random_gain + [0.0])
    figure.suptitle("Full-validation retrieval quality", fontsize=14)
    paths = _save_figure(figure, output)
    plt.close(figure)
    return paths


def _plot_memory_quality(
    protocol_metrics: list[tuple[str, dict[str, Any]]],
    model_label: str,
    frequency_minutes: int,
    output: Path,
) -> list[Path]:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        2,
        len(protocol_metrics),
        figsize=(6.5 * len(protocol_metrics), 7.7),
        constrained_layout=True,
    )
    if len(protocol_metrics) == 1:
        axes = np.asarray(axes).reshape(2, 1)
    for column, (protocol_label, metrics) in enumerate(protocol_metrics):
        memory = metrics["memory_metrics"]
        overall = [
            float(memory["pretrained_memory"]["mae"]),
            float(_raw_memory(metrics)["mae"]),
            float(memory["random_memory"]["mae"]),
        ]
        top = axes[0, column]
        bars = top.bar(
            np.arange(3),
            overall,
            color=[spec[2] for spec in SELECTOR_SPECS],
            width=0.62,
        )
        top.set_xticks(np.arange(3), [model_label, "Raw-L1", "Matched random"])
        top.tick_params(axis="x", labelrotation=10)
        top.set_title(f"{protocol_label}: aggregated Memory MAE")
        top.set_ylabel("MAE in traffic-speed units (lower is better)")
        top.grid(axis="y", alpha=0.25)
        top.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
        _tight_axis(top, overall)
        learned = overall[0]
        top.text(
            0.98,
            0.96,
            f"Reduction vs Raw-L1: {(overall[1] - learned) / overall[1] * 100:.1f}%\n"
            f"Reduction vs random: {(overall[2] - learned) / overall[2] * 100:.1f}%",
            transform=top.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            bbox={"facecolor": "white", "edgecolor": "#DDDDDD", "alpha": 0.9},
        )

        rows = horizon_comparison_rows(metrics, frequency_minutes)
        minutes = [int(row["minutes"]) for row in rows]
        bottom = axes[1, column]
        for selector, default_label, color in SELECTOR_SPECS:
            field = {
                "pretrained": "learned_mae",
                "raw_l1": "raw_l1_mae",
                "random": "random_mae",
            }[selector]
            bottom.plot(
                minutes,
                [float(row[field]) for row in rows],
                color=color,
                marker="o",
                linewidth=2.4 if selector == "pretrained" else 1.7,
                label=model_label if selector == "pretrained" else default_label,
            )
        bottom.set_title("Horizon-wise Memory MAE")
        bottom.set_ylabel("MAE in traffic-speed units")
        bottom.set_xlabel("Forecast horizon (minutes)")
        bottom.set_xticks(minutes[2::3] + ([minutes[-1]] if minutes[-1] not in minutes[2::3] else []))
        bottom.grid(axis="y", alpha=0.25)
        bottom.legend(frameon=False, fontsize=8)
    figure.suptitle("Quality of retrieved future payloads", fontsize=14)
    paths = _save_figure(figure, output)
    plt.close(figure)
    return paths


def _plot_context_evidence(
    *,
    surface_path: str | Path,
    quadrant_path: str | Path,
    partial_path: str | Path,
    output: Path,
) -> tuple[list[Path], list[dict[str, float | str]]]:
    import matplotlib.pyplot as plt

    surface_rows = _read_csv(surface_path)
    bins = max(int(row["context_bin"]) for row in surface_rows)
    surface = np.full((bins, bins), np.nan, dtype=np.float64)
    for row in surface_rows:
        surface[int(row["future_bin"]) - 1, int(row["context_bin"]) - 1] = float(
            row["mean_key_similarity"]
        )
    quadrant_rows = _read_csv(quadrant_path)
    contrasts = context_quadrant_contrast_rows(quadrant_rows)
    partial_rows = _read_csv(partial_path)

    model_specs = (
        ("current", "Final joint-context", "#0072B2"),
        ("reference", "OffsetDecay reference", "#D55E00"),
        ("random", "Random encoder", "#7F7F7F"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(16.2, 4.55), constrained_layout=True)

    finite = surface[np.isfinite(surface)]
    image = axes[0].imshow(
        surface,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=float(finite.min()),
        vmax=float(finite.max()),
    )
    ticks = np.arange(bins)
    axes[0].set_xticks(ticks, labels=[f"Q{i + 1}" for i in ticks])
    axes[0].set_yticks(ticks, labels=[f"Q{i + 1}" for i in ticks])
    axes[0].set_xlabel("Context-shape distance (similar → different)")
    axes[0].set_ylabel("Offset-future distance (similar → different)")
    axes[0].set_title("(a) Final model: controlled relation surface")
    colorbar = figure.colorbar(image, ax=axes[0], fraction=0.046, pad=0.03)
    colorbar.set_label("Mean Key cosine similarity")

    quadrant_order = (
        "context_similar_future_different",
        "context_different_future_similar",
        "context_different_future_different",
    )
    quadrant_labels = ("Context only", "Future only", "Both different")
    positions = np.arange(3, dtype=np.float64)
    offsets = (-0.22, 0.0, 0.22)
    for (model, label, color), offset in zip(model_specs, offsets):
        rows = [row for row in contrasts if row["model"] == model]
        by_quadrant = {str(row["quadrant"]): row for row in rows}
        means = np.asarray(
            [float(by_quadrant[name]["excess_key_distance"]) for name in quadrant_order]
        )
        low = np.asarray([float(by_quadrant[name]["ci_low"]) for name in quadrant_order])
        high = np.asarray([float(by_quadrant[name]["ci_high"]) for name in quadrant_order])
        axes[1].errorbar(
            positions + offset,
            means,
            yerr=np.vstack((means - low, high - means)),
            fmt="o-",
            linewidth=1.5,
            markersize=4.5,
            capsize=2.5,
            color=color,
            label=label,
        )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xticks(positions, labels=quadrant_labels)
    axes[1].set_ylabel("Excess Key distance vs both-similar (95% CI)")
    axes[1].set_title("(b) Within-encoder contrast")
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].legend(frameon=False, fontsize=8)

    factor_order = ("context_shape", "offset_only_future", "context_level")
    factor_labels = ("Context\nshape", "Offset\nfuture", "Context\nlevel")
    width = 0.24
    factor_x = np.arange(3, dtype=np.float64)
    for model_index, (model, label, color) in enumerate(model_specs):
        by_factor = {
            row["factor"]: float(row["partial_spearman"])
            for row in partial_rows
            if row["model"] == model
        }
        axes[2].bar(
            factor_x + (model_index - 1) * width,
            [by_factor[factor] for factor in factor_order],
            width=width,
            color=color,
            label=label,
        )
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set_xticks(factor_x, labels=factor_labels)
    axes[2].set_ylabel("Partial Spearman ρ with Key distance")
    axes[2].set_title("(c) Independent relation after controls")
    axes[2].grid(axis="y", alpha=0.22)
    axes[2].legend(frameon=False, fontsize=8)

    figure.suptitle("Does the final Key preserve both context and future structure?", fontsize=14)
    paths = _save_figure(figure, output)
    plt.close(figure)
    return paths, contrasts


def render_case_study_report_figures(
    *,
    history_path: str | Path,
    broad_metrics_path: str | Path,
    exact_metrics_path: str | Path,
    model_label: str,
    objective: str,
    frequency_minutes: int,
    output_dir: str | Path,
    secondary_protocol_label: str = "Weekday ±1, same slot (deployment-like)",
    context_surface_path: str | Path | None = None,
    context_quadrant_path: str | Path | None = None,
    context_partial_path: str | Path | None = None,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    protocols = [
        ("Broad causal (pretraining-aligned)", load_json(broad_metrics_path)),
        (secondary_protocol_label, load_json(exact_metrics_path)),
    ]
    paths: list[Path] = []
    paths.extend(
        _plot_training(
            history_path,
            model_label,
            objective,
            output / "training_convergence.png",
        )
    )
    paths.extend(
        _plot_rank_profile(
            protocols,
            model_label,
            output / "aggregate_rank_profile.png",
        )
    )
    paths.extend(
        _plot_ranking_comparison(
            protocols,
            model_label,
            output / "full_validation_ranking_comparison.png",
        )
    )
    paths.extend(
        _plot_memory_quality(
            protocols,
            model_label,
            frequency_minutes,
            output / "memory_retrieval_quality.png",
        )
    )

    context_paths = (context_surface_path, context_quadrant_path, context_partial_path)
    if any(path is not None for path in context_paths):
        if not all(path is not None for path in context_paths):
            raise ValueError("all three context-evidence CSV paths must be provided together")
        rendered, contrasts = _plot_context_evidence(
            surface_path=context_surface_path,
            quadrant_path=context_quadrant_path,
            partial_path=context_partial_path,
            output=output / "context_future_key_evidence.png",
        )
        paths.extend(rendered)
        with (output / "context_quadrant_contrast.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=contrasts[0].keys())
            writer.writeheader()
            writer.writerows(contrasts)

    with (output / "ranking_comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = (
            "protocol",
            "metric",
            "learned",
            "raw_l1",
            "random",
            "gain_vs_raw_l1",
            "gain_vs_random",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for protocol_label, metrics in protocols:
            for row in ranking_comparison_rows(metrics):
                writer.writerow({"protocol": protocol_label, **row})

    with (output / "horizon_comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = (
            "protocol",
            "step",
            "minutes",
            "learned_mae",
            "raw_l1_mae",
            "random_mae",
            "gain_vs_raw_l1",
            "gain_vs_random",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for protocol_label, metrics in protocols:
            for row in horizon_comparison_rows(metrics, frequency_minutes):
                writer.writerow({"protocol": protocol_label, **row})

    with (output / "memory_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = ("protocol", "selector", "mae", "rmse", "mape")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for protocol_label, metrics in protocols:
            memory_rows = (
                ("learned", metrics["memory_metrics"]["pretrained_memory"]),
                ("raw_l1", _raw_memory(metrics)),
                ("matched_random", metrics["memory_metrics"]["random_memory"]),
            )
            for selector, values in memory_rows:
                writer.writerow(
                    {
                        "protocol": protocol_label,
                        "selector": selector,
                        "mae": values["mae"],
                        "rmse": values["rmse"],
                        "mape": values["mape"],
                    }
                )
    return paths
