"""Training-history plots used by pretraining case-study reports."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def load_pretraining_history(path: str | Path) -> dict[str, list[float | int]]:
    fields = {
        "epoch": [],
        "train_total": [],
        "val_total": [],
        "train_reconstruction": [],
        "val_reconstruction": [],
        "train_retrieval": [],
        "val_retrieval": [],
        "teacher_keff": [],
        "student_keff": [],
    }
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record: dict[str, Any] = json.loads(line)
        train = record["train"]
        val = record["val"]
        fields["epoch"].append(int(record["epoch"]))
        fields["train_total"].append(float(train["total"]))
        fields["train_reconstruction"].append(float(train["reconstruction"]))
        fields["train_retrieval"].append(float(train["retrieval"]))
        if val is None:
            fields["val_total"].append(float("nan"))
            fields["val_reconstruction"].append(float("nan"))
            fields["val_retrieval"].append(float("nan"))
            fields["teacher_keff"].append(float("nan"))
            fields["student_keff"].append(float("nan"))
            continue
        fields["val_total"].append(float(val["total"]))
        fields["val_reconstruction"].append(float(val["reconstruction"]))
        fields["val_retrieval"].append(float(val["retrieval"]))
        fields["teacher_keff"].append(float(val["teacher_effective_support"]))
        fields["student_keff"].append(float(val["student_effective_support"]))
    if not fields["epoch"]:
        raise ValueError("pretraining history is empty")
    return fields


def _tighten_axis(axis: Any, series: list[list[float | int]]) -> None:
    """Use a data-dependent margin so shallow trends remain visible."""
    values = [
        float(value)
        for sequence in series
        for value in sequence
        if math.isfinite(float(value))
    ]
    if not values:
        return
    lower = min(values)
    upper = max(values)
    span = upper - lower
    margin = max(span * 0.12, 1.0e-3)
    axis.set_ylim(lower - margin, upper + margin)


def render_pretraining_history(
    metrics_path: str | Path,
    output_path: str | Path,
    title: str,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history = load_pretraining_history(metrics_path)
    epochs = history["epoch"]
    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), constrained_layout=True)

    axes[0].plot(epochs, history["train_total"], label="Train total", color="#4C78A8")
    axes[0].plot(epochs, history["val_total"], label="Validation total", color="#C43C39")
    axes[0].set_title("Joint objective")
    _tighten_axis(axes[0], [history["train_total"], history["val_total"]])

    axes[1].plot(
        epochs, history["val_reconstruction"], label="Reconstruction", color="#59A14F"
    )
    axes[1].plot(epochs, history["val_retrieval"], label="Future relation", color="#F28E2B")
    axes[1].set_title("Validation component losses")
    _tighten_axis(axes[1], [history["val_reconstruction"], history["val_retrieval"]])

    axes[2].plot(epochs, history["teacher_keff"], label="Teacher support", color="#111111")
    axes[2].plot(epochs, history["student_keff"], label="Student support", color="#B279A2")
    axes[2].set_title("Effective candidate support")
    _tighten_axis(axes[2], [history["teacher_keff"], history["student_keff"]])

    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    figure.suptitle(title, fontsize=14)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, facecolor="white")
    plt.close(figure)
    return output
