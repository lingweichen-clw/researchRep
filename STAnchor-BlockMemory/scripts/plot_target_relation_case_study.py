"""Plot target-domain retrieval/future relation case-study outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHOD_LABELS = {
    "source_key": "Source learned key",
    "random_key": "Random encoder key",
    "raw_l1": "Raw-L1 context",
    "oracle_future": "Oracle future ranking",
}
METHOD_COLORS = {
    "source_key": "#2166ac",
    "random_key": "#b2182b",
    "raw_l1": "#4d9221",
    "oracle_future": "#762a83",
}


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _plot_density(results: dict[str, dict], output: Path) -> None:
    names = list(results)
    fig, axes = plt.subplots(1, len(names), figsize=(5.1 * len(names), 4.2), squeeze=False)
    axes = axes[0]
    for axis, (dataset, result) in zip(axes, results.items()):
        for method in ("source_key", "random_key"):
            pairs = result["density_pairs"].get(method, {})
            x = np.asarray(pairs.get("key_cosine", []), dtype=np.float32)
            y = np.asarray(pairs.get("future_cosine", []), dtype=np.float32)
            if x.size == 0:
                continue
            axis.hexbin(
                x, y, gridsize=42, mincnt=1, bins="log", cmap="Blues" if method == "source_key" else "Reds",
                alpha=0.52, linewidths=0.0,
            )
        axis.set_title(dataset, fontsize=12, fontweight="bold")
        axis.set_xlabel("Key cosine similarity")
        axis.set_ylabel("Future trend cosine similarity")
        axis.set_xlim(-1.0, 1.0)
        axis.set_ylim(-1.0, 1.0)
        axis.grid(alpha=0.18, linewidth=0.6)
    fig.suptitle("Target-domain key and future-trend relation", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_topk(results: dict[str, dict], output: Path) -> None:
    names = list(results)
    ks = [1, 3, 5, 8, 12]
    fig, axes = plt.subplots(1, len(names), figsize=(5.1 * len(names), 4.2), squeeze=False)
    axes = axes[0]
    for axis, (dataset, result) in zip(axes, results.items()):
        for method in ("source_key", "random_key", "raw_l1", "oracle_future"):
            values = result["methods"][method]["future_cosine_mean"]
            y = [values[str(k)] for k in ks]
            axis.plot(
                ks, y, marker="o", linewidth=2.0, markersize=4.5,
                label=METHOD_LABELS[method], color=METHOD_COLORS[method],
            )
        axis.set_title(dataset, fontsize=12, fontweight="bold")
        axis.set_xlabel("Retrieved candidates K")
        axis.set_ylabel("Mean future-trend cosine")
        axis.set_xticks(ks)
        axis.set_ylim(-1.0, 1.0)
        axis.grid(alpha=0.18, linewidth=0.6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Future-trend similarity of the top-K candidates", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_summary(results: dict[str, dict], output: Path) -> None:
    ks = [1, 3, 5, 8, 12]
    fields = ["dataset", "method", "spearman_mean"]
    fields.extend(f"future_cosine_at_{k}" for k in ks)
    fields.extend(f"oracle_recall_at_{k}" for k in ks)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for dataset, result in results.items():
            for method, values in result["methods"].items():
                row = {"dataset": dataset, "method": method, "spearman_mean": values["spearman_mean"]}
                row.update({f"future_cosine_at_{k}": values["future_cosine_mean"][str(k)] for k in ks})
                row.update({f"oracle_recall_at_{k}": values["oracle_recall_mean"][str(k)] for k in ks})
                writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--datasets", nargs="+", default=["pemsbay", "pems04", "pems08"])
    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    results = {
        name: _load(input_dir / f"{name}_relation_metrics.json")
        for name in args.datasets
    }
    _plot_density(results, output_dir / "target_key_future_density.png")
    _plot_topk(results, output_dir / "target_future_topk_curve.png")
    _write_summary(results, output_dir / "cross_dataset_relation_summary.csv")
    print(f"wrote {output_dir.resolve()}")


if __name__ == "__main__":
    main()
