"""Test whether trained retrieval keys preserve joint context/future similarity."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stanchor.diagnostics.context_key_alignment import (
    add_key_and_control_distances,
    build_quantile_relation_surface,
    partial_rank_correlation,
    select_context_future_quadrants,
    select_level_controlled_quadrants,
    summarize_quadrant_contrasts,
)


AXIS_ARRAYS = (
    "sample_id.npy",
    "weekday.npy",
    "slot.npy",
    "context_start.npy",
    "context_end.npy",
    "future_end.npy",
)


def _load_manifest(bank: Path) -> dict:
    return json.loads((bank / "manifest.json").read_text(encoding="utf-8"))


def _validate_aligned_banks(banks: list[Path], manifests: list[dict]) -> None:
    fields = ("num_events", "num_nodes", "context_length", "horizon", "retrieval_dim")
    for field in fields:
        values = [manifest[field] for manifest in manifests]
        if len(set(values)) != 1:
            raise ValueError(f"banks disagree on manifest field {field}: {values}")
    for filename in AXIS_ARRAYS:
        reference = np.load(banks[0] / filename, mmap_mode="r")
        for bank in banks[1:]:
            candidate = np.load(bank / filename, mmap_mode="r")
            if not np.array_equal(reference, candidate):
                raise ValueError(f"bank event axes differ for {filename}: {bank}")


def _calendar_pairs(weekday: np.ndarray, slot: np.ndarray) -> np.ndarray:
    pairs: list[tuple[int, int]] = []
    for slot_value in np.unique(slot):
        event_ids = np.flatnonzero(slot == slot_value)
        for left, right in combinations(event_ids.tolist(), 2):
            gap = abs(int(weekday[left]) - int(weekday[right])) % 7
            if min(gap, 7 - gap) <= 1:
                pairs.append((left, right))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def _masked_rms(
    values: np.ndarray,
    valid: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    overlap = valid[left] & valid[right]
    counts = overlap.sum(axis=-1)
    delta = values[left] - values[right]
    squared = np.where(overlap, np.square(delta, dtype=np.float32), 0.0).sum(axis=-1)
    distance = np.sqrt(squared / np.maximum(counts, 1))
    return distance, counts / values.shape[-1]


def _build_context_and_future_features(
    *,
    raw_values: np.ndarray,
    raw_observed: np.ndarray,
    sample_ids: np.ndarray,
    node_ids: np.ndarray,
    level_features: np.ndarray,
    future_values: np.ndarray,
    future_masks: np.ndarray,
    manifest: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    history_steps = int(manifest["context_length"])
    if history_steps % 24:
        raise ValueError("context length must be divisible into 24 patches")
    contexts = np.stack(
        [
            raw_values[sample - history_steps + 1 : sample + 1, node_ids].T
            for sample in sample_ids
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    context_valid = np.stack(
        [
            raw_observed[sample - history_steps + 1 : sample + 1, node_ids].T
            for sample in sample_ids
        ],
        axis=0,
    )
    scaler_mean = np.asarray(manifest["scaler"]["mean"], dtype=np.float32).reshape(-1)[node_ids]
    scaler_std = np.asarray(manifest["scaler"]["std"], dtype=np.float32).reshape(-1)[node_ids]
    scaler_eps = float(manifest["scaler"].get("eps", 1.0e-6))
    model_context = (
        contexts - scaler_mean[None, :, None]
    ) / (scaler_std[None, :, None] + scaler_eps)

    window_mean = level_features[..., 0]
    window_std = level_features[..., 1]
    normalized = (model_context - window_mean[..., None]) / (
        window_std[..., None] + 1.0e-6
    )
    normalized = np.where(context_valid, normalized, 0.0)
    event_count, node_count, _ = normalized.shape
    patch_size = history_steps // 24
    patch_values = normalized.reshape(event_count, node_count, 24, patch_size)
    patch_mask = context_valid.reshape(event_count, node_count, 24, patch_size)
    patch_count = patch_mask.sum(axis=-1)
    patch_means = np.where(
        patch_count > 0,
        np.where(patch_mask, patch_values, 0.0).sum(axis=-1)
        / np.maximum(patch_count, 1),
        0.0,
    ).astype(np.float32)
    patch_valid = patch_count > 0

    forecast_context = model_context[..., -int(manifest["horizon"]) :]
    forecast_valid = context_valid[..., -int(manifest["horizon"]) :]
    visible_count = forecast_valid.sum(axis=-1)
    visible_mean = np.where(
        visible_count > 0,
        np.where(forecast_valid, forecast_context, 0.0).sum(axis=-1)
        / np.maximum(visible_count, 1),
        0.0,
    )
    endpoint = np.where(forecast_valid[..., -1], forecast_context[..., -1], visible_mean)
    endpoint_valid = visible_count > 0
    future_model = np.asarray(future_values, dtype=np.float32).transpose(0, 2, 1)
    future_valid = np.asarray(future_masks, dtype=bool).transpose(0, 2, 1)
    signature_valid = future_valid & endpoint_valid[..., None]
    signature = np.where(
        signature_valid,
        future_model - endpoint[..., None],
        0.0,
    ).astype(np.float32)
    return patch_means, patch_valid, signature, signature_valid


def _pair_records(
    context: np.ndarray,
    context_valid: np.ndarray,
    future: np.ndarray,
    future_valid: np.ndarray,
    event_pairs: np.ndarray,
    *,
    max_pairs_per_node: int,
    seed: int,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    node_count = context.shape[1]
    for node in range(node_count):
        pairs = event_pairs
        if len(pairs) > max_pairs_per_node:
            chosen = np.sort(rng.choice(len(pairs), max_pairs_per_node, replace=False))
            pairs = pairs[chosen]
        left, right = pairs[:, 0], pairs[:, 1]
        context_distance, context_overlap = _masked_rms(
            context[:, node], context_valid[:, node], left, right
        )
        future_distance, future_overlap = _masked_rms(
            future[:, node], future_valid[:, node], left, right
        )
        keep = (
            (context_overlap >= 0.80)
            & (future_overlap >= 0.80)
            & np.isfinite(context_distance)
            & np.isfinite(future_distance)
        )
        for pair_index in np.flatnonzero(keep):
            records.append(
                {
                    "node": int(node),
                    "i": int(left[pair_index]),
                    "j": int(right[pair_index]),
                    "context_distance": float(context_distance[pair_index]),
                    "future_distance": float(future_distance[pair_index]),
                    "context_overlap_fraction": float(context_overlap[pair_index]),
                    "future_overlap_fraction": float(future_overlap[pair_index]),
                }
            )
    return records


def _spearman(rows: list[dict[str, object]], left_name: str, right_name: str) -> float:
    if len(rows) < 2:
        return float("nan")
    left = pd.Series([float(row[left_name]) for row in rows])
    right = pd.Series([float(row[right_name]) for row in rows])
    return float(left.corr(right, method="spearman"))


def _bootstrap_similarity_summary(
    rows: list[dict[str, object]],
    key_name: str,
    *,
    samples: int,
    seed: int,
) -> dict[str, float | int]:
    values = 1.0 - np.asarray([float(row[key_name]) for row in rows], dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }
    rng = np.random.default_rng(seed)
    if samples <= 0:
        low = high = float(values.mean())
    else:
        bootstrap = np.empty(samples, dtype=np.float64)
        for index in range(samples):
            bootstrap[index] = rng.choice(values, size=values.size, replace=True).mean()
        low, high = np.quantile(bootstrap, [0.025, 0.975]).tolist()
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
    }


def _surface_rows(surface: dict[str, object]) -> list[dict[str, float | int]]:
    means = np.asarray(surface["mean"], dtype=np.float64)
    counts = np.asarray(surface["count"], dtype=np.int64)
    context_edges = np.asarray(surface["context_edges"], dtype=np.float64)
    future_edges = np.asarray(surface["future_edges"], dtype=np.float64)
    rows: list[dict[str, float | int]] = []
    for future_bin in range(means.shape[0]):
        for context_bin in range(means.shape[1]):
            rows.append(
                {
                    "future_bin": future_bin + 1,
                    "context_bin": context_bin + 1,
                    "future_low": float(future_edges[future_bin]),
                    "future_high": float(future_edges[future_bin + 1]),
                    "context_low": float(context_edges[context_bin]),
                    "context_high": float(context_edges[context_bin + 1]),
                    "count": int(counts[future_bin, context_bin]),
                    "mean_key_distance": float(means[future_bin, context_bin]),
                    "mean_key_similarity": float(1.0 - means[future_bin, context_bin]),
                }
            )
    return rows


def _save_relation_figure(
    output_dir: Path,
    *,
    surface: dict[str, object],
    quadrants: dict[str, list[dict[str, object]]],
    partial_correlations: dict[str, dict[str, float]],
    bootstrap_samples: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[str]]:
    model_specs = (
        ("current", "Final joint-context", "#0072B2"),
        ("reference", "OffsetDecay", "#D55E00"),
        ("random", "Random encoder", "#7F7F7F"),
    )
    quadrant_specs = (
        ("context_similar_future_similar", "Both\nsimilar"),
        ("context_similar_future_different", "Context\nonly"),
        ("context_different_future_similar", "Future\nonly"),
        ("context_different_future_different", "Both\ndifferent"),
    )
    quadrant_rows: list[dict[str, object]] = []
    for model_index, (prefix, model_label, _) in enumerate(model_specs):
        for quadrant_index, (name, display) in enumerate(quadrant_specs):
            summary = _bootstrap_similarity_summary(
                quadrants[name],
                f"{prefix}_key_distance",
                samples=bootstrap_samples,
                seed=seed + 101 * model_index + quadrant_index,
            )
            quadrant_rows.append(
                {
                    "model": prefix,
                    "model_label": model_label,
                    "quadrant": name,
                    "quadrant_label": display.replace("\n", " "),
                    **summary,
                }
            )

    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.5,
        }
    )
    figure, axes = plt.subplots(1, 3, figsize=(16.2, 4.55), constrained_layout=True)

    mean_distance = np.asarray(surface["mean"], dtype=np.float64)
    mean_similarity = 1.0 - mean_distance
    finite = mean_similarity[np.isfinite(mean_similarity)]
    vmin, vmax = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    image = axes[0].imshow(
        mean_similarity,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    bins = mean_similarity.shape[0]
    ticks = np.arange(bins)
    axes[0].set_xticks(ticks, labels=[f"Q{index + 1}" for index in ticks])
    axes[0].set_yticks(ticks, labels=[f"Q{index + 1}" for index in ticks])
    axes[0].set_xlabel("Context-shape distance  (similar → different)")
    axes[0].set_ylabel("Offset-future distance  (similar → different)")
    axes[0].set_title("(a) Level-controlled relation surface")
    colorbar = figure.colorbar(image, ax=axes[0], fraction=0.046, pad=0.03)
    colorbar.set_label("Mean Key cosine similarity")

    x = np.arange(len(quadrant_specs), dtype=np.float64)
    offsets = (-0.22, 0.0, 0.22)
    for model_index, ((prefix, model_label, color), offset) in enumerate(
        zip(model_specs, offsets)
    ):
        model_rows = [row for row in quadrant_rows if row["model"] == prefix]
        means = np.asarray([float(row["mean"]) for row in model_rows])
        low = np.asarray([float(row["ci_low"]) for row in model_rows])
        high = np.asarray([float(row["ci_high"]) for row in model_rows])
        axes[1].errorbar(
            x + offset,
            means,
            yerr=np.vstack((means - low, high - means)),
            fmt="o-",
            linewidth=1.4,
            markersize=4.5,
            capsize=2.5,
            color=color,
            label=model_label,
        )
    axes[1].set_xticks(x, labels=[display for _, display in quadrant_specs])
    axes[1].set_ylabel("Key cosine similarity (mean ± 95% CI)")
    axes[1].set_title("(b) Four controlled pair types")
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].legend(frameon=False, loc="best")

    factor_specs = (
        ("context_shape", "Context\nshape"),
        ("offset_only_future", "Offset\nfuture"),
        ("context_level", "Context\nlevel"),
    )
    width = 0.24
    factor_x = np.arange(len(factor_specs), dtype=np.float64)
    for model_index, (prefix, model_label, color) in enumerate(model_specs):
        values = [float(partial_correlations[prefix][name]) for name, _ in factor_specs]
        axes[2].bar(
            factor_x + (model_index - 1) * width,
            values,
            width=width,
            color=color,
            alpha=0.9,
            label=model_label,
        )
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].set_xticks(factor_x, labels=[display for _, display in factor_specs])
    axes[2].set_ylabel("Partial Spearman ρ with Key distance")
    axes[2].set_title("(c) Independent relation after controls")
    axes[2].grid(axis="y", alpha=0.22)

    png_path = output_dir / "context_future_key_relation.png"
    pdf_path = output_dir / "context_future_key_relation.pdf"
    figure.savefig(png_path, dpi=320, facecolor="white")
    figure.savefig(pdf_path, facecolor="white")
    plt.close(figure)
    return quadrant_rows, [str(png_path), str(pdf_path)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Controlled context-key alignment test for an Offset-only retrieval Bank."
    )
    parser.add_argument("--data", default="../data/METRLA_data/METR-LA.h5")
    parser.add_argument(
        "--bank",
        default="artifacts/case_bank_hn_offset_only_v1_transfer_hidden128_ffn2_b16_seed42_epoch32",
        help="Current Offset-only Bank whose key behavior is being diagnosed.",
    )
    parser.add_argument(
        "--reference-bank",
        default="artifacts/case_bank_hn_offset_decay_v2_transfer_hidden128_ffn2_b16_seed42",
        help="Trained OffsetDecay reference Bank.",
    )
    parser.add_argument(
        "--random-bank",
        default="artifacts/case_bank_hn_offset_decay_v2_transfer_hidden128_ffn2_b16_random_seed42",
        help="Matched-architecture random-encoder Bank.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/diagnostics/context_key_alignment_offset_only_epoch32/context_key_alignment.json",
    )
    parser.add_argument("--num-events", type=int, default=2500)
    parser.add_argument("--num-nodes", type=int, default=64)
    parser.add_argument("--max-pairs-per-node", type=int, default=1500)
    parser.add_argument("--quantile", type=float, default=0.20)
    parser.add_argument("--level-control-quantile", type=float, default=0.30)
    parser.add_argument("--surface-bins", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    banks = [Path(args.bank), Path(args.reference_bank), Path(args.random_bank)]
    manifests = [_load_manifest(bank) for bank in banks]
    _validate_aligned_banks(banks, manifests)
    manifest = manifests[0]
    event_count = int(manifest["num_events"])
    node_count = int(manifest["num_nodes"])
    rng = np.random.default_rng(args.seed)
    selected_events = np.sort(
        rng.choice(event_count, min(event_count, args.num_events), replace=False)
    )
    selected_nodes = np.sort(
        rng.choice(node_count, min(node_count, args.num_nodes), replace=False)
    )
    sample_ids_all = np.load(banks[0] / "sample_id.npy", mmap_mode="r")
    sample_ids = np.asarray(sample_ids_all[selected_events], dtype=np.int64)
    weekday = np.asarray(
        np.load(banks[0] / "weekday.npy", mmap_mode="r")[selected_events],
        dtype=np.int64,
    )
    slot = np.asarray(
        np.load(banks[0] / "slot.npy", mmap_mode="r")[selected_events],
        dtype=np.int64,
    )
    event_pairs = _calendar_pairs(weekday, slot)
    if event_pairs.size == 0:
        raise ValueError("selected events contain no calendar-compatible pairs")

    with pd.HDFStore(Path(args.data), "r") as store:
        raw_values = store.get("/df").to_numpy(dtype=np.float32)
    raw_observed = np.isfinite(raw_values) & (raw_values != 0)
    current_levels = np.asarray(
        np.load(banks[0] / "level_features.npy", mmap_mode="r")[selected_events],
        dtype=np.float32,
    )[:, selected_nodes]
    current_future = np.asarray(
        np.load(banks[0] / "future_values.npy", mmap_mode="r")[selected_events],
        dtype=np.float32,
    )[:, :, selected_nodes, 0]
    current_future_mask = np.asarray(
        np.load(banks[0] / "future_masks.npy", mmap_mode="r")[selected_events],
        dtype=bool,
    )[:, :, selected_nodes, 0]
    for other in banks[1:]:
        other_future = np.asarray(
            np.load(other / "future_values.npy", mmap_mode="r")[selected_events],
            dtype=np.float32,
        )[:, :, selected_nodes, 0]
        other_mask = np.asarray(
            np.load(other / "future_masks.npy", mmap_mode="r")[selected_events],
            dtype=bool,
        )[:, :, selected_nodes, 0]
        if not np.array_equal(current_future_mask, other_mask) or not np.allclose(
            current_future, other_future, rtol=0.0, atol=0.0
        ):
            raise ValueError(f"future payload is not aligned across Banks: {other}")

    context, context_valid, future, future_valid = _build_context_and_future_features(
        raw_values=raw_values,
        raw_observed=raw_observed,
        sample_ids=sample_ids,
        node_ids=selected_nodes,
        level_features=current_levels,
        future_values=current_future,
        future_masks=current_future_mask,
        manifest=manifest,
    )
    pair_records = _pair_records(
        context,
        context_valid,
        future,
        future_valid,
        event_pairs,
        max_pairs_per_node=args.max_pairs_per_node,
        seed=args.seed,
    )
    key_arrays = [
        np.asarray(
            np.load(bank / "node_keys.npy", mmap_mode="r")[selected_events],
            dtype=np.float32,
        )[:, selected_nodes]
        for bank in banks
    ]
    records = add_key_and_control_distances(
        pair_records,
        current_keys=key_arrays[0],
        reference_keys=key_arrays[1],
        random_keys=key_arrays[2],
        level_features=current_levels,
        weekday=weekday,
        slot=slot,
    )
    quadrants, thresholds = select_context_future_quadrants(
        records,
        quantile=args.quantile,
        require_calendar_compatible=True,
    )
    controlled_quadrants, controlled_thresholds = select_level_controlled_quadrants(
        records,
        tail_quantile=args.quantile,
        level_quantile=args.level_control_quantile,
        require_calendar_compatible=True,
    )
    controlled_records = [
        row
        for row in records
        if bool(row["calendar_compatible"])
        and float(row["level_distance"]) <= controlled_thresholds["level_distance_max"]
    ]
    relation_surface = build_quantile_relation_surface(
        controlled_records,
        value_name="current_key_distance",
        bins=args.surface_bins,
    )
    factors = {
        "context_shape": ("context_distance", ("future_distance", "level_distance")),
        "offset_only_future": ("future_distance", ("context_distance", "level_distance")),
        "context_level": ("level_distance", ("context_distance", "future_distance")),
    }
    partial_correlations = {
        prefix: {
            factor: partial_rank_correlation(
                records,
                outcome=f"{prefix}_key_distance",
                predictor=predictor,
                controls=controls,
            )
            for factor, (predictor, controls) in factors.items()
        }
        for prefix in ("current", "reference", "random")
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    quadrant_rows, figure_paths = _save_relation_figure(
        output.parent,
        surface=relation_surface,
        quadrants=controlled_quadrants,
        partial_correlations=partial_correlations,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    surface_path = output.parent / "context_future_key_surface.csv"
    quadrant_path = output.parent / "level_controlled_quadrants.csv"
    partial_path = output.parent / "partial_rank_correlations.csv"
    records_path = output.parent / "level_controlled_pair_records.csv.gz"
    pd.DataFrame(_surface_rows(relation_surface)).to_csv(surface_path, index=False)
    pd.DataFrame(quadrant_rows).to_csv(quadrant_path, index=False)
    pd.DataFrame(
        [
            {"model": model, "factor": factor, "partial_spearman": value}
            for model, factor_values in partial_correlations.items()
            for factor, value in factor_values.items()
        ]
    ).to_csv(partial_path, index=False)
    pd.DataFrame(controlled_records).to_csv(records_path, index=False, compression="gzip")
    result = {
        "protocol": {
            "current_bank": str(banks[0]),
            "reference_bank": str(banks[1]),
            "random_bank": str(banks[2]),
            "encoder_fingerprints": [manifest_["encoder_fingerprint"] for manifest_ in manifests],
            "selected_events": int(len(selected_events)),
            "selected_nodes": int(len(selected_nodes)),
            "calendar_event_pairs": int(len(event_pairs)),
            "evaluated_node_pairs": int(len(records)),
            "seed": int(args.seed),
            "tail_quantile": float(args.quantile),
            "level_control_quantile": float(args.level_control_quantile),
            "calendar_control": "same slot and cyclic weekday distance <= 1",
            "context_shape": "masked RMS over 24 patch means after the encoder's per-window normalization",
            "context_level": "mean absolute distance over Bank [mean,std,last,slope] features",
            "future_signature": "exact Offset-only teacher Y - 12-step forecast-context endpoint in scaler/model units",
            "key_distance": "1 - cosine similarity on the same event-node pair",
        },
        "thresholds": thresholds,
        "counts": {name: len(rows) for name, rows in quadrants.items()},
        "summary": summarize_quadrant_contrasts(quadrants),
        "rank_correlations": {
            prefix: {
                "context_shape": _spearman(records, f"{prefix}_key_distance", "context_distance"),
                "context_level": _spearman(records, f"{prefix}_key_distance", "level_distance"),
                "offset_only_future": _spearman(records, f"{prefix}_key_distance", "future_distance"),
            }
            for prefix in ("current", "reference", "random")
        },
        "level_controlled_relation": {
            "definition": (
                "calendar-compatible same-node pairs restricted to the lowest configured "
                "quantile of [mean,std,last,slope] distance; context and future tails are "
                "then selected independently"
            ),
            "thresholds": controlled_thresholds,
            "counts": {
                name: len(rows) for name, rows in controlled_quadrants.items()
            },
            "summary": summarize_quadrant_contrasts(controlled_quadrants),
            "partial_rank_correlations": partial_correlations,
            "relation_surface": relation_surface,
        },
        "outputs": {
            "figures": figure_paths,
            "surface_csv": str(surface_path),
            "quadrants_csv": str(quadrant_path),
            "partial_correlations_csv": str(partial_path),
            "controlled_pair_records": str(records_path),
        },
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
