"""Measure 24-patch level-to-key association without retraining."""

from __future__ import annotations

import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stanchor.diagnostics.patch_level_key_sensitivity import (
    LEVEL_COMPONENTS,
    pairwise_cosine_key_distance,
    patch_component_spearman_by_node,
    patch_level_statistics,
)


MODEL_NAMES = ("offset_only", "offset_decay", "random")
AXIS_ARRAYS = (
    "sample_id.npy",
    "weekday.npy",
    "slot.npy",
    "context_start.npy",
    "context_end.npy",
    "future_end.npy",
)


def _manifest(bank: Path) -> dict:
    return json.loads((bank / "manifest.json").read_text(encoding="utf-8"))


def _validate_banks(banks: list[Path], manifests: list[dict]) -> None:
    fields = ("num_events", "num_nodes", "context_length", "retrieval_dim")
    for field in fields:
        values = [manifest[field] for manifest in manifests]
        if len(set(values)) != 1:
            raise ValueError(f"banks disagree on {field}: {values}")
    for filename in AXIS_ARRAYS:
        reference = np.load(banks[0] / filename, mmap_mode="r")
        for bank in banks[1:]:
            if not np.array_equal(reference, np.load(bank / filename, mmap_mode="r")):
                raise ValueError(f"bank axes disagree for {filename}: {bank}")


def _calendar_pairs(weekday: np.ndarray, slot: np.ndarray) -> np.ndarray:
    pairs: list[tuple[int, int]] = []
    for slot_value in np.unique(slot):
        ids = np.flatnonzero(slot == slot_value)
        for left, right in combinations(ids.tolist(), 2):
            gap = abs(int(weekday[left]) - int(weekday[right])) % 7
            if min(gap, 7 - gap) <= 1:
                pairs.append((left, right))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def _digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _profile_summary(profile: np.ndarray) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for component, name in enumerate(LEVEL_COMPONENTS):
        curve = profile[:, 0, component]
        finite = np.isfinite(curve)
        if not finite.any():
            result[name] = {"mean_rho": float("nan"), "max_rho": float("nan"), "max_patch": -1}
            continue
        max_index = int(np.nanargmax(curve))
        result[name] = {
            "mean_rho": float(np.nanmean(curve)),
            "mean_abs_rho": float(np.nanmean(np.abs(curve))),
            "recent_3_mean_rho": float(np.nanmean(curve[-3:])),
            "earlier_21_mean_rho": float(np.nanmean(curve[:-3])),
            "max_rho": float(curve[max_index]),
            "max_patch": max_index + 1,
            "max_patch_hours_before_endpoint": 24 - (max_index + 1),
        }
    return result


def _plot_profiles(
    output: Path,
    profiles: dict[str, dict[str, np.ndarray]],
) -> None:
    colors = {"offset_only": "#0072B2", "offset_decay": "#D55E00", "random": "#777777"}
    labels = {"offset_only": "Offset-only", "offset_decay": "OffsetDecay", "random": "Random"}
    figure, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True, sharey="row")
    x = np.arange(1, 25)
    for row, statistic_kind in enumerate(("absolute", "relative")):
        for component, component_name in enumerate(LEVEL_COMPONENTS):
            axis = axes[row, component]
            for model_name in MODEL_NAMES:
                curve = profiles[model_name][statistic_kind][:, 0, component]
                axis.plot(x, curve, color=colors[model_name], label=labels[model_name], linewidth=1.8)
            axis.axhline(0.0, color="#222222", linewidth=0.8, alpha=0.45)
            axis.axvspan(21.5, 24.5, color="#F0E442", alpha=0.15)
            axis.set_title(component_name)
            axis.set_xticks((1, 6, 12, 18, 24))
            axis.grid(alpha=0.18)
            if row == 1:
                axis.set_xlabel("Patch index (1 oldest, 24 most recent)")
        axes[row, 0].set_ylabel(
            f"{statistic_kind.capitalize()} level\nSpearman with key distance"
        )
    axes[0, 3].legend(frameon=False, fontsize=9)
    figure.suptitle("Patch-level history statistics versus retrieval-key distance")
    figure.tight_layout()
    figure.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute absolute and global-relative 24-patch level-to-key curves."
    )
    parser.add_argument("--data", default="../data/METRLA_data/METR-LA.h5")
    parser.add_argument(
        "--bank",
        default="artifacts/case_bank_hn_offset_only_v1_transfer_hidden128_ffn2_b16_seed42_epoch32",
    )
    parser.add_argument(
        "--reference-bank",
        default="artifacts/case_bank_hn_offset_decay_v2_transfer_hidden128_ffn2_b16_seed42",
    )
    parser.add_argument(
        "--random-bank",
        default="artifacts/case_bank_hn_offset_decay_v2_transfer_hidden128_ffn2_b16_random_seed42",
    )
    parser.add_argument(
        "--output",
        default="artifacts/diagnostics/patch_level_key_sensitivity_offset_only_epoch32/patch_level_key_sensitivity.json",
    )
    parser.add_argument("--num-events", type=int, default=2500)
    parser.add_argument("--num-nodes", type=int, default=64)
    parser.add_argument("--max-event-pairs", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    started = perf_counter()

    banks = [Path(args.bank), Path(args.reference_bank), Path(args.random_bank)]
    manifests = [_manifest(bank) for bank in banks]
    _validate_banks(banks, manifests)
    manifest = manifests[0]
    if int(manifest["context_length"]) % 24:
        raise ValueError("the retrieval context must divide into 24 patches")
    rng = np.random.default_rng(args.seed)
    selected_events = np.sort(
        rng.choice(
            int(manifest["num_events"]),
            min(args.num_events, int(manifest["num_events"])),
            replace=False,
        )
    )
    selected_nodes = np.sort(
        rng.choice(
            int(manifest["num_nodes"]),
            min(args.num_nodes, int(manifest["num_nodes"])),
            replace=False,
        )
    )
    weekday = np.asarray(
        np.load(banks[0] / "weekday.npy", mmap_mode="r")[selected_events], dtype=np.int64
    )
    slot = np.asarray(
        np.load(banks[0] / "slot.npy", mmap_mode="r")[selected_events], dtype=np.int64
    )
    event_pairs = _calendar_pairs(weekday, slot)
    if len(event_pairs) > args.max_event_pairs:
        chosen = np.sort(rng.choice(len(event_pairs), args.max_event_pairs, replace=False))
        event_pairs = event_pairs[chosen]
    if len(event_pairs) < 2:
        raise ValueError("not enough calendar-compatible event pairs")

    sample_ids = np.asarray(
        np.load(banks[0] / "sample_id.npy", mmap_mode="r")[selected_events], dtype=np.int64
    )
    history_steps = int(manifest["context_length"])
    with pd.HDFStore(Path(args.data), "r") as store:
        raw_values = store.get("/df").to_numpy(dtype=np.float32)
    raw_observed = np.isfinite(raw_values) & (raw_values != 0)
    contexts = np.stack(
        [
            raw_values[sample - history_steps + 1 : sample + 1, selected_nodes]
            for sample in sample_ids
        ],
        axis=0,
    )[..., None]
    context_observed = np.stack(
        [
            raw_observed[sample - history_steps + 1 : sample + 1, selected_nodes]
            for sample in sample_ids
        ],
        axis=0,
    )[..., None]
    scaler_mean = np.asarray(manifest["scaler"]["mean"], dtype=np.float32).reshape(-1)[selected_nodes]
    scaler_std = np.asarray(manifest["scaler"]["std"], dtype=np.float32).reshape(-1)[selected_nodes]
    scaler_eps = float(manifest["scaler"].get("eps", 1.0e-6))
    contexts -= scaler_mean[None, None, :, None]
    contexts /= scaler_std[None, None, :, None] + scaler_eps
    absolute, relative, patch_valid = patch_level_statistics(
        contexts,
        context_observed,
        num_patches=24,
    )
    global_levels = np.asarray(
        np.load(banks[0] / "level_features.npy", mmap_mode="r")[selected_events],
        dtype=np.float32,
    )[:, selected_nodes]
    global_levels = global_levels[:, None, :, None, :]
    global_valid = np.isfinite(global_levels).all(axis=-1)

    profiles: dict[str, dict[str, np.ndarray]] = {}
    node_profile_std: dict[str, dict[str, np.ndarray]] = {}
    global_profiles: dict[str, np.ndarray] = {}
    for model_name, bank in zip(MODEL_NAMES, banks):
        keys = np.asarray(
            np.load(bank / "node_keys.npy", mmap_mode="r")[selected_events],
            dtype=np.float32,
        )[:, selected_nodes]
        key_distance = pairwise_cosine_key_distance(keys, event_pairs)
        absolute_by_node = patch_component_spearman_by_node(
            key_distance, absolute, patch_valid, event_pairs
        )
        relative_by_node = patch_component_spearman_by_node(
            key_distance, relative, patch_valid, event_pairs
        )
        global_by_node = patch_component_spearman_by_node(
            key_distance, global_levels, global_valid, event_pairs
        )
        profiles[model_name] = {
            "absolute": np.nanmean(absolute_by_node, axis=1),
            "relative": np.nanmean(relative_by_node, axis=1),
        }
        node_profile_std[model_name] = {
            "absolute": np.nanstd(absolute_by_node, axis=1),
            "relative": np.nanstd(relative_by_node, axis=1),
        }
        global_profiles[model_name] = np.nanmean(global_by_node[0], axis=0)[0]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure_path = output.with_name("patch_level_key_sensitivity.png")
    _plot_profiles(figure_path, profiles)
    result = {
        "diagnostic": "patch_level_key_sensitivity",
        "protocol": {
            "current_bank": str(banks[0]),
            "reference_bank": str(banks[1]),
            "random_bank": str(banks[2]),
            "encoder_fingerprints": {
                name: manifest_["encoder_fingerprint"]
                for name, manifest_ in zip(MODEL_NAMES, manifests)
            },
            "events": int(len(selected_events)),
            "nodes": int(len(selected_nodes)),
            "event_pairs": int(len(event_pairs)),
            "node_event_pairs": int(len(event_pairs) * len(selected_nodes)),
            "event_ids_sha256": _digest(selected_events),
            "node_ids_sha256": _digest(selected_nodes),
            "event_pairs_sha256": _digest(event_pairs),
            "seed": int(args.seed),
            "patches": 24,
            "steps_per_patch": history_steps // 24,
            "patch_order": "1 is oldest; 24 is the most recent hour",
            "calendar_control": "same slot and cyclic weekday distance <= 1",
            "absolute_statistics": "mean/std/last/slope in the Bank scaler's model units",
            "relative_statistics": "patch mean/std/last/slope after full-288-step event normalization",
            "key_distance": "1 - cosine similarity",
            "association": "within-node Spearman on fixed event pairs, macro-averaged across nodes; observational, not causal attribution",
            "query_future_used": False,
        },
        "components": list(LEVEL_COMPONENTS),
        "global_component_spearman": {
            model_name: {
                component: float(global_profiles[model_name][index])
                for index, component in enumerate(LEVEL_COMPONENTS)
            }
            for model_name in MODEL_NAMES
        },
        "patch_profiles": {
            model_name: {
                statistic_kind: {
                    component: profiles[model_name][statistic_kind][:, 0, index].tolist()
                    for index, component in enumerate(LEVEL_COMPONENTS)
                }
                for statistic_kind in ("absolute", "relative")
            }
            for model_name in MODEL_NAMES
        },
        "node_profile_std": {
            model_name: {
                statistic_kind: {
                    component: node_profile_std[model_name][statistic_kind][:, 0, index].tolist()
                    for index, component in enumerate(LEVEL_COMPONENTS)
                }
                for statistic_kind in ("absolute", "relative")
            }
            for model_name in MODEL_NAMES
        },
        "profile_summary": {
            model_name: {
                statistic_kind: _profile_summary(profiles[model_name][statistic_kind])
                for statistic_kind in ("absolute", "relative")
            }
            for model_name in MODEL_NAMES
        },
        "figure": str(figure_path),
        "runtime_seconds": float(perf_counter() - started),
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
