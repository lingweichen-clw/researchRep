"""Patch-level history statistics for retrieval-key association diagnostics."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


LEVEL_COMPONENTS = ("mean", "std", "last", "slope")


def _window_statistics(
    values: np.ndarray,
    observed: np.ndarray,
    *,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``[mean,std,last,slope]`` and validity for ``[E,P,S,N,C]``."""
    visible = np.asarray(observed, dtype=bool) & np.isfinite(values)
    counts = visible.sum(axis=2)
    safe_counts = np.maximum(counts, 1)
    mean = np.where(visible, values, 0.0).sum(axis=2) / safe_counts
    centered = np.where(visible, values - mean[:, :, None], 0.0)
    variance = np.square(centered, dtype=np.float64).sum(axis=2) / safe_counts
    std = np.sqrt(variance + eps)

    steps = values.shape[2]
    positions = np.arange(steps, dtype=np.int64).reshape(1, 1, steps, 1, 1)
    first_index = np.where(visible, positions, steps).min(axis=2)
    last_index = np.where(visible, positions, -1).max(axis=2)
    first = np.take_along_axis(
        values,
        np.clip(first_index, 0, steps - 1)[:, :, None],
        axis=2,
    ).squeeze(2)
    last = np.take_along_axis(
        values,
        np.clip(last_index, 0, steps - 1)[:, :, None],
        axis=2,
    ).squeeze(2)
    valid = counts > 0
    statistics = np.stack((mean, std, last, last - first), axis=-1)
    statistics = np.where(valid[..., None], statistics, 0.0).astype(np.float32)
    return statistics, valid


def patch_level_statistics(
    values: np.ndarray,
    observed: np.ndarray,
    *,
    num_patches: int = 24,
    eps: float = 1.0e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute absolute and global-relative patch statistics.

    Inputs are model/scaler-unit histories ``[E,T,N,C]``. Outputs are
    ``[E,P,N,C,4]`` ordered as mean, std, last, slope. Relative statistics
    are computed after normalizing each full ``T``-step event/node window,
    not by independently normalizing each patch.
    """
    values = np.asarray(values, dtype=np.float32)
    observed = np.asarray(observed, dtype=bool)
    if values.ndim != 4 or observed.shape != values.shape:
        raise ValueError("values and observed must be aligned [E,T,N,C] arrays")
    if num_patches <= 0 or values.shape[1] % num_patches:
        raise ValueError("num_patches must be positive and divide the time axis")
    if eps < 0:
        raise ValueError("eps must be non-negative")
    events, time, nodes, channels = values.shape
    patch_steps = time // num_patches
    patch_values = values.reshape(events, num_patches, patch_steps, nodes, channels)
    patch_observed = observed.reshape(events, num_patches, patch_steps, nodes, channels)
    absolute, valid = _window_statistics(patch_values, patch_observed, eps=eps)

    global_values = values.reshape(events, 1, time, nodes, channels)
    global_observed = observed.reshape(events, 1, time, nodes, channels)
    global_statistics, global_valid = _window_statistics(
        global_values,
        global_observed,
        eps=eps,
    )
    global_mean = global_statistics[:, 0, :, :, 0]
    global_std = global_statistics[:, 0, :, :, 1]
    normalized = (values - global_mean[:, None]) / (global_std[:, None] + eps)
    normalized = np.where(observed & global_valid[:, 0, None], normalized, 0.0)
    relative_values = normalized.reshape(events, num_patches, patch_steps, nodes, channels)
    relative, relative_valid = _window_statistics(
        relative_values,
        patch_observed,
        eps=eps,
    )
    return absolute, relative, valid & relative_valid


def _validate_pairs(event_pairs: np.ndarray, event_count: int) -> np.ndarray:
    pairs = np.asarray(event_pairs, dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("event_pairs must be [R,2]")
    if pairs.size and (pairs.min() < 0 or pairs.max() >= event_count):
        raise IndexError("event pair lies outside the event axis")
    return pairs


def pairwise_cosine_key_distance(
    keys: np.ndarray,
    event_pairs: np.ndarray,
) -> np.ndarray:
    """Return cosine distances as ``[pair,node]`` on fixed event pairs."""
    keys = np.asarray(keys, dtype=np.float32)
    if keys.ndim != 3:
        raise ValueError("keys must be [event,node,retrieval_dim]")
    pairs = _validate_pairs(event_pairs, keys.shape[0])
    left = keys[pairs[:, 0]]
    right = keys[pairs[:, 1]]
    numerator = np.sum(left * right, axis=-1, dtype=np.float64)
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    similarity = numerator / np.maximum(denominator, 1.0e-12)
    return (1.0 - np.clip(similarity, -1.0, 1.0)).astype(np.float32)


def patch_component_spearman(
    key_distance: np.ndarray,
    level_statistics: np.ndarray,
    level_valid: np.ndarray,
    event_pairs: np.ndarray,
) -> np.ndarray:
    """Correlate key distance with every patch/component distance.

    The result is ``[patch,channel,4]`` and uses the same event/node pairs for
    every patch. Query futures are neither required nor accepted.
    """
    statistics = np.asarray(level_statistics, dtype=np.float32)
    valid = np.asarray(level_valid, dtype=bool)
    if statistics.ndim != 5 or statistics.shape[-1] != len(LEVEL_COMPONENTS):
        raise ValueError("level_statistics must be [event,patch,node,channel,4]")
    if valid.shape != statistics.shape[:-1]:
        raise ValueError("level_valid must match level_statistics without components")
    pairs = _validate_pairs(event_pairs, statistics.shape[0])
    key_distance = np.asarray(key_distance, dtype=np.float32)
    expected_key_shape = (len(pairs), statistics.shape[2])
    if key_distance.shape != expected_key_shape:
        raise ValueError(f"key_distance must be {expected_key_shape}")

    left = statistics[pairs[:, 0]]
    right = statistics[pairs[:, 1]]
    component_distance = np.abs(left - right)
    pair_valid = valid[pairs[:, 0]] & valid[pairs[:, 1]]
    patches, channels = statistics.shape[1], statistics.shape[3]
    output = np.full((patches, channels, len(LEVEL_COMPONENTS)), np.nan, dtype=np.float64)
    for patch in range(patches):
        for channel in range(channels):
            usable = pair_valid[:, patch, :, channel] & np.isfinite(key_distance)
            key_values = key_distance[usable]
            for component in range(len(LEVEL_COMPONENTS)):
                level_values = component_distance[:, patch, :, channel, component][usable]
                finite = np.isfinite(level_values)
                if finite.sum() < 2:
                    continue
                selected_key = key_values[finite]
                selected_level = level_values[finite]
                if np.ptp(selected_key) <= 1.0e-12 or np.ptp(selected_level) <= 1.0e-12:
                    continue
                output[patch, channel, component] = float(
                    spearmanr(selected_key, selected_level).statistic
                )
    return output


def patch_component_spearman_by_node(
    key_distance: np.ndarray,
    level_statistics: np.ndarray,
    level_valid: np.ndarray,
    event_pairs: np.ndarray,
) -> np.ndarray:
    """Return within-node correlations as ``[patch,node,channel,4]``."""
    statistics = np.asarray(level_statistics, dtype=np.float32)
    valid = np.asarray(level_valid, dtype=bool)
    if statistics.ndim != 5 or statistics.shape[-1] != len(LEVEL_COMPONENTS):
        raise ValueError("level_statistics must be [event,patch,node,channel,4]")
    if valid.shape != statistics.shape[:-1]:
        raise ValueError("level_valid must match level_statistics without components")
    pairs = _validate_pairs(event_pairs, statistics.shape[0])
    key_distance = np.asarray(key_distance, dtype=np.float32)
    expected_key_shape = (len(pairs), statistics.shape[2])
    if key_distance.shape != expected_key_shape:
        raise ValueError(f"key_distance must be {expected_key_shape}")

    left = statistics[pairs[:, 0]]
    right = statistics[pairs[:, 1]]
    component_distance = np.abs(left - right)
    pair_valid = valid[pairs[:, 0]] & valid[pairs[:, 1]]
    patches, nodes, channels = (
        statistics.shape[1],
        statistics.shape[2],
        statistics.shape[3],
    )
    output = np.full(
        (patches, nodes, channels, len(LEVEL_COMPONENTS)),
        np.nan,
        dtype=np.float64,
    )
    for patch in range(patches):
        for node in range(nodes):
            for channel in range(channels):
                usable = pair_valid[:, patch, node, channel] & np.isfinite(
                    key_distance[:, node]
                )
                selected_key = key_distance[usable, node]
                for component in range(len(LEVEL_COMPONENTS)):
                    selected_level = component_distance[
                        usable, patch, node, channel, component
                    ]
                    finite = np.isfinite(selected_level)
                    if finite.sum() < 2:
                        continue
                    key_values = selected_key[finite]
                    level_values = selected_level[finite]
                    if np.ptp(key_values) <= 1.0e-12 or np.ptp(level_values) <= 1.0e-12:
                        continue
                    output[patch, node, channel, component] = float(
                        spearmanr(key_values, level_values).statistic
                    )
    return output
