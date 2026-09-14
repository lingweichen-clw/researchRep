"""Controlled context/future alignment diagnostics for retrieval keys."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np


QUADRANT_NAMES = (
    "context_similar_future_similar",
    "context_similar_future_different",
    "context_different_future_similar",
    "context_different_future_different",
)
KEY_PREFIXES = ("current", "reference", "random")


def _cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1.0e-12:
        return float("nan")
    similarity = float(np.dot(left, right) / denominator)
    return float(1.0 - np.clip(similarity, -1.0, 1.0))


def add_key_and_control_distances(
    records: Iterable[Mapping[str, object]],
    *,
    current_keys: np.ndarray,
    reference_keys: np.ndarray,
    random_keys: np.ndarray,
    level_features: np.ndarray,
    weekday: np.ndarray,
    slot: np.ndarray,
) -> list[dict[str, object]]:
    """Measure all key variants on exactly the same ``(event,node)`` pairs.

    Keys and level features must be indexed as ``[event,node,feature]``.  The
    calendar flag matches the broad causal protocol's semantic calendar pool:
    equal time-of-day slot and cyclic weekday distance at most one day.
    """
    key_arrays = (current_keys, reference_keys, random_keys)
    if any(array.ndim != 3 for array in key_arrays):
        raise ValueError("key arrays must be [event,node,retrieval_dim]")
    if any(array.shape[:2] != current_keys.shape[:2] for array in key_arrays):
        raise ValueError("all key arrays must share event and node axes")
    if level_features.ndim != 3 or level_features.shape[:2] != current_keys.shape[:2]:
        raise ValueError("level_features must align with key event and node axes")
    event_count = current_keys.shape[0]
    weekday = np.asarray(weekday)
    slot = np.asarray(slot)
    if weekday.shape != (event_count,) or slot.shape != (event_count,):
        raise ValueError("weekday and slot must align with the event axis")

    enriched: list[dict[str, object]] = []
    for source in records:
        row = dict(source)
        node, left, right = int(row["node"]), int(row["i"]), int(row["j"])
        if not (0 <= left < event_count and 0 <= right < event_count):
            raise IndexError("pair event index lies outside the key arrays")
        if not 0 <= node < current_keys.shape[1]:
            raise IndexError("pair node index lies outside the key arrays")
        distances = [
            _cosine_distance(array[left, node], array[right, node])
            for array in key_arrays
        ]
        level_distance = float(
            np.mean(
                np.abs(
                    np.asarray(level_features[left, node], dtype=np.float64)
                    - np.asarray(level_features[right, node], dtype=np.float64)
                )
            )
        )
        weekday_gap = abs(int(weekday[left]) - int(weekday[right])) % 7
        weekday_gap = min(weekday_gap, 7 - weekday_gap)
        row.update(
            current_key_distance=distances[0],
            reference_key_distance=distances[1],
            random_key_distance=distances[2],
            level_distance=level_distance,
            calendar_compatible=(
                int(slot[left]) == int(slot[right]) and weekday_gap <= 1
            ),
        )
        if all(np.isfinite(value) for value in (*distances, level_distance)):
            enriched.append(row)
    return enriched


def select_context_future_quadrants(
    records: Iterable[Mapping[str, object]],
    *,
    quantile: float = 0.20,
    require_calendar_compatible: bool = True,
) -> tuple[dict[str, list[dict[str, object]]], dict[str, float]]:
    """Split controlled pairs into the four context/future similarity tails."""
    if not 0.0 < quantile < 0.5:
        raise ValueError("quantile must lie strictly between 0 and 0.5")
    rows = [
        dict(row)
        for row in records
        if not require_calendar_compatible or bool(row.get("calendar_compatible", False))
    ]
    if not rows:
        raise ValueError("no eligible pair records")
    names = ("context_distance", "level_distance", "future_distance")
    values = {
        name: np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        for name in names
    }
    if any(not np.isfinite(array).all() for array in values.values()):
        raise ValueError("pair distances must be finite")
    thresholds: dict[str, float] = {}
    for name, array in values.items():
        thresholds[f"{name}_low"] = float(np.quantile(array, quantile))
        thresholds[f"{name}_high"] = float(np.quantile(array, 1.0 - quantile))

    quadrants = {name: [] for name in QUADRANT_NAMES}
    for row in rows:
        context_low = (
            float(row["context_distance"]) <= thresholds["context_distance_low"]
            and float(row["level_distance"]) <= thresholds["level_distance_low"]
        )
        context_high = (
            float(row["context_distance"]) >= thresholds["context_distance_high"]
            and float(row["level_distance"]) >= thresholds["level_distance_high"]
        )
        future_low = float(row["future_distance"]) <= thresholds["future_distance_low"]
        future_high = float(row["future_distance"]) >= thresholds["future_distance_high"]
        if context_low and future_low:
            quadrants[QUADRANT_NAMES[0]].append(row)
        elif context_low and future_high:
            quadrants[QUADRANT_NAMES[1]].append(row)
        elif context_high and future_low:
            quadrants[QUADRANT_NAMES[2]].append(row)
        elif context_high and future_high:
            quadrants[QUADRANT_NAMES[3]].append(row)
    return quadrants, thresholds


def _distance_summary(rows: list[Mapping[str, object]], name: str) -> dict[str, float | int]:
    if not rows:
        return {"count": 0, "mean": float("nan"), "median": float("nan"), "std": float("nan")}
    values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
    }


def summarize_quadrant_contrasts(
    quadrants: Mapping[str, list[Mapping[str, object]]],
) -> dict[str, object]:
    """Summarize key distances and the three decision-bearing contrasts."""
    joint = list(quadrants.get(QUADRANT_NAMES[0], []))
    context_only = list(quadrants.get(QUADRANT_NAMES[1], []))
    future_only = list(quadrants.get(QUADRANT_NAMES[2], []))
    summaries: dict[str, object] = {"quadrants": {}}
    quadrant_summaries: dict[str, object] = {}
    for quadrant_name in QUADRANT_NAMES:
        rows = list(quadrants.get(quadrant_name, []))
        quadrant_summaries[quadrant_name] = {
            prefix: _distance_summary(rows, f"{prefix}_key_distance")
            for prefix in KEY_PREFIXES
        }
    summaries["quadrants"] = quadrant_summaries

    for prefix in KEY_PREFIXES:
        joint_mean = float(_distance_summary(joint, f"{prefix}_key_distance")["mean"])
        context_mean = float(
            _distance_summary(context_only, f"{prefix}_key_distance")["mean"]
        )
        future_mean = float(
            _distance_summary(future_only, f"{prefix}_key_distance")["mean"]
        )
        summaries[prefix] = {
            "joint_similar_minus_context_similar_future_different": joint_mean
            - context_mean,
            "joint_similar_minus_context_different_future_similar": joint_mean
            - future_mean,
        }
    joint_current = float(_distance_summary(joint, "current_key_distance")["mean"])
    joint_reference = float(_distance_summary(joint, "reference_key_distance")["mean"])
    joint_random = float(_distance_summary(joint, "random_key_distance")["mean"])
    summaries["joint_similar_current_minus_reference"] = joint_current - joint_reference
    summaries["joint_similar_current_minus_random"] = joint_current - joint_random
    return summaries
