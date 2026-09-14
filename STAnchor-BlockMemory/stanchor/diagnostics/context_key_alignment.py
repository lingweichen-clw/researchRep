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


def select_level_controlled_quadrants(
    records: Iterable[Mapping[str, object]],
    *,
    tail_quantile: float = 0.20,
    level_quantile: float = 0.30,
    require_calendar_compatible: bool = True,
) -> tuple[dict[str, list[dict[str, object]]], dict[str, float]]:
    """Split context/future tails after restricting pairs to similar levels.

    This differs from :func:`select_context_future_quadrants`: level is held
    approximately constant by a low-distance filter, so the context axis
    measures normalized temporal shape instead of a joint shape-and-level
    condition.
    """
    if not 0.0 < tail_quantile < 0.5:
        raise ValueError("tail_quantile must lie strictly between 0 and 0.5")
    if not 0.0 < level_quantile <= 1.0:
        raise ValueError("level_quantile must lie in (0, 1]")
    rows = [
        dict(row)
        for row in records
        if not require_calendar_compatible or bool(row.get("calendar_compatible", False))
    ]
    if not rows:
        raise ValueError("no eligible pair records")
    for name in ("context_distance", "level_distance", "future_distance"):
        if not np.isfinite([float(row[name]) for row in rows]).all():
            raise ValueError("pair distances must be finite")

    level_limit = float(
        np.quantile(
            np.asarray([float(row["level_distance"]) for row in rows], dtype=np.float64),
            level_quantile,
        )
    )
    matched = [row for row in rows if float(row["level_distance"]) <= level_limit]
    if not matched:
        raise ValueError("level control removed every pair")
    context_values = np.asarray(
        [float(row["context_distance"]) for row in matched], dtype=np.float64
    )
    future_values = np.asarray(
        [float(row["future_distance"]) for row in matched], dtype=np.float64
    )
    thresholds = {
        "level_distance_max": level_limit,
        "context_distance_low": float(np.quantile(context_values, tail_quantile)),
        "context_distance_high": float(np.quantile(context_values, 1.0 - tail_quantile)),
        "future_distance_low": float(np.quantile(future_values, tail_quantile)),
        "future_distance_high": float(np.quantile(future_values, 1.0 - tail_quantile)),
    }
    quadrants = {name: [] for name in QUADRANT_NAMES}
    for row in matched:
        context_low = float(row["context_distance"]) <= thresholds["context_distance_low"]
        context_high = float(row["context_distance"]) >= thresholds["context_distance_high"]
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


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    del unique
    ends = np.cumsum(counts)
    starts = ends - counts
    average = (starts + ends - 1) / 2.0 + 1.0
    return average[inverse]


def partial_rank_correlation(
    records: Iterable[Mapping[str, object]],
    *,
    outcome: str,
    predictor: str,
    controls: tuple[str, ...] = (),
) -> float:
    """Return a residualized Spearman correlation with named controls."""
    names = (outcome, predictor, *controls)
    matrix = np.asarray(
        [[float(row[name]) for name in names] for row in records], dtype=np.float64
    )
    if matrix.ndim != 2 or matrix.shape[0] < 3:
        return float("nan")
    matrix = matrix[np.isfinite(matrix).all(axis=1)]
    if matrix.shape[0] < 3:
        return float("nan")
    ranked = np.column_stack([_average_ranks(matrix[:, column]) for column in range(matrix.shape[1])])
    if controls:
        design = np.column_stack((np.ones(matrix.shape[0]), ranked[:, 2:]))
        outcome_residual = ranked[:, 0] - design @ np.linalg.lstsq(
            design, ranked[:, 0], rcond=None
        )[0]
        predictor_residual = ranked[:, 1] - design @ np.linalg.lstsq(
            design, ranked[:, 1], rcond=None
        )[0]
    else:
        outcome_residual = ranked[:, 0] - ranked[:, 0].mean()
        predictor_residual = ranked[:, 1] - ranked[:, 1].mean()
    denominator = float(
        np.linalg.norm(outcome_residual) * np.linalg.norm(predictor_residual)
    )
    if denominator <= 1.0e-12:
        return float("nan")
    return float(np.dot(outcome_residual, predictor_residual) / denominator)


def build_quantile_relation_surface(
    records: Iterable[Mapping[str, object]],
    *,
    value_name: str,
    bins: int = 8,
    context_name: str = "context_distance",
    future_name: str = "future_distance",
) -> dict[str, object]:
    """Aggregate a value over context-by-future quantile cells."""
    if bins < 2:
        raise ValueError("bins must be at least 2")
    matrix = np.asarray(
        [
            [float(row[context_name]), float(row[future_name]), float(row[value_name])]
            for row in records
        ],
        dtype=np.float64,
    )
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("no pair records for relation surface")
    matrix = matrix[np.isfinite(matrix).all(axis=1)]
    if matrix.shape[0] == 0:
        raise ValueError("no finite pair records for relation surface")
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    context_edges = np.quantile(matrix[:, 0], quantiles)
    future_edges = np.quantile(matrix[:, 1], quantiles)
    context_bins = np.searchsorted(context_edges[1:-1], matrix[:, 0], side="right")
    future_bins = np.searchsorted(future_edges[1:-1], matrix[:, 1], side="right")
    sums = np.zeros((bins, bins), dtype=np.float64)
    counts = np.zeros((bins, bins), dtype=np.int64)
    np.add.at(sums, (future_bins, context_bins), matrix[:, 2])
    np.add.at(counts, (future_bins, context_bins), 1)
    means = np.full((bins, bins), np.nan, dtype=np.float64)
    np.divide(sums, counts, out=means, where=counts > 0)
    return {
        "mean": means.tolist(),
        "count": counts.tolist(),
        "context_edges": context_edges.tolist(),
        "future_edges": future_edges.tolist(),
    }


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
