"""Validation-only metrics for target-domain key/future relation studies."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def build_trend_signatures(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Return level-invariant signatures for arrays whose final axis is time."""
    values = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if values.ndim < 1 or valid.shape != values.shape:
        raise ValueError("values and valid must have the same shape with a time axis")
    horizon = values.shape[-1]
    if horizon <= 0:
        raise ValueError("time axis must be non-empty")
    positions = np.arange(horizon, dtype=np.int64)
    previous = np.where(valid, positions, 0)
    previous = np.maximum.accumulate(previous, axis=-1)
    following = np.where(valid, positions, horizon - 1)
    following = np.minimum.accumulate(following[..., ::-1], axis=-1)[..., ::-1]
    left = np.take_along_axis(values, previous, axis=-1)
    right = np.take_along_axis(values, following, axis=-1)
    span = following - previous
    fraction = np.divide(
        positions - previous,
        span,
        out=np.zeros_like(values, dtype=np.float32),
        where=span > 0,
    )
    filled = left + (right - left) * fraction
    no_valid = ~valid.any(axis=-1)
    filled = np.where(no_valid[..., None], 0.0, filled)
    centered = filled - filled[..., :1]
    scale = np.maximum(centered.std(axis=-1, keepdims=True), 1.0e-6)
    return (centered / scale).astype(np.float32)


def future_similarity(
    query_values: np.ndarray,
    query_valid: np.ndarray,
    candidate_values: np.ndarray,
    candidate_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute query/candidate future trend cosine.

    Shapes are ``query=[B,N,H]`` and ``candidate=[B,N,R,H]``.  The returned
    score and pair-valid mask both have shape ``[B,N,R]``.
    """
    query_values = np.asarray(query_values, dtype=np.float32)
    query_valid = np.asarray(query_valid, dtype=bool)
    candidate_values = np.asarray(candidate_values, dtype=np.float32)
    candidate_valid = np.asarray(candidate_valid, dtype=bool)
    if query_values.ndim != 3 or query_valid.shape != query_values.shape:
        raise ValueError("query values and valid must be [B,N,H]")
    if candidate_values.ndim != 4 or candidate_valid.shape != candidate_values.shape:
        raise ValueError("candidate values and valid must be [B,N,R,H]")
    if (
        candidate_values.shape[0] != query_values.shape[0]
        or candidate_values.shape[1] != query_values.shape[1]
        or candidate_values.shape[-1] != query_values.shape[-1]
    ):
        raise ValueError("query and candidate future dimensions do not align")
    query_signature = build_trend_signatures(query_values, query_valid)
    candidate_signature = build_trend_signatures(candidate_values, candidate_valid)
    numerator = (candidate_signature * query_signature[:, :, None, :]).sum(axis=-1)
    query_norm = np.linalg.norm(query_signature, axis=-1, keepdims=True)
    candidate_norm = np.linalg.norm(candidate_signature, axis=-1)
    denominator = query_norm * candidate_norm
    similarity = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float32),
        where=denominator > 1.0e-8,
    )
    pair_valid = query_valid.any(axis=-1, keepdims=True) & candidate_valid.any(axis=-1)
    return np.where(pair_valid, similarity, 0.0).astype(np.float32), pair_valid


def _ordinal_ranks(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, axis=-1, kind="stable")
    ranks = np.empty_like(order, dtype=np.float32)
    ordinal = np.broadcast_to(
        np.arange(scores.shape[-1], dtype=np.float32), scores.shape
    )
    np.put_along_axis(ranks, order, ordinal, axis=-1)
    return ranks


def _spearman_by_node(
    method_scores: np.ndarray,
    future_scores: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    method_ranks = _ordinal_ranks(np.where(valid, method_scores, -np.inf))
    future_ranks = _ordinal_ranks(np.where(valid, future_scores, -np.inf))
    mask = valid.astype(np.float32)
    count = mask.sum(axis=-1)
    method_ranks = np.where(valid, method_ranks, 0.0)
    future_ranks = np.where(valid, future_ranks, 0.0)
    method_mean = method_ranks.sum(axis=-1) / np.maximum(count, 1.0)
    future_mean = future_ranks.sum(axis=-1) / np.maximum(count, 1.0)
    method_centered = method_ranks - method_mean[..., None]
    future_centered = future_ranks - future_mean[..., None]
    covariance = (mask * method_centered * future_centered).sum(axis=-1)
    method_scale = np.sqrt((mask * np.square(method_centered)).sum(axis=-1))
    future_scale = np.sqrt((mask * np.square(future_centered)).sum(axis=-1))
    denominator = method_scale * future_scale
    result = np.divide(
        covariance,
        denominator,
        out=np.full_like(covariance, np.nan, dtype=np.float32),
        where=(denominator > 1.0e-8) & (count >= 3),
    )
    return result


def relation_metric_arrays(
    method_scores: np.ndarray,
    future_scores: np.ndarray,
    valid: np.ndarray,
    ks: Iterable[int] = (1, 3, 5, 8, 12),
) -> dict[str, np.ndarray | dict[str, np.ndarray]]:
    """Return query-level arrays for one ranking method.

    Inputs use ``[B,N,R]``.  Ranking is descending; callers use ``-raw_l1``
    for the raw-L1 baseline.  Node-level values are macro-averaged to one
    value per query, so large-node datasets do not receive extra weight.
    """
    method_scores = np.asarray(method_scores, dtype=np.float32)
    future_scores = np.asarray(future_scores, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if method_scores.ndim != 3 or future_scores.shape != method_scores.shape or valid.shape != method_scores.shape:
        raise ValueError("method_scores, future_scores, and valid must be [B,N,R]")
    if not np.isfinite(np.where(valid, method_scores, 0.0)).all():
        raise ValueError("valid method scores contain NaN or Inf")
    ks = tuple(int(k) for k in ks)
    if not ks or any(k <= 0 for k in ks):
        raise ValueError("ks must contain positive values")
    relation = _spearman_by_node(method_scores, future_scores, valid)
    with np.errstate(invalid="ignore"):
        relation_query = np.nanmean(relation, axis=1)
    relation_query = np.where(np.isfinite(relation_query), relation_query, np.nan)
    method_order = np.argsort(-np.where(valid, method_scores, -np.inf), axis=-1, kind="stable")
    oracle_order = np.argsort(-np.where(valid, future_scores, -np.inf), axis=-1, kind="stable")
    counts = valid.sum(axis=-1)
    future_cosine: dict[str, np.ndarray] = {}
    recall: dict[str, np.ndarray] = {}
    for k in ks:
        take = min(k, method_scores.shape[-1])
        selected = method_order[..., :take]
        selected_valid = np.take_along_axis(valid, selected, axis=-1)
        selected_future = np.take_along_axis(future_scores, selected, axis=-1)
        node_denominator = selected_valid.sum(axis=-1)
        node_cosine = np.divide(
            (selected_future * selected_valid).sum(axis=-1),
            node_denominator,
            out=np.full_like(node_denominator, np.nan, dtype=np.float32),
            where=node_denominator > 0,
        )
        with np.errstate(invalid="ignore"):
            future_cosine[str(k)] = np.nanmean(node_cosine, axis=1)

        oracle_selected = oracle_order[..., :take]
        oracle_valid = np.take_along_axis(valid, oracle_selected, axis=-1)
        matches = (
            selected[:, :, :, None] == oracle_selected[:, :, None, :]
        ) & selected_valid[:, :, :, None] & oracle_valid[:, :, None, :]
        matched = matches.any(axis=-1).sum(axis=-1)
        denominator = np.minimum(k, counts).astype(np.float32)
        node_recall = np.divide(
            matched,
            denominator,
            out=np.full_like(matched, np.nan, dtype=np.float32),
            where=denominator > 0,
        )
        with np.errstate(invalid="ignore"):
            recall[str(k)] = np.nanmean(node_recall, axis=1)
    return {
        "query_spearman": relation_query.astype(np.float32),
        "query_future_cosine": future_cosine,
        "query_oracle_recall": recall,
    }


def summarize_query_metrics(
    arrays: list[dict[str, np.ndarray | dict[str, np.ndarray]]],
    *,
    seed: int = 42,
    bootstrap_samples: int = 200,
) -> dict:
    """Concatenate query arrays and summarize means plus reproducible CIs."""
    if not arrays:
        raise ValueError("at least one metric array is required")
    spearman = np.concatenate([np.asarray(row["query_spearman"]) for row in arrays])
    cosine_keys = tuple(arrays[0]["query_future_cosine"].keys())
    recall_keys = tuple(arrays[0]["query_oracle_recall"].keys())

    def _summary(values: np.ndarray) -> dict[str, float | list[float]]:
        values = np.asarray(values, dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return {"mean": float("nan"), "std": float("nan"), "ci95": [float("nan"), float("nan")], "count": 0}
        rng = np.random.default_rng(seed)
        if bootstrap_samples > 0:
            indices = rng.integers(0, finite.size, size=(bootstrap_samples, finite.size))
            bootstrap = finite[indices].mean(axis=1)
            ci = np.quantile(bootstrap, [0.025, 0.975])
        else:
            ci = np.asarray([np.nan, np.nan])
        return {
            "mean": float(finite.mean()),
            "std": float(finite.std()),
            "ci95": [float(ci[0]), float(ci[1])],
            "count": int(finite.size),
        }

    result = {
        "spearman": _summary(spearman),
        "future_cosine_at_k": {},
        "oracle_recall_at_k": {},
        "query_count": int(spearman.shape[0]),
    }
    # Keep the flat aliases used by the existing diagnostics/report helpers.
    result["spearman_mean"] = result["spearman"]["mean"]
    for key in cosine_keys:
        values = np.concatenate(
            [np.asarray(row["query_future_cosine"][key]) for row in arrays]
        )
        result["future_cosine_at_k"][key] = _summary(values)
    for key in recall_keys:
        values = np.concatenate(
            [np.asarray(row["query_oracle_recall"][key]) for row in arrays]
        )
        result["oracle_recall_at_k"][key] = _summary(values)
    result["future_cosine_mean"] = {
        key: value["mean"] for key, value in result["future_cosine_at_k"].items()
    }
    result["oracle_recall_mean"] = {
        key: value["mean"] for key, value in result["oracle_recall_at_k"].items()
    }
    return result


def summarize_rank_relation(
    method_scores: np.ndarray,
    future_scores: np.ndarray,
    valid: np.ndarray,
    ks: Iterable[int] = (1, 3, 5, 8, 12),
) -> dict:
    """Convenience wrapper returning aggregate relation metrics."""
    arrays = relation_metric_arrays(method_scores, future_scores, valid, ks)
    summary = summarize_query_metrics([arrays], bootstrap_samples=0)
    return {
        "query_count": summary["query_count"],
        "spearman_mean": summary["spearman"]["mean"],
        "future_cosine_at_k": {
            key: value["mean"]
            for key, value in summary["future_cosine_at_k"].items()
        },
        "oracle_recall_at_k": {
            key: value["mean"]
            for key, value in summary["oracle_recall_at_k"].items()
        },
    }
