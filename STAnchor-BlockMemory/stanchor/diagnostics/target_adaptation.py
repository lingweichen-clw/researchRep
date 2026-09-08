"""Small helpers for source-vs-target-adapted retrieval reports."""

from __future__ import annotations

from typing import Mapping


def summarize_adaptation_deltas(
    source: Mapping,
    adapted: Mapping,
) -> dict:
    """Return adapted-minus-source deltas for relation metrics."""
    source_cosine = source["future_cosine_mean"]
    adapted_cosine = adapted["future_cosine_mean"]
    source_recall = source["oracle_recall_mean"]
    adapted_recall = adapted["oracle_recall_mean"]
    if set(source_cosine) != set(adapted_cosine):
        raise ValueError("source and adapted future-cosine keys do not match")
    if set(source_recall) != set(adapted_recall):
        raise ValueError("source and adapted recall keys do not match")
    return {
        "spearman": float(adapted["spearman_mean"] - source["spearman_mean"]),
        "future_cosine_at_k": {
            key: float(adapted_cosine[key] - source_cosine[key])
            for key in source_cosine
        },
        "oracle_recall_at_k": {
            key: float(adapted_recall[key] - source_recall[key])
            for key in source_recall
        },
    }


def summarize_hn_adaptation_deltas(
    source: Mapping,
    adapted: Mapping,
) -> dict:
    """Return adapted-minus-source deltas for HN-OffsetDecay v2 metrics.

    The source-domain CaseStudy reports two complementary views: global
    distance alignment and anchor-wise candidate ranking. Keep the delta
    calculation explicit so target-domain reports cannot silently switch to a
    different metric family.
    """
    source_alignment = source["alignment"]
    adapted_alignment = adapted["alignment"]
    source_ranking = source["ranking"]
    adapted_ranking = adapted["ranking"]
    scalar_alignment = (
        "spearman",
        "future_neighbor_recall_at_5",
    )
    scalar_ranking = (
        "spearman_mean",
        "kendall_mean",
        "recall_at_1_mean",
        "ndcg_at_5_mean",
        "recall_at_5_mean",
    )
    return {
        "alignment": {
            key: float(adapted_alignment[key] - source_alignment[key])
            for key in scalar_alignment
        },
        "ranking": {
            key: float(adapted_ranking[key] - source_ranking[key])
            for key in scalar_ranking
        },
    }


def summarize_hn_selector_deltas(
    random: Mapping,
    source: Mapping,
    finetuned: Mapping,
) -> dict:
    """Compare all three HN selectors using the same metric fields."""
    return {
        "source_minus_random": summarize_hn_adaptation_deltas(random, source),
        "finetuned_minus_source": summarize_hn_adaptation_deltas(source, finetuned),
    }
