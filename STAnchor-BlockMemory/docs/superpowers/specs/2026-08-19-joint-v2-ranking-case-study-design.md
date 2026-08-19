# Joint v2 Ranking Case Study Design

## Scope

This case study evaluates the current 26-epoch **Joint v2** checkpoint. Joint v2
uses the future-relation objective with weight `0.1` together with masked
reconstruction. The unfinished Relation-only run is a separate experiment and
must not be mixed into these metrics or figures.

## Research question

Does the learned 48-dimensional history key order candidates according to the
future relation used during pretraining, and does that ordering improve the
retrieved OffsetDecay memory?

## Candidate protocols

1. `pretrain_broad_causal` (primary mechanism protocol): for each validation
   query, use only Bank events whose future ends before the query context starts;
   select a deterministic chronological subset of at most `event_top_r` events.
   This has no weekday/slot restriction and matches the pairwise non-overlap
   relation teacher used by pretraining.
2. `exact_calendar` (deployment protocol): retain the existing same-weekday and
   same-slot causal candidate pool as a secondary deployment check.

Both pretrained and random controls use the identical event axis. Query future
is never used for candidate construction, key encoding, or ranking.

## Primary metrics

For each valid anchor `(query q, node n)` with `M` candidates, obtain the key
ranking and the OffsetDecay teacher ranking from their distances.

- **Anchor-wise Spearman**: compute Spearman correlation within each anchor and
  average over anchors with at least two valid candidates.
- **Anchor-wise Kendall**: compute the pairwise concordance statistic within
  each anchor and average over the same anchors. Ties are excluded from the
  denominator.
- **Recall@1**: whether the key top-1 candidate equals the teacher top-1.
- **NDCG@5**: use `exp(-teacher_distance / temperature)` as graded relevance;
  normalize DCG by the teacher ideal ordering, with `k=min(5,M)`.

Recall@5 remains a secondary set-coverage diagnostic only. Its random expected
value is recorded from the actual candidate count (`1/M` for Recall@1 and the
expected random Top-5 overlap for Recall@5), rather than interpreted as an
absolute retrieval accuracy.

## Prediction metrics and cases

For both protocols, compute memory MAE/RMSE/MAPE after ranking. Select strong-win,
representative, and failure cases by deterministic quantiles of the pretrained
minus random anchor MAE gain. Include ranking tables and the retrieved payload
curves in PNG figures.

## Outputs

Each complete run writes `metrics.json`, `cases.json`, `alignment_bins.csv`,
`ranking_metrics.csv`, PNG figures, and a self-contained Markdown report under
`doc/诊断报告合集/`. The report states the Joint v2 boundary, candidate
protocol, metric definitions, future-information boundary, and the fact that
Relation-only is pending.

## Decision rule

- Keep the mechanism evidence only if Joint v2 beats the matched random control
  on local ranking metrics under `pretrain_broad_causal` and the direction is
  not contradicted by memory MAE.
- Treat exact-calendar results as deployment evidence, not as the sole proof of
  pretraining alignment.
- Do not claim Relation-only superiority until its completed checkpoint and
  matched controls are evaluated separately.
