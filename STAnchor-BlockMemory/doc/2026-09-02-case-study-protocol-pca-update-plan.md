# METR-LA Final CaseStudy Protocol and PCA Update Plan

## 1. Scope

This update modifies only the completed METR-LA hidden128 HN-OffsetDecay v2 CaseStudy. It does not retrain the encoder, rebuild the Bank, change the model, or redesign the mirage-case selection experiment.

The target report is `doc/诊断报告合集/E5-TGGE-HN-OffsetDecay-v2-hidden128-最终CaseStudy报告.md`.

## 2. Required changes

1. Keep `pretrain_broad_causal` as the pretraining-aligned analysis protocol.
2. Replace the obsolete `exact_calendar` deployment analysis with `weekday_radius1_overlap`.
3. Use `event_top_r=96` so the three-weekday legal pool is not truncated, and use node-level Top-12.
4. Add a Raw-L1 Top-12 baseline to both retained protocols. Raw-L1 uses the same legal event pool and the same masked 288-step observed history, ranks candidates by raw-context L1 distance, and uniformly aggregates their original Bank futures. It does not use the learned key, learned weights, or OffsetDecay.
5. Do not report Learned Uniform Top-12.
6. Remove the v1-versus-v2 comparison section from the final report.
7. Preserve the existing mirage A/B definitions and selected evidence contract. Re-render only the population key PCA.

## 3. PCA display contract

Future-trend clustering and all original 64-D statistics remain unchanged. The PCA is fitted on the same deterministic Bank-key population sample. For readability, every retained cluster displays at most 40 actual event-node keys selected deterministically nearest to its 64-D cluster centroid.

The main PCA figure uses visible colored points and a centroid marker, following the visual grammar of ST-SSDL Figure 5. The dense gray Bank background is removed from the main panel because it obscures cluster points. The figure is qualitative; full cluster sizes and within/between statistics remain the quantitative evidence.

## 4. Future-information boundary

Query future is allowed only for validation metrics, offline cluster construction, deterministic case selection, and plots. Query future is not used to produce query keys, filter candidates, rank candidates, or update the Bank.

## 5. Verification and cleanup boundary

- Add regression tests before production changes.
- Run complete METR-LA validation diagnostics for both retained protocols without `--max-batches`.
- Visually inspect the regenerated PCA and case figures.
- Update report values only from formal JSON artifacts.
- Do not delete exact-calendar or v1 artifacts in this change. Delete them only after the new report and figures are verified, in a separate cleanup step.
