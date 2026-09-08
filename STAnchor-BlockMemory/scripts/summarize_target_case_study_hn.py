"""Summarize complete target-domain HN-OffsetDecay v2 CaseStudies."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "artifacts" / "cross_dataset_case_study_hn"
OUTPUT = INPUT_DIR / "summary.json"
SELECTORS = ("random", "source", "finetuned")


def compact_method(method: dict) -> dict:
    alignment = method["alignment"]
    ranking = method["ranking"]
    return {
        "global_spearman": alignment["spearman"],
        "valid_pairs": alignment["valid_pairs"],
        "future_neighbor_recall_at_5": alignment["future_neighbor_recall_at_5"],
        "anchor_spearman": ranking["spearman_mean"],
        "kendall": ranking["kendall_mean"],
        "recall_at_1": ranking["recall_at_1_mean"],
        "ndcg_at_5": ranking["ndcg_at_5_mean"],
        "recall_at_5": ranking["recall_at_5_mean"],
        "eligible_anchors": ranking["spearman_eligible_anchors"],
    }


def main() -> None:
    records = []
    for path in sorted(INPUT_DIR.glob("*_hn.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not payload.get("complete_validation", False):
            continue
        records.append(
            {
                "file": path.name,
                "dataset": payload["dataset"],
                "candidate_protocol": payload["candidate_protocol"],
                "event_top_r": payload["event_top_r"],
                "node_top_k": payload["node_top_k"],
                "queries": payload["queries"],
                "batches": payload["batches"],
                "evaluation_batch_size": payload["evaluation_batch_size"],
                "candidate_pool": payload["candidate_pool"],
                "teacher_signature": payload["teacher_signature"],
                "methods": {name: compact_method(payload["methods"][name]) for name in SELECTORS},
                "deltas": payload["deltas"],
                "elapsed_seconds": payload["elapsed_seconds"],
            }
        )
    if len(records) != 6:
        raise RuntimeError(f"expected six complete target results, found {len(records)}")
    result = {
        "schema_version": 1,
        "study": "target_source_random_t1_hn_offset_decay_v2",
        "metric_contract": "HN-OffsetDecay-v2; future is post-ranking teacher only",
        "complete_results": len(records),
        "results": records,
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "complete_results": len(records)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
