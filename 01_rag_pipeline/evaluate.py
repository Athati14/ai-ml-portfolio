"""Lightweight retrieval/answer evaluation without external graders.

Computes: context hit-rate (did we retrieve a gold passage?),
MRR, and a simple groundedness check (answer tokens covered by context).
For a fuller harness, plug in ragas and feed it this same record format.
"""
from __future__ import annotations
import json
from typing import List, Dict


def mrr(retrieved_ids: List[str], gold_id: str) -> float:
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid == gold_id:
            return 1.0 / rank
    return 0.0


def groundedness(answer: str, contexts: List[str]) -> float:
    ctx = " ".join(contexts).lower()
    toks = [t for t in answer.lower().split() if len(t) > 3]
    if not toks:
        return 0.0
    return sum(t in ctx for t in toks) / len(toks)


def evaluate(records: List[Dict]) -> Dict[str, float]:
    """records: [{retrieved_ids, gold_id, answer, contexts}, ...]"""
    hit = sum(r["gold_id"] in r["retrieved_ids"] for r in records) / len(records)
    mrr_score = sum(mrr(r["retrieved_ids"], r["gold_id"]) for r in records) / len(records)
    grnd = sum(groundedness(r["answer"], r["contexts"]) for r in records) / len(records)
    return {"context_hit_rate": round(hit, 3),
            "mrr": round(mrr_score, 3),
            "groundedness": round(grnd, 3)}


if __name__ == "__main__":
    with open("eval_records.json", encoding="utf-8") as fh:
        data = json.load(fh)
    print(json.dumps(evaluate(data), indent=2))
