"""Run an evaluation matrix of (model x prompt-template) over a dataset."""
from __future__ import annotations
import time, json
from dataclasses import dataclass, asdict
from typing import Callable, List, Dict

# rough per-1K-token prices; update to current values before reporting
PRICE_PER_1K = {"gpt-4o-mini": 0.00015, "gpt-4o": 0.005}

@dataclass
class EvalResult:
    model: str
    template: str
    question: str
    answer: str
    latency_s: float
    est_cost: float
    correct: bool
    grounded: bool

def estimate_cost(model: str, prompt: str, answer: str) -> float:
    toks = (len(prompt) + len(answer)) / 4  # ~4 chars/token
    return round((toks / 1000) * PRICE_PER_1K.get(model, 0.0), 6)

def score_correct(answer: str, expected: str) -> bool:
    return expected.lower().strip() in answer.lower()

def score_grounded(answer: str, context: str) -> bool:
    if not context:
        return True
    toks = [t for t in answer.lower().split() if len(t) > 4]
    if not toks:
        return True
    return sum(t in context.lower() for t in toks) / len(toks) > 0.4

def run_matrix(dataset: List[Dict], models: List[str],
               templates: Dict[str, str],
               call_llm: Callable[[str, str], str]) -> List[dict]:
    results = []
    for item in dataset:
        for model in models:
            for tname, tmpl in templates.items():
                prompt = tmpl.format(**item)
                t0 = time.time()
                ans = call_llm(model, prompt)
                dt = round(time.time() - t0, 3)
                results.append(asdict(EvalResult(
                    model=model, template=tname, question=item["question"],
                    answer=ans, latency_s=dt,
                    est_cost=estimate_cost(model, prompt, ans),
                    correct=score_correct(ans, item.get("expected", "")),
                    grounded=score_grounded(ans, item.get("context", "")),
                )))
    return results

if __name__ == "__main__":
    data = [{"question": "Capital of France?", "expected": "Paris", "context": ""}]
    templates = {"plain": "{question}",
                 "strict": "Answer concisely. {question}"}
    fake = lambda m, p: "The capital is Paris."
    res = run_matrix(data, ["gpt-4o-mini"], templates, fake)
    json.dump(res, open("eval_results.json", "w"), indent=2)
    print(f"Wrote {len(res)} rows.")
