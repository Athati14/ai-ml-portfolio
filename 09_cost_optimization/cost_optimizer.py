"""Cost-saving LLM wrapper: semantic cache + tiered model routing."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List
import numpy as np
from sentence_transformers import SentenceTransformer

PRICE = {"small": 0.00015, "large": 0.005}  # $/1K tokens; update before reporting


@dataclass
class Stats:
    calls: int = 0
    cache_hits: int = 0
    small_calls: int = 0
    large_calls: int = 0
    naive_cost: float = 0.0
    actual_cost: float = 0.0


class SemanticCache:
    def __init__(self, threshold=0.92):
        self.embed = SentenceTransformer("all-MiniLM-L6-v2")
        self.keys: List[np.ndarray] = []
        self.vals: List[str] = []
        self.threshold = threshold

    def get(self, q: str):
        if not self.keys:
            return None
        qv = self.embed.encode([q])[0]
        sims = [float(np.dot(qv, k) / (np.linalg.norm(qv) * np.linalg.norm(k)))
                for k in self.keys]
        i = int(np.argmax(sims))
        return self.vals[i] if sims[i] >= self.threshold else None

    def put(self, q: str, a: str):
        self.keys.append(self.embed.encode([q])[0])
        self.vals.append(a)


def est_cost(tier: str, prompt: str, answer: str) -> float:
    toks = (len(prompt) + len(answer)) / 4
    return (toks / 1000) * PRICE[tier]


class CostOptimizer:
    def __init__(self, small_llm: Callable[[str], str],
                 large_llm: Callable[[str], str],
                 confidence: Callable[[str], float]):
        self.small, self.large, self.confidence = small_llm, large_llm, confidence
        self.cache, self.stats = SemanticCache(), Stats()

    def ask(self, prompt: str) -> str:
        self.stats.calls += 1
        cached = self.cache.get(prompt)
        if cached is not None:
            self.stats.cache_hits += 1
            return cached
        ans = self.small(prompt)
        self.stats.small_calls += 1
        cost = est_cost("small", prompt, ans)
        if self.confidence(ans) < 0.6:          # escalate
            ans = self.large(prompt)
            self.stats.large_calls += 1
            cost += est_cost("large", prompt, ans)
        self.stats.actual_cost += cost
        self.stats.naive_cost += est_cost("large", prompt, ans)  # all-large baseline
        self.cache.put(prompt, ans)
        return ans

    def savings(self) -> dict:
        s = self.stats
        saved = s.naive_cost - s.actual_cost
        pct = (saved / s.naive_cost * 100) if s.naive_cost else 0
        return {"calls": s.calls, "cache_hits": s.cache_hits,
                "small": s.small_calls, "large": s.large_calls,
                "naive_cost": round(s.naive_cost, 5),
                "actual_cost": round(s.actual_cost, 5),
                "saved_pct": round(pct, 1)}


if __name__ == "__main__":
    opt = CostOptimizer(
        small_llm=lambda p: "short answer",
        large_llm=lambda p: "detailed answer",
        confidence=lambda a: 0.9 if "short" in a else 0.4)
    for q in ["hi", "hi there", "explain transformers"]:
        opt.ask(q)
    print(opt.savings())
