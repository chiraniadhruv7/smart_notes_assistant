from typing import List


def recall_at_k(relevant: List[str], retrieved: List[str], k: int) -> float:
    retrieved_k = retrieved[:k]
    hits = sum(1 for r in retrieved_k if r in relevant)
    return hits / len(relevant) if relevant else 0.0


def mean_reciprocal_rank(relevant: List[str], retrieved: List[str]) -> float:
    for idx, item in enumerate(retrieved):
        if item in relevant:
            return 1.0 / (idx + 1)
    return 0.0
