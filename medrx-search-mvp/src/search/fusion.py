# src/search/fusion.py
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x  # порожній вхід → порожній вихід
    x = x / max(temperature, 1e-6)
    x = x - x.max()  # стабільність
    ex = np.exp(x)
    denom = ex.sum()
    if denom <= 0:
        return np.zeros_like(x)
    return ex / denom


def normalize_scores(scores: Dict[str, float],
                     method: str = "none",
                     temperature: float = 0.5) -> Dict[str, float]:
    # критично: якщо кандидатів нема → повертаємо порожній словник
    if not scores:
        return {}

    if method == "none":
        return scores

    vals = np.array(list(scores.values()), dtype=float)
    if method == "minmax":
        mn, mx = vals.min(), vals.max()
        if mx - mn < 1e-12:
            norm = np.zeros_like(vals)
        else:
            norm = (vals - mn) / (mx - mn)
    elif method == "softmax":
        norm = _softmax(vals, temperature)
    else:
        raise ValueError(f"Unknown norm: {method}")

    return {k: float(v) for k, v in zip(scores.keys(), norm)}

def rrf(rank_lists: Dict[str, Dict[str, int]], k: int = 60) -> Dict[str, float]:
    """
    rank_lists: {"bm25": {doc_id: rank}, "dense": {...}}
    Важливо: rank = 1..N (1 = найкращий)
    """
    agg: Dict[str, float] = {}
    for ranks in rank_lists.values():
        for doc, r in ranks.items():
            agg[doc] = agg.get(doc, 0.0) + 1.0 / (k + r)
    return agg

def weighted_rrf(rank_lists: Dict[str, Dict[str, int]],
                 weights: Dict[str, float],
                 k: int = 60) -> Dict[str, float]:
    """
    weights, наприклад: {"bm25": 0.6, "dense": 0.4}
    """
    agg: Dict[str, float] = {}
    for name, ranks in rank_lists.items():
        w = float(weights.get(name, 1.0))
        for doc, r in ranks.items():
            agg[doc] = agg.get(doc, 0.0) + w * (1.0 / (k + r))
    return agg

def dedup_keep_best(ranked_pairs: List[Tuple[str, float]],
                    key_by: Dict[str, str]) -> List[Tuple[str, float]]:
    """
    ranked_pairs: [(doc_id, score), ...] у спадному порядку
    key_by: map doc_id -> entity_key (наприклад, INN+route)
    Залишаємо найкращий doc для кожного entity_key.
    """
    seen = set()
    out = []
    for doc, s in ranked_pairs:
        k = key_by.get(doc, doc)
        if k in seen:
            continue
        seen.add(k)
        out.append((doc, s))
    return out
