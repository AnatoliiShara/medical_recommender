# -*- coding: utf-8 -*-
from __future__ import annotations
from math import inf
from typing import Dict, List, Tuple

def rrf(ranks_a: Dict[int, int], ranks_b: Dict[int, int], alpha: float = 60.0) -> List[Tuple[int, float]]:
    ids = set(ranks_a) | set(ranks_b)
    out = {}
    for d in ids:
        ra = ranks_a.get(d, inf)
        rb = ranks_b.get(d, inf)
        sa = 0.0 if ra is inf else 1.0 / (alpha + ra)
        sb = 0.0 if rb is inf else 1.0 / (alpha + rb)
        out[d] = sa + sb
    return sorted(out.items(), key=lambda x: x[1], reverse=True)

def weighted_rrf(ranks_bm25: Dict[int, int], ranks_dense: Dict[int, int],
                 alpha: float = 60.0, w_bm25: float = 1.0, w_dense: float = 1.0) -> List[Tuple[int, float]]:
    ids = set(ranks_bm25) | set(ranks_dense)
    out = {}
    for d in ids:
        rb = ranks_bm25.get(d, inf)
        rd = ranks_dense.get(d, inf)
        sb = 0.0 if rb is inf else (w_bm25 / (alpha + rb))
        sd = 0.0 if rd is inf else (w_dense / (alpha + rd))
        out[d] = sb + sd
    return sorted(out.items(), key=lambda x: x[1], reverse=True)
