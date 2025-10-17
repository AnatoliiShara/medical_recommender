# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any, Dict
from search.wrrf_fusion import rrf, weighted_rrf

def _get(cfg: Any, key: str, default: float|bool) -> float|bool:
    # Підтримує dataclass, Namespace, dict
    if hasattr(cfg, key):
        return getattr(cfg, key)
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return default

def fuse_candidates(ranks_bm25: Dict[int,int], ranks_dense: Dict[int,int], cfg: Any):
    use_weighted = bool(_get(cfg, "use_weighted_rrf", False))
    alpha = float(_get(cfg, "rrf_alpha", 90.0))
    if use_weighted:
        w_bm25 = float(_get(cfg, "w_bm25", 2.0))
        w_dense = float(_get(cfg, "w_dense", 1.0))
        return weighted_rrf(ranks_bm25=ranks_bm25, ranks_dense=ranks_dense,
                            alpha=alpha, w_bm25=w_bm25, w_dense=w_dense)
    else:
        return rrf(ranks_a=ranks_bm25, ranks_b=ranks_dense, alpha=alpha)
