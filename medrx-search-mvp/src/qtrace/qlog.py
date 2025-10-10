# src/qtrace/qlog.py
from __future__ import annotations
import json, os, time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = ["QLog"]

@dataclass
class QLog:
    query_id: str
    out_dir: str
    query_text: str
    gold_ids: List[int] = field(default_factory=list)

    # optional trace fields — заповнюємо по можливості
    intent: Optional[str] = None
    intent_info: Dict[str, Any] = field(default_factory=dict)

    rewrite_applied: bool = False
    query_rewritten: Optional[str] = None
    alias_hits: List[str] = field(default_factory=list)

    priors_matched: bool = False
    prior_terms: List[str] = field(default_factory=list)
    prior_mode: Optional[str] = None
    prior_intent_allowed: bool = True

    doc_prefilter: Dict[str, Any] = field(default_factory=dict)

    bm25: List[Dict[str, Any]] = field(default_factory=list)   # [{doc_id, section, score}]
    dense: List[Dict[str, Any]] = field(default_factory=list)  # FAISS-passages
    rrf:   List[Dict[str, Any]] = field(default_factory=list)

    ce_top_in: int = 0
    ce: List[Dict[str, Any]] = field(default_factory=list)     # [{doc_id, ce_score}]

    gate: Dict[str, Any] = field(default_factory=dict)         # {mode, sections, kept, dropped, info}

    final: List[Dict[str, Any]] = field(default_factory=list)  # [{doc_id, rank, score, section, violation?}]

    meta: Dict[str, Any] = field(default_factory=dict)

    def add_candidates(self, stage: str, items: List[Dict[str, Any]]) -> None:
        if stage == "bm25": self.bm25 = items
        elif stage == "dense": self.dense = items
        elif stage == "rrf": self.rrf = items
        elif stage == "ce":  self.ce = items

    def set_gate(self, mode: str, sections: List[str], kept: int, dropped: int, info: Dict[str, Any] = None) -> None:
        self.gate = {"mode": mode, "sections": list(sections or []), "kept": int(kept), "dropped": int(dropped), "info": info or {}}

    def dump(self) -> str:
        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(self.out_dir, f"q_{self.query_id}.json")
        obj = {
            "query_id": self.query_id,
            "ts_ms": int(time.time() * 1000),
            "query": self.query_text,
            "gold": self.gold_ids,

            "intent": self.intent,
            "intent_info": self.intent_info,

            "rewrite_applied": self.rewrite_applied,
            "query_rewritten": self.query_rewritten,
            "alias_hits": self.alias_hits,

            "priors": {
                "matched": self.priors_matched,
                "terms": self.prior_terms,
                "mode": self.prior_mode,
                "intent_allowed": self.prior_intent_allowed,
            },

            "doc_prefilter": self.doc_prefilter,
            "bm25": self.bm25, "dense": self.dense, "rrf": self.rrf,
            "ce_top_in": self.ce_top_in, "ce": self.ce,
            "gate": self.gate,
            "final": self.final,
            "meta": self.meta,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        return path
