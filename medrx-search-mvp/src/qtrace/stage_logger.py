# src/qtrace/stage_logger.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage Logger for MedRx pipeline.
Logs per-stage candidate lists and parameters, and can emit predictions.jsonl.
"""

from pathlib import Path
from typing import List, Dict, Optional, Any
import json
import time


class StageLogger:
    def __init__(self, qid: str, query_text: str, out_dir: Path):
        self.qid = str(qid)
        self.query_text = query_text
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.stages: List[Dict[str, Any]] = []
        self.final_results: Optional[List[Dict[str, Any]]] = None

    def log_stage(self, name: str, candidates: List[Dict[str, Any]], params: Optional[Dict[str, Any]] = None):
        # Keep only lightweight snapshot
        snap = []
        for c in candidates:
            snap.append({
                "id": c.get("id") or c.get("doc_id"),
                "bm25": c.get("bm25") or c.get("score_bm25"),
                "dense": c.get("dense") or c.get("score_dense"),
                "ce": c.get("ce") or c.get("score_ce"),
                "evidence": c.get("evidence") or c.get("score_evidence"),
                "final": c.get("final") or c.get("score"),
            })
        self.stages.append({
            "name": name,
            "params": params or {},
            "candidates": snap,
            "ts": time.time(),
        })

    def set_final(self, results: List[Dict[str, Any]]):
        self.final_results = results

    def save(self):
        path = self.out_dir / f"qid_{self.qid}.stages.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "qid": self.qid,
                "query": self.query_text,
                "stages": self.stages,
                "final_size": len(self.final_results or []),
            }, f, ensure_ascii=False, indent=2)

    def dump_predictions_jsonl(self, pred_path: Path, run_id: str, rerank_top: int = 25,
                               k_bm25: int = 0, k_dense: int = 0, source: str = "hybrid_wrrf"):
        if not self.final_results:
            return
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pred_path, "a", encoding="utf-8") as fp:
            for rank, cand in enumerate(self.final_results, start=1):
                row = {
                    "qid": self.qid,
                    "query": self.query_text,
                    "tags": [],
                    "doc_id": cand.get("id") or cand.get("doc_id"),
                    "rank": rank,
                    "scores": {
                        "bm25": safe_float(cand.get("bm25") or cand.get("score_bm25")),
                        "dense": safe_float(cand.get("dense") or cand.get("score_dense")),
                        "ce": safe_float(cand.get("ce") or cand.get("score_ce")),
                        "evidence": safe_float(cand.get("evidence") or cand.get("score_evidence")) or 1.0,
                        "final": safe_float(cand.get("final") or cand.get("score")),
                    },
                    "meta": {
                        "source": source,
                        "run_id": run_id,
                        "k_bm25": int(k_bm25),
                        "k_dense": int(k_dense),
                        "k_ce": int(rerank_top),
                    },
                }
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None
