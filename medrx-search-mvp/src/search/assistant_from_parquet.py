# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
import gc
import json
import math
import time
import argparse
import unicodedata
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

# -------- FAISS (optional) --------
try:
    import faiss  # type: ignore
    FAISS_OK = True
except Exception:
    FAISS_OK = False

# -------- Query rewrite modules --------
try:
    from search.query_reform.normalizer import Normalizer
    from search.query_reform.alias_expander import AliasExpander
except Exception:
    class Normalizer:  # type: ignore
        def normalize(self, s: str) -> str:
            if not isinstance(s, str):
                return ""
            s = unicodedata.normalize("NFKC", s)
            s = s.replace("\u00A0", " ")
            s = s.replace("’", "'").replace("“", '"').replace("”", '"')
            s = re.sub(r"\s+", " ", s).strip().lower()
            return s

    class AliasExpander:  # type: ignore
        def __init__(self, csv_path: Optional[str] = None):
            self.map = {}
            if csv_path and os.path.isfile(csv_path):
                import csv
                with open(csv_path, "r", encoding="utf-8") as f:
                    rdr = csv.DictReader(f)
                    for r in rdr:
                        a = (r.get("alias") or "").strip().lower()
                        t = (r.get("target") or "").strip().lower()
                        if a and t:
                            self.map.setdefault(a, set()).add(t)
            for a, ts in list(self.map.items()):
                for t in ts:
                    self.map.setdefault(t, set()).add(a)

        def expand(self, text: str, max_terms: int = 5) -> List[str]:
            out = set()
            toks = re.split(r"[^\w\u0400-\u04FF]+", text.lower())
            for tok in toks:
                tok = tok.strip()
                if not tok:
                    continue
                out.add(tok)
                if tok in self.map:
                    out |= set(list(self.map[tok])[:max_terms])
            return list(out)[:max_terms]

# -------- Clinical priors (optional) --------
def load_priors(jsonl_path: Optional[str]) -> List[Dict[str, Any]]:
    if not jsonl_path:
        return []
    if not os.path.isfile(jsonl_path):
        print(f"[WARN] priors jsonl not found: {jsonl_path}")
        return []
    priors = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                priors.append(json.loads(line))
            except Exception:
                pass
    return priors

def match_priors(query_norm: str, priors: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not priors:
        return None
    q = query_norm.lower()
    for p in priors:
        trig = p.get("triggers", []) or []
        neg = p.get("neg_triggers", []) or []
        if any(t.lower() in q for t in neg):
            continue
        if any(t.lower() in q for t in trig):
            return p
    return None

# -------- intent-policy --------
import yaml

@dataclass
class GateConfig:
    mode: str = "prefer"  # 'none'|'prefer'|'require'
    sections: List[str] = field(default_factory=lambda: ["Показання", "Спосіб застосування та дози"])

@dataclass
class IntentParams:
    name: str = "indication"
    doc_topN: int = 400
    k: int = 300
    gate: GateConfig = field(default_factory=GateConfig)

def load_intent_policy(path: Optional[str]) -> Dict[str, IntentParams]:
    defaults = IntentParams()
    if not path or not os.path.isfile(path):
        return {"_default": defaults}
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    out: Dict[str, IntentParams] = {}
    for intent, cfg in raw.items():
        mode = (cfg.get("gate_mode") or defaults.gate.mode)
        sections = cfg.get("gate_sections") or defaults.gate.sections
        out[intent] = IntentParams(
            name=intent,
            doc_topN=int(cfg.get("doc_topN", defaults.doc_topN)),
            k=int(cfg.get("k", defaults.k)),
            gate=GateConfig(mode=mode, sections=list(sections)),
        )
    if "_default" not in out:
        out["_default"] = defaults
    return out

# -------- utils --------
def _bm25_tokens(s: str) -> List[str]:
    s = s.lower()
    s = re.sub(r"[^\w\u0400-\u04FF]+", " ", s)
    return s.split()

# -------- Engine --------
class ParquetHybridEngine:
    def __init__(
        self,
        index_dir: str,
        embed_model: str,
        doc_index_dir: Optional[str] = None,
        ce_model: Optional[str] = None,
        rrf_alpha: float = 60.0,
        dump_dir: Optional[str] = None,
        priors_jsonl: Optional[str] = None,
    ):
        self.index_dir = index_dir
        self.embed_model_name = embed_model
        self.doc_index_dir = doc_index_dir or index_dir
        self.rrf_alpha = float(rrf_alpha)
        self.dump_dir = dump_dir
        if self.dump_dir:
            os.makedirs(self.dump_dir, exist_ok=True)

        # load chunks metadata & BM25
        chunks_parquet = os.path.join(index_dir, "chunks.parquet")
        table = pq.read_table(
            chunks_parquet,
            columns=["doc_id", "section", "text"],
        )
        self.doc_ids = table.column("doc_id").to_numpy().astype(np.int64)
        self.sections = table.column("section").to_pylist()
        self.passages = table.column("text").to_pylist()

        tokenized = [_bm25_tokens(t) for t in self.passages]
        self.bm25 = BM25Okapi(tokenized)

        # FAISS chunk-level
        self.faiss_index = None
        self.vec_dim = 0
        if FAISS_OK:
            fa_path = os.path.join(index_dir, "faiss.index")
            if os.path.isfile(fa_path):
                self.faiss_index = faiss.read_index(fa_path)
                self.vec_dim = self.faiss_index.d
                print(f"[INFO] FAISS loaded: faiss.index | dim={self.vec_dim}")
            else:
                print("[WARN] FAISS chunk index not found; only BM25 will be used.")
        else:
            print("[WARN] FAISS is not available; only BM25 will be used.")

        # Query encoder
        self.encoder = SentenceTransformer(self.embed_model_name)
        print(f"[INFO] Query encoder ready: {self.embed_model_name}")

        # Doc prefilter
        self.doc_faiss = None
        self.doc_id_map = None
        if FAISS_OK:
            doc_index_path = os.path.join(self.doc_index_dir, "doc.index")
            doc_ids_path = os.path.join(self.doc_index_dir, "doc_ids.npy")
            if os.path.isfile(doc_index_path) and os.path.isfile(doc_ids_path):
                self.doc_faiss = faiss.read_index(doc_index_path)
                self.doc_id_map = np.load(doc_ids_path)
                print(f"[INFO] Doc prefilter ready: n={self.doc_id_map.shape[0]}, dim={self.doc_faiss.d}")
            else:
                print("[WARN] Doc prefilter files not found; skipping doc-level filtering.")

        # CrossEncoder
        self.ce = None
        if ce_model:
            self.ce = CrossEncoder(ce_model, max_length=512)
            print(f"[INFO] CrossEncoder active: {ce_model}")

        # Priors
        self.priors = load_priors(priors_jsonl)
        if self.priors:
            print(f"[INFO] Clinical priors loaded: {len(self.priors)}")

    def _rrf(self, bm: List[int], fa: List[int], k: float) -> Dict[int, float]:
        rank_bm = {pid: r + 1 for r, pid in enumerate(bm)}
        rank_fa = {pid: r + 1 for r, pid in enumerate(fa)}
        ids = set(rank_bm) | set(rank_fa)
        out: Dict[int, float] = {}
        for pid in ids:
            s = 0.0
            if pid in rank_bm: s += 1.0 / (k + rank_bm[pid])
            if pid in rank_fa: s += 1.0 / (k + rank_fa[pid])
            out[pid] = s
        return out

    def _apply_gate(self, pids: List[int], mode: str, sections: List[str]) -> List[int]:
        if mode == "none":
            return pids
        want = set(s.strip() for s in sections if s.strip())
        filtered = [pid for pid in pids if self.sections[pid] in want]
        if mode == "require":
            return filtered
        return filtered if filtered else pids  # prefer

    def _doc_prefilter(self, query: str, topN: int) -> Optional[set]:
        if not self.doc_faiss:
            return None
        qv = self.encoder.encode([query], normalize_embeddings=True).astype("float32")
        D, I = self.doc_faiss.search(qv, topN)
        idxs = I[0].tolist()
        idxs = [i for i in idxs if i >= 0]
        if not idxs:
            return None
        return set(self.doc_id_map[idxs].tolist())

    def _prior_boost_mask(self, pids: List[int], prior: Dict[str, Any]) -> np.ndarray:
        if not prior:
            return np.zeros(len(pids), dtype="float32")
        boost = float(prior.get("boost", 0.05))
        terms = set()
        for t in prior.get("inn", []) + prior.get("brands_opt", []):
            tt = (t or "").strip().lower()
            if tt:
                terms.add(tt)
        if not terms:
            return np.zeros(len(pids), dtype="float32")
        mask = np.zeros(len(pids), dtype="float32")
        for i, pid in enumerate(pids):
            txt = (self.passages[pid] or "").lower()
            if any(term in txt for term in terms):
                mask[i] = boost
        return mask

    def search_once(
        self,
        query: str,
        intent_params: IntentParams,
        use_rewrite: bool = False,
        normalizer: Optional[Normalizer] = None,
        aliaser: Optional[AliasExpander] = None,
        priors: Optional[List[Dict[str, Any]]] = None,
        ce_top: int = 200,
        ce_min: float = 0.15,
        ce_weight: float = 0.75,
        rrf_alpha: float = 60.0,
        dump_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        # rewrite
        q_norm = normalizer.normalize(query) if (use_rewrite and normalizer) else query
        expanded_terms: List[str] = []
        if use_rewrite and aliaser:
            expanded_terms = aliaser.expand(q_norm, max_terms=5)

        # priors
        prior_hit = match_priors(q_norm, priors or [])

        # base query with expansions
        extra_terms = []
        if expanded_terms:
            extra_terms += expanded_terms
        if prior_hit:
            extra_terms += (prior_hit.get("inn", []) or []) + (prior_hit.get("brands_opt", []) or [])
        query_for_search = q_norm if q_norm else query
        if extra_terms:
            query_for_search = query_for_search + " " + " ".join(sorted(set([t for t in extra_terms if t])))

        # doc prefilter
        doc_keep: Optional[set] = self._doc_prefilter(query_for_search, intent_params.doc_topN)

        # BM25
        bm_scores = self.bm25.get_scores(_bm25_tokens(query_for_search))
        bm_idx = np.argsort(-bm_scores)
        if doc_keep is not None:
            bm_idx = [int(pid) for pid in bm_idx if self.doc_ids[int(pid)] in doc_keep]
        else:
            bm_idx = [int(pid) for pid in bm_idx]
        bm_idx = list(bm_idx[:max(1, intent_params.k)])

        # FAISS
        fa_idx: List[int] = []
        if self.faiss_index is not None:
            q_vec = self.encoder.encode([query_for_search], normalize_embeddings=True).astype("float32")
            D, I = self.faiss_index.search(q_vec, max(1, intent_params.k))
            fa_idx = [int(i) for i in I[0].tolist() if i >= 0]
            if doc_keep is not None:
                fa_idx = [pid for pid in fa_idx if self.doc_ids[pid] in doc_keep]

        # RRF
        rrf = self._rrf(bm_idx, fa_idx, k=float(rrf_alpha))
        base_ranked = sorted(rrf.items(), key=lambda x: -x[1])
        pids = [int(pid) for pid, _ in base_ranked[:intent_params.k]]
        base_scores = np.array([rrf[pid] for pid in pids], dtype="float32")

        # Section gate
        gated_pids = self._apply_gate(pids, intent_params.gate.mode, intent_params.gate.sections)
        if not gated_pids:
            gated_pids = pids
        gate_mask = np.array([1.0 if pid in gated_pids else 0.0 for pid in pids], dtype="float32")

        # Prior bonus
        prior_bonus = self._prior_boost_mask(pids, prior_hit)

        # CrossEncoder
        ce_scores = None
        fused = base_scores.copy()
        if self.ce is not None and ce_top > 0:
            topN = min(ce_top, len(pids))
            pairs = [(query_for_search, self.passages[p]) for p in pids[:topN]]
            try:
                raw = self.ce.predict(pairs)
                if isinstance(raw, list):
                    raw = np.array(raw, dtype="float32")
                else:
                    raw = raw.astype("float32")  # type: ignore
            except Exception:
                raw = np.zeros(topN, dtype="float32")
            raw_adj = np.maximum(0.0, raw - float(ce_min))
            if raw_adj.size > 0:
                rmin, rmax = float(raw_adj.min()), float(raw_adj.max())
                ce_norm = (raw_adj - rmin) / (rmax - rmin + 1e-9)
            else:
                ce_norm = np.zeros_like(raw_adj)
            ce_scores = np.zeros(len(pids), dtype="float32")
            ce_scores[:topN] = ce_norm
            fused = (1.0 - float(ce_weight)) * fused + float(ce_weight) * ce_scores

        # Gate prefer masking
        if intent_params.gate.mode == "prefer":
            fused = fused * np.maximum(0.8, gate_mask)
        # Prior bonus add
        fused = fused + prior_bonus

        # Final sort
        order = np.argsort(-fused)
        final_pids = [pids[i] for i in order]
        final_scores = fused[order]

        # ---- logging dump (STRICT Python ints) ----
        dump: Dict[str, Any] = {
            "query": query,
            "query_norm": q_norm,
            "query_expanded_terms": expanded_terms,
            "prior_hit": prior_hit,
            "intent_params": {
                "intent": intent_params.name,
                "doc_topN": int(intent_params.doc_topN),
                "k": int(intent_params.k),
                "gate_mode": intent_params.gate.mode,
                "gate_sections": list(intent_params.gate.sections),
                "rrf_alpha": float(rrf_alpha),
            },
            "bm25_top": [int(x) for x in bm_idx[:50]],
            "faiss_top": [int(x) for x in fa_idx[:50]],
            "rrf_top": [int(x) for x in pids[:50]],
            "gate_mode_used": intent_params.gate.mode,
            "gate_sections_used": list(intent_params.gate.sections),
            "final_topk": [],
        }
        if ce_scores is not None:
            dump["ce_top"] = int(min(ce_top, len(pids)))
            dump["ce_scores"] = [float(v) for v in ce_scores[:int(min(ce_top, len(pids)))].tolist()]

        for i in range(min(intent_params.k, len(final_pids))):
            pid = int(final_pids[i])
            dump["final_topk"].append({
                "rank": int(i + 1),
                "pid": pid,
                "doc_id": int(self.doc_ids[pid]),
                "section": self.sections[pid],
                "score": float(final_scores[i]),
                "text": self.passages[pid][:512],
            })

        if dump_path:
            with open(dump_path, "w", encoding="utf-8") as f:
                json.dump(dump, f, ensure_ascii=False, indent=2)

        return dump

# -------- intent detect (rule-based, простий) --------
def guess_intent(q: str) -> str:
    s = q.lower()
    if any(w in s for w in ["доза", "скільки приймати", "як приймати"]):
        return "dosage"
    if any(w in s for w in ["протипоказан", "кому не можна", "заборонено"]):
        return "contra"
    if any(w in s for w in ["побічн", "побочка", "побочки", "небажан"]):
        return "side_effects"
    if any(w in s for w in ["взаємод", "несумісн", "разом з"]):
        return "interactions"
    return "indication"

# -------- EVAL --------
def evaluate(
    engine: ParquetHybridEngine,
    queries_path: str,
    intent_policy: Dict[str, IntentParams],
    use_rewrite: bool,
    normalizer: Optional[Normalizer],
    aliaser: Optional[AliasExpander],
    ce_top: int,
    ce_min: float,
    ce_weight: float,
    rrf_alpha: float,
    dump_dir: Optional[str] = None,
) -> Dict[str, float]:
    rows = []
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    print(f"[INFO] Loaded queries: {len(rows)}")

    prec_hits = 0.0
    recall_hits = 0.0
    ndcg_sum = 0.0
    total_q = 0

    run_dir = None
    if dump_dir:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(dump_dir, f"run_{stamp}")
        os.makedirs(run_dir, exist_ok=True)

    for qi, item in enumerate(tqdm(rows, desc="Eval", unit="q")):
        q = item.get("query", "")
        gold_ids = item.get("gold_doc_ids", []) or item.get("gold_drugs", [])
        gold_ids = [int(x) for x in gold_ids] if gold_ids else []

        intent = item.get("intent") or guess_intent(q)
        ip = intent_policy.get(intent) or intent_policy["_default"]

        dump_path = os.path.join(run_dir, f"q{qi:04d}.json") if run_dir else None
        res = engine.search_once(
            q,
            ip,
            use_rewrite=use_rewrite,
            normalizer=normalizer,
            aliaser=aliaser,
            priors=engine.priors,
            ce_top=ce_top,
            ce_min=ce_min,
            ce_weight=ce_weight,
            rrf_alpha=rrf_alpha,
            dump_path=dump_path,
        )

        top_docs = [int(r["doc_id"]) for r in res["final_topk"]]
        k = len(top_docs)
        rels = set(gold_ids)
        hits = [1 if d in rels else 0 for d in top_docs]

        p_at_k = sum(hits) / max(1, k)
        r_at_k = (len(rels.intersection(set(top_docs))) / max(1, len(rels))) if rels else 0.0

        dcg = 0.0
        for rank, h in enumerate(hits, start=1):
            if h:
                dcg += 1.0 / math.log2(rank + 1.0)
        ideal = 0.0
        if rels:
            ideal_count = min(len(rels), k)
            for rank in range(1, ideal_count + 1):
                ideal += 1.0 / math.log2(rank + 1.0)
        ndcg = (dcg / ideal) if ideal > 0 else 0.0

        prec_hits += p_at_k
        recall_hits += r_at_k
        ndcg_sum += ndcg
        total_q += 1

    metrics = {
        "Precision@k": float(prec_hits / max(1, total_q)),
        "Recall@k": float(recall_hits / max(1, total_q)),
        "nDCG@k": float(ndcg_sum / max(1, total_q)),
        "queries": int(total_q),
    }
    if run_dir:
        with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
    return metrics

# -------- CLI --------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Assistant from Parquet — hybrid BM25 + FAISS + RRF + CE (+ rewrite + priors) eval"
    )
    p.add_argument("--index_dir", required=True)
    p.add_argument("--doc_index_dir", default=None)
    p.add_argument("--embed_model", default="intfloat/multilingual-e5-base")
    p.add_argument("--ce_model", default="BAAI/bge-reranker-v2-m3")
    p.add_argument("--queries", required=True)
    p.add_argument("--aliases", default=None)
    p.add_argument("--rrf_alpha", type=float, default=60.0)

    # rewrite
    p.add_argument("--use_rewrite", action="store_true")
    p.add_argument("--rewrite_aliases_csv", default=None)
    p.add_argument("--rewrite_max_terms", type=int, default=5)

    # intent policy
    p.add_argument("--intent_policy", default=None)

    # priors
    p.add_argument("--priors_jsonl", default=None)

    # CE params
    p.add_argument("--ce_top", type=int, default=200)
    p.add_argument("--ce_min", type=float, default=0.15)
    p.add_argument("--ce_weight", type=float, default=0.75)

    # dump
    p.add_argument("--dump_eval_dir", default=None)

    return p.parse_args()

def main():
    args = parse_args()

    policy = load_intent_policy(args.intent_policy)
    normalizer = Normalizer() if args.use_rewrite else None
    aliaser = AliasExpander(args.rewrite_aliases_csv) if args.use_rewrite else None

    engine = ParquetHybridEngine(
        index_dir=args.index_dir,
        embed_model=args.embed_model,
        doc_index_dir=args.doc_index_dir,
        ce_model=args.ce_model,
        rrf_alpha=args.rrf_alpha,
        dump_dir=args.dump_eval_dir,
        priors_jsonl=args.priors_jsonl,
    )

    metrics = evaluate(
        engine=engine,
        queries_path=args.queries,
        intent_policy=policy,
        use_rewrite=args.use_rewrite,
        normalizer=normalizer,
        aliaser=aliaser,
        ce_top=args.ce_top,
        ce_min=args.ce_min,
        ce_weight=args.ce_weight,
        rrf_alpha=args.rrf_alpha,
        dump_dir=args.dump_eval_dir,
    )

    print("\n=== EVAL @k ===")
    print(f"Precision@k: {metrics['Precision@k']:.4f}")
    print(f"Recall@k:    {metrics['Recall@k']:.4f}")
    print(f"nDCG@k:      {metrics['nDCG@k']:.4f}")

if __name__ == "__main__":
    main()
