# -*- coding: utf-8 -*-
"""
assistant_from_parquet.py — P0 end-to-end search pipeline (raw/norm split).

Pipeline:
  Load corpus (parquet) → build doc blobs (RAW)
  Clinical assets (lexicons, patterns, inverted index; RAW+NORM)
  BM25 over NORM tokens
  Dense (E5) FAISS over RAW (load or build & autosave)
  Fusion (weighted or WRRF) + optional UNION boost/force-include
  CrossEncoder rerank on RAW
  CE-bias from condition-specific positives/penalties
  Output TOP-N (JSONL)
"""

from __future__ import annotations
import os
import sys
import json
import math
import argparse
from typing import List, Tuple, Dict, Any, Optional, Iterable, Set

import numpy as np
import pandas as pd

# --- External libs
try:
    import faiss
except Exception:
    faiss = None
try:
    import torch
    from sentence_transformers import SentenceTransformer, CrossEncoder
except Exception:
    SentenceTransformer = None
    CrossEncoder = None
    torch = None

# === P0 clinical imports ===
from src.search.clinical.p0_runtime import (
    build_clinical_assets,
    detect_conditions,
    union_candidates_for_conditions,
    ce_bias_for_doc,
)
from src.search.clinical.normalizer import (
    normalize_text, tokenize, normalize_tokens, tokens  # tokens: alias for compatibility
)
# === end P0 clinical imports ===


# ---------------------------
# Utilities / IO
# ---------------------------
def read_parquet_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset parquet not found: {path}")
    return pd.read_parquet(path)


def build_doc_blob_raw(row: pd.Series, max_chars: Optional[int] = None) -> str:
    """
    Compose a RAW search text blob from typical columns in Compendium.
    NO normalization here — keep natural language for Dense/CE and regex.
    """
    fields = [
        "Назва препарату", "Назва", "Name",
        "Форма", "Лікарська форма",
        "Показання", "Показання до застосування",
        "Склад", "Склад (qual/quant)",
        "Фармакотерапевтична група", "Фармакологічні властивості",
        "Протипоказання", "Особливості застосування",
        "Спосіб застосування та дози", "Вагітність", "Вплив на реакцію",
        "Текст",  # іноді зустрічається зібране поле
    ]
    parts: List[str] = []
    for f in fields:
        if f in row and isinstance(row[f], str) and row[f].strip():
            parts.append(row[f].strip())
    blob = " \n".join(parts)
    if max_chars and len(blob) > max_chars:
        blob = blob[:max_chars]
    return blob


def load_queries(path: str) -> List[Dict[str, Any]]:
    """
    Expects JSONL with at least {"query": "..."}.
    """
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                obj = {"query": line}
            if "query" not in obj:
                for k in ("q", "text", "question"):
                    if k in obj:
                        obj["query"] = obj[k]
                        break
            if "query" in obj:
                out.append(obj)
    return out


# ---------------------------
# BM25 (Okapi)
# ---------------------------
class BM25Okapi:
    def __init__(self, docs_tokens: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.docs_tokens = docs_tokens
        self.N = len(docs_tokens)
        self.k1 = k1
        self.b = b

        self.doc_freq: Dict[str, int] = {}
        self.doc_len = np.array([len(doc) for doc in docs_tokens], dtype=np.float32)
        self.avgdl = float(self.doc_len.mean()) if self.N else 0.0

        self.tf: List[Dict[str, int]] = []
        for toks in docs_tokens:
            tf_i: Dict[str, int] = {}
            for t in toks:
                tf_i[t] = tf_i.get(t, 0) + 1
            self.tf.append(tf_i)
            for t in tf_i.keys():
                self.doc_freq[t] = self.doc_freq.get(t, 0) + 1

        self.idf: Dict[str, float] = {}
        for t, df in self.doc_freq.items():
            self.idf[t] = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)

    def _score_doc(self, q_tokens: List[str], i: int) -> float:
        score = 0.0
        tf_i = self.tf[i]
        dl = self.doc_len[i] if self.doc_len.size else 0.0
        for t in q_tokens:
            if t not in tf_i:
                continue
            f = tf_i[t]
            denom = f + self.k1 * (1.0 - self.b + self.b * (dl / (self.avgdl + 1e-9)))
            score += self.idf.get(t, 0.0) * ((f * (self.k1 + 1.0)) / (denom + 1e-9))
        return score

    def search(self, query: str, top_k: int = 50) -> List[Tuple[int, float]]:
        # IMPORTANT: BM25 works over normalized tokens
        q_toks = normalize_tokens(tokenize(query))
        if not q_toks or self.N == 0:
            return []
        scores = np.zeros(self.N, dtype=np.float32)
        for i in range(self.N):
            scores[i] = self._score_doc(q_toks, i)
        if top_k >= self.N:
            idx = np.argsort(-scores)
        else:
            idx = np.argpartition(-scores, top_k - 1)[:top_k]
            idx = idx[np.argsort(-scores[idx])]
        return [(int(i), float(scores[i])) for i in idx if scores[i] > 0.0]


# ---------------------------
# Dense Index (FAISS over RAW)
# ---------------------------
class DenseIndex:
    def __init__(self, model_name: str, faiss_index_path: Optional[str] = None):
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.index = None
        self.dim = None
        self.embeddings: Optional[np.ndarray] = None
        self.faiss_index_path = faiss_index_path
        self.ntotal: Optional[int] = None

    def _ensure_model(self):
        if self.model is None:
            if SentenceTransformer is None:
                raise RuntimeError("sentence_transformers not available")
            self.model = SentenceTransformer(self.model_name)

    def build_or_load(self, docs_raw: List[str], rebuild_if_missing: bool = True):
        # Try load
        if self.faiss_index_path and os.path.exists(self.faiss_index_path) and faiss is not None:
            self.index = faiss.read_index(self.faiss_index_path)
            self.dim = self.index.d
            try:
                self.ntotal = int(self.index.ntotal)
            except Exception:
                self.ntotal = None
            print(f"[FAISS] loaded index: {self.faiss_index_path}", file=sys.stderr)
            return

        if not rebuild_if_missing:
            raise FileNotFoundError(f"FAISS index not found: {self.faiss_index_path}")

        # Build from scratch on RAW docs
        self._ensure_model()
        with torch.no_grad():
            embs = self.model.encode(
                docs_raw, batch_size=64, convert_to_numpy=True,
                show_progress_bar=True, normalize_embeddings=True
            )
        self.embeddings = embs
        self.dim = embs.shape[1]
        if faiss is None:
            raise RuntimeError("faiss not available to build index")
        index = faiss.IndexFlatIP(self.dim)
        index.add(embs)
        self.index = index
        try:
            self.ntotal = int(self.index.ntotal)
        except Exception:
            self.ntotal = len(docs_raw)

        # autosave
        if self.faiss_index_path:
            os.makedirs(os.path.dirname(self.faiss_index_path), exist_ok=True)
            try:
                faiss.write_index(self.index, self.faiss_index_path)
                print(f"[FAISS] saved index to: {self.faiss_index_path}", file=sys.stderr)
            except Exception as e:
                print(f"[WARN] failed to save FAISS index: {e}", file=sys.stderr)

    def search(self, query_raw: str, top_k: int = 50) -> List[Tuple[int, float]]:
        self._ensure_model()
        with torch.no_grad():
            q = self.model.encode([query_raw], convert_to_numpy=True, normalize_embeddings=True)
        if self.index is None:
            raise RuntimeError("Dense index is not built/loaded")
        D, I = self.index.search(q, top_k)
        res = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            res.append((int(idx), float(score)))
        return res


# ---------------------------
# Fusion
# ---------------------------
def minmax_norm(scores: Dict[int, float]) -> Dict[int, float]:
    if not scores:
        return {}
    vals = np.array(list(scores.values()), dtype=np.float32)
    lo, hi = float(vals.min()), float(vals.max())
    if hi <= lo + 1e-9:
        return {k: 0.5 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def rrf_fuse(rank_lists: List[List[int]], alpha: int = 60, weights: Optional[List[float]] = None) -> Dict[int, float]:
    if weights is None:
        weights = [1.0] * len(rank_lists)
    scores: Dict[int, float] = {}
    for rlist, w in zip(rank_lists, weights):
        for r, doc_id in enumerate(rlist):
            scores[doc_id] = scores.get(doc_id, 0.0) + w * (1.0 / (alpha + r + 1))
    return scores


def fuse_scores(
    bm25: List[Tuple[int, float]],
    dense: List[Tuple[int, float]],
    method: str = "weighted",
    w_bm25: float = 0.7,
    w_dense: float = 0.3,
    norm: str = "minmax",
    rrf_alpha: int = 60,
) -> Dict[int, float]:
    bdict = {i: s for i, s in bm25}
    ddict = {i: s for i, s in dense}
    if method == "wrrf":
        b_rank = [i for i, _ in bm25]
        d_rank = [i for i, _ in dense]
        return rrf_fuse([b_rank, d_rank], alpha=rrf_alpha, weights=[w_bm25, w_dense])

    if norm == "minmax":
        bnorm = minmax_norm(bdict)
        dnorm = minmax_norm(ddict)
    else:
        bnorm, dnorm = bdict, ddict

    doc_ids = set(bdict) | set(ddict)
    fused: Dict[int, float] = {}
    for i in doc_ids:
        fused[i] = w_bm25 * bnorm.get(i, 0.0) + w_dense * dnorm.get(i, 0.0)
    return fused


# ---------------------------
# CE rerank (RAW)
# ---------------------------
class CEReranker:
    def __init__(self, model_name: str):
        if CrossEncoder is None:
            raise RuntimeError("sentence_transformers CrossEncoder not available")
        self.model = CrossEncoder(model_name, max_length=512)

    def rerank(self, query_raw: str, docs_raw: List[str], batch_size: int = 16) -> List[float]:
        pairs = [(query_raw, d) for d in docs_raw]
        scores = self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        return [float(x) for x in scores]


# ---------------------------
# Helpers
# ---------------------------
def _filter_valid(results: List[Tuple[int, float]], n_docs: int) -> List[Tuple[int, float]]:
    return [(i, s) for (i, s) in results if 0 <= i < n_docs]


def _filter_ids(ids: Iterable[int], n_docs: int) -> List[int]:
    return [i for i in ids if 0 <= i < n_docs]


# ---------------------------
# Main pipeline
# ---------------------------
def run_pipeline(args: argparse.Namespace):
    # 1) Load dataset
    df = read_parquet_dataset(args.dataset)

    # 2) Build RAW/NORM docs
    max_chars = args.max_doc_chars if args.max_doc_chars is not None else None
    doc_texts_raw: List[str] = [build_doc_blob_raw(row, max_chars=max_chars) for _, row in df.iterrows()]
    # NORM only for BM25 + dictionaries (normalizer handles tokenization/lemma)
    doc_texts_norm: List[str] = [normalize_text(t) for t in doc_texts_raw]
    N = len(doc_texts_raw)

    # 3) Clinical assets (build from RAW; NORM computed inside as well)
    clinical = build_clinical_assets(doc_texts_raw, dict_root=args.dict_root)

    # 4) BM25 on NORM tokens
    bm25_index = BM25Okapi([normalize_tokens(tokenize(t)) for t in doc_texts_raw])

    # 5) Dense index (RAW) load or build FAISS
    dense = DenseIndex(model_name=args.embed_model, faiss_index_path=args.faiss_index)
    dense.build_or_load(doc_texts_raw, rebuild_if_missing=True)
    if dense.ntotal is not None and dense.ntotal != N:
        print(f"[WARN] FAISS index ntotal={dense.ntotal} != corpus={N}. "
              f"Filtering out-of-range ids.", file=sys.stderr)

    # 6) CE model (RAW)
    ce = CEReranker(args.ce_model)

    # 7) Load queries
    queries = load_queries(args.queries)
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as fout:
        for qi, qobj in enumerate(queries):
            q_raw = str(qobj["query"]).strip()
            q_norm = normalize_text(q_raw)
            print(f"\n=== Query[{qi}] {q_norm} ===", file=sys.stderr)

            # Detect clinical conditions (P0) — function normalizes всередині
            matched_conditions = detect_conditions(q_raw, clinical)
            if matched_conditions:
                print(f"[P0] clinical matched: {matched_conditions}", file=sys.stderr)

            # Retrieve
            bm25_top_raw = bm25_index.search(q_norm, top_k=args.top_k)  # BM25 on NORM
            bm25_top = _filter_valid(bm25_top_raw, N)
            bm25_top_ids: Set[int] = {i for i, _ in bm25_top}

            try:
                dense_top_raw = dense.search(q_raw, top_k=args.top_k)     # Dense on RAW
            except Exception as e:
                print(f"[WARN] Dense search failed: {e}. Falling back to BM25-only.", file=sys.stderr)
                dense_top_raw = []
            dense_top = _filter_valid(dense_top_raw, N)

            # Heuristic UNION (condition → docs)
            union_ids_set = union_candidates_for_conditions(
                assets=clinical,
                matched_conditions=matched_conditions,
                bm25_top_ids=bm25_top_ids,
                union_cap=args.heuristic_union_cap,
                force_include=args.union_force_include,
            )
            union_ids = set(_filter_ids(union_ids_set, N))
            if union_ids:
                print(f"[P0] heuristic UNION CAPPED added: {len(union_ids)}", file=sys.stderr)

            # Fusion
            fused = fuse_scores(
                bm25=bm25_top,
                dense=dense_top,
                method=("wrrf" if args.fusion.lower() == "wrrf" else "weighted"),
                w_bm25=args.w_bm25,
                w_dense=args.w_dense,
                norm=args.norm,
                rrf_alpha=args.rrf_alpha,
            )
            # Soft boost to UNION docs (so they surface into candidate cut)
            if union_ids and args.union_boost > 0.0:
                for did in union_ids:
                    fused[did] = fused.get(did, 0.0) + float(args.union_boost)

            # Candidates for CE
            cand_sorted = sorted(fused.items(), key=lambda x: x[1], reverse=True)
            cand_ids: List[int] = [i for i, _ in cand_sorted[:max(args.rerank_top, 1)]]
            cand_ids = _filter_ids(cand_ids, N)

            # Optionally force-include UNION docs into CE set
            if args.union_force_include and union_ids:
                merged: List[int] = []
                seen: Set[int] = set()
                for i in cand_ids + list(union_ids):
                    if i not in seen:
                        seen.add(i)
                        merged.append(i)
                cand_ids = merged[:max(args.rerank_top, 1)]

            if not cand_ids and bm25_top:
                cand_ids = _filter_ids([i for i, _ in bm25_top[:args.rerank_top]], N)
            if not cand_ids:
                fout.write(json.dumps({"query": q_raw, "matched_conditions": matched_conditions, "top": []},
                                      ensure_ascii=False) + "\n")
                print("[WARN] No candidates after filtering; emitted empty result.", file=sys.stderr)
                continue

            # CE on RAW
            cand_texts_raw = [doc_texts_raw[i] for i in cand_ids]
            ce_scores = ce.rerank(q_raw, cand_texts_raw, batch_size=args.ce_batch)

            # CE-bias (RAW doc; patterns & norm come from assets)
            B_POS = float(getattr(args, "heuristic_ce_bias_pos", 0.15))
            B_PEN = float(getattr(args, "heuristic_ce_bias_pen", 0.20))
            ce_scored: List[Tuple[int, float]] = []
            for doc_id, ce_s in zip(cand_ids, ce_scores):
                delta = ce_bias_for_doc(
                    doc_text=doc_texts_raw[doc_id],
                    assets=clinical,
                    matched_conditions=matched_conditions,
                    bias_pos=B_POS,
                    bias_pen=B_PEN,
                )
                ce_scored.append((doc_id, ce_s + delta))
            ce_scored.sort(key=lambda x: x[1], reverse=True)

            # Output
            top_n = min(10, len(ce_scored))
            top_items = ce_scored[:top_n]
            recs = []
            for rank, (doc_id, score) in enumerate(top_items, start=1):
                title = ""
                if "Назва препарату" in df.columns:
                    try:
                        title = str(df.iloc[doc_id]["Назва препарату"])
                    except Exception:
                        title = ""
                elif "Назва" in df.columns:
                    try:
                        title = str(df.iloc[doc_id]["Назва"])
                    except Exception:
                        title = ""
                recs.append({
                    "rank": rank,
                    "doc_id": int(doc_id),
                    "score": float(score),
                    "title": title,
                })
            out = {"query": q_raw, "matched_conditions": matched_conditions, "top": recs}
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            print(f"[OUT] wrote {len(recs)} items", file=sys.stderr)

    print(f"\nSaved results to: {args.out_json}", file=sys.stderr)


# ---------------------------
# CLI
# ---------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("assistant_from_parquet (P0, raw/norm split)")

    # Core inputs
    p.add_argument("--dataset", type=str, default="data/raw/compendium_all.parquet",
                   help="Path to compendium_all.parquet")
    p.add_argument("--queries", type=str, required=True, help="JSONL with {'query': ...}")
    p.add_argument("--out_json", type=str, default="data/eval/predictions.jsonl")

    # Retrieval & models
    p.add_argument("--embed_model", type=str, default="intfloat/multilingual-e5-base")
    p.add_argument("--faiss_index", type=str, default="",
                   help="Path to faiss index. If missing, will be built & autosaved.")
    p.add_argument("--ce_model", type=str, default="BAAI/bge-reranker-v2-m3")
    p.add_argument("--top_k", type=int, default=60)
    p.add_argument("--rerank_top", type=int, default=25)
    p.add_argument("--ce_batch", type=int, default=8)

    # Fusion
    p.add_argument("--fusion", type=str, default="weighted", choices=["weighted", "wrrf"])
    p.add_argument("--norm", type=str, default="minmax", choices=["minmax", "none"])
    p.add_argument("--w_bm25", type=float, default=0.55)
    p.add_argument("--w_dense", type=float, default=0.45)
    p.add_argument("--rrf_alpha", type=int, default=60)
    p.add_argument("--temperature", type=float, default=1.0)

    # P0 clinical config
    p.add_argument("--dict_root", type=str, default="data/dicts/clinical")
    p.add_argument("--heuristic_union_cap", type=int, default=20)
    p.add_argument("--heuristic_ce_bias_pos", type=float, default=0.15)
    p.add_argument("--heuristic_ce_bias_pen", type=float, default=0.20)
    p.add_argument("--union_force_include", action="store_true",
                   help="Force include UNION doc ids into CE candidate set")
    p.add_argument("--union_boost", type=float, default=0.05,
                   help="Soft boost added to fused score for UNION docs")

    # Text building
    p.add_argument("--max_doc_chars", nargs="?", const=4000, type=int, default=None,
                   help="Trim RAW doc blob to this many chars (default: None). If flag used without value, uses 4000.")

    # Back-compat no-op flags
    p.add_argument("--index_dir", type=str, default="")
    p.add_argument("--doc_index_dir", type=str, default="")
    p.add_argument("--use_rewrite", action="store_true")
    p.add_argument("--rewrite_aliases_csv", type=str, default="")
    p.add_argument("--rewrite_max_terms", type=int, default=5)
    p.add_argument("--intent_policy", type=str, default="")
    p.add_argument("--dump_eval_dir", type=str, default="")

    return p


def main():
    args = build_argparser().parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
