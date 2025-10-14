# -*- coding: utf-8 -*-
from __future__ import annotations
import json, argparse, math
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import pyarrow.parquet as pq

from search.wrrf_fusion import rrf, weighted_rrf

def load_index_dir(index_dir: Path) -> Dict[str, Path]:
    index_dir = index_dir.expanduser().resolve()

    # FAISS
    faiss_cands = [index_dir / "faiss.index", index_dir / "index" / "faiss.index"]
    faiss_path = next((p for p in faiss_cands if p.exists()), None)
    if not faiss_path:
        raise FileNotFoundError(f"FAISS not found in: {[str(p) for p in faiss_cands]}")

    # chunks
    chunks_cands = [index_dir / "chunks.parquet", index_dir.parent / "chunks.parquet"]
    chunks_path = next((p for p in chunks_cands if p.exists()), None)
    if not chunks_path:
        raise FileNotFoundError(f"chunks.parquet not found near {index_dir}")

    # schema → визначаємо колонки
    schema = pq.ParquetFile(chunks_path).schema_arrow
    cols = set(schema.names)
    text_col = "text" if "text" in cols else ("chunk_text" if "chunk_text" in cols else None)
    if text_col is None:
        raise RuntimeError(f"Не знайшов колонку тексту у {chunks_path}; наявні: {sorted(cols)}")

    # колонки для мапи PID→DOC
    doc_idx_col = "doc_idx" if "doc_idx" in cols else None
    doc_id_col  = "doc_id"  if "doc_id"  in cols else None

    # додаткові файли
    doc_ids_cands   = [index_dir / "doc_ids.npy", index_dir.parent / "doc_ids.npy"]
    docs_meta_cands = [index_dir / "docs_meta.parquet", index_dir.parent / "docs_meta.parquet"]
    doc_ids_path = next((p for p in doc_ids_cands   if p.exists()), None)
    docs_meta_path = next((p for p in docs_meta_cands if p.exists()), None)

    return {
        "faiss": faiss_path, "chunks": chunks_path, "text_col": text_col,
        "doc_idx_col": doc_idx_col, "doc_id_col": doc_id_col,
        "doc_ids": doc_ids_path, "docs_meta": docs_meta_path,
    }

def bm25_build(corpus_texts: List[str]):
    tokenized = [t.lower().split() for t in corpus_texts]
    return BM25Okapi(tokenized), tokenized

def bm25_topk(bm25, tokenized, query: str, k: int) -> List[Tuple[int, float]]:
    qtok = query.lower().split()
    scores = bm25.get_scores(qtok)
    idx = np.argpartition(scores, -k)[-k:]
    idx = idx[np.argsort(scores[idx])[::-1]]
    return [(int(i), float(scores[i])) for i in idx]

def dense_topk(index: faiss.Index, qvec: np.ndarray, k: int) -> List[Tuple[int, float]]:
    q = qvec.astype(np.float32)[None, :]
    D, I = index.search(q, k)
    return [(int(i), float(d)) for i, d in zip(I[0], D[0]) if i >= 0]

def load_queries(path: Path) -> List[Dict[str, Any]]:
    out = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line: continue
            out.append(json.loads(line))
    return out

def get_golds(q: Dict[str, Any]) -> List[int]:
    for key in ("gold_doc_ids","gold_ids","golds","relevants","relevant_ids"):
        if key in q and isinstance(q[key], (list, tuple)):
            try: return [int(x) for x in q[key]]
            except: return []
    return []

def detect_gold_level(golds: List[int], num_passages: int, doc_id_series: Optional[np.ndarray]) -> str:
    if not golds: return "none"
    if doc_id_series is not None:
        doc_set = set(np.unique(doc_id_series).tolist())
        if any(g in doc_set for g in golds):
            return "doc"
    if any(0 <= g < num_passages for g in golds):
        return "passage"
    return "unknown"

def dedup_order(seq: List[int]) -> List[int]:
    seen=set(); out=[]
    for x in seq:
        if x is None: continue
        if x in seen: continue
        seen.add(x); out.append(x)
    return out

def metrics_at_k(pred_ids: List[int], golds: List[int], k: int=10) -> Tuple[float,float]:
    if not golds: return math.nan, math.nan
    topk = pred_ids[:k]
    hit = 1.0 if any(p in golds for p in topk) else 0.0
    rr = 0.0
    for rank, p in enumerate(topk, start=1):
        if p in golds:
            rr = 1.0/rank; break
    return hit, rr

def build_doc_id_series(df_chunks: pd.DataFrame, text_col: str,
                        doc_idx_col: Optional[str], doc_id_col: Optional[str],
                        doc_ids_path: Optional[Path], docs_meta_path: Optional[Path]) -> Optional[np.ndarray]:
    """
    Повертає np.ndarray довжиною = #passages, де кожний елемент — DOC_ID для відповідного PID.
    """
    n = len(df_chunks)
    # 1) прямий doc_id у chunks
    if doc_id_col and doc_id_col in df_chunks.columns:
        s = df_chunks[doc_id_col].to_numpy(dtype=np.int64, copy=False)
        if len(s)==n: return s

    # 2) doc_idx у chunks + doc_ids.npy (мапа doc_idx -> DOC_ID)
    if doc_idx_col and doc_idx_col in df_chunks.columns:
        doc_idx = df_chunks[doc_idx_col].to_numpy(dtype=np.int64, copy=False)
        if doc_ids_path and doc_ids_path.exists():
            doc_ids = np.load(str(doc_ids_path))
            if doc_idx.max() < len(doc_ids):
                return doc_ids[doc_idx]
        # якщо doc_ids.npy немає — повертаємо doc_idx як сурогат doc_id
        return doc_idx

    # 3) відновлення з docs_meta.parquet (якщо є counts per doc)
    if docs_meta_path and docs_meta_path.exists():
        dfm = pd.read_parquet(docs_meta_path)
        # шукаємо колонку з кількістю чанків по документу
        count_col = None
        for c in ("num_chunks","chunks","n_chunks","chunk_count"):
            if c in dfm.columns:
                count_col = c; break
        if count_col is None:
            return None
        counts = dfm[count_col].astype(int).tolist()
        doc_idx = np.repeat(np.arange(len(counts), dtype=np.int64), counts)
        if len(doc_idx) != n:
            return None
        if doc_ids_path and doc_ids_path.exists():
            doc_ids = np.load(str(doc_ids_path))
            if len(doc_ids) == len(counts):
                return doc_ids[doc_idx]
        return doc_idx

    return None

def main():
    ap = argparse.ArgumentParser("WRRF eval (FAISS+BM25, no CE) with doc-level golds")
    ap.add_argument("--index_dir", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--embed_model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--k_dense", type=int, default=100)
    ap.add_argument("--k_bm25", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=60.0)
    ap.add_argument("--w_bm25", type=float, default=1.0)
    ap.add_argument("--w_dense", type=float, default=1.0)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--out_csv", default="data/eval/wrrf_preds.csv")
    ap.add_argument("--metrics_csv", default="data/eval/wrrf_metrics.csv")
    args = ap.parse_args()

    paths = load_index_dir(Path(args.index_dir))
    print(f"[INFO] FAISS: {paths['faiss']}")
    print(f"[INFO] chunks: {paths['chunks']} (text_col={paths['text_col']})")

    index = faiss.read_index(str(paths["faiss"]))
    print(f"[INFO] FAISS dim={index.d}")

    cols_to_read = [paths["text_col"]]
    if paths["doc_idx_col"]: cols_to_read.append(paths["doc_idx_col"])
    if paths["doc_id_col"] and paths["doc_id_col"] not in cols_to_read:
        cols_to_read.append(paths["doc_id_col"])

    df = pd.read_parquet(paths["chunks"], columns=cols_to_read)
    texts = df[paths["text_col"]].astype(str).tolist()
    num_passages = len(texts)
    print(f"[INFO] passages={num_passages:,}")

    doc_id_series = build_doc_id_series(
        df, paths["text_col"], paths["doc_idx_col"], paths["doc_id_col"],
        paths.get("doc_ids"), paths.get("docs_meta")
    )
    if doc_id_series is not None:
        print(f"[INFO] PID→DOC map ready: shape={doc_id_series.shape}")
    else:
        print("[WARN] Не вдалося побудувати PID→DOC мапу — doc-level метрики недоступні.")

    bm25, tokenized = bm25_build(texts)
    enc = SentenceTransformer(args.embed_model)

    queries = load_queries(Path(args.queries))
    print(f"[INFO] queries={len(queries)}")

    rows, met = [], []
    for qi, q in enumerate(tqdm(queries, desc="Eval")):
        qid = q.get("id") or q.get("qid") or q.get("query_id") or str(qi+1)
        qtext = q.get("query") or q.get("text") or q.get("q") or ""
        if not qtext:
            continue

        qvec = enc.encode("query: " + qtext, normalize_embeddings=True)
        dense = dense_topk(index, np.array(qvec), args.k_dense)
        bm = bm25_topk(bm25, tokenized, qtext, args.k_bm25)

        ranks_dense = {pid: r+1 for r,(pid,_) in enumerate(dense)}
        ranks_bm25  = {pid: r+1 for r,(pid,_) in enumerate(bm)}

        fused_rrf  = rrf(ranks_bm25, ranks_dense, alpha=args.alpha)[:args.topk]
        fused_wrrf = weighted_rrf(ranks_bm25, ranks_dense, alpha=args.alpha,
                                  w_bm25=args.w_bm25, w_dense=args.w_dense)[:args.topk]

        golds = get_golds(q)
        level = detect_gold_level(golds, num_passages, doc_id_series)

        # пасажові топи
        pred_pass_rrf  = [pid for pid,_ in fused_rrf]
        pred_pass_wrrf = [pid for pid,_ in fused_wrrf]

        # метрики
        if level == "doc" and doc_id_series is not None:
            pred_doc_rrf  = dedup_order([int(doc_id_series[pid]) for pid in pred_pass_rrf])
            pred_doc_wrrf = dedup_order([int(doc_id_series[pid]) for pid in pred_pass_wrrf])
            p_rrf, m_rrf  = metrics_at_k(pred_doc_rrf,  golds, k=args.topk)
            p_w,   m_w    = metrics_at_k(pred_doc_wrrf, golds, k=args.topk)
        elif level == "passage":
            p_rrf, m_rrf  = metrics_at_k(pred_pass_rrf,  golds, k=args.topk)
            p_w,   m_w    = metrics_at_k(pred_pass_wrrf, golds, k=args.topk)
        else:
            p_rrf = m_rrf = p_w = m_w = math.nan

        met.append({
            "qid": qid, "gold_level": level,
            f"P@{args.topk}": p_w, f"MRR@{args.topk}": m_w,
            f"P@{args.topk}_rrf": p_rrf, f"MRR@{args.topk}_rrf": m_rrf
        })

        rows.append({
            "qid": qid, "query": qtext, "gold_level": level,
            "alpha": args.alpha, "w_bm25": args.w_bm25, "w_dense": args.w_dense,
            "rrf_pids":  [int(pid) for pid,_ in fused_rrf],
            "wrrf_pids": [int(pid) for pid,_ in fused_wrrf],
        })

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_csv, index=False, sep=",")
    print(f"[OK] predictions -> {args.out_csv}")

    pd.DataFrame(met).to_csv(args.metrics_csv, index=False, sep=",")
    print(f"[OK] metrics -> {args.metrics_csv}")
if __name__ == "__main__":
    main()
