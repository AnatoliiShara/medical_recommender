#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import time
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import faiss
import pyarrow.parquet as pq

from sentence_transformers import SentenceTransformer


QUERY_FIELDS = ["query", "text", "original_query", "q", "question", "user_query"]
QID_FIELDS = ["qid", "query_id", "id", "uuid"]
GOLD_FIELDS = ["gold_doc_ids", "gold", "gold_ids", "relevant_doc_ids", "relevant_ids"]


def _first_present(d: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for k in keys:
        if k in d and d[k] is not None and str(d[k]).strip() != "":
            return k
    return None


def _extract_gold(d: Dict[str, Any]) -> List[str]:
    k = _first_present(d, GOLD_FIELDS)
    if not k:
        return []
    v = d.get(k)
    if v is None:
        return []
    # allow single str, list[str], list[int], etc.
    if isinstance(v, (str, int)):
        return [str(v)]
    if isinstance(v, list):
        out = []
        for x in v:
            if x is None:
                continue
            out.append(str(x))
        return out
    return []


def _maybe_prefix(texts: List[str], mode: str, kind: str) -> List[str]:
    """
    mode:
      - none: do nothing
      - e5: add 'query: ' for queries, 'passage: ' for docs (here we only use queries)
      - auto: if kind=query and text doesn't start with 'query:' -> add for e5-ish models
    """
    if mode == "none":
        return texts
    if mode == "e5":
        pref = "query: " if kind == "query" else "passage: "
        return [t if t.lower().startswith(pref.strip()) else pref + t for t in texts]
    if mode == "auto":
        # conservative: only prefix queries; docs already embedded in index
        if kind != "query":
            return texts
        pref = "query: "
        return [t if t.lower().startswith("query:") else pref + t for t in texts]
    raise ValueError(f"Unknown prefix_mode: {mode}")


def _load_docs_meta(docs_meta_path: str) -> Dict[str, str]:
    # optional mapping doc_id -> drug_name (for debugging)
    if not docs_meta_path or not os.path.exists(docs_meta_path):
        return {}
    t = pq.read_table(docs_meta_path, columns=[c for c in ["doc_id", "drug_name"] if c in pq.ParquetFile(docs_meta_path).schema.names])
    cols = t.column_names
    if "doc_id" not in cols:
        return {}
    name_col = "drug_name" if "drug_name" in cols else None
    if not name_col:
        return {}
    doc_ids = [str(x) for x in t["doc_id"].to_pylist()]
    names = ["" if x is None else str(x) for x in t[name_col].to_pylist()]
    return dict(zip(doc_ids, names))


def _read_queries_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception as e:
                raise RuntimeError(f"Bad JSONL at {path}:{line_no} -> {e}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", required=True, help="Directory with faiss.index + doc_ids.npy (and optional docs_meta.parquet)")
    ap.add_argument("--model", required=True, help="SentenceTransformer model name or path (for QUERY encoder)")
    ap.add_argument("--queries", required=True, help="Queries JSONL with gold doc_ids")
    ap.add_argument("--out", required=True, help="Output predictions JSONL")
    ap.add_argument("--top_k", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--prefix_mode", choices=["none", "e5", "auto"], default="e5")
    args = ap.parse_args()

    faiss_path = os.path.join(args.index_dir, "faiss.index")
    doc_ids_path = os.path.join(args.index_dir, "doc_ids.npy")
    docs_meta_path = os.path.join(args.index_dir, "docs_meta.parquet")

    if not os.path.exists(faiss_path):
        raise FileNotFoundError(f"Missing {faiss_path}")
    if not os.path.exists(doc_ids_path):
        raise FileNotFoundError(f"Missing {doc_ids_path}")

    print(f"[INFO] Loading FAISS: {faiss_path}")
    index = faiss.read_index(faiss_path)
    print(f"[INFO] FAISS ntotal={index.ntotal} dim={index.d}")

    doc_ids = np.load(doc_ids_path, allow_pickle=True).tolist()
    if len(doc_ids) != index.ntotal:
        raise ValueError(f"doc_ids.npy length {len(doc_ids)} != faiss.ntotal {index.ntotal}")

    doc_id_str = [str(x) for x in doc_ids]
    docname = _load_docs_meta(docs_meta_path)

    print(f"[INFO] Loading queries: {args.queries}")
    queries = _read_queries_jsonl(args.queries)
    print(f"[INFO] Queries loaded: {len(queries)}")

    q_text_key = None
    q_id_key = None
    # infer keys from first non-empty
    for q in queries:
        q_text_key = _first_present(q, QUERY_FIELDS)
        q_id_key = _first_present(q, QID_FIELDS)
        if q_text_key:
            break
    if not q_text_key:
        raise ValueError(f"Cannot find query text field among {QUERY_FIELDS} in {args.queries}")

    print(f"[INFO] Using query_text field: {q_text_key} | qid field: {q_id_key or '(auto)'}")

    print(f"[INFO] Loading model: {args.model} | device={args.device} | max_length={args.max_length}")
    model = SentenceTransformer(args.model, device=args.device)
    # best-effort max length
    try:
        model.max_seq_length = args.max_length
    except Exception:
        pass

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    t0 = time.time()
    written = 0

    # encode in batches
    q_texts_all = [str(q[q_text_key]) for q in queries]
    q_texts_all = _maybe_prefix(q_texts_all, args.prefix_mode, kind="query")

    # SentenceTransformer.encode returns np.ndarray (float32/float64 depending)
    embs = model.encode(
        q_texts_all,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,  # we normalize ourselves to match IndexFlatIP+cos
    ).astype("float32")

    # L2 normalize -> cosine similarity via IP
    faiss.normalize_L2(embs)

    # search
    D, I = index.search(embs, args.top_k)

    # write predictions
    with open(args.out, "w", encoding="utf-8") as f:
        for qi, q in enumerate(queries):
            qid = str(q.get(q_id_key)) if q_id_key and q.get(q_id_key) is not None else f"q{qi:05d}"
            qtext = str(q[q_text_key])

            hits = []
            for rank, (idx, score) in enumerate(zip(I[qi].tolist(), D[qi].tolist()), start=1):
                if idx < 0:
                    continue
                did = doc_id_str[idx]
                hit = {"rank": rank, "doc_id": did, "score": float(score)}
                if did in docname:
                    hit["drug_name"] = docname[did]
                hits.append(hit)

            out_obj = {
                "qid": qid,
                "query": qtext,
                "gold_doc_ids": _extract_gold(q),
                "hits": hits,
            }
            f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            written += 1

    dt = time.time() - t0
    # quick sanity summary
    finite_scores = np.isfinite(D)
    print(f"[OK] Wrote predictions: {args.out}")
    print(f"[INFO] Done: {written} queries | top_k={args.top_k} | time={dt:.2f}s")
    print(f"[INFO] Scores: min={float(np.min(D[finite_scores])):.6f} mean={float(np.mean(D[finite_scores])):.6f} max={float(np.max(D[finite_scores])):.6f}")


if __name__ == "__main__":
    main()
