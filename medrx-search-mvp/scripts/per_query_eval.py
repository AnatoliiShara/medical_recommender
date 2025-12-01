#!/usr/bin/env python3
"""
Per-query evaluator for predictions.jsonl.

Robustness goals:
- gold_doc_ids and hit doc_ids are ALWAYS compared as strings
- hits can be:
  - list[dict] with {"doc_id": ..., "score": ..., "rank": ...}
  - list[str|int] (doc_id directly, or row_idx)
- optional idx2docid map: row_idx -> real doc_id (string)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows


def canon_doc_id(x: Any) -> Optional[str]:
    if x is None:
        return None
    # unwrap dicts like {"doc_id": ...}
    if isinstance(x, dict):
        x = x.get("doc_id", None)
        if x is None:
            return None
    # bool is subclass of int; but doc_id should not be True/False
    if isinstance(x, bool):
        return str(int(x))
    return str(x).strip()


def load_idx2docid(path: Optional[str]) -> Optional[Dict[int, str]]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"--idx2docid not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    # allow { "0": "6875", "1": "..." } or {0: "..."}
    out: Dict[int, str] = {}
    for k, v in obj.items():
        try:
            ki = int(k)
        except Exception:
            continue
        cv = canon_doc_id(v)
        if cv is not None:
            out[ki] = cv
    return out


def extract_qid(row: Dict[str, Any]) -> str:
    # accept qid / id / query_id in different files
    for k in ("qid", "id", "query_id"):
        if k in row and row[k] is not None:
            return str(row[k])
    # fallback: stable-ish
    return str(row.get("query", row.get("text", "")))[:50]


def extract_query_text(row: Dict[str, Any]) -> str:
    for k in ("query", "text", "q", "question"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def extract_gold_ids(row: Dict[str, Any]) -> List[str]:
    # support multiple possible field names
    cand = None
    for k in ("gold_doc_ids", "gold", "gold_ids", "gold_docid", "gold_doc_id"):
        if k in row:
            cand = row[k]
            break
    if cand is None:
        return []

    # normalize to list
    if isinstance(cand, (str, int)):
        cand = [cand]
    if not isinstance(cand, list):
        return []

    out: List[str] = []
    for x in cand:
        cx = canon_doc_id(x)
        if cx is not None and cx != "":
            out.append(cx)
    # keep unique but stable order
    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


def extract_hit_doc_ids(
    hits: Any,
    *,
    idx2docid: Optional[Dict[int, str]] = None,
    offset: int = 0,
) -> List[str]:
    if hits is None:
        return []
    if not isinstance(hits, list):
        return []

    out: List[str] = []
    for h in hits:
        # dict hit
        if isinstance(h, dict):
            did = canon_doc_id(h.get("doc_id"))
            if did is None:
                # sometimes {doc:..} etc
                did = canon_doc_id(h.get("doc"))
            if did is not None:
                out.append(did)
            continue

        # direct doc_id or row_idx
        if idx2docid is not None:
            # treat as row index first
            try:
                ridx = int(h) + int(offset)
                mapped = idx2docid.get(ridx)
                if mapped is not None:
                    out.append(mapped)
                    continue
            except Exception:
                pass

        did = canon_doc_id(h)
        if did is not None:
            out.append(did)

    return out


def recall_at_k(hit_ids: List[str], gold_set: set[str], k: int) -> float:
    if not gold_set:
        return 0.0
    top = hit_ids[:k]
    got = sum(1 for x in top if x in gold_set)
    return got / float(len(gold_set))


def precision_at_k(hit_ids: List[str], gold_set: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = hit_ids[:k]
    got = sum(1 for x in top if x in gold_set)
    return got / float(k)


def first_hit_rank(hit_ids: List[str], gold_set: set[str]) -> int:
    # 1-based rank; 0 means no hit anywhere
    for i, x in enumerate(hit_ids, start=1):
        if x in gold_set:
            return i
    return 0


def mrr_from_first_rank(r: int) -> float:
    return 0.0 if r <= 0 else 1.0 / float(r)


def ndcg_at_k(hit_ids: List[str], gold_set: set[str], k: int) -> float:
    # binary relevance; DCG = sum(1/log2(i+1))
    # IDCG = best possible with min(len(gold), k) ones at top
    import math

    if not gold_set or k <= 0:
        return 0.0

    top = hit_ids[:k]
    dcg = 0.0
    for i, x in enumerate(top, start=1):
        if x in gold_set:
            dcg += 1.0 / math.log2(i + 1.0)

    m = min(len(gold_set), k)
    idcg = sum(1.0 / math.log2(i + 1.0) for i in range(1, m + 1))
    return 0.0 if idcg == 0.0 else dcg / idcg


def main():
    ap = argparse.ArgumentParser("Per-query evaluator for predictions.jsonl")
    ap.add_argument("--pred", required=True, help="predictions.jsonl (with hits)")
    ap.add_argument("--queries", default=None, help="optional queries jsonl to take gold from (qid/id + gold_*)")
    ap.add_argument("--idx2docid", default=None, help="optional map row_idx->real_doc_id json")
    ap.add_argument("--offset", type=int, default=0, help="row_idx offset (0 means hit doc_id is 0-based idx)")
    ap.add_argument("--ks", type=int, nargs="+", default=[10, 20, 50, 100, 200], help="K values")
    ap.add_argument("--out_csv", required=True, help="output per-query csv")
    args = ap.parse_args()

    pred_rows = read_jsonl(args.pred)
    idx2docid = load_idx2docid(args.idx2docid)

    # optional queries file for gold/query text
    gold_by_qid: Dict[str, List[str]] = {}
    query_text_by_qid: Dict[str, str] = {}
    if args.queries:
        qrows = read_jsonl(args.queries)
        for r in qrows:
            qid = extract_qid(r)
            gold = extract_gold_ids(r)
            if gold:
                gold_by_qid[qid] = gold
            qt = extract_query_text(r)
            if qt:
                query_text_by_qid[qid] = qt

    out: List[Dict[str, Any]] = []
    miss_any = 0
    with_gold = 0

    for r in pred_rows:
        qid = extract_qid(r)
        query = extract_query_text(r) or query_text_by_qid.get(qid, "")

        # gold can be inside pred row OR from queries file
        gold = extract_gold_ids(r)
        if not gold and gold_by_qid.get(qid):
            gold = gold_by_qid[qid]

        gold = [g for g in (canon_doc_id(x) for x in gold) if g]
        gold_set = set(gold)

        if gold_set:
            with_gold += 1

        hits = r.get("hits", r.get("documents", r.get("docs")))
        hit_ids = extract_hit_doc_ids(hits, idx2docid=idx2docid, offset=args.offset)

        fh = first_hit_rank(hit_ids, gold_set) if gold_set else 0
        if gold_set and fh == 0:
            miss_any += 1

        row: Dict[str, Any] = {
            "qid": qid,
            "query": query,
            "gold_n": len(gold_set),
            "hits_n": len(hit_ids),
            "first_hit_rank": fh,
            "mrr": mrr_from_first_rank(fh),
        }

        for k in args.ks:
            row[f"recall@{k}"] = recall_at_k(hit_ids, gold_set, k)
            row[f"precision@{k}"] = precision_at_k(hit_ids, gold_set, k)
            row[f"ndcg@{k}"] = ndcg_at_k(hit_ids, gold_set, k)

        out.append(row)

    df = pd.DataFrame(out)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote: {args.out_csv}")

    print(f"Queries: {len(pred_rows)}")
    print(f"Queries with gold: {with_gold}")
    print(f"Miss (no gold found anywhere in hits list): {miss_any}")

    # macro only over queries that have gold_n > 0
    df_eval = df[df["gold_n"] > 0].copy()
    for k in args.ks:
        r = df_eval[f"recall@{k}"].mean() if f"recall@{k}" in df_eval.columns else 0.0
        p = df_eval[f"precision@{k}"].mean() if f"precision@{k}" in df_eval.columns else 0.0
        nd = df_eval[f"ndcg@{k}"].mean() if f"ndcg@{k}" in df_eval.columns else 0.0
        mrr = df_eval["mrr"].mean() if "mrr" in df_eval.columns else 0.0
        print(f"Macro@{k}: Recall={r:.6f} | P={p:.6f} | MRR={mrr:.6f} | nDCG={nd:.6f}")


if __name__ == "__main__":
    main()
