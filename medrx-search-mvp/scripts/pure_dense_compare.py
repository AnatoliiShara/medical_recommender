#!/usr/bin/env python3
"""
Compare two PURE dense setups (baseline vs finetuned) with paired bootstrap.

Inputs:
- baseline: (faiss.index + doc_ids.npy) + bi-encoder for query encoding
- finetuned: same
- queries jsonl: each line must have a query text + gold doc ids list.
  Script autodetects common field names.

Outputs:
- metric means, delta, 95% CI, p-values (one-sided + two-sided)
- optional JSON dump
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
from tqdm import tqdm

try:
    import faiss  # type: ignore
except Exception as e:
    raise SystemExit("faiss import failed. Install faiss-cpu (or faiss-gpu).") from e

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise SystemExit("sentence-transformers import failed. Is it installed in your venv?") from e


QUERY_KEYS = ("query", "q", "text", "user_query", "original_query")
GOLD_KEYS = ("gold_ids", "gold_doc_ids", "gold", "relevant_ids", "relevant_doc_ids", "positives", "positive_ids")


def _load_index(index_dir: str) -> Tuple["faiss.Index", np.ndarray]:
    idx = faiss.read_index(os.path.join(index_dir, "faiss.index"))
    doc_ids = np.load(os.path.join(index_dir, "doc_ids.npy"), allow_pickle=True)
    doc_ids = np.asarray([str(x) for x in doc_ids], dtype=object)
    return idx, doc_ids


def _read_jsonl(path: str) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _pick_fields(ex: dict) -> Tuple[str, str]:
    qk = next((k for k in QUERY_KEYS if k in ex), None)
    if qk is None:
        raise SystemExit(f"Cannot find query field in json keys={list(ex.keys())}. Tried: {QUERY_KEYS}")
    gk = next((k for k in GOLD_KEYS if k in ex), None)
    if gk is None:
        raise SystemExit(f"Cannot find gold field in json keys={list(ex.keys())}. Tried: {GOLD_KEYS}")
    return qk, gk


def _normalize_gold(gold_val) -> List[str]:
    if gold_val is None:
        return []
    if isinstance(gold_val, (str, int)):
        return [str(gold_val)]
    if isinstance(gold_val, list):
        if not gold_val:
            return []
        if isinstance(gold_val[0], dict):
            for key in ("doc_id", "id", "docid"):
                if key in gold_val[0]:
                    return [str(x.get(key)) for x in gold_val if x.get(key) is not None]
            return []
        return [str(x) for x in gold_val if x is not None]
    return []


def _encode_queries(model: SentenceTransformer, queries: List[str], query_prefix: str, batch_size: int) -> np.ndarray:
    q = [query_prefix + s for s in queries]
    emb = model.encode(q, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=False, show_progress_bar=True)
    emb = emb.astype(np.float32, copy=False)
    faiss.normalize_L2(emb)
    return emb


def _metrics_for_hits(hits: List[int], gold_size: int, k: int) -> Dict[str, float]:
    k = min(k, len(hits))
    h = hits[:k]
    s = int(sum(h))
    recall = s / gold_size if gold_size > 0 else 0.0

    mrr = 0.0
    for i, v in enumerate(h, start=1):
        if v:
            mrr = 1.0 / i
            break

    dcg = 0.0
    for i, v in enumerate(h, start=1):
        if v:
            dcg += 1.0 / np.log2(i + 1.0)

    ideal = min(gold_size, k)
    idcg = 0.0
    for i in range(1, ideal + 1):
        idcg += 1.0 / np.log2(i + 1.0)

    ndcg = (dcg / idcg) if idcg > 0 else 0.0
    return {"recall": recall, "mrr": mrr, "ndcg": ndcg}


def _score_run(
    index: "faiss.Index",
    doc_ids: np.ndarray,
    model_name_or_path: str,
    device: str,
    query_texts: List[str],
    gold_sets: List[set],
    ks: List[int],
    query_prefix: str,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    model = SentenceTransformer(model_name_or_path, device=device)
    q_emb = _encode_queries(model, query_texts, query_prefix=query_prefix, batch_size=batch_size)

    kmax = max(ks)
    _, I = index.search(q_emb, kmax)

    out: Dict[str, np.ndarray] = {}
    nq = len(query_texts)
    for k in ks:
        out[f"recall@{k}"] = np.zeros((nq,), dtype=np.float32)
        out[f"mrr@{k}"] = np.zeros((nq,), dtype=np.float32)
        out[f"ndcg@{k}"] = np.zeros((nq,), dtype=np.float32)

    for qi in tqdm(range(nq), desc=f"scoring ({os.path.basename(str(model_name_or_path))})", unit="q"):
        retrieved = doc_ids[I[qi]].tolist()
        gold = gold_sets[qi]
        hits = [1 if rid in gold else 0 for rid in retrieved]

        for k in ks:
            m = _metrics_for_hits(hits, gold_size=len(gold), k=k)
            out[f"recall@{k}"][qi] = m["recall"]
            out[f"mrr@{k}"][qi] = m["mrr"]
            out[f"ndcg@{k}"][qi] = m["ndcg"]
    return out


def _bootstrap_paired(a: np.ndarray, b: np.ndarray, n_boot: int, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    n = a.shape[0]
    diffs = np.empty((n_boot,), dtype=np.float32)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = float(np.mean(b[idx]) - np.mean(a[idx]))
    diffs.sort()
    lo = float(np.percentile(diffs, 2.5))
    hi = float(np.percentile(diffs, 97.5))
    delta = float(np.mean(b) - np.mean(a))
    p_one = float((1.0 + np.sum(diffs <= 0.0)) / (n_boot + 1.0))
    p_two = float(min(1.0, 2.0 * min(p_one, 1.0 - p_one)))
    return {"delta": delta, "ci_lo": lo, "ci_hi": hi, "p_one_sided": p_one, "p_two_sided": p_two}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True, help="jsonl with query + gold")
    ap.add_argument("--baseline_index", required=True)
    ap.add_argument("--ft_index", required=True)
    ap.add_argument("--baseline_model", required=True)
    ap.add_argument("--ft_model", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--query_prefix", default="query: ", help='E5-style query prefix. Use "" to disable.')
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--ks", default="1,5,10,20")
    ap.add_argument("--bootstrap", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="")
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",") if x.strip()]

    qrows = _read_jsonl(args.queries)
    qk, gk = _pick_fields(qrows[0])

    b_index, b_doc_ids = _load_index(args.baseline_index)
    f_index, f_doc_ids = _load_index(args.ft_index)

    b_set = set(b_doc_ids.tolist())
    f_set = set(f_doc_ids.tolist())
    if b_set != f_set:
        raise SystemExit(f"Baseline/FT doc_id sets differ: baseline={len(b_set):,}, ft={len(f_set):,}. Rebuild both from the same chunks.")

    universe = b_set
    print(f"[INFO] doc universe: {len(universe):,} docs (baseline == ft)")

    query_texts: List[str] = []
    gold_sets: List[set] = []
    dropped_empty_query = 0
    dropped_no_gold = 0

    for r in qrows:
        q = r.get(qk, "")
        if not isinstance(q, str) or not q.strip():
            dropped_empty_query += 1
            continue
        gold = set(_normalize_gold(r.get(gk)))
        gold = set([gid for gid in gold if gid in universe])
        if not gold:
            dropped_no_gold += 1
            continue
        query_texts.append(q.strip())
        gold_sets.append(gold)

    print(f"[INFO] queries loaded: {len(qrows):,}")
    print(f"[INFO] eval queries kept: {len(query_texts):,} | dropped_empty_query={dropped_empty_query:,} | dropped_no_gold_after_filter={dropped_no_gold:,}")

    baseline_scores = _score_run(
        index=b_index, doc_ids=b_doc_ids, model_name_or_path=args.baseline_model, device=args.device,
        query_texts=query_texts, gold_sets=gold_sets, ks=ks, query_prefix=args.query_prefix, batch_size=args.batch_size
    )
    ft_scores = _score_run(
        index=f_index, doc_ids=f_doc_ids, model_name_or_path=args.ft_model, device=args.device,
        query_texts=query_texts, gold_sets=gold_sets, ks=ks, query_prefix=args.query_prefix, batch_size=args.batch_size
    )

    results = {"n_queries": len(query_texts), "ks": ks, "bootstrap": args.bootstrap, "seed": args.seed, "metrics": {}}

    print("\n=== PURE DENSE: baseline vs finetuned (paired bootstrap) ===")
    for k in ks:
        for metric in ("recall", "mrr", "ndcg"):
            key = f"{metric}@{k}"
            a = baseline_scores[key]
            b = ft_scores[key]
            s = _bootstrap_paired(a, b, n_boot=args.bootstrap, seed=args.seed)
            results["metrics"][key] = {
                "baseline_mean": float(np.mean(a)),
                "ft_mean": float(np.mean(b)),
                **s,
            }
            v = results["metrics"][key]
            print(
                f"{key:10s}  baseline={v['baseline_mean']:.4f}  ft={v['ft_mean']:.4f}  "
                f"Δ={v['delta']:+.4f}  95%CI=[{v['ci_lo']:+.4f},{v['ci_hi']:+.4f}]  p={v['p_one_sided']:.4g}"
            )

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        out_path = os.path.join(args.out_dir, "pure_dense_compare_results.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] wrote: {out_path}")


if __name__ == "__main__":
    main()
