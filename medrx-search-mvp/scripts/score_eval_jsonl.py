#!/usr/bin/env python3
import argparse, json, math
from pathlib import Path

def load_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                yield json.loads(line)

def get_id(x: dict):
    v = x.get("id") or x.get("query_id") or x.get("qid")
    return None if v is None else str(v)

def get_gold_ids(q: dict):
    g = q.get("gold_ids") or q.get("gold") or q.get("gold_id") or q.get("relevant_ids")
    if g is None:
        return []
    if isinstance(g, (str, int)):
        return [str(g)]
    if isinstance(g, list):
        return [str(t) for t in g]
    return []

def get_pred_ids(r: dict):
    for key in ("pred_ids","doc_ids","ids","candidates","top_ids"):
        v = r.get(key)
        if isinstance(v, list):
            return [str(x) for x in v if x is not None]
    hits = r.get("hits") or r.get("results") or r.get("retrieved") or r.get("docs")
    if isinstance(hits, list):
        out=[]
        for h in hits:
            if isinstance(h, (str,int)):
                out.append(str(h)); continue
            if isinstance(h, dict):
                out.append(str(h.get("doc_id") or h.get("id") or h.get("docid") or h.get("pid") or h.get("product_id")))
        return [x for x in out if x and x!="None"]
    return []

def dcg(binary_rels):
    s=0.0
    for i,rel in enumerate(binary_rels, start=1):
        if rel:
            s += 1.0 / math.log2(i+1)
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, nargs="+", default=[1,3,5,10,20,60])
    args = ap.parse_args()

    queries = list(load_jsonl(args.queries))
    preds   = list(load_jsonl(args.predictions))

    q_ids = [get_id(q) for q in queries]
    p_ids = [get_id(r) for r in preds]

    gold_by_id = {}
    order_ids = []
    for q in queries:
        qid = get_id(q)
        order_ids.append(qid)
        gold_by_id[qid] = set(get_gold_ids(q))

    pred_by_id = {}
    for r in preds:
        rid = get_id(r)
        if rid is not None:
            pred_by_id[rid] = get_pred_ids(r)

    use_id_join = (
        all(x is not None for x in q_ids) and
        all(x is not None for x in p_ids) and
        len(set(order_ids) & set(pred_by_id.keys())) >= max(1, len(order_ids)//5)
    )

    # align
    aligned = []
    if use_id_join:
        for qid in order_ids:
            aligned.append((qid, gold_by_id.get(qid,set()), pred_by_id.get(qid, [])))
    else:
        # fallback: align by row order
        m = min(len(queries), len(preds))
        for i in range(m):
            q = queries[i]
            r = preds[i]
            qid = get_id(q) or str(i)
            aligned.append((qid, set(get_gold_ids(q)), get_pred_ids(r)))

    ks = args.k
    metrics = {K: {"recall":0.0, "mrr":0.0, "ndcg":0.0} for K in ks}
    n = 0
    any_pred_nonempty = 0

    for qid, gset, plist in aligned:
        if not gset:
            continue
        n += 1
        if plist:
            any_pred_nonempty += 1
        for K in ks:
            top = plist[:K]
            hits = [1 if t in gset else 0 for t in top]
            metrics[K]["recall"] += 1.0 if any(hits) else 0.0

            rr = 0.0
            for i,t in enumerate(top, start=1):
                if t in gset:
                    rr = 1.0/i
                    break
            metrics[K]["mrr"]  += rr

            ideal = 1.0 / math.log2(2)  # one relevant at rank 1
            metrics[K]["ndcg"] += (dcg(hits) / ideal) if ideal > 0 else 0.0

    for K in ks:
        for m in metrics[K]:
            metrics[K][m] = metrics[K][m] / max(1, n)

    report = {
        "queries_file": str(args.queries),
        "predictions_file": str(args.predictions),
        "aligned_mode": "id_join" if use_id_join else "row_order_fallback",
        "rows_queries": len(queries),
        "rows_predictions": len(preds),
        "scored_queries": n,
        "queries_with_nonempty_preds": any_pred_nonempty,
        "metrics": metrics,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"aligned_mode={report['aligned_mode']}")
    print(f"rows_queries={report['rows_queries']} rows_predictions={report['rows_predictions']} scored_queries={n} nonempty_preds={any_pred_nonempty}")
    for K in ks:
        print(f"K={K:>2}  Recall@K={metrics[K]['recall']:.3f}  MRR@K={metrics[K]['mrr']:.3f}  nDCG@K={metrics[K]['ndcg']:.3f}")

if __name__ == "__main__":
    main()
