import argparse, json, os, math
from typing import Dict, List, Set, Tuple

def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def pick_predictions_file(run_dir: str) -> str:
    # Try common names first; else pick the first jsonl that contains pred_ids/candidates
    preferred = [
        "predictions.jsonl",
        "results.jsonl",
        "run.jsonl",
        "answers.jsonl",
        "responses.jsonl",
    ]
    for name in preferred:
        p = os.path.join(run_dir, name)
        if os.path.exists(p):
            return p
    # heuristic fallback
    for name in os.listdir(run_dir):
        if name.endswith(".jsonl"):
            p = os.path.join(run_dir, name)
            try:
                row = read_jsonl(p)[0]
            except Exception:
                continue
            if any(k in row for k in ("pred_ids", "candidates", "doc_ids", "results")):
                return p
    raise SystemExit(f"[FAIL] cannot find predictions file in {run_dir}")

def extract_pred_ids(row) -> List[int]:
    if "pred_ids" in row and isinstance(row["pred_ids"], list):
        return [int(x) for x in row["pred_ids"]]
    if "doc_ids" in row and isinstance(row["doc_ids"], list):
        return [int(x) for x in row["doc_ids"]]
    if "candidates" in row and isinstance(row["candidates"], list):
        out = []
        for c in row["candidates"]:
            if isinstance(c, dict):
                if "doc_id" in c: out.append(int(c["doc_id"]))
                elif "id" in c: out.append(int(c["id"]))
        return out
    if "results" in row and isinstance(row["results"], list):
        out = []
        for c in row["results"]:
            if isinstance(c, dict):
                if "doc_id" in c: out.append(int(c["doc_id"]))
                elif "id" in c: out.append(int(c["id"]))
            else:
                # sometimes results is list[int]
                try:
                    out.append(int(c))
                except Exception:
                    pass
        return out
    raise ValueError("Unknown prediction format: expected pred_ids/doc_ids/candidates/results")

def precision_at_k(pred: List[int], gold: Set[int], k: int) -> float:
    if k <= 0: return 0.0
    top = pred[:k]
    if not top: return 0.0
    hits = sum(1 for x in top if x in gold)
    return hits / len(top)

def recall_at_k(pred: List[int], gold: Set[int], k: int) -> float:
    if not gold: return 0.0
    hits = sum(1 for x in pred[:k] if x in gold)
    return hits / len(gold)

def hit_at_k(pred: List[int], gold: Set[int], k: int) -> float:
    return 1.0 if any(x in gold for x in pred[:k]) else 0.0

def mrr_at_k(pred: List[int], gold: Set[int], k: int) -> float:
    for i, x in enumerate(pred[:k], 1):
        if x in gold:
            return 1.0 / i
    return 0.0

def ndcg_at_k(pred: List[int], gold: Set[int], k: int) -> float:
    # binary relevance
    def dcg(vals):
        s = 0.0
        for i, rel in enumerate(vals, 1):
            s += (2**rel - 1) / math.log2(i + 1)
        return s

    rels = [1 if x in gold else 0 for x in pred[:k]]
    dcg_v = dcg(rels)
    ideal = [1]*min(len(gold), k) + [0]*max(0, k - min(len(gold), k))
    idcg = dcg(ideal) if ideal else 0.0
    return (dcg_v / idcg) if idcg > 0 else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True, help="jsonl with gold_ids")
    ap.add_argument("--predictions", default=None, help="jsonl with predictions")
    ap.add_argument("--run_dir", default=None, help="dir that contains predictions jsonl")
    ap.add_argument("--k", default="1,3,5,10,20", help="comma-separated cutoffs")
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    ks = [int(x) for x in args.k.split(",") if x.strip()]

    qrows = read_jsonl(args.queries)
    q_gold: Dict[int, Set[int]] = {}
    for r in qrows:
        if "gold_ids" not in r:
            continue
        q_gold[int(r["id"])] = set(int(x) for x in r["gold_ids"])

    if not q_gold:
        raise SystemExit("[FAIL] no gold_ids found in queries file")

    pred_path = args.predictions
    if pred_path is None:
        if not args.run_dir:
            raise SystemExit("[FAIL] provide --predictions or --run_dir")
        pred_path = pick_predictions_file(args.run_dir)

    prows = read_jsonl(pred_path)
    preds: Dict[int, List[int]] = {}
    for r in prows:
        if "id" not in r:
            continue
        rid = int(r["id"])
        try:
            preds[rid] = extract_pred_ids(r)
        except Exception:
            continue

    common_ids = [i for i in q_gold.keys() if i in preds]
    if not common_ids:
        raise SystemExit(f"[FAIL] no overlapping ids between gold and predictions. pred_file={pred_path}")

    metrics = {"n": len(common_ids), "k": ks, "precision": {}, "recall": {}, "hit": {}, "mrr": {}, "ndcg": {}}
    for k in ks:
        p = []
        r = []
        h = []
        m = []
        n = []
        for qid in common_ids:
            gold = q_gold[qid]
            pred = preds[qid]
            p.append(precision_at_k(pred, gold, k))
            r.append(recall_at_k(pred, gold, k))
            h.append(hit_at_k(pred, gold, k))
            m.append(mrr_at_k(pred, gold, k))
            n.append(ndcg_at_k(pred, gold, k))
        metrics["precision"][str(k)] = sum(p)/len(p)
        metrics["recall"][str(k)] = sum(r)/len(r)
        metrics["hit"][str(k)] = sum(h)/len(h)
        metrics["mrr"][str(k)] = sum(m)/len(m)
        metrics["ndcg"][str(k)] = sum(n)/len(n)

    print(f"[OK] scored: {pred_path}")
    print(f"  n_queries = {metrics['n']}")
    for k in ks:
        k = str(k)
        print(f"  @ {k:>2} | hit={metrics['hit'][k]:.4f} mrr={metrics['mrr'][k]:.4f} ndcg={metrics['ndcg'][k]:.4f} "
              f"p={metrics['precision'][k]:.4f} r={metrics['recall'][k]:.4f}")

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"[OK] wrote {args.out_json}")

if __name__ == "__main__":
    main()
