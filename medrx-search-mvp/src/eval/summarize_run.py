# src/eval/summarize_run.py
# Strict run summarizer: counts only logs that contain one of the allowed
# result containers (final/results/topk). No fallback to "any list field".

import argparse
import datetime
import glob
import json
import math
import os
import re
import subprocess
import sys
from typing import Any, Dict, List, Set, Tuple

import pandas as pd


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _get_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"

def _pick_first(d: dict, keys: List[str], default=None):
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] is not None:
            return d[k]
    return default


# ---------------- queries/gold ----------------

def load_query_gold(queries_files: List[str]) -> Dict[str, Set[str]]:
    mapping: Dict[str, Set[str]] = {}
    for qf in queries_files:
        if not os.path.exists(qf):
            print(f"[WARN] queries file not found: {qf}", file=sys.stderr)
            continue
        with open(qf, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                qid = _pick_first(obj, ["query_id", "id", "qid", "uid", "hash"], None)
                if qid is None:
                    qid = str(abs(hash(_pick_first(obj, ["query", "text"], ""))))
                gold = _pick_first(obj, ["gold", "gold_ids", "target_ids"], [])
                if isinstance(gold, str):
                    gold = [gold]
                mapping[str(qid)] = set(str(x) for x in gold)
    return mapping


# ---------------- log parsing ----------------

def _extract_results_strict(obj: Any, containers: List[str]) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Return (query_id, results_list, meta) ONLY if one of `containers` exists.
    No generic fallback to "any list[dict]" to avoid picking intermediate logs.
    """
    qid = None
    meta: Dict[str, Any] = {}
    results_raw = None

    if isinstance(obj, dict):
        qid = _pick_first(obj, ["query_id", "id", "qid", "uid", "hash"], None)
        meta = obj.get("meta", {})
        for key in containers:     # prefer the first matching container
            if key in obj and isinstance(obj[key], list):
                results_raw = obj[key]
                break
    elif isinstance(obj, list):
        # top-level list is accepted ONLY if caller explicitly allows "top" (i.e., containers includes "__top__")
        if "__top__" in containers:
            results_raw = obj

    if results_raw is None:
        return (qid, [], meta)

    parsed: List[Dict[str, Any]] = []
    for r in results_raw:
        if not isinstance(r, dict):
            continue
        docid = _pick_first(r, ["doc_id", "document_id", "id", "doc", "passage_id"], None)
        if docid is None:
            docid = _pick_first(_pick_first(r, ["document", "doc"], {}) or {}, ["id","doc_id","document_id"], None)
        if docid is None:
            continue
        rank  = _pick_first(r, ["rank","position","idx"], None)
        score = _pick_first(r, ["score","ce_score","rrf_score","bm25_score","dense_score"], None)

        violation = False
        v_direct = _pick_first(r, ["violation","safety_violation"], None)
        if isinstance(v_direct, bool):
            violation = v_direct
        else:
            flags = _pick_first(r, ["flags","attrs"], {})
            if isinstance(flags, dict):
                violation = bool(_pick_first(flags, ["safety_violation","violation"], False))

        parsed.append({"doc_id": str(docid), "rank": rank, "score": score, "violation": violation})
    return (qid, parsed, meta)


def parse_run_logs(run_dir: str,
                   filename_pattern: str,
                   containers_csv: str) -> Dict[str, Dict[str, Any]]:
    """
    Scan run_dir recursively for JSON logs; keep only those having one of the
    allowed result containers (comma-separated list).
    If query_id is missing -> fallback to unique relpath-based id.
    """
    per_query: Dict[str, Dict[str, Any]] = {}
    all_files = glob.glob(os.path.join(run_dir, "**", "*.json"), recursive=True)
    rx = re.compile(filename_pattern)
    containers = [c.strip() for c in containers_csv.split(",") if c.strip()]
    if not containers:
        containers = ["final", "results", "topk"]

    for path in all_files:
        base = os.path.basename(path)
        if not rx.match(base):
            # still read; we will filter by presence of allowed container
            pass
        try:
            obj = _read_json(path)
        except Exception:
            continue

        qid, results, meta = _extract_results_strict(obj, containers)
        if not results:
            continue

        if not qid:
            rel = os.path.relpath(path, run_dir)
            qid = rel.replace(os.sep, "__")

        # avoid collisions
        orig = qid
        c = 1
        while qid in per_query:
            c += 1
            qid = f"{orig}__dup{c}"

        per_query[str(qid)] = {"results": results, "meta": meta, "src": path}

    return per_query


# ---------------- metrics ----------------

def dcg_at_k(binary_rels: List[int], k: int) -> float:
    s = 0.0
    for i, rel in enumerate(binary_rels[:k]):
        if rel:
            s += 1.0 / math.log2(i + 2)
    return s

def ndcg_at_k(binary_rels: List[int], k: int, ideal_size: int) -> float:
    dcg = dcg_at_k(binary_rels, k)
    ideal = dcg_at_k([1] * max(1, ideal_size), k)
    return (dcg / ideal) if ideal > 0 else 0.0

def evaluate(per_query_logs: Dict[str, Dict[str, Any]],
             gold_map: Dict[str, Set[str]],
             ks: List[int]):
    rows: List[Dict[str, Any]] = []
    for qid, data in per_query_logs.items():
        preds = [r["doc_id"] for r in data["results"]]
        gold  = gold_map.get(qid, set())
        if not preds and not gold:
            continue

        bin_list   = [1 if p in gold else 0 for p in preds]
        total_gold = len(gold)

        viol_flags = [bool(r.get("violation", False)) for r in data["results"]]

        row: Dict[str, Any] = {"query_id": qid, "total_gold": total_gold}
        for k in ks:
            topk = bin_list[:k]
            correct = sum(topk)
            prec = correct / max(1, k)
            rec  = (correct / total_gold) if total_gold > 0 else None
            nd   = ndcg_at_k(bin_list, k, total_gold)

            vk = sk = None
            if len(viol_flags) > 0:
                v_any = any(viol_flags[:k])
                vk = 1.0 if v_any else 0.0
                sk = 1.0 if not v_any else 0.0

            row.update({
                f"Prec@{k}": prec,
                f"Recall@{k}": rec,
                f"nDCG@{k}": nd,
                f"Violation@{k}": vk,
                f"Safe@{k}": sk
            })
        rows.append(row)

    df = pd.DataFrame(rows)
    agg = (df[[c for c in df.columns if c not in ("query_id", "total_gold")]]
           .mean(numeric_only=True)) if not df.empty else pd.Series(dtype=float)
    return df, agg


# ---------------- entrypoint ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="Directory with per-query logs")
    ap.add_argument("--queries", nargs="+", required=True, help="*.jsonl with gold labels")
    ap.add_argument("--k_eval", nargs="+", type=int, default=[10, 20])
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--pattern", default=r".*\.json$",
                    help="Regex for filenames to consider (default: any *.json)")
    ap.add_argument("--containers", default="final,results,topk",
                    help="Comma-separated list of allowed result containers (default: final,results,topk)")
    args = ap.parse_args()

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or os.path.join("data", "eval", "baselines", stamp)
    os.makedirs(out_dir, exist_ok=True)

    gold = load_query_gold(args.queries)
    logs = parse_run_logs(args.run_dir, args.pattern, args.containers)
    if not logs:
        print(f"[ERROR] no logs parsed from {args.run_dir}. Check --run_dir / --pattern / --containers.", file=sys.stderr)
        sys.exit(2)

    df, agg = evaluate(logs, gold, args.k_eval)

    perq_csv = os.path.join(out_dir, "per_query_metrics.csv")
    agg_csv  = os.path.join(out_dir, "aggregate_metrics.csv")
    meta_json = os.path.join(out_dir, "meta.json")

    df.to_csv(perq_csv, index=False)
    agg.to_csv(agg_csv, header=["value"])

    meta = {
        "timestamp": stamp,
        "commit": _get_commit_hash(),
        "run_dir": args.run_dir,
        "queries": args.queries,
        "k_eval": args.k_eval,
        "num_queries": int(df.shape[0]),
        "num_files_scanned": len(glob.glob(os.path.join(args.run_dir, "**", "*.json"), recursive=True)),
        "pattern": args.pattern,
        "containers": args.containers
    }
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[OK] Saved:")
    print("  ", perq_csv)
    print("  ", agg_csv)
    print("  ", meta_json)

if __name__ == "__main__":
    main()
