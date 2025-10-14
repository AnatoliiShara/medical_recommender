# -*- coding: utf-8 -*-
import argparse, json, glob, math, os.path as P, os
import numpy as np
import pyarrow.parquet as pq

def precision_at_k(pred_ids, gold_ids, k=10):
    if not pred_ids: return 0.0
    pred = pred_ids[:k]
    inter = len(set(pred) & set(gold_ids))
    return inter / float(k)

def recall_at_k(pred_ids, gold_ids, k=300):
    if not gold_ids: return 0.0
    pred = pred_ids[:k]
    inter = len(set(pred) & set(gold_ids))
    return inter / float(len(gold_ids))

def ndcg_at_k(pred_ids, gold_ids, k=300):
    if not gold_ids: return 0.0
    dcg = 0.0
    for r, pid in enumerate(pred_ids[:k]):
        if pid in gold_ids:
            dcg += 1.0 / math.log2(r + 2)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold_ids), k)))
    return (dcg / ideal) if ideal > 0 else 0.0

def load_gold_list(jsonl_path):
    out=[]
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip(): out.append([]); continue
            o = json.loads(line)
            g = o.get("gold_doc_ids") or []
            out.append([int(x) for x in g if str(x).isdigit()])
    return out

def try_read_doc_ids_npy(chunks_path):
    base_dir = P.dirname(chunks_path)
    alt_dir  = P.dirname(base_dir)
    for cand in (P.join(base_dir,"doc_ids.npy"), P.join(alt_dir,"doc_ids.npy")):
        if P.exists(cand):
            return np.load(cand), cand
    return None, None

def choose_doc_column(pf):
    cols = [f.name for f in pf.schema_arrow]
    # 1) найкраще — уже канонічне поле
    for c in ("doc_id","document_id","docId"):
        if c in cols: return c, True
    # 2) резерв — локальний індекс
    for c in ("doc_idx","doc_index","doc"):
        if c in cols: return c, False
    raise RuntimeError(f"Не знайшов doc-колонку. Є: {cols}")

def build_pid2canonical(chunks_path, gold_set_preview=None, debug=False):
    pf = pq.ParquetFile(chunks_path)
    col, is_canonical = choose_doc_column(pf)
    arr = pf.read(columns=[col]).column(0).to_numpy(zero_copy_only=False)  # per passage

    if debug:
        print(f"[INFO] Using column '{col}' from chunks (canonical={is_canonical})")

    if is_canonical:
        # значення з chunks уже канонічні
        return arr.astype(np.int64), f"chunks:{col}"

    # інакше пробуємо npy-мапу
    doc_ids, path = try_read_doc_ids_npy(chunks_path)
    if doc_ids is not None:
        max_idx = int(np.max(arr))
        if max_idx < len(doc_ids):
            pid2doc = doc_ids[arr.astype(np.int64)]
            src = f"doc_ids.npy:{path}"
            # sanity: чи перетинаються gold’и (якщо дали прев’ю)
            if debug and gold_set_preview:
                inter = len(set(pid2doc[:500]) & gold_set_preview)
                print(f"[DBG] overlap(pid2doc[:500], gold_preview)={inter}")
            return pid2doc.astype(np.int64), src

    # fallback: значення виглядають як локальні індекси без мапи -> це поганий варіант
    # але інколи у chunks є "велике" поле під іншим ім'ям; спробуємо знайти
    cols = [f.name for f in pf.schema_arrow]
    for alt in ("doc_id","document_id"):
        if alt in cols:
            pid2doc = pf.read(columns=[alt]).column(0).to_numpy(zero_copy_only=False)
            return pid2doc.astype(np.int64), f"chunks:{alt}"

    # як крайній випадок — віддамо arr як є (майже напевно дасть нульові метрики)
    return arr.astype(np.int64), f"chunks:{col}(local_idx)"

def extract_pred_ids(j):
    for k in ("pred_doc_ids","doc_ids","docids"):
        if k in j and isinstance(j[k], list):
            return [int(x) for x in j[k] if str(x).isdigit()]
    ID_KEYS=("id","doc_id","docId","document_id")
    ARR_KEYS=("results","fused","preds","docs","items","final","top_docs","ranked","hits")
    for arr_key in ARR_KEYS:
        if arr_key in j and isinstance(j[arr_key], list):
            out=[]
            for it in j[arr_key]:
                if isinstance(it, dict):
                    for idk in ID_KEYS:
                        if idk in it and str(it[idk]).isdigit():
                            out.append(int(it[idk])); break
            if out: return out
    if "passages" in j and isinstance(j["passages"], list):
        out=[]
        for it in j["passages"]:
            if isinstance(it, dict) and "doc_id" in it and str(it["doc_id"]).isdigit():
                out.append(int(it["doc_id"]))
        if out: return out
    return []

def map_passages_to_canonical(pred_ids, pid2canon):
    L = len(pid2canon)
    out=[]; seen=set()
    for pid in pred_ids:
        if 0 <= pid < L:
            doc = int(pid2canon[pid])
        else:
            doc = int(pid)
        if doc not in seen:
            seen.add(doc)
            out.append(doc)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("queries_jsonl")
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--debug", type=int, default=0)
    ap.add_argument("--out_csv", default="")
    args = ap.parse_args()

    paths = sorted(glob.glob(P.join(args.run_dir, "q*.json")))
    if not paths:
        print(f"[ERR] Не знайшов q*.json у {args.run_dir}")
        return

    gold_list = load_gold_list(args.queries_jsonl)
    gold_preview = set(gold_list[0] if gold_list else [])
    pid2canon, src = build_pid2canonical(args.chunks, gold_preview, debug=(args.debug>0))
    if args.debug:
        print(f"[INFO] pid→canonical source: {src}")

    n_q = min(len(paths), len(gold_list))
    p10s=[]; r300s=[]; n300s=[]; rows=[]
    for i in range(n_q):
        with open(paths[i], encoding="utf-8") as f:
            j = json.load(f)
        pred_raw = extract_pred_ids(j)
        pred_doc = map_passages_to_canonical(pred_raw, pid2canon)
        gold = set(gold_list[i])

        if args.debug>0 and i < args.debug:
            print(f"[DBG q{i}] pass/raw_len={len(pred_raw)} doc_len={len(pred_doc)} gold_len={len(gold)}")
            print(f"  raw_head={pred_raw[:12]}")
            print(f"  doc_head={pred_doc[:12]}")
            print(f"  gold_head={list(gold)[:12]}")

        if not gold: 
            continue

        p10 = precision_at_k(pred_doc, gold, 10)
        r300 = recall_at_k(pred_doc, gold, 300)
        n300 = ndcg_at_k(pred_doc, gold, 300)
        p10s.append(p10); r300s.append(r300); n300s.append(n300)
        rows.append({"qid": i, "P@10": p10, "Recall@300": r300, "nDCG@300": n300})

    if rows:
        mp10 = sum(p10s)/len(p10s)
        mr300 = sum(r300s)/len(r300s)
        mn300 = sum(n300s)/len(n300s)
        print("\n=== POST-HOC DOC-LEVEL METRICS (canonical) ===")
        print(f"Queries with gold  : {len(rows)} / {n_q}")
        print(f"Precision@10       : {mp10:.4f}")
        print(f"Recall@300         : {mr300:.4f}")
        print(f"nDCG@300           : {mn300:.4f}")
    else:
        print("[WARN] Порожні gold’и — нічого рахувати.")

    if args.out_csv:
        import pandas as pd
        pd.DataFrame(rows).to_csv(args.out_csv, index=False)
        print(f"[OK] per-query metrics -> {args.out_csv}")

if __name__ == "__main__":
    main()
