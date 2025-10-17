# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import sys
import json
import argparse
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

# дозволяємо запуск як "python src/eval/evaluate_bench.py"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.search.integrated_medical_assistant import AssistantIndex, SearchConfig

# ---- метрики ----

def dcg(rels: List[float]) -> float:
    return sum((rel / np.log2(i+2) for i, rel in enumerate(rels)))

def ndcg_at_k(golds: List[str], preds: List[str], k: int) -> float:
    rels = [1.0 if p in golds else 0.0 for p in preds[:k]]
    idcg = dcg(sorted(rels, reverse=True))
    return 0.0 if idcg == 0 else dcg(rels) / idcg

def precision_at_k(golds: List[str], preds: List[str], k: int) -> float:
    if k == 0: return 0.0
    return sum(1 for p in preds[:k] if p in golds) / float(k)

# ---- бенч ----

def load_queries(path: str) -> List[Dict]:
    qs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                js = json.loads(line)
            except Exception:
                continue
            q = js.get("query", "").strip()
            golds = js.get("gold_drugs", [])
            qs.append({"query": q, "gold": [str(g).lower() for g in golds]})
    return qs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--product_map", required=False)
    ap.add_argument("--top_k", type=int, default=120)
    ap.add_argument("--show", type=int, default=120)
    ap.add_argument("--alpha", type=float, default=0.35)
    ap.add_argument("--ks", nargs="+", type=int, default=[1,3,5,10])
    ap.add_argument("--w_acc", type=float, default=0.5)
    ap.add_argument("--prefer_indications", action="store_true")
    ap.add_argument("--encoder_model", type=str, default="models/medical-search-ua-full")  # локальна бі-енкодерна модель
    ap.add_argument("--reranker_model", type=str, default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--rerank_top", type=int, default=100)
    ap.add_argument("--rerank_batch", type=int, default=16)  # лишаємо для сумісності, не використовується явно
    ap.add_argument("--ce_min", type=float, default=0.15)
    ap.add_argument("--ce_weight", type=float, default=0.8)
    ap.add_argument("--attach_contra", action="store_true")  # для сумісності з попередніми прапорцями
    ap.add_argument("--warnings", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out_json", default="")
    ap.add_argument("--out_csv", default="")
    args = ap.parse_args()

    print(f"[INFO] Завантаження датасету: {args.dataset}")
    df = pd.read_parquet(args.dataset)

    idx = AssistantIndex()
    idx.build_from_dataframe(
        df,
        encoder_model=args.encoder_model,       # ВАЖЛИВО: саме це вмикає FAISS
        prefer_indications=args.prefer_indications,
        chunk_size=1200,
        overlap=200,
        show_batches=True,
    )
    ## [PATCH] skip CE when none
    if args.reranker_model and str(args.reranker_model).lower() not in {'none','off'} and args.rerank_top>0:
        idx.enable_crossencoder(args.reranker_model, args.rerank_top, args.ce_min, args.ce_weight)

    qs = load_queries(args.queries)
    print(f"[INFO] Запитів у бенчі: {len(qs)}")

    cfg = SearchConfig(
        rrf_alpha=args.alpha,
        top_k=args.top_k,
        show=args.show,
        prefer_indications=args.prefer_indications,
        ce_min=args.ce_min,
        ce_weight=args.ce_weight,
    )

    all_rows = []
    P = {k: [] for k in args.ks}
    N = {k: [] for k in args.ks}
    MA = {k: [] for k in args.ks}

    for i, q in enumerate(qs, 1):
        query = q["query"]
        golds = [g.lower() for g in q["gold"]]
        res = idx.search(query, cfg)

        names = [r["name"].lower() for r in res]
        row = {"query": query, "top": [f'{r["name"]}' for r in res[:10]]}

        if args.verbose:
            print(f"\n[{i}/{len(qs)}] '{query}'")
        for k in args.ks:
            p = precision_at_k(golds, names, k)
            n = ndcg_at_k(golds, names, k)
            # must_avoid у цьому baseline вимикаємо (усе 0.0)
            ma = 0.0
            P[k].append(p); N[k].append(n); MA[k].append(ma)
            if args.verbose:
                print(f"  P@{k}={p:.3f}  nDCG@{k}={n:.3f}  MA@{k}={int(ma)}")
        if args.verbose:
            top10 = ", ".join(row["top"])
            print(f"  top: {top10}")

        all_rows.append(row)

    # агрегати
    print("\n================= ЗВІТ ПО БЕНЧУ =================")
    print(f"Кількість запитів: {len(qs)}")
    print(f"CE: ON  (model={args.reranker_model}, top={args.rerank_top}, ce_min={args.ce_min}, ce_w={args.ce_weight})")
    print(f"RRF alpha={args.alpha}, top_k={args.top_k}, show={args.show}, prefer_indications={args.prefer_indications}")
    print("--------------------------------------------------")
    for k in args.ks:
        print(f"P@{k}: {np.mean(P[k]):.3f}    nDCG@{k}: {np.mean(N[k]):.3f}    MustAvoid@{k}: {np.mean(MA[k]):.3f}")
    print("==================================================")

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(all_rows, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Збережено детальні результати у: {args.out_json}")
    if args.out_csv:
        rows = []
        for k in args.ks:
            rows.append({"k": k, "P": np.mean(P[k]), "nDCG": np.mean(N[k]), "MustAvoid": np.mean(MA[k])})
        pd.DataFrame(rows).to_csv(args.out_csv, index=False)
        print(f"[INFO] Збережено зведені метрики у: {args.out_csv}")

if __name__ == "__main__":
    main()
