# -*- coding: utf-8 -*-
import argparse
import sys
import re
import pandas as pd
from typing import List, Dict
sys.path.append('src')

from search.enhanced_medical_assistant import EnhancedMedicalAssistant, SearchConfig

# Прості запити та списки підрядків, які вважаємо релевантними в "Показання"
DEFAULT_QUERIES = {
    "артеріальна гіпертензія": ["гіпертензі", "артеріальн", "високий тиск", "підвищений тиск"],
    "серцева недостатність": ["серцев недостатн"],
    "цукровий діабет": ["цукровий діабет", "діабет"],
    "головний біль": ["головн біл", "мігрен", "цефалгі"],
    "астма": ["астм", "бронхіальна астма"],
}

def build_engine(df: pd.DataFrame, max_chunk_tokens: int) -> EnhancedMedicalAssistant:
    eng = EnhancedMedicalAssistant()
    eng.build_from_dataframe(
        df,
        encoder_model=None,
        medical_chunking=True,
        max_chunk_tokens=max_chunk_tokens
    )
    return eng

def is_relevant(row: pd.Series, substrings: List[str]) -> bool:
    txt = str(row.get("Показання", "")).lower()
    return any(ss in txt for ss in substrings)

def ndcg_at_k(gains: List[int], k: int = 10) -> float:
    import math
    gains = gains[:k]
    if not gains:
        return 0.0
    dcg = sum((g / math.log2(i+2)) for i, g in enumerate(gains))
    ideal = sorted(gains, reverse=True)
    idcg = sum((g / math.log2(i+2)) for i, g in enumerate(ideal))
    return float(dcg / (idcg + 1e-9))

def recall_at_k(gains: List[int], total_relevant: int, k: int = 10) -> float:
    return float(min(sum(gains[:k]), total_relevant) / (total_relevant + 1e-9))

def evaluate(df: pd.DataFrame, queries: Dict[str, List[str]], rrf_values: List[int], k: int = 10, max_chunk_tokens: int = 128):
    # Попередньо визначимо релевантні doc_ids на основі показань
    gt: Dict[str, set] = {}
    for q, subs in queries.items():
        rel_ids = set()
        for idx, row in df.iterrows():
            if is_relevant(row, subs):
                rel_ids.add(idx)
        gt[q] = rel_ids

    print(f"\nEvaluating RRF alpha values: {rrf_values}  (k={k}, max_chunk_tokens={max_chunk_tokens})")
    results = []
    for alpha in rrf_values:
        eng = build_engine(df, max_chunk_tokens=max_chunk_tokens)
        cfg = SearchConfig(rrf_alpha=float(alpha), top_k=100, show=100, max_chunk_tokens=max_chunk_tokens)

        agg_ndcg, agg_recall, count = 0.0, 0.0, 0
        for q, subs in queries.items():
            res = eng.search(q, cfg)
            ranked_doc_ids = [r["doc_id"] for r in res]
            # Маркуємо релевантність
            rel_set = gt[q]
            gains = [1 if did in rel_set else 0 for did in ranked_doc_ids]
            ndcg = ndcg_at_k(gains, k=k)
            recall = recall_at_k(gains, total_relevant=len(rel_set), k=k)
            agg_ndcg += ndcg
            agg_recall += recall
            count += 1
        results.append((alpha, agg_ndcg / count, agg_recall / count))

    print("\nalpha\t nDCG@10\t Recall@10")
    for alpha, ndcg, recall in results:
        print(f"{alpha}\t {ndcg:.4f}\t\t {recall:.4f}")

    best = max(results, key=lambda x: (x[1], x[2]))
    print(f"\n🏆 Best alpha by nDCG@10 (tie-break Recall@10): {best[0]}  nDCG@10={best[1]:.4f}  Recall@10={best[2]:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, help="Path to compendium_all.parquet (або підвибірку)")
    ap.add_argument("--rows", type=int, default=1000, help="Скільки рядків взяти для швидкої оцінки")
    ap.add_argument("--k", type=int, default=10, help="k для nDCG/Recall")
    ap.add_argument("--alpha", type=str, default="60,90,120", help="Список RRF alpha, комою")
    ap.add_argument("--max_chunk_tokens", type=int, default=128, help="Ліміт токенів на чанк")
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet).head(args.rows).copy()
    rrf_values = [int(x) for x in args.alpha.split(",") if x.strip()]
    evaluate(df, DEFAULT_QUERIES, rrf_values, k=args.k, max_chunk_tokens=args.max_chunk_tokens)
