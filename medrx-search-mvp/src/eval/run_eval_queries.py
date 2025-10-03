# -*- coding: utf-8 -*-
"""
Енд-ту-енд евал: dense + CE + section gate → метрики на queries_medrx_ua.jsonl

Вимоги:
- enhanced_medical_assistant.py (наш останній варіант) у src/search/
- В середовищі встановлено: rank_bm25, faiss-cpu (або faiss-gpu), sentence-transformers

Параметри:
--parquet         шлях до compendium_all.parquet
--queries         шлях до queries_medrx_ua.jsonl
--aliases         (опц.) шлях до brand_inn_aliases.csv
--rows            скільки рядків взяти з parquet (для швидкого прогону)
--embed_model     модель ембеддингів (деф.: intfloat/multilingual-e5-base)
--reranker_model  модель CE (деф.: BAAI/bge-reranker-v2-m3)
--k               cut-off для метрик (деф.: 10; якщо в JSONL є k, має пріоритет)
--rrf_alpha       к (RRF) (деф.: 60)
--max_chunk_tokens  макс. токенів на чанк (деф.: 128)
--gate_mode       none|prefer|hard (деф.: prefer)
--gate_sections   комою: які секції вважати цільовими (деф.: "Показання,Спосіб застосування та дози")

Вихід:
- зведені метрики (macro-avg): Precision@k, Recall@k, nDCG@k, violations (must_avoid у топ-k)
- короткий лог по кожному запиту
"""
import argparse
import json
import sys
import re
from typing import List, Dict, Tuple, Set
import pandas as pd
import math

sys.path.append('src')
from search.enhanced_medical_assistant import EnhancedMedicalAssistant, SearchConfig

# ---------------- Utils ----------------

def normalize_name(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"[®™©]", " ", s)
    s = s.replace("’", "'")
    s = re.sub(r"[^a-zа-яіїєґ0-9' -]+", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_alias_map(csv_path: str) -> Dict[str, str]:
    """
    Очікуваний формат: alias,target[,type]
    Повертає мапу alias_norm -> target_norm
    """
    alias_map = {}
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        alias = normalize_name(str(row.get("alias", "")))
        target = normalize_name(str(row.get("target", "")))
        if alias and target:
            alias_map[alias] = target
    return alias_map

def map_alias(name: str, alias_map: Dict[str, str]) -> str:
    nm = normalize_name(name)
    return alias_map.get(nm, nm)

def ndcg_at_k(gains: List[int], k: int = 10) -> float:
    gains = gains[:k]
    if not gains:
        return 0.0
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(gains, reverse=True)
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return float(dcg / (idcg + 1e-9))

def precision_recall_at_k(ranked: List[str], gold: Set[str], k: int) -> Tuple[float, float]:
    top = ranked[:k]
    hits = sum(1 for x in top if x in gold)
    prec = hits / max(1, len(top))
    rec = hits / max(1, len(gold)) if gold else 0.0
    return float(prec), float(rec)

def section_gate(groups: List[Dict], preferred: Set[str], mode: str = "prefer", boost: float = 0.15) -> List[Dict]:
    """
    - prefer: +boost до best_score, якщо серед passages є секція з preferred
    - hard:   відфільтровуємо групи, де немає жодного passage з preferred
    - none:   без змін
    """
    if mode == "none":
        return groups
    if mode == "hard":
        gated = []
        for g in groups:
            has_pref = any(p.get("section") in preferred for p in g.get("passages", []))
            if has_pref:
                gated.append(g)
        # зберігаємо порядок (у нас вони вже відсортовані за score)
        return gated
    # prefer
    out = []
    for g in groups:
        has_pref = any(p.get("section") in preferred for p in g.get("passages", []))
        g2 = dict(g)
        if has_pref:
            g2["best_score"] = g2.get("best_score", 0.0) + boost
        out.append(g2)
    # пересортувати за best_score спадаюче
    out.sort(key=lambda x: -x.get("best_score", 0.0))
    return out

# ---------------- Runner ----------------

def run_eval(
    parquet_path: str,
    queries_path: str,
    aliases_path: str = None,
    rows: int = 2000,
    embed_model: str = "intfloat/multilingual-e5-base",
    reranker_model: str = "BAAI/bge-reranker-v2-m3",
    default_k: int = 10,
    rrf_alpha: float = 60.0,
    max_chunk_tokens: int = 128,
    gate_mode: str = "prefer",
    gate_sections: List[str] = None,
):
    print("=== MED-RX: Hybrid (BM25+dense) + RRF + CE + SectionGate — Evaluation ===")
    print(f"Data: {parquet_path}  |  Queries: {queries_path}")
    print(f"Rows: {rows} | Embed: {embed_model} | CE: {reranker_model}")
    print(f"k(default): {default_k} | RRF k: {rrf_alpha} | chunk_tokens: {max_chunk_tokens}")
    print(f"Gate: {gate_mode} on sections: {gate_sections}")

    # Load data
    df = pd.read_parquet(parquet_path).head(rows).copy()

    # Build engine with dense
    eng = EnhancedMedicalAssistant()
    eng.build_from_dataframe(
        df,
        encoder_model=embed_model,          # <— включає FAISS, якщо faiss встановлено
        medical_chunking=True,
        max_chunk_tokens=max_chunk_tokens
    )
    # Enable CE
    eng.enable_crossencoder(reranker_model, ce_top=100, ce_min=0.15, ce_weight=0.8)

    # Alias map (optional)
    alias_map = load_alias_map(aliases_path) if aliases_path else {}

    # Load queries
    queries = []
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            queries.append(item)

    metrics = {
        "precision": [],
        "recall": [],
        "ndcg": [],
        "violations": 0,   # must_avoid у топ-k
        "count": 0
    }

    preferred_set = set(gate_sections or ["Показання", "Спосіб застосування та дози"])

    for qi, q in enumerate(queries, 1):
        raw_query = q.get("query", "")
        k = int(q.get("k", default_k))

        # Ground-truth sets (normalised and alias-mapped)
        gold_list = [map_alias(x, alias_map) for x in q.get("gold_drugs", [])]
        acceptable_list = [map_alias(x, alias_map) for x in q.get("acceptable", [])]
        must_avoid_list = [map_alias(x, alias_map) for x in q.get("must_avoid", [])]

        gold: Set[str] = set(gold_list + acceptable_list)  # рахуємо як релевант
        must_avoid: Set[str] = set(must_avoid_list)

        # Search
        cfg = SearchConfig(
            rrf_alpha=rrf_alpha,
            top_k=120,
            show=120,
            prefer_indications=True,
            indications_boost=0.05,
            max_chunk_tokens=max_chunk_tokens,
            enable_safety_filter=False  # для чистоти евалу вимкнемо, можна включити пізніше
        )
        groups = eng.search(raw_query, cfg)

        # Section gate
        groups = section_gate(groups, preferred=preferred_set, mode=gate_mode, boost=0.15)

        # Ranked drug names (normalized/alias-mapped)
        ranked = [map_alias(g["drug_name"], alias_map) for g in groups]

        # Binary gains для nDCG (1 якщо у gold, інакше 0)
        gains = [1 if name in gold else 0 for name in ranked]

        # Metrics
        prec, rec = precision_recall_at_k(ranked, gold, k=k)
        ndcg = ndcg_at_k(gains, k=k)
        violations = sum(1 for name in ranked[:k] if name in must_avoid)

        metrics["precision"].append(prec)
        metrics["recall"].append(rec)
        metrics["ndcg"].append(ndcg)
        metrics["violations"] += violations
        metrics["count"] += 1

        # Log per query (коротко)
        print(f"\n[{qi}/{len(queries)}] Q: {raw_query} | k={k}")
        print(f"  gold(~{len(gold)}): {list(gold)[:5]}{'...' if len(gold)>5 else ''}")
        print(f"  top@{k}: {ranked[:k]}")
        if violations:
            print(f"  ⚠️ must_avoid in top@{k}: {violations}")

    # Aggregate
    n = max(1, metrics["count"])
    P = sum(metrics["precision"]) / n
    R = sum(metrics["recall"]) / n
    N = sum(metrics["ndcg"]) / n
    V = metrics["violations"]

    print("\n==================== SUMMARY ====================")
    print(f"Queries: {n}")
    print(f"Precision@k: {P:.4f}")
    print(f"Recall@k:    {R:.4f}")
    print(f"nDCG@k:      {N:.4f}")
    print(f"must_avoid violations@k (total): {V}")
    print("=================================================")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--aliases", default=None)
    ap.add_argument("--rows", type=int, default=2000)
    ap.add_argument("--embed_model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--reranker_model", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--rrf_alpha", type=float, default=60.0)
    ap.add_argument("--max_chunk_tokens", type=int, default=128)
    ap.add_argument("--gate_mode", choices=["none", "prefer", "hard"], default="prefer")
    ap.add_argument("--gate_sections", default="Показання,Спосіб застосування та дози")
    args = ap.parse_args()

    gate_sections = [s.strip() for s in args.gate_sections.split(",") if s.strip()]
    run_eval(
        parquet_path=args.parquet,
        queries_path=args.queries,
        aliases_path=args.aliases,
        rows=args.rows,
        embed_model=args.embed_model,
        reranker_model=args.reranker_model,
        default_k=args.k,
        rrf_alpha=args.rrf_alpha,
        max_chunk_tokens=args.max_chunk_tokens,
        gate_mode=args.gate_mode,
        gate_sections=gate_sections,
    )
