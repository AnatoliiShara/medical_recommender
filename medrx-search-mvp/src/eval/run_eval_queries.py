# -*- coding: utf-8 -*-
"""
run_eval_queries.py
-------------------
Евалює гібридний пошук (BM25 + Dense + Weighted-RRF + optional CrossEncoder)
на основі EnhancedMedicalAssistant.

Приклад:
    PYTHONPATH=src python src/eval/run_eval_queries.py \
        --parquet data/raw/compendium_all.parquet \
        --queries data/eval/queries_all.enhanced.jsonl \
        --aliases data/dicts/brand_inn_aliases.PATCHED2.csv \
        --rows 0 \
        --embed_model intfloat/multilingual-e5-base \
        --ce_model cross-encoder/ms-marco-MiniLM-L-6-v2 \
        --ce_top 150 --ce_weight 0.70 \
        --k 10 --rrf_alpha 60.0 --w_bm25 1.0 --w_dense 1.0 \
        --gate_mode prefer \
        --gate_sections "Показання,Спосіб застосування та дози" \
        --out_csv data/eval/preds_hybrid.csv \
        --metrics_csv data/eval/summary_hybrid.csv
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

try:
    # Імпорт із нашого модуля пошуку
    from search.enhanced_medical_assistant import EnhancedMedicalAssistant, SearchConfig
except Exception as e:
    raise RuntimeError(
        "Не вдалося імпортувати EnhancedMedicalAssistant. "
        "Переконайтесь, що PYTHONPATH включає 'src'.\n"
        "Напр.: export PYTHONPATH=src:$PYTHONPATH"
    ) from e


# ----------------------------- JSONL utils ----------------------------------


def read_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # м'яко пропускаємо биті рядки
                continue
    return rows


def pick_query(obj: dict) -> Optional[str]:
    for key in ("query", "q", "text", "input"):
        if key in obj and isinstance(obj[key], str) and obj[key].strip():
            return obj[key].strip()
    return None


def pick_gold_names(obj: dict) -> List[str]:
    """
    Повертає список gold-найменувань (назв препаратів) із різних можливих ключів.
    Підтримує:
      - gold, golds, gold_names, answers, expected (списки рядків або одиночні рядки)
      - {"gold": {"names": [...]}}   # вкладені структури
    """
    candidates: List = []
    for key in ("gold", "golds", "gold_names", "answers", "expected"):
        if key in obj:
            candidates.append(obj[key])

    out: List[str] = []

    def _flatten(x):
        if x is None:
            return
        if isinstance(x, str):
            s = x.strip()
            if s:
                out.append(s)
        elif isinstance(x, (list, tuple)):
            for z in x:
                _flatten(z)
        elif isinstance(x, dict):
            # пробуємо типові вкладені поля
            for k in ("names", "gold", "expected"):
                if k in x:
                    _flatten(x[k])

    for c in candidates:
        _flatten(c)

    # dedup з збереженням порядку
    seen: Set[str] = set()
    uniq: List[str] = []
    for s in out:
        if s not in seen:
            uniq.append(s)
            seen.add(s)
    return uniq


# ----------------------------- Normalization --------------------------------

import re

_WS_RE = re.compile(r"\s+", flags=re.U)
_PUNCT_RE = re.compile(r"[^\w\u0400-\u04FF]+", flags=re.U)  # лат + кирилиця


def normalize_name(s: str) -> str:
    """
    Дуже проста нормалізація імені препарату (для співставлення з gold'ами).
    """
    s = s.strip().lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s)
    return s.strip()


def load_alias_map(path: Optional[str]) -> Dict[str, str]:
    """
    Опційна мапа alias->canon із CSV (колонки: term, canon, type).
    Якщо файл не задано/відсутній — повертаємо порожню мапу.
    """
    if not path:
        return {}
    if not os.path.exists(path):
        print(f"[WARN] aliases file not found: {path}")
        return {}
    try:
        df = pd.read_csv(path)
        # очікуємо колонки: term, canon
        term_col = "term" if "term" in df.columns else df.columns[0]
        canon_col = "canon" if "canon" in df.columns else df.columns[1]
        mp: Dict[str, str] = {}
        for _, row in df.iterrows():
            term = str(row.get(term_col, "")).strip()
            canon = str(row.get(canon_col, "")).strip()
            if not term or not canon:
                continue
            mp[normalize_name(term)] = normalize_name(canon)
        return mp
    except Exception as e:
        print(f"[WARN] failed to load aliases: {e}")
        return {}


def canon_name(name: str, alias_map: Dict[str, str]) -> str:
    n = normalize_name(name)
    return alias_map.get(n, n)


# ----------------------------- Metrics --------------------------------------


@dataclass
class Metrics:
    p_at_k: float
    r_at_k: float
    ndcg_at_k: float


def dcg_at_k(rels: List[int], k: int) -> float:
    """Binary DCG@k"""
    s = 0.0
    for i, r in enumerate(rels[:k]):
        if r <= 0:
            continue
        s += (2.0**r - 1.0) / math.log2(i + 2.0)
    return s


def ndcg_at_k(rels: List[int], k: int) -> float:
    dcg = dcg_at_k(rels, k)
    ideal = dcg_at_k(sorted(rels, reverse=True), k)
    if ideal <= 0.0:
        return 0.0
    return dcg / ideal


def eval_one(pred_names: List[str], gold_names: Set[str], k: int) -> Metrics:
    """Обчислює P@k, R@k, nDCG@k (бінарні релевантності)"""
    k = min(k, len(pred_names)) if pred_names else k
    hits = 0
    rels: List[int] = []
    for i, name in enumerate(pred_names[:k]):
        rel = 1 if name in gold_names else 0
        rels.append(rel)
        if rel == 1:
            hits += 1
    p = float(hits) / float(k) if k > 0 else 0.0
    r = float(hits) / float(len(gold_names)) if gold_names else 0.0
    return Metrics(p_at_k=p, r_at_k=r, ndcg_at_k=ndcg_at_k(rels, k))


# ----------------------------- Main logic -----------------------------------


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, help="Parquet із сирим Compendium (усі колонки секцій).")
    ap.add_argument("--queries", required=True, help="JSONL із полями query + gold-ами (у різних форматах).")
    ap.add_argument("--aliases", default=None, help="CSV (term,canon,...) для канонізації назв (опційно).")
    ap.add_argument("--rows", type=int, default=0, help="Ліміт рядків із parquet (0 = всі).")

    # Моделі
    ap.add_argument("--embed_model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--ce_model", default=None, help="Напр., cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--ce_top", type=int, default=0)
    ap.add_argument("--ce_weight", type=float, default=0.70)

    # Параметри пошуку/ф’южну
    ap.add_argument("--k", type=int, default=10, help="Top-K на рівні груп (препаратів).")
    ap.add_argument("--rrf_alpha", type=float, default=60.0)
    ap.add_argument("--w_bm25", type=float, default=1.0)
    ap.add_argument("--w_dense", type=float, default=1.0)
    ap.add_argument("--max_chunk_tokens", type=int, default=128)

    # Gate
    ap.add_argument("--gate_mode", choices=["none", "prefer", "require"], default="none")
    ap.add_argument(
        "--gate_sections",
        default="",
        help="Кома-сепарований список секцій, напр.: 'Показання,Спосіб застосування та дози'",
    )
    ap.add_argument("--prefer_boost", type=float, default=1.10)

    # Вивід
    ap.add_argument("--out_csv", default=None, help="Куди зберегти preds (CSV).")
    ap.add_argument("--metrics_csv", default=None, help="Куди зберегти summary-метрики (CSV).")

    return ap.parse_args()


def main():
    args = parse_args()

    # --- Завантаження даних
    df = pd.read_parquet(args.parquet)
    if args.rows and args.rows > 0:
        df = df.head(args.rows).copy()
    print(f"[INFO] Loaded parquet: {args.parquet} rows={len(df):,}")

    alias_map = load_alias_map(args.aliases)
    if alias_map:
        print(f"[INFO] Aliases loaded: {len(alias_map):,}")

    queries = read_jsonl(args.queries)
    print(f"[INFO] Loaded queries: {len(queries):,}")

    # --- Побудова індексів
    ema = EnhancedMedicalAssistant()
    ema.build_from_dataframe(
        df,
        encoder_model=args.embed_model,
        medical_chunking=True,
        max_chunk_tokens=int(args.max_chunk_tokens),
    )

    # --- Конфіг пошуку
    gate_sections = [s.strip() for s in args.gate_sections.split(",") if s.strip()]
    cfg = SearchConfig(
        top_k=int(args.k),
        rrf_alpha=float(args.rrf_alpha),
        w_bm25=float(args.w_bm25),
        w_dense=float(args.w_dense),
        gate_mode=args.gate_mode,
        gate_sections=gate_sections,
        prefer_boost=float(args.prefer_boost),
        ce_model=args.ce_model,
        ce_top=int(args.ce_top),
        ce_weight=float(args.ce_weight),
        max_chunk_tokens=int(args.max_chunk_tokens),
    )

    print(
        f"[CONFIG] k={cfg.top_k}  alpha={cfg.rrf_alpha}  w_bm25={cfg.w_bm25}  w_dense={cfg.w_dense}  "
        f"gate={cfg.gate_mode}:{','.join(cfg.gate_sections) if cfg.gate_sections else '-'}  "
        f"CE={'ON' if (cfg.ce_model and cfg.ce_top>0) else 'OFF'}"
    )

    # --- Евал
    preds_rows: List[Dict[str, object]] = []
    macro_p: List[float] = []
    macro_r: List[float] = []
    macro_ndcg: List[float] = []

    n_total = 0
    n_with_gold = 0

    for qi, obj in enumerate(queries):
        q = pick_query(obj)
        if not q:
            continue
        gold_list = pick_gold_names(obj)
        gold_canon: Set[str] = set(canon_name(g, alias_map) for g in gold_list if g and g.strip())

        n_total += 1
        if gold_canon:
            n_with_gold += 1

        groups = ema.search(q, cfg)

        # підготуємо предикти і локальні метрики
        pred_names = [canon_name(g["name"], alias_map) for g in groups]
        # збережемо подробиці (перших cfg.top_k)
        for rank, g in enumerate(groups[: cfg.top_k], start=1):
            name_c = canon_name(g["name"], alias_map)
            is_hit = int(name_c in gold_canon) if gold_canon else None
            preds_rows.append(
                {
                    "qid": qi,
                    "query": q,
                    "rank": rank,
                    "name": g["name"],
                    "name_canon": name_c,
                    "best_section": g.get("best_section", ""),
                    "score": float(g.get("best_score", g.get("score", 0.0))),
                    "is_hit": is_hit,
                }
            )

        # метрики тільки коли є gold
        if gold_canon:
            m = eval_one(pred_names, gold_canon, cfg.top_k)
            macro_p.append(m.p_at_k)
            macro_r.append(m.r_at_k)
            macro_ndcg.append(m.ndcg_at_k)

    # --- Зведення
    def _avg(x: List[float]) -> float:
        return float(np.mean(x)) if x else 0.0

    summary = {
        "queries_total": n_total,
        "queries_with_gold": n_with_gold,
        "p_at_k": round(_avg(macro_p), 6),
        "r_at_k": round(_avg(macro_r), 6),
        "ndcg_at_k": round(_avg(macro_ndcg), 6),
    }

    print("\n[SUMMARY]")
    for k, v in summary.items():
        print(f"- {k}: {v}")

    # --- Запис виходів
    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        pd.DataFrame(preds_rows).to_csv(args.out_csv, index=False, sep=",", encoding="utf-8")
        print(f"[OK] preds CSV saved: {args.out_csv}")

    if args.metrics_csv:
        os.makedirs(os.path.dirname(args.metrics_csv), exist_ok=True)
        pd.DataFrame([summary]).to_csv(args.metrics_csv, index=False, sep=",", encoding="utf-8")
        print(f"[OK] summary CSV saved: {args.metrics_csv}")


if __name__ == "__main__":
    main()
