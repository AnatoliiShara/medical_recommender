# src/search/assistant_from_parquet.py
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Tuple, Optional, Iterable, Any

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

# локальні утиліти
from src.search.fusion import rrf, normalize_scores, dedup_keep_best  # normalize_scores/dedup_keep_best — на майбутнє

# ------------------------------- IO / COLS -------------------------------

DEFAULT_TEXT_COLUMNS = [
    "Назва препарату",
    "Лікарська форма",
    "Фармакотерапевтична група",
    "Фармакологічні властивості",
    "Показання",
    "Протипоказання",
    "Особливості застосування",
    "Взаємодія з іншими лікарськими засобами",
    "Спосіб застосування та дози",
    "Побічні реакції",
    "Термін придатності",
    "Умови зберігання",
    "Упаковка",
    "Склад",
    "Виробник",
]

NAME_COL_CANDIDATES = ["Назва препарату", "Назва", "Препарат"]


def read_parquet_dataset(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    print(f"[INFO] Loaded dataset: {path} rows={len(df)}")
    name_cols = [c for c in NAME_COL_CANDIDATES if c in df.columns]
    if name_cols:
        print(f"[INFO] name columns detected: {name_cols}")
    return df


def make_doc_text(df: pd.DataFrame) -> List[str]:
    cols = [c for c in DEFAULT_TEXT_COLUMNS if c in df.columns]
    texts = []
    for _, row in df[cols].fillna("").iterrows():
        blob = " | ".join(str(row[c]) for c in cols)
        texts.append(blob)
    return texts


def get_name_series(df: pd.DataFrame) -> pd.Series:
    for c in NAME_COL_CANDIDATES:
        if c in df.columns:
            return df[c].fillna("").astype(str)
    # fallback
    return df[DEFAULT_TEXT_COLUMNS[0]].fillna("").astype(str)


# ------------------------------- Tokenize -------------------------------

WORD_RE = re.compile(r"[a-zа-яіїєґ0-9]+", re.IGNORECASE)


def tokenize_uk(text: str) -> List[str]:
    return WORD_RE.findall((text or "").lower())


# ------------------------------- Heuristics -------------------------------
# ТРИГЕРИ ЗАПИТУ
ORAL_THRUSH_TRIGGER = re.compile(
    r"(кандидоз(?!\s+шкiри)|кандидозної\s+інфекції\s+ротоглотк|молочниця|thrush|oropharyn(g|ge)al|ротоглотк(и|и))",
    re.IGNORECASE,
)

ACID_QUERY_TRIGGER = re.compile(
    r"(гастрит|підвищен(а|ої)\s+кислотн|рефлюкс|печія|виразк(а|и))",
    re.IGNORECASE,
)

# ТЕРМІНИ В ДОКУМЕНТАХ
ANTIFUNGAL_TERMS = re.compile(
    r"(клотримазол|ністатин|натаміцин|міконазол|кетоконазол|флуконазол|ітраконазол|вориконазол|амфотерицин|деквалінію|ніфурател|пімафуцин)",
    re.IGNORECASE,
)

# Лише локальні/оромукозні форми (без системних таб/капсул)
ORAL_ALLOWED_FORM = re.compile(
    r"(спрей(\s+для\s+ротової\s+порожнини)?|для\s+ротової\s+порожнини|ополіскувач|полоскання|"
    r"льодяник(и)?|пастилк(а|и)|таблетк(а|и)\s+для\s+розсмоктування|"
    r"розчин\s+для\s+ротової\s+порожнини|аерозоль\s+для\s+ротової\s+порожнини)",
    re.IGNORECASE,
)

# СИСТЕМНІ ПЕРОРАЛЬНІ ФОРМИ (для відсікання коли просили «місцеве»)
ORAL_SYSTEMIC_FORM = re.compile(
    r"(таблетк(а|и)\b(?!\s+для\s+розсмоктування)|капсул(а|и)|суспензія|сироп|пероральн|оральн(?!\s+порожнини)|"
    r"розчин\s+для\s+інфузій|інфузійн(ий|а))",
    re.IGNORECASE,
)

# Конкуренти (антисептики для горла)
THROAT_ANTISEPTIC = re.compile(
    r"(бензидамін|хлоргексидин|цетилпіридин(і|и)й|амілметакрезол|2,4-ди(х|хл)лорбензилов(ий|ого)\s+спирт|лор|антисептик)",
    re.IGNORECASE,
)

# Дерматологічні форми, які не підходять для ротоглотки
DERM_BLOCK_FORM = re.compile(
    r"(мазь|крем(?!\s+для\s+ротової)|шампунь|крем-гель|лосьйон(?!\s+для\s+ротової))",
    re.IGNORECASE,
)

# ACID POS терміни (контроль кислотності)
ACID_POS_TERMS = re.compile(
    r"(омепразол|пантопразол|рабепразол|лансопразол|езомепразол|де-нол|сукральфат|антацид|гавіскон|"
    r"альгінат|альгінова\s+кислота|фамотидин|ранитидин|протонн(ий|і)\s+насос|ppi|h2-блокатор)",
    re.IGNORECASE,
)


def heuristic_filter_ids(query: str, df: pd.DataFrame) -> Tuple[Optional[set], Optional[str], str]:
    """
    Повертає (id_set, tag, mode), де:
      - id_set: множина id документів, на які слід накласти filter/boost;
      - tag: назва евристики;
      - mode: "filter" або "boost".
    Якщо евристика не тригериться — (None, None, "none").
    """
    # ORAL THRUSH — локальна (ротоглотка)
    if ORAL_THRUSH_TRIGGER.search(query):
        comp_series = df.get("Склад", pd.Series([""] * len(df))).fillna("").astype(str)
        form_series = df.get("Лікарська форма", pd.Series([""] * len(df))).fillna("").astype(str)
        text_series = (
            df.get("Показання", pd.Series([""] * len(df))).fillna("").astype(str)
            + " "
            + comp_series
            + " "
            + form_series
        )

        strict, sys_any, soft_oral, throat_antiseptic = set(), set(), set(), set()

        for i, (comp, form, blob) in enumerate(zip(comp_series, form_series, text_series)):
            if ANTIFUNGAL_TERMS.search(comp) or ANTIFUNGAL_TERMS.search(blob):
                if ORAL_ALLOWED_FORM.search(form) or ORAL_ALLOWED_FORM.search(blob):
                    strict.add(i)
                if ORAL_SYSTEMIC_FORM.search(form) or ORAL_SYSTEMIC_FORM.search(blob):
                    sys_any.add(i)
                soft_oral.add(i)
            if THROAT_ANTISEPTIC.search(comp) or THROAT_ANTISEPTIC.search(blob):
                throat_antiseptic.add(i)

        if len(strict) >= 3:
            return strict, "oral_thrush", "filter"
        if 1 <= len(strict) < 3:
            return strict, "oral_thrush", "boost"
        if len(sys_any) >= 1:
            return sys_any, "oral_thrush_sys", "boost"
        if len(soft_oral) >= 1:
            soft_clean = soft_oral.difference(throat_antiseptic)
            if len(soft_clean) >= 1:
                return soft_clean, "oral_thrush_soft_clean", "boost"
            return soft_oral, "oral_thrush_soft", "boost"
        return None, None, "none"

    # ACID CONTROL — гастрит/кислотність
    if ACID_QUERY_TRIGGER.search(query):
        cols = [c for c in ["Показання", "Фармакологічні властивості", "Склад", "Лікарська форма"] if c in df.columns]
        series = (df[cols].fillna("").astype(str)).agg(" ".join, axis=1) if cols else pd.Series([""] * len(df))
        keep = {i for i, blob in enumerate(series) if ACID_POS_TERMS.search(blob)}
        return (keep if keep else None), "gastritis_acid", ("filter" if keep else "none")

    return None, None, "none"


# ------------------------------- Rewrite -------------------------------

def load_aliases_csv(path: Optional[str]) -> Dict[str, str]:
    if not path or not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path)
        lower_cols = {c.lower(): c for c in df.columns}
        alias_c = lower_cols.get("alias") or list(df.columns)[0]
        target_c = lower_cols.get("target") or list(df.columns)[1]
        mapping = {}
        for _, r in df[[alias_c, target_c]].fillna("").iterrows():
            a = str(r[alias_c]).strip().lower()
            t = str(r[target_c]).strip()
            if a:
                mapping[a] = t
        return mapping
    except Exception:
        return {}


def rewrite_query(q: str, aliases: Dict[str, str]) -> Tuple[str, bool]:
    orig = q
    q_low = q.lower()
    if ORAL_THRUSH_TRIGGER.search(q_low) and "clotrimazole" not in q_low and "клотримазол" not in q_low:
        q = f"{q} clotrimazole"
    tokens = tokenize_uk(q)
    rewritten, changed = [], False
    for t in tokens:
        if t in aliases:
            rewritten.append(aliases[t])
            changed = True
        else:
            rewritten.append(t)
    new_q = " ".join(rewritten) if changed else q
    return new_q, (new_q != orig)


# ------------------------------- BM25 -----------------------------------

class BM25Wrapper:
    def __init__(self, docs: List[str]):
        self.docs = docs
        self.tokens = [tokenize_uk(t) for t in docs]
        self.engine = BM25Okapi(self.tokens)

    def search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        qtok = tokenize_uk(query)
        if not qtok:
            return []
        scores = self.engine.get_scores(qtok)
        if len(scores) == 0:
            return []
        k = min(top_k, len(scores))
        idx = np.argpartition(scores, -k)[-k:]
        idx = idx[np.argsort(scores[idx])[::-1]]
        return [(int(i), float(scores[i])) for i in idx]


# ------------------------------- DENSE (FAISS) ---------------------------

class DenseIndex:
    def __init__(self, faiss_index_path: str, doc_ids_path: str, embed_model_name: str):
        import faiss  # імпортуємо тут, щоб не ламати інсталяції без faiss

        if not os.path.exists(faiss_index_path):
            raise FileNotFoundError(f"FAISS index not found: {faiss_index_path}")
        if not os.path.exists(doc_ids_path):
            raise FileNotFoundError(f"doc_ids.npy not found: {doc_ids_path}")

        self.faiss = faiss
        self.index = faiss.read_index(faiss_index_path)
        self.doc_ids = np.load(doc_ids_path, allow_pickle=True)
        self.model = SentenceTransformer(embed_model_name)
        self.embed_model_name = embed_model_name

    def search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        # E5: краще додати префікс "query: "
        q_text = f"query: {query}"
        q_vec = self.model.encode(q_text, normalize_embeddings=True)
        if not isinstance(q_vec, np.ndarray):
            q_vec = np.asarray(q_vec)
        q_vec = q_vec.astype(np.float32, copy=False)
        D, I = self.index.search(q_vec[None, :], top_k)
        if I is None or len(I) == 0:
            return []
        I = I[0]
        D = D[0]
        out: List[Tuple[int, float]] = []
        for idx, score in zip(I, D):
            if idx < 0:
                continue
            # мапимо індекс FAISS -> id документа в parquet
            did = int(self.doc_ids[idx]) if idx < len(self.doc_ids) else int(idx)
            out.append((did, float(score)))
        # FAISS вже повертає за спаданням score; на всяк випадок — відсортуємо
        out.sort(key=lambda kv: kv[1], reverse=True)
        return out


# ------------------------------- CE Rerank -------------------------------

def ce_rerank(ce: CrossEncoder, query: str, docs: List[str], batch: int = 8) -> List[float]:
    if not docs:
        return []
    pairs = [[query, d] for d in docs]
    scores = ce.predict(pairs, batch_size=batch, show_progress_bar=False)
    return scores.tolist() if isinstance(scores, np.ndarray) else list(scores)


# ------------------------------- Utils -----------------------------------

def as_ranked_list(rrf_out: Any) -> List[int]:
    """
    Приводить будь-який вихід rrf(...) до списку doc_id у порядку зменшення балів.
    Підтримує:
      - dict {doc_id: score}
      - list[int] (вже відсортований)
      - list[tuple(id, score)]
      - numpy масиви
    """
    if rrf_out is None:
        return []
    # dict: сортуємо за score desc
    if isinstance(rrf_out, dict):
        return [int(doc) for doc, _ in sorted(rrf_out.items(), key=lambda kv: kv[1], reverse=True)]
    # list/tuple/np.array
    if isinstance(rrf_out, (list, tuple, np.ndarray)):
        seq = list(rrf_out)
        if len(seq) == 0:
            return []
        first = seq[0]
        if isinstance(first, (tuple, list)) and len(first) >= 2:
            return [int(doc) for doc, _ in sorted(seq, key=lambda kv: kv[1], reverse=True)]
        return [int(x) for x in seq]
    # щось інше — консервативний fallback
    try:
        return [int(x) for x in rrf_out]  # type: ignore
    except Exception:
        return []


def safe_head(text: str, n: int = 80) -> str:
    s = (text or "").replace("\n", " ")
    return (s[:n] + "…") if len(s) > n else s


# ------------------------------- Main ------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/raw/compendium_all.parquet")
    p.add_argument("--index_dir", required=True)
    p.add_argument("--faiss_index", default=None)
    p.add_argument("--embed_model", default=None)
    p.add_argument("--ce_model", default="BAAI/bge-reranker-v2-m3")
    p.add_argument("--ce_batch", type=int, default=4)

    p.add_argument("--queries", required=True)
    p.add_argument("--dump_eval_dir", required=True)

    p.add_argument("--intent_policy", default=None)

    p.add_argument("--use_rewrite", action="store_true")
    p.add_argument("--rewrite_aliases_csv", default=None)
    p.add_argument("--rewrite_max_terms", type=int, default=5)

    p.add_argument("--fusion", choices=["rrf", "weighted"], default="rrf")
    p.add_argument("--rrf_k", type=int, default=60)
    p.add_argument("--rrf_alpha", type=float, default=0.0)  # зарезервовано
    p.add_argument("--w_bm25", type=float, default=0.5)
    p.add_argument("--w_dense", type=float, default=0.5)

    p.add_argument("--norm", choices=["none", "minmax", "softmax"], default="softmax")
    p.add_argument("--temperature", type=float, default=0.3)

    p.add_argument("--dedup_by", default=None)

    p.add_argument("--top_k", type=int, default=60)
    p.add_argument("--rerank_top", type=int, default=20)
    p.add_argument("--max_doc_chars", type=int, default=1200)
    p.add_argument("--limit_queries", type=int, default=0)

    p.add_argument("--enable_heuristics", action="store_true")
    p.add_argument("--heuristic_ce_bias", type=float, default=0.35)

    args = p.parse_args()
    os.makedirs(args.dump_eval_dir, exist_ok=True)

    # DATA
    df = read_parquet_dataset(args.dataset)
    names = get_name_series(df).tolist()
    docs_raw = make_doc_text(df)

    # BM25
    print("[INFO] Lexical engine: bm25")
    bm25 = BM25Wrapper(docs_raw)

    # DENSE (FAISS)
    use_dense = False
    dense: Optional[DenseIndex] = None
    faiss_path = args.faiss_index or os.path.join(args.index_dir, "faiss.index")
    doc_ids_path = os.path.join(args.index_dir, "doc_ids.npy")
    if args.embed_model and os.path.exists(faiss_path) and os.path.exists(doc_ids_path):
        try:
            dense = DenseIndex(faiss_path, doc_ids_path, args.embed_model)
            use_dense = True
            print(f"[INFO] Dense retrieval enabled via FAISS + {args.embed_model} (ntotal={dense.index.ntotal})")
        except Exception as e:
            print(f"[WARN] Dense retrieval disabled: {e}")
    else:
        if not args.embed_model:
            print("[INFO] Dense retrieval disabled: --embed_model not provided")
        if not os.path.exists(faiss_path):
            print(f"[INFO] Dense retrieval disabled: no FAISS index at {faiss_path}")
        if not os.path.exists(doc_ids_path):
            print(f"[INFO] Dense retrieval disabled: no doc_ids at {doc_ids_path}")

    # CE
    ce = CrossEncoder(args.ce_model)
    print(f"[INFO] CrossEncoder loaded: {args.ce_model}; batch={args.ce_batch}")

    # Queries
    queries = []
    with open(args.queries, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            queries.append(item)
    if args.limit_queries:
        queries = queries[: args.limit_queries]
    print(f"[INFO] Loaded queries: {len(queries)}")

    # Aliases
    aliases = load_aliases_csv(args.rewrite_aliases_csv) if args.use_rewrite else {}

    out_path = os.path.join(args.dump_eval_dir, "predictions.jsonl")
    with open(out_path, "w", encoding="utf-8") as fout:
        for qi, qobj in enumerate(queries, 1):
            q = qobj.get("query") or qobj.get("text") or ""
            intent = qobj.get("intent") or "indication"

            # optional rewrite
            if args.use_rewrite:
                new_q, changed = rewrite_query(q, aliases)
                if changed:
                    print(f"[Q{qi}] rewrite: '{q}' -> '{new_q}'")
                q = new_q

            print(f"[Q{qi}] {intent}: {q}")

            # евристики
            filt_ids, heuristic_tag, heuristic_mode = heuristic_filter_ids(q, df)

            # BM25 retrieve
            bm25_pairs = bm25.search(q, top_k=args.top_k)
            if not bm25_pairs:
                print(f"[Q{qi}] WARN: no BM25 results")
                continue
            bm25_ids = [i for i, _ in bm25_pairs]
            bm25_scores = dict(bm25_pairs)

            # Dense retrieve
            dense_pairs: List[Tuple[int, float]] = []
            if use_dense and dense is not None:
                dense_pairs = dense.search(q, top_k=args.top_k)
            dense_ids = [i for i, _ in dense_pairs]
            dense_scores = dict(dense_pairs) if dense_pairs else {}

            # --------------------- Fusion ---------------------
            fused: List[int] = []
            fusion_label = ""
            if args.fusion == "rrf":
                # RRF очікує словники ранґів: {doc_id: rank}
                bm25_ranks = {doc_id: rank for rank, doc_id in enumerate(bm25_ids, start=1)}
                rank_lists = {"bm25": bm25_ranks}
                if use_dense and dense_ids:
                    dense_ranks = {doc_id: rank for rank, doc_id in enumerate(dense_ids, start=1)}
                    rank_lists["dense"] = dense_ranks
                    fusion_label = "rrf (hybrid)"
                else:
                    fusion_label = "rrf (bm25-only)"
                fused_out = rrf(rank_lists, k=args.rrf_k)
                fused = as_ranked_list(fused_out)
            elif args.fusion == "weighted":
                bm25_norm = normalize_scores(bm25_scores, method=args.norm, temperature=args.temperature)
                dense_norm = normalize_scores(dense_scores, method=args.norm, temperature=args.temperature) if dense_scores else {}
                fused_scores: Dict[int, float] = {}
                for doc in set(list(bm25_norm.keys()) + list(dense_norm.keys())):
                    s_b = bm25_norm.get(doc, 0.0)
                    s_d = dense_norm.get(doc, 0.0)
                    fused_scores[doc] = args.w_bm25 * s_b + args.w_dense * s_d
                fused = [doc for doc, _ in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)]
                fusion_label = f"weighted  w_bm25={args.w_bm25}  w_dense={args.w_dense}"
            else:
                raise ValueError(f"Unknown fusion mode: {args.fusion}")

            print(f"[Q{qi}] fusion={fusion_label} k={args.rrf_k}")

            # застосування евристики до списку кандидатів перед CE
            if filt_ids and heuristic_tag:
                if heuristic_mode == "filter":
                    fused2 = [i for i in fused if i in filt_ids]
                    print(f"[Q{qi}] heuristic filter applied [{heuristic_tag}]: kept {len(fused2)} docs")
                    fused = fused2 if fused2 else bm25_ids[:]  # без результатів — fallback до bm25

            # обираємо topN для CE
            cand_ids = fused[: args.rerank_top] if fused else bm25_ids[: args.rerank_top]
            if not cand_ids:
                print(f"[Q{qi}] WARN: empty candidate set after fusion; skipping.")
                continue

            cand_docs = []
            for idx in cand_ids:
                blob = docs_raw[idx] if 0 <= idx < len(docs_raw) else ""
                if args.max_doc_chars:
                    blob = blob[: args.max_doc_chars]
                cand_docs.append(blob)

            # CE scores
            ce_scores = ce_rerank(ce, q, cand_docs, batch=args.ce_batch)

            # евристичний bias у CE-бали (boost режим)
            if heuristic_tag and heuristic_mode != "filter" and args.heuristic_ce_bias > 0:
                bias = float(args.heuristic_ce_bias)
                ce_adj = []
                for idx, s, blob in zip(cand_ids, ce_scores, cand_docs):
                    s_adj = s
                    doc_text = blob
                    if heuristic_tag in ("oral_thrush", "oral_thrush_soft", "oral_thrush_sys", "oral_thrush_soft_clean"):
                        if ANTIFUNGAL_TERMS.search(doc_text) and (
                            ORAL_ALLOWED_FORM.search(doc_text) or ORAL_SYSTEMIC_FORM.search(doc_text)
                        ):
                            s_adj += bias * 1.00
                        if THROAT_ANTISEPTIC.search(doc_text):
                            s_adj -= bias * 1.20
                        if DERM_BLOCK_FORM.search(doc_text):
                            s_adj -= bias * 0.60
                    if heuristic_tag == "gastritis_acid":
                        if ACID_POS_TERMS.search(doc_text):
                            s_adj += bias * 0.75
                    ce_adj.append(s_adj)
                ce_scores = ce_adj
                if heuristic_mode == "boost":
                    print(f"[Q{qi}] heuristic BOOST applied [{heuristic_tag}]: top {len(cand_ids)} promoted")

            # фінальний порядок topN за CE
            order = list(np.argsort(ce_scores)[::-1]) if ce_scores else list(range(len(cand_ids)))
            final_ids = [cand_ids[i] for i in order][:10]

            # друк TOP10
            print(f"[Q{qi}] TOP10:")
            for rank, did in enumerate(final_ids, 1):
                nm = names[did] if 0 <= did < len(names) else f"doc_{did}"
                print(f"  {rank:02d}. {nm}")

            # збереження (додаємо трохи ширший “tail” для аналізу)
            tail = fused[:15] if fused else bm25_ids[:15]
            rec = {
                "query_id": str(qobj.get("id") or qi - 1),
                "query": q,
                "intent": intent,
                "predictions": [str(i) for i in final_ids + tail],
                "gold": qobj.get("gold", []),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[DONE] wrote predictions -> {out_path}")


if __name__ == "__main__":
    sys.exit(main())
