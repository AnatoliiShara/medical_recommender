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

# ------------------------------- Local fusion utils -------------------------------

def _softmax(x: np.ndarray, T: float = 1.0) -> np.ndarray:
    if x.size == 0:
        return x
    z = (x - np.max(x)) / max(T, 1e-8)
    e = np.exp(z)
    s = e.sum()
    return e / s if s > 0 else np.zeros_like(x)

def normalize_scores(scores: Dict[int, float], mode: str = "none", temperature: float = 1.0) -> Dict[int, float]:
    if not scores:
        return {}
    idx = np.array(list(scores.keys()))
    val = np.array([scores[i] for i in idx], dtype=float)
    if mode == "none":
        norm = val
    elif mode == "minmax":
        lo, hi = float(val.min()), float(val.max())
        if hi > lo:
            norm = (val - lo) / (hi - lo)
        else:
            norm = np.zeros_like(val)
    elif mode == "softmax":
        norm = _softmax(val, T=max(temperature, 1e-8))
    else:
        norm = val
    return {int(i): float(v) for i, v in zip(idx, norm)}

def weighted_fusion(sources: List[Dict[int, float]], weights: List[float]) -> Dict[int, float]:
    out: Dict[int, float] = {}
    for src, w in zip(sources, weights):
        if not src or w == 0:
            continue
        for doc, sc in src.items():
            out[doc] = out.get(doc, 0.0) + w * sc
    return out

def rrf_fuse(rank_lists: List[List[int]], k: int = 60) -> Dict[int, float]:
    # Reciprocal Rank Fusion
    scores: Dict[int, float] = {}
    for ranks in rank_lists:
        for r, doc in enumerate(ranks, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + r)
    return scores

def as_ranked_list(rrf_out: Any) -> List[int]:
    """
    Приводить будь-який вихід до списку doc_id у порядку зменшення балів.
    Підтримує:
      - dict {doc_id: score}
      - list[int] (вже відсортований)
      - list[tuple(id, score)]
      - numpy масиви
    """
    if rrf_out is None:
        return []
    if isinstance(rrf_out, dict):
        return [int(doc) for doc, _ in sorted(rrf_out.items(), key=lambda kv: kv[1], reverse=True)]
    if isinstance(rrf_out, (list, tuple, np.ndarray)):
        seq = list(rrf_out)
        if not seq:
            return []
        first = seq[0]
        if isinstance(first, (tuple, list)) and len(first) >= 2:
            return [int(doc) for doc, _ in sorted(seq, key=lambda kv: kv[1], reverse=True)]
        return [int(x) for x in seq]
    try:
        return [int(x) for x in rrf_out]  # type: ignore
    except Exception:
        return []

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

# ------------------------------- Heuristics (defaults) -------------------------------
# ТРИГЕРИ ЗАПИТУ
ORAL_THRUSH_TRIGGER = re.compile(
    r"(кандидоз(?!\s+шкiри)|кандидозної\s+інфекції\s+ротоглотк|молочниця|thrush|oropharyn(g|ge)al|ротоглотк(и|и))",
    re.IGNORECASE,
)

ACID_QUERY_TRIGGER = re.compile(
    r"(гастрит|підвищен(а|ої)\s+кислотн|рефлюкс|печія|виразк(а|и))",
    re.IGNORECASE,
)

# Базові терміни (fallback), замінюємо словниками, якщо вони є
ANTIFUNGAL_TERMS = re.compile(
    r"(клотримазол|ністатин|натаміцин|міконазол|кетоконазол|флуконазол|ітраконазол|вориконазол|амфотерицин|деквалінію|ніфурател|пімафуцин)",
    re.IGNORECASE,
)

# Лише локальні/оромукозні форми (без системних таб/капсул) — fallback
ORAL_ALLOWED_FORM = re.compile(
    r"(спрей(\s+для\s+ротової\s+порожнини)?|для\s+ротової\s+порожнини|ополіскувач|полоскання|"
    r"льодяник(и)?|пастилк(а|и)|таблетк(а|и)\s+для\s+розсмоктування|"
    r"розчин\s+для\s+ротової\s+порожнини|аерозоль\s+для\s+ротової\s+порожнини|"
    r"оральн(ий|а)\s+гель|оральна\s+паста|суспензія\s+для\s+ротової\s+порожнини)",
    re.IGNORECASE,
)

# СИСТЕМНІ ПЕРОРАЛЬНІ ФОРМИ (для довідки/відсікання в bias)
ORAL_SYSTEMIC_FORM = re.compile(
    r"(таблетк(а|и)\b(?!\s+для\s+розсмоктування)|капсул(а|и)|суспензія|сироп|пероральн|оральн(?!\s+порожнини)|"
    r"розчин\s+для\s+інфузій|інфузійн(ий|а))",
    re.IGNORECASE,
)

# Конкуренти (антисептики для горла) — fallback
THROAT_ANTISEPTIC = re.compile(
    r"(бензидамін|хлоргексидин|цетилпіридин(і|и)й|амілметакрезол|2,4-ди(х|хл)лорбензилов(ий|ого)\s+спирт|лор|антисептик)",
    re.IGNORECASE,
)

# Дерматологічні форми — fallback
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

# ------------------------------- Heuristic logic -------------------------------

def heuristic_filter_ids(
    query: str,
    df: pd.DataFrame,
    actives_rx: Optional[re.Pattern],
    allowed_form_rx: Optional[re.Pattern],
    antiseptic_rx: Optional[re.Pattern],
) -> Tuple[Optional[set], Optional[str], str]:
    """
    Повертає (id_set, tag, mode), де:
      - id_set: множина id документів, на які слід накласти union/boost;
      - tag: назва евристики;
      - mode: "filter" або "boost" (зараз використовується лише як індикатор; у CE робимо UNION).
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

        ARX = actives_rx or ANTIFUNGAL_TERMS
        FRX = allowed_form_rx or ORAL_ALLOWED_FORM
        ANRX = antiseptic_rx or THROAT_ANTISEPTIC

        for i, (comp, form, blob) in enumerate(zip(comp_series, form_series, text_series)):
            comp = comp or ""
            form = form or ""
            blob = blob or ""

            if ARX.search(comp) or ARX.search(blob):
                if FRX.search(form) or FRX.search(blob):
                    strict.add(i)
                if ORAL_SYSTEMIC_FORM.search(form) or ORAL_SYSTEMIC_FORM.search(blob):
                    sys_any.add(i)
                soft_oral.add(i)

            if ANRX.search(comp) or ANRX.search(blob):
                throat_antiseptic.add(i)

        # Tagging decisions (режими історичні — фактично ми робимо UNION у CE)
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

# ------------------------------- Dense (FAISS) -----------------------------------

class DenseRetriever:
    def __init__(self, faiss_index_path: Optional[str], doc_ids_path: Optional[str], embed_model: Optional[str], metric: str = "ip"):
        self.enabled = False
        self.index = None
        self.idmap: Optional[np.ndarray] = None
        self.encoder: Optional[SentenceTransformer] = None
        self.metric = metric.lower()

        if faiss_index_path and os.path.exists(faiss_index_path):
            try:
                import faiss  # type: ignore
                self.faiss = faiss
                self.index = faiss.read_index(faiss_index_path)
                # id map
                if doc_ids_path and os.path.exists(doc_ids_path):
                    self.idmap = np.load(doc_ids_path, allow_pickle=True)
                # encoder
                if embed_model:
                    self.encoder = SentenceTransformer(embed_model)
                self.enabled = True
            except Exception as e:
                print(f"[WARN] Dense retriever disabled (FAISS init failed): {e}")

    def search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        if not self.enabled or not self.encoder or self.index is None:
            return []
        vec = self.encoder.encode([query], normalize_embeddings=True)
        D, I = self.index.search(np.array(vec).astype(np.float32), min(top_k, self.index.ntotal))
        dist = D[0]
        ids = I[0]
        out: List[Tuple[int, float]] = []
        for idx, d in zip(ids, dist):
            if idx < 0:
                continue
            # metric handling: for IP — більший краще; для L2 — менший краще (інвертуємо знак)
            score = float(d if self.metric == "ip" else -d)
            doc_id = int(self.idmap[idx]) if self.idmap is not None else int(idx)
            out.append((doc_id, score))
        # Вже у порядку від кращого до гіршого завдяки FAISS, але на всяк — пересортуємо.
        out.sort(key=lambda x: x[1], reverse=True)
        return out

# ------------------------------- CE Rerank -------------------------------

def ce_rerank(ce: CrossEncoder, query: str, docs: List[str], batch: int = 8) -> List[float]:
    if not docs:
        return []
    pairs = [[query, d] for d in docs]
    scores = ce.predict(pairs, batch_size=batch, show_progress_bar=False)
    return scores.tolist() if isinstance(scores, np.ndarray) else list(scores)

# ------------------------------- Utils -----------------------------------

def _load_lines(path: str) -> List[str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = []
            for ln in f:
                s = ln.strip()
                if not s or s.startswith("#"):
                    continue
                lines.append(s)
            return lines
    except Exception:
        return []

def compile_regex_from_file(path: str, mode: str = "words") -> Optional[re.Pattern]:
    """
    mode:
      - 'words' -> кожен рядок як слово/фраза, екранізуємо та огортаємо у \b...\b
      - 'regex' -> рядки трактуються як готові regex-патерни
    """
    if not path or not os.path.exists(path):
        return None
    lines = _load_lines(path)
    if not lines:
        return None
    if mode == "words":
        escaped = [r"\b" + re.escape(s) + r"\b" for s in lines]
        pat = "(?:" + "|".join(escaped) + ")"
    else:  # regex
        pat = "(?:" + "|".join(lines) + ")"
    try:
        return re.compile(pat, re.IGNORECASE)
    except Exception:
        return None

def safe_head(text: str, n: int = 80) -> str:
    s = (text or "").replace("\n", " ")
    return (s[:n] + "…") if len(s) > n else s

# ------------------------------- Main ------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="data/raw/compendium_all.parquet")
    p.add_argument("--index_dir", required=True)  # для сумісності з існуючими командами (шлях до ембеддингів/метаданих)
    p.add_argument("--faiss_index", default=None)
    p.add_argument("--embed_model", default=None)
    p.add_argument("--faiss_metric", choices=["ip", "l2"], default="ip")

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
    p.add_argument("--w_bm25", type=float, default=0.5)
    p.add_argument("--w_dense", type=float, default=0.5)

    p.add_argument("--norm", choices=["none", "minmax", "softmax"], default="softmax")
    p.add_argument("--temperature", type=float, default=0.3)

    p.add_argument("--dedup_by", default=None)

    p.add_argument("--top_k", type=int, default=60)
    p.add_argument("--rerank_top", type=int, default=20)
    p.add_argument("--max_doc_chars", type=int, default=1200)
    p.add_argument("--limit_queries", type=int, default=0)

    p.add_argument("--enable_heuristics", action="store_true")  # зарезервовано
    p.add_argument("--heuristic_ce_bias", type=float, default=0.35)

    # Словники для ротоглоткового кандидозу
    p.add_argument("--dict_allowed_forms", default="data/dicts/oral_thrush_allowed_forms.txt")
    p.add_argument("--dict_actives",       default="data/dicts/oral_thrush_actives.txt")
    p.add_argument("--dict_antiseptics",   default="data/dicts/throat_antiseptics.txt")
    p.add_argument("--dict_derm_block",    default="data/dicts/derm_block_forms.txt")

    args = p.parse_args()
    os.makedirs(args.dump_eval_dir, exist_ok=True)

    # DATA
    df = read_parquet_dataset(args.dataset)
    names = get_name_series(df).tolist()
    docs_raw = make_doc_text(df)

    # BM25
    print("[INFO] Lexical engine: bm25")
    bm25 = BM25Wrapper(docs_raw)

    # Dense
    doc_ids_path = os.path.join(args.index_dir, "doc_ids.npy")
    dense = DenseRetriever(args.faiss_index, doc_ids_path, args.embed_model, metric=args.faiss_metric)
    if dense.enabled:
        try:
            ntotal = dense.index.ntotal  # type: ignore
        except Exception:
            ntotal = len(df)
        print(f"[INFO] Dense retrieval enabled via FAISS + {args.embed_model} (ntotal={ntotal})")

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

    # Dictionaries -> regex overrides
    global ORAL_ALLOWED_FORM, THROAT_ANTISEPTIC, DERM_BLOCK_FORM, ANTIFUNGAL_TERMS
    _allowed_forms_rx = compile_regex_from_file(args.dict_allowed_forms, mode="regex")
    _actives_rx       = compile_regex_from_file(args.dict_actives,       mode="words")
    _antiseptics_rx   = compile_regex_from_file(args.dict_antiseptics,   mode="words")
    _derm_block_rx    = compile_regex_from_file(args.dict_derm_block,    mode="words")
    if _allowed_forms_rx:
        ORAL_ALLOWED_FORM = _allowed_forms_rx
    if _antiseptics_rx:
        THROAT_ANTISEPTIC = _antiseptics_rx
    if _derm_block_rx:
        DERM_BLOCK_FORM = _derm_block_rx
    # ACTIVES_RX для bias (якщо нема файлу — fallback на ANTIFUNGAL_TERMS)
    ACTIVES_RX = _actives_rx if _actives_rx else ANTIFUNGAL_TERMS

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

            # евристики -> беремо id для UNION
            filt_ids, heuristic_tag, heuristic_mode = heuristic_filter_ids(
                q, df, ACTIVES_RX, ORAL_ALLOWED_FORM, THROAT_ANTISEPTIC
            )

            # --- BM25 retrieve
            bm25_pairs = bm25.search(q, top_k=args.top_k)
            bm25_ids = [i for i, _ in bm25_pairs]
            bm25_scores = {i: s for i, s in bm25_pairs}

            # --- DENSE retrieve (optional)
            dense_pairs: List[Tuple[int, float]] = []
            dense_ids: List[int] = []
            dense_scores: Dict[int, float] = {}
            if dense.enabled:
                dense_pairs = dense.search(q, top_k=args.top_k)
                dense_ids = [i for i, _ in dense_pairs]
                dense_scores = {i: s for i, s in dense_pairs}

            # --- Fusion
            fused: List[int] = []
            if args.fusion == "weighted":
                # normalize & weight
                bm25_norm = normalize_scores(bm25_scores, mode=args.norm, temperature=args.temperature)
                dense_norm = normalize_scores(dense_scores, mode=args.norm, temperature=args.temperature) if dense_scores else {}
                combined = weighted_fusion([bm25_norm, dense_norm], [args.w_bm25, args.w_dense])
                fused = as_ranked_list(combined)
                if dense.enabled:
                    print(f"[Q{qi}] fusion=weighted  w_bm25={args.w_bm25}  w_dense={args.w_dense} k={args.rrf_k}")
                else:
                    print(f"[Q{qi}] fusion=weighted (bm25-only) w_bm25={args.w_bm25} k={args.rrf_k}")
            else:
                # rrf
                bm25_ranks = list(bm25_scores.keys())  # already ordered by score desc
                if dense.enabled and dense_ids:
                    rank_lists = [bm25_ids, dense_ids]
                    fused_out = rrf_fuse(rank_lists, k=args.rrf_k)
                    fused = as_ranked_list(fused_out)
                    print(f"[Q{qi}] fusion=rrf (hybrid) k={args.rrf_k}")
                else:
                    fused_out = rrf_fuse([bm25_ids], k=args.rrf_k)
                    fused = as_ranked_list(fused_out)
                    print(f"[Q{qi}] fusion=rrf (bm25-only) k={args.rrf_k}")

            # ---- UNION-кандидати перед CE (замість жорсткого filter)
            base = fused[: args.rerank_top] if fused else bm25_ids[: args.rerank_top]
            if filt_ids and heuristic_tag:
                seen = set()
                union = []
                for kdoc in list(base) + list(filt_ids):
                    if kdoc not in seen:
                        union.append(kdoc)
                        seen.add(kdoc)
                cand_ids = union[: args.rerank_top]
                print(f"[Q{qi}] heuristic UNION applied [{heuristic_tag}]: base {len(base)} + heur {len(filt_ids)} -> {len(cand_ids)}")
            else:
                cand_ids = base

            if not cand_ids:
                print(f"[Q{qi}] WARN: empty candidate set after fusion; skipping.")
                continue

            # Підготовка текстів для CE
            cand_docs = []
            for idx in cand_ids:
                blob = docs_raw[idx] if 0 <= idx < len(docs_raw) else ""
                if args.max_doc_chars:
                    blob = blob[: args.max_doc_chars]
                cand_docs.append(blob)

            # CE scores
            ce_scores = ce_rerank(ce, q, cand_docs, batch=args.ce_batch)

            # евристичний bias у CE-бали (boost)
            if heuristic_tag and args.heuristic_ce_bias > 0:
                bias = float(args.heuristic_ce_bias)
                ce_adj = []
                for idx, s, blob in zip(cand_ids, ce_scores, cand_docs):
                    s_adj = s
                    doc_text = blob or ""
                    if heuristic_tag in ("oral_thrush", "oral_thrush_soft", "oral_thrush_sys", "oral_thrush_soft_clean"):
                        # + активи, + дозволені форми; − антисептики, − дерматоформи
                        if ACTIVES_RX and ACTIVES_RX.search(doc_text):
                            s_adj += bias * 0.60
                        if ORAL_ALLOWED_FORM and ORAL_ALLOWED_FORM.search(doc_text):
                            s_adj += bias * 0.40
                        if THROAT_ANTISEPTIC and THROAT_ANTISEPTIC.search(doc_text):
                            s_adj -= bias * 0.80
                        if DERM_BLOCK_FORM and DERM_BLOCK_FORM.search(doc_text):
                            s_adj -= bias * 0.60
                    if heuristic_tag == "gastritis_acid":
                        if ACID_POS_TERMS.search(doc_text):
                            s_adj += bias * 0.75
                    ce_adj.append(s_adj)
                ce_scores = ce_adj
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
            tail = (fused[:15] if fused else bm25_ids[:15])
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
