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

# optional faiss (dense retrieval)
try:
    import faiss  # type: ignore
    _FAISS_OK = True
except Exception:
    _FAISS_OK = False

# локальні утиліти
from src.search.fusion import rrf, normalize_scores


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
    present = [c for c in DEFAULT_TEXT_COLUMNS if c in df.columns]
    if present:
        print(f"[INFO] text columns used for BM25/CE: {present}")
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


# ------------------------------- Dict utils -------------------------------

def _clean_line(s: str) -> str:
    return s.strip().strip('"').strip("'").lower()


def load_lines(path: Optional[str]) -> List[str]:
    if not path or not os.path.exists(path):
        return []
    try:
        out = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = _clean_line(line)
                if not s or s.startswith("#"):
                    continue
                out.append(s)
        return out
    except Exception:
        return []


def any_term_in(text: str, terms: List[str]) -> bool:
    if not terms:
        return False
    t = (text or "").lower()
    return any(term in t for term in terms)


# ------------------------------- Heuristics: SPECIAL (oral thrush) -------------------------------
ORAL_THRUSH_TRIGGER = re.compile(
    r"(кандидоз(?!\s+шкiри)|кандидозної\s+інфекції\s+ротоглотк|молочниця|thrush|oropharyn(g|ge)al|ротоглотк(и|и))",
    re.IGNORECASE,
)

ANTIFUNGAL_TERMS = re.compile(
    r"(клотримазол|ністатин|натаміцин|міконазол|кетоконазол|флуконазол|ітраконазол|вориконазол|амфотерицин|декваліній|dequalinium|ніфурател|пімафуцин)",
    re.IGNORECASE,
)

ORAL_ALLOWED_FORM = re.compile(
    r"(спрей(\s+для\s+ротової\s+порожнини)?|для\s+ротової\s+порожнини|ополіскувач|полоскання|"
    r"льодяник(и)?|пастилк(а|и)|таблетк(а|и)\s+для\s+розсмоктування|"
    r"розчин\s+для\s+ротової\s+порожнини|аерозоль\s+для\s+ротової\s+порожнини|оральн(ий|а)\s+гель|оральна\s+паста|"
    r"суспензія\s+для\s+ротової\s+порожнини)",
    re.IGNORECASE,
)

ORAL_SYSTEMIC_FORM = re.compile(
    r"(таблетк(а|и)\b(?!\s+для\s+розсмоктування)|капсул(а|и)|суспензія\b(?!\s+для\s+ротової)|сироп\b|пероральн(ий|а)\b(?!\s+порожнини)|"
    r"розчин\s+для\s+інфузій|інфузійн(ий|а))",
    re.IGNORECASE,
)

THROAT_ANTISEPTIC = re.compile(
    r"(бензидамін|хлоргексидин|цетилпіридин(і|и)й|амілметакрезол|2,4-ди(х|хл)лорбензилов(ий|ого)\s+спирт|гексаспрей|biclotymol|біклотимол)",
    re.IGNORECASE,
)

DERM_BLOCK_FORM = re.compile(
    r"(мазь|крем(?!\s+для\s+ротової)|шампунь|крем-гель|лосьйон(?!\s+для\s+ротової)|"
    r"вагінальн|овул(я|і)|вагінальні\s+супозиторії|свічк(и)?\s+вагінальн(і|а|ий)|"
    r"крем\s+вагінальний|гель\s+вагінальний|таблетк(а|и)\s+вагінальн(і|а))",
    re.IGNORECASE,
)


# ------------------------------- Heuristics: GENERIC CLINICAL DICTS -------------------------------

class ClinicalDicts:
    """Завантажує data/dicts/clinical/<condition>/{*_trigger,_positive,_penalty}.txt"""
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.conditions: Dict[str, Dict[str, List[str]]] = {}
        self._load()

    def _load(self) -> None:
        if not self.root_dir or not os.path.isdir(self.root_dir):
            return
        for cond in sorted(os.listdir(self.root_dir)):
            cdir = os.path.join(self.root_dir, cond)
            if not os.path.isdir(cdir):
                continue
            entry: Dict[str, List[str]] = {"trigger": [], "positive": [], "penalty": []}
            for fname in os.listdir(cdir):
                path = os.path.join(cdir, fname)
                low = fname.lower()
                if low.endswith("_trigger.txt"):
                    entry["trigger"].extend(load_lines(path))
                elif low.endswith("_positive.txt"):
                    entry["positive"].extend(load_lines(path))
                elif low.endswith("_penalty.txt"):
                    entry["penalty"].extend(load_lines(path))
            if any(entry.values()):
                self.conditions[cond] = entry

    def detect(self, query: str) -> List[str]:
        q = (query or "").lower()
        matched = []
        for cond, d in self.conditions.items():
            trg = d.get("trigger", [])
            if trg and any_term_in(q, trg):
                matched.append(cond)
        return matched

    def positives(self, cond: str) -> List[str]:
        return self.conditions.get(cond, {}).get("positive", []) or []

    def penalties(self, cond: str) -> List[str]:
        return self.conditions.get(cond, {}).get("penalty", []) or []


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


# ------------------------------- Dense (FAISS) -------------------------------

class DenseFaiss:
    def __init__(self, faiss_index_path: str, index_dir: str, embed_model_name: str):
        self.index = faiss.read_index(faiss_index_path)
        self.doc_ids = np.load(os.path.join(index_dir, "doc_ids.npy"), allow_pickle=True)
        self.encoder = SentenceTransformer(embed_model_name)
        self._normalize = True  # e5 краще з normalize_embeddings=True

    def encode_query(self, q: str) -> np.ndarray:
        v = self.encoder.encode([q], normalize_embeddings=self._normalize)
        return v.astype("float32")

    def search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        qv = self.encode_query(query)
        D, I = self.index.search(qv, min(top_k, len(self.doc_ids)))
        idxs = I[0].tolist()
        dists = D[0].tolist()
        out = []
        for i, s in zip(idxs, dists):
            if i < 0:
                continue
            out.append((int(i), float(s)))
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


def safe_head(text: str, n: int = 80) -> str:
    s = (text or "").replace("\n", " ")
    return (s[:n] + "…") if len(s) > n else s


def dedup_preserve_order(seq: Iterable[int]) -> List[int]:
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(int(x))
    return out


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
    p.add_argument("--rrf_alpha", type=float, default=0.0)
    p.add_argument("--w_bm25", type=float, default=0.5)
    p.add_argument("--w_dense", type=float, default=0.5)

    p.add_argument("--norm", choices=["none", "minmax", "softmax"], default="softmax")
    p.add_argument("--temperature", type=float, default=0.3)

    p.add_argument("--dedup_by", default=None)

    p.add_argument("--top_k", type=int, default=60)
    p.add_argument("--rerank_top", type=int, default=20)
    p.add_argument("--max_doc_chars", type=int, default=1200)
    p.add_argument("--limit_queries", type=int, default=0)

    # Heuristics / dicts
    p.add_argument("--enable_heuristics", action="store_true")  # історичний прапорець
    p.add_argument("--heuristic_ce_bias", type=float, default=0.35)

    # NEW: клінічні словники
    p.add_argument("--use_clinical_dicts", action="store_true")
    p.add_argument("--clinical_dicts_dir", default="data/dicts/clinical")
    p.add_argument("--heuristic_union_cap", type=int, default=96)
    p.add_argument("--heuristic_bm25_gate_top", type=int, default=64)

    args = p.parse_args()
    os.makedirs(args.dump_eval_dir, exist_ok=True)

    # DATA
    df = read_parquet_dataset(args.dataset)
    names = get_name_series(df).tolist()
    docs_raw = make_doc_text(df)
    docs_raw_lower = [d.lower() for d in docs_raw]

    # BM25
    print("[INFO] Lexical engine: bm25")
    bm25 = BM25Wrapper(docs_raw)

    # Dense
    dense = None
    if _FAISS_OK and args.faiss_index and os.path.exists(args.faiss_index) and args.embed_model:
        try:
            dense = DenseFaiss(args.faiss_index, args.index_dir, args.embed_model)
            print(f"[INFO] Dense retrieval enabled via FAISS + {args.embed_model} (ntotal={len(dense.doc_ids)})")
        except Exception as e:
            print(f"[WARN] Dense retrieval init failed: {e}")

    # CE
    ce = CrossEncoder(args.ce_model)
    print(f"[INFO] CrossEncoder loaded: {args.ce_model}; batch={args.ce_batch}")

    # Dicts: загальні та спеціальні
    dict_dir = os.path.join("data", "dicts")
    oral_forms_allowed = load_lines(os.path.join(dict_dir, "oral_thrush_allowed_forms.txt"))
    oral_actives = load_lines(os.path.join(dict_dir, "oral_thrush_actives.txt"))
    throat_antiseptics_ls = load_lines(os.path.join(dict_dir, "throat_antiseptics.txt"))
    derm_block_ls = load_lines(os.path.join(dict_dir, "derm_block_forms.txt"))
    penalty_general = load_lines(os.path.join(dict_dir, "penalty_general.txt"))

    # NEW: клінічні словники
    clinical = ClinicalDicts(args.clinical_dicts_dir) if args.use_clinical_dicts else None
    if clinical and clinical.conditions:
        print(f"[INFO] Clinical dicts loaded: {len(clinical.conditions)} conditions -> {sorted(clinical.conditions.keys())}")

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
            q_low = q.lower()

            # --- 1) BM25 retrieve
            bm25_pairs = bm25.search(q, top_k=args.top_k)
            bm25_ids = [i for i, _ in bm25_pairs]
            bm25_scores = {i: s for i, s in bm25_pairs}

            # --- 2) Dense retrieve (optional)
            dense_pairs: List[Tuple[int, float]] = []
            dense_scores: Dict[int, float] = {}
            dense_ids: List[int] = []
            if dense is not None:
                try:
                    dense_pairs = dense.search(q, top_k=args.top_k)
                    dense_ids = [i for i, _ in dense_pairs]
                    dense_scores = {i: s for i, s in dense_pairs}
                except Exception as e:
                    print(f"[Q{qi}] WARN: dense retrieval failed: {e}")

            # --- 3) Fusion
            if args.fusion == "rrf":
                bm25_ranks = {doc_id: rank for rank, doc_id in enumerate(bm25_ids, start=1)}
                sources = {"bm25": bm25_ranks}
                if dense_ids:
                    dense_ranks = {doc_id: rank for rank, doc_id in enumerate(dense_ids, start=1)}
                    sources["dense"] = dense_ranks
                fused_out = rrf(sources, k=args.rrf_k)
                fused = as_ranked_list(fused_out)
                print(f"[Q{qi}] fusion=rrf  k={args.rrf_k}")
            else:
                all_ids = set(bm25_ids) | set(dense_ids)
                if not all_ids:
                    print(f"[Q{qi}] WARN: no candidates from BM25/Dense")
                    continue
                bm25_norm = normalize_scores({i: bm25_scores.get(i, 0.0) for i in all_ids},
                                             method=args.norm, temperature=args.temperature)
                dense_norm = normalize_scores({i: dense_scores.get(i, 0.0) for i in all_ids},
                                              method=args.norm, temperature=args.temperature)
                fused_scores = {}
                for i in all_ids:
                    fused_scores[i] = args.w_bm25 * bm25_norm.get(i, 0.0) + args.w_dense * dense_norm.get(i, 0.0)
                fused = [i for i, _ in sorted(fused_scores.items(), key=lambda kv: kv[1], reverse=True)]
                print(f"[Q{qi}] fusion=weighted  w_bm25={args.w_bm25}  w_dense={args.w_dense} k={args.top_k}")

            # --- 4) UNION логіка від клінічних словників + спеціальний thrush
            base = fused[: args.rerank_top] if fused else bm25_ids[: args.rerank_top]
            cand_ids = list(base)
            heuristic_tags_applied: List[str] = []

            matched_conditions: List[str] = []
            if clinical and clinical.conditions:
                matched_conditions = clinical.detect(q_low)
                if matched_conditions:
                    print(f"[Q{qi}] clinical HEUR matched: {', '.join(matched_conditions)}")
                for cond in matched_conditions:
                    pos_terms = clinical.positives(cond)
                    if not pos_terms:
                        continue

                    # знайти документи за позитивними термінами
                    heur_ids = [i for i, blob in enumerate(docs_raw_lower) if any_term_in(blob, pos_terms)]
                    if not heur_ids:
                        continue

                    # BM25-gate: беремо тільки з top-N BM25 (чистий перетин)
                    gateN = max(0, min(args.heuristic_bm25_gate_top, args.top_k))
                    gate_set = set(bm25_ids[:gateN])
                    picked = [i for i in heur_ids if i in gate_set][:gateN]

                    # cap
                    cap = max(0, args.heuristic_union_cap)
                    union_before = len(cand_ids)
                    for i in picked[:cap]:
                        if i not in cand_ids:
                            cand_ids.append(i)
                    print(
                        f"[Q{qi}] heuristic UNION CAPPED [{cond}]: base {union_before} + heur {len(heur_ids)} -> {len(cand_ids)} "
                        f"(cap={cap}, picked={len(picked)} by BM25-gate={gateN})"
                    )
                    heuristic_tags_applied.append(cond)

            # 4.b) SPECIAL: oral thrush
            heuristic_tag_special = None
            if ORAL_THRUSH_TRIGGER.search(q_low):
                strict_ids = []
                for i, blob in enumerate(docs_raw):
                    comp_form_text = blob
                    has_antifungal = bool(ANTIFUNGAL_TERMS.search(comp_form_text))
                    has_allowed_oral = bool(ORAL_ALLOWED_FORM.search(comp_form_text)) or any_term_in(
                        comp_form_text.lower(), oral_forms_allowed
                    )
                    if has_antifungal and has_allowed_oral:
                        strict_ids.append(i)
                if strict_ids:
                    gateN = max(0, min(args.heuristic_bm25_gate_top, args.top_k))
                    gate_set = set(bm25_ids[:gateN])
                    picked = [i for i in strict_ids if i in gate_set][:gateN]
                    cap = max(0, args.heuristic_union_cap)
                    union_before = len(cand_ids)
                    for i in picked[:cap]:
                        if i not in cand_ids:
                            cand_ids.append(i)
                    print(
                        f"[Q{qi}] heuristic UNION CAPPED [oral_thrush]: base {union_before} + heur {len(strict_ids)} -> {len(cand_ids)} "
                        f"(cap={cap}, picked={len(picked)} by BM25-gate={gateN})"
                    )
                    heuristic_tag_special = "oral_thrush"

            if not cand_ids:
                print(f"[Q{qi}] WARN: empty candidate set after fusion; skipping.")
                continue

            # --- 5) CE rerank
            cand_docs = []
            for idx in cand_ids:
                blob = docs_raw[idx] if 0 <= idx < len(docs_raw) else ""
                if args.max_doc_chars:
                    blob = blob[: args.max_doc_chars]
                cand_docs.append(blob)

            ce_scores = ce_rerank(ce, q, cand_docs, batch=args.ce_batch)

            # --- 6) CE-bias
            if args.heuristic_ce_bias > 0:
                bias = float(args.heuristic_ce_bias)
                ce_adj: List[float] = []

                for idx, s, blob in zip(cand_ids, ce_scores, cand_docs):
                    s_adj = float(s)
                    doc_text = (blob or "").lower()

                    # Загальні штрафи
                    if penalty_general and any_term_in(doc_text, penalty_general):
                        s_adj -= bias * 0.60

                    # SPECIAL patch: oral thrush
                    if heuristic_tag_special == "oral_thrush":
                        has_antifungal = bool(ANTIFUNGAL_TERMS.search(doc_text))
                        has_allowed_oral = bool(ORAL_ALLOWED_FORM.search(doc_text)) or any_term_in(
                            doc_text, oral_forms_allowed
                        )
                        has_systemic_oral = bool(ORAL_SYSTEMIC_FORM.search(doc_text))
                        has_antiseptic = bool(THROAT_ANTISEPTIC.search(doc_text)) or any_term_in(
                            doc_text, throat_antiseptics_ls
                        )
                        has_derm = bool(DERM_BLOCK_FORM.search(doc_text)) or any_term_in(doc_text, derm_block_ls)
                        has_active_dict = any_term_in(doc_text, oral_actives)

                        if has_allowed_oral and (has_antifungal or has_active_dict):
                            s_adj += bias * 1.00
                        if has_systemic_oral and not has_allowed_oral:
                            s_adj -= bias * 0.75
                        if has_antiseptic:
                            s_adj -= bias * 1.20
                        if has_derm:
                            s_adj -= bias * 0.90

                    # GENERIC: уже знайдені теми
                    if matched_conditions:
                        for cond in matched_conditions:
                            pos = clinical.positives(cond) if clinical else []
                            neg = clinical.penalties(cond) if clinical else []
                            if pos and any_term_in(doc_text, pos):
                                s_adj += bias * 0.80
                            if neg and any_term_in(doc_text, neg):
                                s_adj -= bias * 0.80

                    ce_adj.append(s_adj)

                ce_scores = ce_adj
                if heuristic_tag_special or heuristic_tags_applied:
                    msg = []
                    if heuristic_tag_special:
                        msg.append(heuristic_tag_special)
                    msg.extend(heuristic_tags_applied)
                    print(f"[Q{qi}] heuristic BOOST applied [{', '.join(msg)}]: top {len(cand_ids)} promoted")

            # --- 7) Фінал
            order = list(np.argsort(ce_scores)[::-1]) if ce_scores else list(range(len(cand_ids)))
            final_ids = [cand_ids[i] for i in order][:10]

            print(f"[Q{qi}] TOP10:")
            for rank, did in enumerate(final_ids, 1):
                nm = names[did] if 0 <= did < len(names) else f"doc_{did}"
                print(f"  {rank:02d}. {nm}")

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
