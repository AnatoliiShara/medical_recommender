# -*- coding: utf-8 -*-
"""
assistant_from_parquet_gemini_ft_v2.py

Hybrid search (BM25 + Dense/FAISS) + CrossEncoder rerank + optional Gemini final rerank.

Key fixes vs v1:
- Gemini used as *ranker* (ordered selection), not as absolute-scoring gate (prevents many 0.0 scores).
- Better candidate context for Gemini (title + indications + pharm group + a tiny contraindications hint).
- Robust fallbacks: if Gemini output is partial/broken, we backfill from CE ranking.
- Non-drug / non-medicine heuristic filter to reduce toothpaste/shampoo/etc candidates when possible.
- Safer & more consistent normalization (min-max for fusion / CE).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import faiss  # type: ignore
except Exception as e:
    raise RuntimeError("faiss is required (pip install faiss-cpu / faiss-gpu)") from e

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder  # type: ignore
except Exception as e:
    raise RuntimeError("sentence-transformers is required") from e

# Optional: keep compatibility with your repo clinical runtime
try:
    from src.search.clinical import p0_runtime  # type: ignore
except Exception:
    p0_runtime = None


_WS_RE = re.compile(r"\s+")
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def norm_ws(s: str) -> str:
    return _WS_RE.sub(" ", (s or "").strip())


def safe_trunc(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[: n - 1] + "…"


def minmax_norm(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx <= mn + 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mn) / (mx - mn)).astype(np.float32)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def try_load_doc_ids(index_dir: Path) -> Optional[np.ndarray]:
    p = index_dir / "doc_ids.npy"
    if p.exists():
        try:
            return np.load(p)
        except Exception:
            return None
    return None


def detect_e5_prefix_mode(emb_model_name: str, docs_text_sample: Optional[str]) -> str:
    """
    Returns one of: 'none', 'query_passage'
    Heuristic:
      - If docs text starts with 'passage:' -> use query/passsage prefixes.
      - Else if model name contains 'e5' -> likely query/passsage.
      - Else -> none
    """
    if docs_text_sample:
        s = docs_text_sample.strip().lower()
        if s.startswith("passage:"):
            return "query_passage"
    if "e5" in (emb_model_name or "").lower():
        return "query_passage"
    return "none"


def format_for_e5_query(q: str, mode: str) -> str:
    q = norm_ws(q)
    if mode == "query_passage":
        if not q.lower().startswith("query:"):
            return "query: " + q
    return q


DRUG_FORM_HINTS = (
    "таблет", "капсул", "саше", "порош", "суспен", "розчин", "спрей", "крапл", "супозитор",
    "ін'єкц", "інфуз", "крем", "маз", "гел", "паста", "аерозол", "пластир", "сироп",
    "вакцин", "ліофілізат"
)

NON_DRUG_HINTS = (
    "зубна щітка", "зубна паста", "шампун", "антиперсп", "дезодорант", "гел для душу",
    "крем-присипка", "підгуз", "дитяче харчування", "батончик", "маска для волосся",
    "shampoo", "toothbrush", "toothpaste"
)


def is_druglike_row(row: pd.Series) -> bool:
    """
    Heuristic filter to drop obvious non-meds from compendium.
    Conservative: if unsure, keep.
    """
    name = str(row.get("Назва препарату", "") or "").lower()
    form = str(row.get("Лікарська форма", "") or "").lower()
    group = str(row.get("Фармакотерапевтична група", "") or "").lower()
    ind = str(row.get("Показання", "") or "").lower()

    if any(h in name for h in NON_DRUG_HINTS):
        return False

    if form.strip() or group.strip() or ind.strip():
        return True

    if any(h in name for h in DRUG_FORM_HINTS):
        return True

    return False


def build_doc_blob_raw(row: pd.Series) -> str:
    """Richer text for BM25/CE/Gemini context (NOT for embeddings)."""
    name = norm_ws(str(row.get("Назва препарату", "") or ""))
    group = norm_ws(str(row.get("Фармакотерапевтична група", "") or ""))
    form = norm_ws(str(row.get("Лікарська форма", "") or ""))
    ind = norm_ws(str(row.get("Показання", "") or ""))
    contra = norm_ws(str(row.get("Протипоказання", "") or ""))
    inter = norm_ws(str(row.get("Взаємодія з іншими лікарськими засобами та інші види взаємодій", "") or ""))

    parts = [name]
    if form:
        parts.append(f"Лікарська форма: {form}")
    if group:
        parts.append(f"Фармгрупа: {group}")
    if ind:
        parts.append(f"Показання: {safe_trunc(ind, 900)}")
    if contra:
        parts.append(f"Протипоказання: {safe_trunc(contra, 350)}")
    if inter:
        parts.append(f"Взаємодії: {safe_trunc(inter, 250)}")
    return "\n".join(parts).strip()


def build_gemini_card(doc_id: int, row: pd.Series) -> str:
    """Compact card to keep token budget sane."""
    name = norm_ws(str(row.get("Назва препарату", "") or ""))
    form = norm_ws(str(row.get("Лікарська форма", "") or ""))
    group = norm_ws(str(row.get("Фармакотерапевтична група", "") or ""))
    ind = norm_ws(str(row.get("Показання", "") or ""))
    contra = norm_ws(str(row.get("Протипоказання", "") or ""))

    ind = safe_trunc(ind, 380)
    contra = safe_trunc(contra, 180)

    lines = [f"[{doc_id}] {name}"]
    if form:
        lines.append(f"Form: {form}")
    if group:
        lines.append(f"Group: {safe_trunc(group, 160)}")
    if ind:
        lines.append(f"Indications: {ind}")
    if contra:
        lines.append(f"Contra: {contra}")
    return "\n".join(lines)


def tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    text = re.sub(r"[^\w\u0400-\u04FF]+", " ", text, flags=re.UNICODE)  # keep Cyrillic
    return [t for t in text.split() if len(t) >= 2]


class BM25Index:
    """
    Simple BM25 with precomputed per-doc term frequencies.
    Fast enough for ~30k docs.
    """
    def __init__(self, docs: Sequence[str]) -> None:
        self.docs = docs
        self.N = len(docs)
        self.tf: List[Dict[str, int]] = []
        self.dl = np.zeros(self.N, dtype=np.int32)

        df: Dict[str, int] = {}
        for i, d in enumerate(docs):
            toks = tokenize(d)
            self.dl[i] = len(toks)
            d_tf: Dict[str, int] = {}
            for t in toks:
                d_tf[t] = d_tf.get(t, 0) + 1
            self.tf.append(d_tf)
            for t in d_tf.keys():
                df[t] = df.get(t, 0) + 1

        self.df = df
        self.avgdl = float(np.mean(self.dl)) if self.N else 0.0

        self.k1 = 1.2
        self.b = 0.75

    def search(self, query: str, top_k: int = 200) -> List[Tuple[int, float]]:
        q = tokenize(query)
        if not q or self.N == 0:
            return []
        q_terms = list(dict.fromkeys(q))
        scores = np.zeros(self.N, dtype=np.float32)

        for term in q_terms:
            n_qi = self.df.get(term, 0)
            if n_qi == 0:
                continue
            idf = np.log(1 + (self.N - n_qi + 0.5) / (n_qi + 0.5))
            for doc_id, d_tf in enumerate(self.tf):
                f_qi = d_tf.get(term, 0)
                if f_qi == 0:
                    continue
                dl = float(self.dl[doc_id])
                denom = f_qi + self.k1 * (1 - self.b + self.b * (dl / (self.avgdl + 1e-9)))
                scores[doc_id] += idf * (f_qi * (self.k1 + 1) / (denom + 1e-9))

        if top_k >= self.N:
            idx = np.argsort(-scores)
        else:
            idx = np.argpartition(-scores, kth=top_k)[:top_k]
            idx = idx[np.argsort(-scores[idx])]
        return [(int(i), float(scores[i])) for i in idx if scores[i] > 0]


@dataclass
class Candidate:
    doc_id: int
    title: str
    fused_score: float
    ce_score: float = 0.0
    ce_norm: float = 0.0


class CrossEncoderReranker:
    def __init__(self, model_name: str, batch_size: int = 16) -> None:
        self.model = CrossEncoder(model_name)
        self.batch_size = batch_size

    def score(self, query: str, candidates: Sequence[Candidate], doc_texts: Sequence[str]) -> np.ndarray:
        pairs = [(query, doc_texts[c.doc_id]) for c in candidates]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return np.array(scores, dtype=np.float32)


@dataclass
class GeminiPick:
    doc_id: int
    safety: str = "ok"  # ok | caution | contraindicated
    why: str = ""


class GeminiFinalReranker:
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_output_tokens: int = 2048,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.called = 0
        self.success = 0
        self.failed = 0

        if api_key is None:
            api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_APIKEY")
        if not api_key:
            raise RuntimeError("Gemini API key not found. Set GOOGLE_API_KEY (or GEMINI_API_KEY).")

        import google.generativeai as genai  # type: ignore
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model_name)

    def _build_prompt(self, query: str, cards: List[str], top_k: int) -> str:
        return f"""
You are a clinical triage assistant for an OTC/medicine recommender.
Task: from the candidate list, pick the BEST {top_k} products to recommend for the user's query.

Query: {query}

Rules:
- Prefer products that directly address the query/condition/symptoms.
- Avoid clearly irrelevant items (cosmetics, toothpaste, detergents, etc).
- If a candidate looks unsafe / contraindicated for generic adult use for this query, mark it as "contraindicated".
- Output STRICT JSON ONLY, no markdown, no extra text.

JSON schema:
{{
  "selected": [
    {{"doc_id": 123, "safety": "ok|caution|contraindicated", "why": "short reason (<=20 words)"}},
    ...
  ]
}}

Candidates:
{chr(10).join(cards)}
""".strip()

    def rank(self, query: str, cards: List[str], top_k: int) -> List[GeminiPick]:
        self.called += 1
        prompt = self._build_prompt(query=query, cards=cards, top_k=top_k)
        try:
            resp = self._model.generate_content(
                prompt,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_output_tokens,
                },
            )
            text = getattr(resp, "text", "") or ""
            m = _JSON_RE.search(text)
            if not m:
                raise ValueError("No JSON object found in Gemini response.")
            obj = json.loads(m.group(0))
            selected = obj.get("selected", [])
            picks: List[GeminiPick] = []
            for item in selected:
                if not isinstance(item, dict) or "doc_id" not in item:
                    continue
                did = int(item["doc_id"])
                safety = str(item.get("safety", "ok")).strip().lower()
                if safety not in ("ok", "caution", "contraindicated"):
                    safety = "ok"
                why = safe_trunc(str(item.get("why", "")), 120)
                picks.append(GeminiPick(doc_id=did, safety=safety, why=why))
            self.success += 1
            return picks[:top_k]
        except Exception:
            self.failed += 1
            return []


def dense_search(index: Any, qvec: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
    D, I = index.search(qvec.astype(np.float32), top_k)
    ids = I[0]
    scores = D[0]
    out: List[Tuple[int, float]] = []
    for did, sc in zip(ids, scores):
        if int(did) < 0:
            continue
        out.append((int(did), float(sc)))
    return out


def fuse_weighted(
    bm25_hits: List[Tuple[int, float]],
    dense_hits: List[Tuple[int, float]],
    w_bm25: float,
    w_dense: float,
) -> Dict[int, float]:
    ids = sorted(set([i for i, _ in bm25_hits] + [i for i, _ in dense_hits]))
    if not ids:
        return {}
    bm = np.zeros(len(ids), dtype=np.float32)
    dn = np.zeros(len(ids), dtype=np.float32)
    bm_map = dict(bm25_hits)
    dn_map = dict(dense_hits)
    for j, did in enumerate(ids):
        bm[j] = float(bm_map.get(did, 0.0))
        dn[j] = float(dn_map.get(did, 0.0))

    bm_n = minmax_norm(bm)
    dn_n = minmax_norm(dn)
    fused = w_bm25 * bm_n + w_dense * dn_n
    return {int(did): float(fused[j]) for j, did in enumerate(ids)}


def apply_clinical_union(
    query: str,
    base_scores: Dict[int, float],
    runtime: Any,
    doc_texts_l: Sequence[str],
    max_extra: int = 400,
) -> Dict[int, float]:
    if runtime is None:
        return base_scores
    try:
        conds = runtime.detect_conditions(query)
    except Exception:
        return base_scores

    conds = sorted(conds, key=lambda x: x[1], reverse=True)[:3]
    extra: Dict[int, float] = {}
    for cond, strength in conds:
        try:
            ids = runtime.union_candidates_for_conditions([cond], doc_texts_l)
        except Exception:
            continue
        for did in ids[:max_extra]:
            extra[int(did)] = max(extra.get(int(did), 0.0), float(0.05 + 0.15 * strength))

    out = dict(base_scores)
    for did, s in extra.items():
        out[did] = max(out.get(did, 0.0), s)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True, help="JSONL with {id, query, ...}")
    ap.add_argument("--out_json", required=True, help="Output JSONL")
    ap.add_argument("--compendium", default="data/raw/compendium_all.parquet", help="Compendium parquet")
    ap.add_argument("--embed_model", default="models/finetuned/e5-medrx-stage1", help="SentenceTransformer model path/name")
    ap.add_argument("--ce_model", default="cross-encoder/ms-marco-MiniLM-L-6-v2", help="CrossEncoder model")
    ap.add_argument("--faiss_index", default="data/processed/embeddings/eval_e5_finetuned_docs/faiss.index", help="FAISS index path")
    ap.add_argument("--dense_top_k", type=int, default=200, help="Dense candidates")
    ap.add_argument("--bm25_top_k", type=int, default=400, help="BM25 candidates")
    ap.add_argument("--rerank_top", type=int, default=25, help="How many to CrossEncoder-rerank")
    ap.add_argument("--final_top_k", type=int, default=20, help="Final recommendations")
    ap.add_argument("--w_bm25", type=float, default=0.55)
    ap.add_argument("--w_dense", type=float, default=0.45)
    ap.add_argument("--use_gemini", action="store_true")
    ap.add_argument("--gemini_model", default="gemini-2.5-pro")
    ap.add_argument("--gemini_top_k", type=int, default=10, help="How many Gemini selects (<= final_top_k)")
    ap.add_argument("--gemini_max_output_tokens", type=int, default=2048)
    ap.add_argument("--gemini_temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_drug_filter", action="store_true", help="Disable non-drug heuristic filtering")
    args = ap.parse_args()

    np.random.seed(args.seed)

    df = pd.read_parquet(args.compendium).reset_index(drop=True)
    if len(df) == 0:
        raise RuntimeError(f"Empty compendium: {args.compendium}")

    faiss_path = Path(args.faiss_index)
    index_dir = faiss_path.parent
    index = faiss.read_index(str(faiss_path))
    print(f"[FAISS] loaded index: {faiss_path}")

    doc_ids_map = try_load_doc_ids(index_dir)
    if doc_ids_map is not None:
        doc_ids_map = doc_ids_map.astype(np.int64)
        if len(doc_ids_map) != index.ntotal:
            doc_ids_map = None

    enc = SentenceTransformer(args.embed_model)
    e5_mode = detect_e5_prefix_mode(args.embed_model, docs_text_sample=None)

    keep_mask = np.ones(len(df), dtype=bool)
    if not args.no_drug_filter:
        keep_mask = df.apply(is_druglike_row, axis=1).to_numpy()

    doc_texts_raw: List[str] = []
    for i, row in df.iterrows():
        if keep_mask[i]:
            doc_texts_raw.append(build_doc_blob_raw(row))
        else:
            doc_texts_raw.append(norm_ws(str(row.get("Назва препарату", "") or "")))
    doc_texts_l = [t.lower() for t in doc_texts_raw]

    print(f"[INFO] Loaded {len(df)} documents")

    bm25 = BM25Index(doc_texts_raw)
    ce = CrossEncoderReranker(args.ce_model, batch_size=16)

    runtime = None
    if p0_runtime is not None:
        try:
            runtime = p0_runtime.ClinicalRuntime("data/dicts/clinical")
        except Exception:
            runtime = None

    gemini = None
    if args.use_gemini:
        gemini = GeminiFinalReranker(
            model_name=args.gemini_model,
            temperature=args.gemini_temperature,
            max_output_tokens=args.gemini_max_output_tokens,
        )

    queries = load_jsonl(args.queries)
    print(f"[INFO] Loaded {len(queries)} queries")

    outputs: List[Dict[str, Any]] = []
    for idx_q, qobj in enumerate(queries, start=1):
        qid = qobj.get("id", qobj.get("qid", idx_q))
        query = str(qobj.get("query", "") or "").strip()
        if not query:
            continue

        q_embed_text = format_for_e5_query(query, e5_mode)
        qvec = enc.encode([q_embed_text], normalize_embeddings=True)
        dense_hits = dense_search(index, np.array(qvec, dtype=np.float32), top_k=args.dense_top_k)

        if doc_ids_map is not None:
            dense_hits = [(int(doc_ids_map[i]), sc) for i, sc in dense_hits
                          if 0 <= i < len(doc_ids_map) and 0 <= int(doc_ids_map[i]) < len(df)]

        bm25_hits = bm25.search(query, top_k=args.bm25_top_k)
        fused_map = fuse_weighted(bm25_hits, dense_hits, w_bm25=args.w_bm25, w_dense=args.w_dense)
        fused_map = apply_clinical_union(query, fused_map, runtime, doc_texts_l)

        if not fused_map:
            outputs.append({"id": qid, "query": query, "top": []})
            continue

        pool_size = max(args.rerank_top * 4, args.final_top_k * 4, args.gemini_top_k * 6)
        cand_ids = [d for d in sorted(fused_map.keys(), key=lambda d: fused_map[d], reverse=True)
                    if 0 <= int(d) < len(df)][:pool_size]

        if not args.no_drug_filter:
            filtered = [d for d in cand_ids if keep_mask[d]]
            if len(filtered) >= max(args.rerank_top, args.gemini_top_k, args.final_top_k):
                cand_ids = filtered

        candidates = [Candidate(doc_id=int(d),
                                title=str(df.iloc[int(d)].get("Назва препарату", "")),
                                fused_score=float(fused_map[int(d)]))
                      for d in cand_ids]

        ce_window = candidates[:max(args.rerank_top, args.gemini_top_k, args.final_top_k)]
        ce_scores = ce.score(query, ce_window, doc_texts_raw)
        ce_norm = minmax_norm(ce_scores)

        for c, sc, scn in zip(ce_window, ce_scores, ce_norm):
            c.ce_score = float(sc)
            c.ce_norm = float(scn)

        ce_sorted = sorted(ce_window, key=lambda c: (c.ce_norm, c.fused_score), reverse=True)

        final_ids: List[int] = []
        safety_map: Dict[int, str] = {}
        why_map: Dict[int, str] = {}

        if gemini is not None:
            gem_pool = ce_sorted[:max(args.gemini_top_k * 3, 18)]
            cards = [build_gemini_card(c.doc_id, df.iloc[c.doc_id]) for c in gem_pool]
            picks = gemini.rank(query=query, cards=cards, top_k=min(args.gemini_top_k, args.final_top_k))
            for p in picks:
                if p.doc_id in final_ids:
                    continue
                final_ids.append(int(p.doc_id))
                safety_map[int(p.doc_id)] = p.safety
                why_map[int(p.doc_id)] = p.why

        for c in ce_sorted:
            if len(final_ids) >= args.final_top_k:
                break
            if c.doc_id in final_ids:
                continue
            final_ids.append(c.doc_id)

        ce_norm_map = {c.doc_id: c.ce_norm for c in ce_sorted}
        fused_score_map = {c.doc_id: c.fused_score for c in ce_sorted}

        top_rows = []
        for rank, did in enumerate(final_ids[:args.final_top_k], start=1):
            base = float(ce_norm_map.get(did, 0.0))
            if did in safety_map:
                bump = 1.0 - (rank - 1) / max(1, args.final_top_k)
                score = 0.65 * bump + 0.35 * base
                if safety_map[did] == "contraindicated":
                    score *= 0.25
                elif safety_map[did] == "caution":
                    score *= 0.75
            else:
                score = base

            title = norm_ws(str(df.iloc[did].get("Назва препарату", "") or ""))
            top_rows.append({
                "rank": rank,
                "doc_id": int(did),
                "score": float(score),
                "title": title,
                "gemini_safety": safety_map.get(did),
                "gemini_why": why_map.get(did),
                "fused_score": float(fused_score_map.get(did, 0.0)),
                "ce_norm": float(base),
            })

        outputs.append({"id": qid, "query": query, "top": top_rows})

        if idx_q % 10 == 0 or idx_q == len(queries):
            print(f"[PROGRESS] {idx_q}/{len(queries)} queries processed")

    write_jsonl(args.out_json, outputs)
    print(f"\n[DONE] Saved results to: {args.out_json}")
    if gemini is not None:
        print(f"[GEMINI STATS] Called: {gemini.called}, Success: {gemini.success}, Failed: {gemini.failed}")


if __name__ == "__main__":
    main()
