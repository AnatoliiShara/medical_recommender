# -*- coding: utf-8 -*-
"""
Stratified eval for MED-RX: human vs auto query sets with detailed run logs.
Pipeline: optional doc-prefilter (FAISS over doc embeddings) -> hybrid (BM25 + passage FAISS) -> RRF -> CrossEncoder -> section-gate.
Outputs: Precision@k, Recall@k, nDCG@k per subset and overall; per-query JSON dumps with all intermediate artifacts.
"""

from __future__ import annotations
import os
import sys
import json
import math
import argparse
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import pyarrow.parquet as pq
import pandas as pd
from tqdm import tqdm
from rank_bm25 import BM25Okapi

# embeddings + FAISS + CE
try:
    import faiss  # type: ignore
    FAISS_OK = True
except Exception:
    FAISS_OK = False

from sentence_transformers import SentenceTransformer, CrossEncoder


# ---------------- utils ----------------

def norm(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00A0", " ")
    s = s.lower()
    for ch in ["®", "™", "©", "’", "‘", "'", "`", "“", "”", "«", "»", "(", ")", "[", "]", "{", "}", ";", ","]:
        s = s.replace(ch, " ")
    s = " ".join(s.split())
    return s


def bm25_tokens(s: str) -> List[str]:
    s = norm(s)
    # дозволяємо латиницю+кирилицю+цифри
    import re
    s = re.sub(r"[^\w\u0400-\u04FF]+", " ", s)
    return s.split()


def ndcg_at_k(gains: List[int], k: int) -> float:
    """gains — 1/0 (relevant by doc_id) за відсортованим списком результатів."""
    k = min(k, len(gains))
    if k == 0:
        return 0.0
    gains_k = gains[:k]
    dcg = 0.0
    for i, g in enumerate(gains_k, start=1):
        dcg += (2**g - 1) / math.log2(i + 1)
    # ideal DCG
    ideal = sorted(gains, reverse=True)[:k]
    idcg = 0.0
    for i, g in enumerate(ideal, start=1):
        idcg += (2**g - 1) / math.log2(i + 1)
    return float(dcg / (idcg + 1e-9))


def precision_at_k(hits: List[int], k: int) -> float:
    k = min(k, len(hits))
    if k == 0:
        return 0.0
    return float(sum(hits[:k]) / max(1, k))


def recall_at_k(hits: List[int], gold_total: int, k: int) -> float:
    if gold_total <= 0:
        return 0.0
    k = min(k, len(hits))
    return float(sum(hits[:k]) / float(gold_total))


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ---------------- configs ----------------

@dataclass
class GatePolicy:
    mode: str = "none"                    # 'none' | 'prefer' | 'require'
    sections: List[str] = field(default_factory=list)

@dataclass
class RerankConfig:
    ce_model: Optional[str] = None
    ce_top: int = 100
    ce_min: float = 0.15
    ce_weight: float = 0.75

@dataclass
class SearchConfig:
    rrf_alpha: float = 60.0
    k: int = 300                          # final @k for metrics
    gate: GatePolicy = field(default_factory=GatePolicy)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    doc_topN: int = 0                     # 0 disables doc prefilter
    restrict_bm25_by_doc: bool = False    # якщо True — BM25 кандидати лише з doc_topN


# ---------------- loaders ----------------

def load_passage_meta(index_dir: Path) -> pd.DataFrame:
    """
    Читаємо chunks.parquet і беремо лише необхідні колонки.
    Очікується схема: doc_id, drug_name, section, text, embedding, norm (embedding не читаємо тут).
    """
    pq_path = index_dir / "chunks.parquet"
    table = pq.read_table(pq_path, columns=["doc_id", "drug_name", "section", "text"])
    df = table.to_pandas()  # потрібно для BM25 токенізації
    df["doc_id"] = df["doc_id"].astype(int)
    df["section"] = df["section"].astype(str)
    df["text"] = df["text"].astype(str)
    return df


def load_faiss_index(index_dir: Path) -> faiss.Index:
    idx_path = index_dir / "faiss.index"
    if not idx_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {idx_path}")
    return faiss.read_index(str(idx_path))


def load_doc_prefilter(doc_index_dir: Path) -> Tuple[faiss.Index, np.ndarray, pd.DataFrame]:
    idx_path = doc_index_dir / "doc.index"
    ids_path = doc_index_dir / "doc_ids.npy"
    meta_path = doc_index_dir / "docs_meta.parquet"
    if not idx_path.exists():
        raise FileNotFoundError(f"Doc FAISS index not found: {idx_path}")
    if not ids_path.exists():
        raise FileNotFoundError(f"Doc ids npy not found: {ids_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Doc meta not found: {meta_path}")
    idx = faiss.read_index(str(idx_path))
    ids = np.load(str(ids_path))
    meta = pq.read_table(meta_path).to_pandas()
    return idx, ids, meta


def load_queries_labeled(args_pairs: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """
    --queries human:/path/file.jsonl auto:/path/file.jsonl
    JSONL формат очікується з полями: query (str), gold_doc_ids (list[int], optional).
    """
    out: Dict[str, List[Dict[str, Any]]] = {}
    for pair in args_pairs:
        if ":" not in pair:
            raise ValueError(f"Invalid --queries entry: {pair}. Use subset:/path/file.jsonl")
        subset, path = pair.split(":", 1)
        subset = subset.strip()
        p = Path(path.strip())
        items: List[Dict[str, Any]] = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                q = str(obj.get("query", "")).strip()
                gold_ids = obj.get("gold_doc_ids", []) or []
                # інколи gold може бути в іншому полі — підстрахуємося
                if not gold_ids and "gold_drugs" in obj:
                    # не мапимо тут заради простоти; у нас має бути вже enriched з gold_doc_ids
                    gold_ids = []
                items.append({
                    "query": q,
                    "gold_doc_ids": list(map(int, gold_ids)) if gold_ids else [],
                })
        out[subset] = items
    return out


# ---------------- core pipeline ----------------

class StratifiedSearch:
    def __init__(self,
                 meta_df: pd.DataFrame,
                 faiss_index: Optional[faiss.Index],
                 encoder: SentenceTransformer,
                 doc_prefilter: Optional[Tuple[faiss.Index, np.ndarray, pd.DataFrame]] = None):
        self.meta = meta_df
        self.encoder = encoder
        self.faiss_index = faiss_index
        self.doc_prefilter = doc_prefilter

        # BM25
        tokenized = [bm25_tokens(t) for t in meta_df["text"].tolist()]
        self.bm25 = BM25Okapi(tokenized)

        # passage<->id map
        self.n_passages = len(meta_df)
        self.pass_texts = meta_df["text"].tolist()
        self.pass_sections = meta_df["section"].tolist()
        self.pass_doc_ids = meta_df["doc_id"].astype(int).to_numpy()
        self.pass_names = meta_df["drug_name"].astype(str).tolist()

    def _doc_filter_ids(self, query: str, topN: int) -> Optional[set]:
        if not self.doc_prefilter or topN <= 0:
            return None
        idx, ids, _meta = self.doc_prefilter
        qv = self.encoder.encode([query], normalize_embeddings=True).astype("float32")
        D, I = idx.search(qv, topN)
        doc_ids = set(int(ids[j]) for j in I[0] if j >= 0)
        return doc_ids

    def _faiss_passage_search(self, query: str, topN: int, restrict_docs: Optional[set]) -> List[Tuple[int, float]]:
        if self.faiss_index is None:
            return []
        q = self.encoder.encode([query], normalize_embeddings=True).astype("float32")
        n = min(topN, self.n_passages)
        D, I = self.faiss_index.search(q, n)
        pairs: List[Tuple[int, float]] = []
        for j, pid in enumerate(I[0]):
            if pid < 0:
                continue
            if restrict_docs is not None:
                if int(self.pass_doc_ids[pid]) not in restrict_docs:
                    continue
            pairs.append((int(pid), float(D[0, j])))
        return pairs

    def _bm25_search(self, query: str, topN: int, restrict_docs: Optional[set]) -> List[Tuple[int, float]]:
        scores = self.bm25.get_scores(bm25_tokens(query))
        idxs = np.argsort(-scores)[:min(topN, len(scores))]
        pairs: List[Tuple[int, float]] = []
        for pid in idxs:
            if restrict_docs is not None:
                if int(self.pass_doc_ids[pid]) not in restrict_docs:
                    continue
            pairs.append((int(pid), float(scores[pid])))
        return pairs

    @staticmethod
    def _rrf_fusion(bm: List[Tuple[int, float]],
                    fa: List[Tuple[int, float]],
                    k_rrf: float,
                    topN: int) -> List[Tuple[int, float]]:
        r_bm = {pid: r + 1 for r, (pid, _) in enumerate(bm)}
        r_fa = {pid: r + 1 for r, (pid, _) in enumerate(fa)}
        all_ids = set(r_bm) | set(r_fa)
        scores: Dict[int, float] = {}
        for pid in all_ids:
            s = 0.0
            if pid in r_bm:
                s += 1.0 / (k_rrf + r_bm[pid])
            if pid in r_fa:
                s += 1.0 / (k_rrf + r_fa[pid])
            scores[pid] = s
        ranked = sorted(scores.items(), key=lambda x: -x[1])[:topN]
        return ranked

    @staticmethod
    def _apply_section_gate(items: List[Tuple[int, float]],
                            sections: List[str],
                            mode: str,
                            pass_sections: List[str]) -> Tuple[List[Tuple[int, float]], Dict[str, Any]]:
        sections_norm = {norm(s) for s in sections}
        if mode == "none" or not sections_norm:
            return items, {"mode": mode, "kept": len(items), "filtered": 0, "note": "off"}

        def is_good(pid: int) -> bool:
            sec = norm(pass_sections[pid])
            return sec in sections_norm

        if mode == "require":
            kept = [(pid, sc) for (pid, sc) in items if is_good(pid)]
            return kept, {"mode": mode, "kept": len(kept), "filtered": len(items) - len(kept)}

        # prefer: спочатку секції з білого списку, потім решта
        good = [(pid, sc) for (pid, sc) in items if is_good(pid)]
        bad  = [(pid, sc) for (pid, sc) in items if not is_good(pid)]
        return (good + bad), {"mode": mode, "good_first": len(good), "bad_after": len(bad)}

    def search_one(self, query: str, cfg: SearchConfig) -> Dict[str, Any]:
        # 1) doc prefilter
        restrict_docs = self._doc_filter_ids(query, cfg.doc_topN)

        # 2) BM25 + FAISS over passages
        baseN = max(cfg.k, cfg.rerank.ce_top)
        bm = self._bm25_search(query, topN=baseN * 5, restrict_docs=restrict_docs if cfg.restrict_bm25_by_doc else None)
        fa = self._faiss_passage_search(query, topN=baseN * 5, restrict_docs=restrict_docs)

        # 3) RRF
        fused = self._rrf_fusion(bm, fa, k_rrf=cfg.rrf_alpha, topN=max(baseN, cfg.k))

        # 4) section-gate
        gated, gate_dbg = self._apply_section_gate(
            fused,
            sections=cfg.gate.sections,
            mode=cfg.gate.mode,
            pass_sections=self.pass_sections
        )

        # 5) CrossEncoder rerank (top ce_top з gated)
        ce_pairs = []
        ce_scores = []
        reranked: List[Tuple[int, float]] = gated
        if cfg.rerank.ce_model:
            ceN = min(cfg.rerank.ce_top, len(gated))
            pairs = [(query, self.pass_texts[pid]) for pid, _ in gated[:ceN]]
            ce_pairs = [int(pid) for pid, _ in gated[:ceN]]
            ce = CrossEncoder(cfg.rerank.ce_model, max_length=512)
            scores = ce.predict(pairs).tolist()
            ce_scores = [float(s) for s in scores]

            # нормалізація і комбінація з RRF
            rrf_vals = np.array([sc for _, sc in gated[:ceN]], dtype="float32")
            if len(rrf_vals) == 0:
                alpha = 0.0
                rrf_n = np.zeros_like(rrf_vals)
            else:
                rrf_n = (rrf_vals - rrf_vals.min()) / (rrf_vals.max() - rrf_vals.min() + 1e-9)
            scores_arr = np.array(scores, dtype="float32")
            scores_adj = np.maximum(0.0, scores_arr - cfg.rerank.ce_min)
            if len(scores_adj) > 0:
                ce_n = (scores_adj - scores_adj.min()) / (scores_adj.max() - scores_adj.min() + 1e-9)
            else:
                ce_n = scores_adj
            comb = (1.0 - cfg.rerank.ce_weight) * rrf_n + cfg.rerank.ce_weight * ce_n
            ranked_ce = list(zip([pid for pid, _ in gated[:ceN]], comb.tolist()))
            # додаємо хвіст без CE в кінці
            tail = gated[ceN:]
            reranked = sorted(ranked_ce, key=lambda x: -x[1]) + tail

        # 6) формуємо @k
        topk = reranked[:cfg.k]
        out = {
            "bm25": bm,
            "faiss": fa,
            "fused": fused,
            "gated": gated,
            "gate_dbg": gate_dbg,
            "ce_pairs": ce_pairs,
            "ce_scores": ce_scores,
            "final": topk,
        }
        return out


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser("Stratified eval (human vs auto) with detailed dumps")
    ap.add_argument("--index_dir", required=True)
    ap.add_argument("--doc_index_dir", default=None)
    ap.add_argument("--queries", nargs="+", required=True, help="Format: human:/path/file.jsonl auto:/path/file.jsonl")
    ap.add_argument("--embed_model", required=True)
    ap.add_argument("--k", type=int, default=300)
    ap.add_argument("--rrf_alpha", type=float, default=60.0)
    ap.add_argument("--doc_topN", type=int, default=0)
    ap.add_argument("--restrict_bm25_by_doc", action="store_true")

    ap.add_argument("--gate_mode", choices=["none", "prefer", "require"], default="none")
    ap.add_argument("--gate_sections", default="")

    ap.add_argument("--ce_model", default=None)
    ap.add_argument("--ce_top", type=int, default=100)
    ap.add_argument("--ce_min", type=float, default=0.15)
    ap.add_argument("--ce_weight", type=float, default=0.75)

    ap.add_argument("--dump_eval_dir", default=None)

    args = ap.parse_args()

    index_dir = Path(args.index_dir)
    # 1) meta (passages)
    meta_df = load_passage_meta(index_dir)
    print(f"[INFO] Loaded metadata: {len(meta_df)} passages")

    # 2) FAISS index (passage)
    if not FAISS_OK:
        raise RuntimeError("faiss is not available in this environment.")
    faiss_index = load_faiss_index(index_dir)
    print(f"[INFO] FAISS loaded: faiss.index")

    # 3) encoder
    encoder = SentenceTransformer(args.embed_model)
    print(f"[INFO] Query encoder ready: {args.embed_model}")

    # 4) doc prefilter (optional)
    doc_pf = None
    if args.doc_index_dir:
        doc_pf = load_doc_prefilter(Path(args.doc_index_dir))
        print(f"[INFO] Doc prefilter ready: n={doc_pf[0].ntotal}, dim={doc_pf[0].d}")

    # 5) queries
    subsets = load_queries_labeled(args.queries)
    for key in subsets:
        print(f"[INFO] Subset '{key}': {len(subsets[key])} queries")

    # 6) pipeline
    cfg = SearchConfig(
        rrf_alpha=args.rrf_alpha,
        k=args.k,
        gate=GatePolicy(
            mode=args.gate_mode,
            sections=[s.strip() for s in args.gate_sections.split(",") if s.strip()]
        ),
        rerank=RerankConfig(
            ce_model=args.ce_model,
            ce_top=args.ce_top,
            ce_min=args.ce_min,
            ce_weight=args.ce_weight
        ),
        doc_topN=args.doc_topN,
        restrict_bm25_by_doc=args.restrict_bm25_by_doc
    )

    engine = StratifiedSearch(
        meta_df=meta_df,
        faiss_index=faiss_index,
        encoder=encoder,
        doc_prefilter=doc_pf
    )

    # 7) eval per subset
    dump_root = Path(args.dump_eval_dir) if args.dump_eval_dir else None
    if dump_root:
        ensure_dir(dump_root)

    overall_hits: List[int] = []
    overall_gains: List[int] = []
    overall_gold_total = 0

    results_summary: Dict[str, Dict[str, float]] = {}

    for subset_name, qlist in subsets.items():
        hits_all: List[int] = []
        gains_all: List[int] = []
        gold_total = 0

        subset_dump = None
        if dump_root:
            subset_dump = dump_root / subset_name
            ensure_dir(subset_dump)

        for qi, item in enumerate(tqdm(qlist, desc=f"Eval[{subset_name}]")):
            q = item["query"]
            gold_ids = set(item.get("gold_doc_ids", []))
            gold_total += len(gold_ids)

            run = engine.search_one(q, cfg)

            # фінальні @k
            final = run["final"]
            final_doc_ids = [int(engine.pass_doc_ids[pid]) for pid, _ in final]
            # hits/gains по @k
            hit_vec = [1 if d in gold_ids else 0 for d in final_doc_ids]
            hits_all.extend(hit_vec)
            gains_all.extend(hit_vec)

            # dump per query
            if subset_dump is not None:
                dump = {
                    "query": q,
                    "gold_doc_ids": sorted(list(gold_ids)),
                    "doc_prefilter": {
                        "topN": args.doc_topN,
                    },
                    "bm25": [{"pid": int(pid), "score": float(sc),
                              "doc_id": int(engine.pass_doc_ids[pid]),
                              "section": engine.pass_sections[pid]}
                             for pid, sc in run["bm25"][:50]],
                    "faiss": [{"pid": int(pid), "score": float(sc),
                               "doc_id": int(engine.pass_doc_ids[pid]),
                               "section": engine.pass_sections[pid]}
                              for pid, sc in run["faiss"][:50]],
                    "rrf_fused_top": [{"pid": int(pid), "rrf": float(sc),
                                       "doc_id": int(engine.pass_doc_ids[pid]),
                                       "section": engine.pass_sections[pid]}
                                      for pid, sc in run["fused"][:100]],
                    "gate": run["gate_dbg"],
                    "ce": {
                        "pairs_pids": run["ce_pairs"],
                        "scores": run["ce_scores"]
                    },
                    "final@k": [{"pid": int(pid),
                                 "score": float(sc),
                                 "doc_id": int(engine.pass_doc_ids[pid]),
                                 "section": engine.pass_sections[pid],
                                 "hit": 1 if int(engine.pass_doc_ids[pid]) in gold_ids else 0}
                                for pid, sc in final]
                }
                out_path = subset_dump / f"q{qi:04d}.json"
                with out_path.open("w", encoding="utf-8") as f:
                    json.dump(dump, f, ensure_ascii=False, indent=2)

        # метрики по піднабору
        P = precision_at_k(hits_all, args.k)
        R = recall_at_k(hits_all, gold_total, args.k)
        nDCG = ndcg_at_k(gains_all, args.k)
        results_summary[subset_name] = {"Precision@k": P, "Recall@k": R, "nDCG@k": nDCG}

        overall_hits.extend(hits_all)
        overall_gains.extend(gains_all)
        overall_gold_total += gold_total

    # overall
    overall_P = precision_at_k(overall_hits, args.k)
    overall_R = recall_at_k(overall_hits, overall_gold_total, args.k)
    overall_nDCG = ndcg_at_k(overall_gains, args.k)
    results_summary["OVERALL"] = {"Precision@k": overall_P, "Recall@k": overall_R, "nDCG@k": overall_nDCG}

    print("\n=== STRATIFIED EVAL ===")
    for name, m in results_summary.items():
        print(f"[{name}] Precision@{args.k}: {m['Precision@k']:.4f} | Recall@{args.k}: {m['Recall@k']:.4f} | nDCG@{args.k}: {m['nDCG@k']:.4f}")

    # збережемо summary у dump_dir, якщо задано
    if dump_root:
        summary_path = dump_root / "summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(results_summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # дозволяємо запуск з кореня проєкту
    sys.exit(main())
