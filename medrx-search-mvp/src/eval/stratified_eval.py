# -*- coding: utf-8 -*-
from __future__ import annotations
import os, sys, json, math, argparse, unicodedata, re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import pyarrow.parquet as pq
import pandas as pd
from tqdm import tqdm
from rank_bm25 import BM25Okapi

# FAISS
try:
    import faiss
    FAISS_OK = True
except Exception:
    FAISS_OK = False

from sentence_transformers import SentenceTransformer, CrossEncoder

# --- optional rewrite ---
try:
    from rewrite.rewrite import load_config as load_rw_config, Rewriter
except Exception:
    load_rw_config = None
    Rewriter = None

# ---------- utils ----------
def norm(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00A0", " ")
    s = s.lower()
    for ch in ["®","™","©","’","‘","'","`","“","”","«","»","(",")","[","]","{","}",";",","]:
        s = s.replace(ch, " ")
    s = " ".join(s.split())
    return s

def bm25_tokens(s: str) -> List[str]:
    s = norm(s)
    s = re.sub(r"[^\w\u0400-\u04FF]+", " ", s)
    return s.split()

def precision_k(hits: List[int], k: int) -> float:
    k = min(k, len(hits))
    return float(sum(hits[:k]) / max(1, k)) if k>0 else 0.0

def recall_k(hits: List[int], gold_total: int, k: int) -> float:
    if gold_total <= 0:
        return 0.0
    k = min(k, len(hits))
    return float(sum(hits[:k]) / float(gold_total)) if k>0 else 0.0

def ndcg_hits_k(hits: List[int], gold_total: int, k: int) -> float:
    k = min(k, len(hits))
    if k == 0:
        return 0.0
    gains = hits[:k]
    dcg = 0.0
    for i,g in enumerate(gains):
        dcg += (2**g - 1) / math.log2(i+2)
    idcg = 0.0
    for i in range(min(k, gold_total)):
        idcg += (2**1 - 1) / math.log2(i+2)
    return float(dcg / (idcg + 1e-9))

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

# ---------- configs ----------
@dataclass
class GatePolicy:
    mode: str = "none"            # none | prefer | require
    sections: List[str] = field(default_factory=list)

@dataclass
class RerankConfig:
    ce_model: Optional[str] = None
    ce_top: int = 0
    ce_min: float = 0.15
    ce_weight: float = 0.75

@dataclass
class SearchConfig:
    rrf_alpha: float = 60.0
    k: int = 300
    gate: GatePolicy = field(default_factory=GatePolicy)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    doc_topN: int = 0
    restrict_bm25_by_doc: bool = False

# ---------- loaders ----------
def load_passage_meta(index_dir: Path) -> pd.DataFrame:
    pq_path = index_dir / "chunks.parquet"
    table = pq.read_table(pq_path, columns=["doc_id","drug_name","section","text"])
    df = table.to_pandas()
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
    if not idx_path.exists():   raise FileNotFoundError(f"Doc FAISS index not found: {idx_path}")
    if not ids_path.exists():   raise FileNotFoundError(f"Doc ids npy not found: {ids_path}")
    if not meta_path.exists():  raise FileNotFoundError(f"Doc meta not found: {meta_path}")
    idx = faiss.read_index(str(idx_path))
    ids = np.load(str(ids_path))
    meta = pq.read_table(meta_path).to_pandas()
    return idx, ids, meta

def load_queries_labeled(args_pairs: List[str]) -> Dict[str, List[Dict[str, Any]]]:
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
                q = str(obj.get("query","")).strip()
                gold_ids = obj.get("gold_doc_ids", []) or []
                items.append({"query": q, "gold_doc_ids": list(map(int, gold_ids)) if gold_ids else []})
        out[subset] = items
    return out

# ---------- core pipeline ----------
class StratifiedSearch:
    def __init__(self, meta_df: pd.DataFrame, faiss_index: Optional[faiss.Index],
                 encoder: SentenceTransformer,
                 doc_prefilter: Optional[Tuple[faiss.Index, np.ndarray, pd.DataFrame]] = None):
        self.meta = meta_df
        self.encoder = encoder
        self.faiss_index = faiss_index
        self.doc_prefilter = doc_prefilter
        tokenized = [bm25_tokens(t) for t in meta_df["text"].tolist()]
        self.bm25 = BM25Okapi(tokenized)
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
        return set(int(ids[j]) for j in I[0] if j >= 0)

    def _faiss_passage_search(self, query: str, topN: int, restrict_docs: Optional[set]) -> List[Tuple[int, float]]:
        if self.faiss_index is None:
            return []
        q = self.encoder.encode([query], normalize_embeddings=True).astype("float32")
        n = min(topN, self.n_passages)
        D, I = self.faiss_index.search(q, n)
        out: List[Tuple[int, float]] = []
        for j, pid in enumerate(I[0]):
            if pid < 0: continue
            if restrict_docs is not None and int(self.pass_doc_ids[pid]) not in restrict_docs:
                continue
            out.append((int(pid), float(D[0, j])))
        return out

    def _bm25_search(self, query: str, topN: int, restrict_docs: Optional[set]) -> List[Tuple[int, float]]:
        scores = self.bm25.get_scores(bm25_tokens(query))
        idxs = np.argsort(-scores)[:min(topN, len(scores))]
        out: List[Tuple[int, float]] = []
        for pid in idxs:
            if restrict_docs is not None and int(self.pass_doc_ids[pid]) not in restrict_docs:
                continue
            out.append((int(pid), float(scores[pid])))
        return out

    @staticmethod
    def _rrf_fusion(bm: List[Tuple[int, float]], fa: List[Tuple[int, float]], k_rrf: float, topN: int) -> List[Tuple[int, float]]:
        r_bm = {pid: r + 1 for r, (pid, _) in enumerate(bm)}
        r_fa = {pid: r + 1 for r, (pid, _) in enumerate(fa)}
        all_ids = set(r_bm) | set(r_fa)
        scores: Dict[int, float] = {}
        for pid in all_ids:
            s = 0.0
            if pid in r_bm: s += 1.0 / (k_rrf + r_bm[pid])
            if pid in r_fa: s += 1.0 / (k_rrf + r_fa[pid])
            scores[pid] = s
        return sorted(scores.items(), key=lambda x: -x[1])[:topN]

    @staticmethod
    def _apply_section_gate(items: List[Tuple[int, float]], sections: List[str], mode: str,
                            pass_sections: List[str]) -> Tuple[List[Tuple[int, float]], Dict[str, Any]]:
        sections_norm = {norm(s) for s in sections}
        if mode == "none" or not sections_norm:
            return items, {"mode": mode, "kept": len(items), "filtered": 0}
        def is_good(pid: int) -> bool:
            return norm(pass_sections[pid]) in sections_norm
        if mode == "require":
            kept = [(pid, sc) for (pid, sc) in items if is_good(pid)]
            return kept, {"mode": mode, "kept": len(kept), "filtered": len(items)-len(kept)}
        good = [(pid, sc) for (pid, sc) in items if is_good(pid)]
        bad  = [(pid, sc) for (pid, sc) in items if not is_good(pid)]
        return (good + bad), {"mode": mode, "good_first": len(good), "bad_after": len(bad)}

    def search_one(self, query: str, cfg: SearchConfig) -> Dict[str, Any]:
        restrict_docs = self._doc_filter_ids(query, cfg.doc_topN)
        baseN = max(cfg.k, cfg.rerank.ce_top, 1)
        bm = self._bm25_search(query, topN=baseN*5, restrict_docs=restrict_docs if cfg.restrict_bm25_by_doc else None)
        fa = self._faiss_passage_search(query, topN=baseN*5, restrict_docs=restrict_docs)
        fused = self._rrf_fusion(bm, fa, k_rrf=cfg.rrf_alpha, topN=max(baseN, cfg.k))
        gated, gate_dbg = self._apply_section_gate(fused, cfg.gate.sections, cfg.gate.mode, self.pass_sections)

        # CE (за потребою)
        reranked: List[Tuple[int, float]] = gated
        ce_pairs: List[int] = []
        ce_scores: List[float] = []
        if cfg.rerank.ce_model:
            ceN = min(cfg.rerank.ce_top, len(gated))
            pairs = [(query, self.pass_texts[pid]) for pid, _ in gated[:ceN]]
            ce_pairs = [int(pid) for pid,_ in gated[:ceN]]
            ce = CrossEncoder(cfg.rerank.ce_model, max_length=512)
            scores = ce.predict(pairs).tolist()
            ce_scores = [float(s) for s in scores]
            rrf_vals = np.array([sc for _,sc in gated[:ceN]], dtype="float32")
            rrf_n = (rrf_vals - rrf_vals.min()) / (rrf_vals.max() - rrf_vals.min() + 1e-9) if len(rrf_vals) else rrf_vals
            scores_arr = np.array(scores, dtype="float32")
            scores_adj = np.maximum(0.0, scores_arr - cfg.rerank.ce_min)
            ce_n = (scores_adj - scores_adj.min()) / (scores_adj.max() - scores_adj.min() + 1e-9) if len(scores_adj) else scores_adj
            comb = (1.0 - cfg.rerank.ce_weight)*rrf_n + cfg.rerank.ce_weight*ce_n
            head = list(zip([pid for pid,_ in gated[:ceN]], comb.tolist()))
            tail = gated[ceN:]
            reranked = sorted(head, key=lambda x: -x[1]) + tail

        topk = reranked[:cfg.k]
        return {
            "bm25": bm,
            "faiss": fa,
            "fused": fused,
            "gated": gated,
            "gate_dbg": gate_dbg,
            "ce_pairs": ce_pairs,
            "ce_scores": ce_scores,
            "final": topk,
        }

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser("Stratified eval with optional query rewrite")
    ap.add_argument("--index_dir", required=True)
    ap.add_argument("--doc_index_dir", required=True)
    ap.add_argument("--queries", nargs="+", required=True)
    ap.add_argument("--embed_model", required=True)

    ap.add_argument("--k", type=int, default=300)
    ap.add_argument("--rrf_alpha", type=float, default=60.0)
    ap.add_argument("--doc_topN", type=int, default=0)
    ap.add_argument("--restrict_bm25_by_doc", action="store_true")

    ap.add_argument("--gate_mode", choices=["none","prefer","require"], default="none")
    ap.add_argument("--gate_sections", default="")

    ap.add_argument("--ce_model", default=None)
    ap.add_argument("--ce_top", type=int, default=0)
    ap.add_argument("--ce_min", type=float, default=0.15)
    ap.add_argument("--ce_weight", type=float, default=0.75)

    ap.add_argument("--rewrite_cfg", default=None)
    ap.add_argument("--dump_eval_dir", default=None)

    args = ap.parse_args()

    index_dir = Path(args.index_dir)
    meta_df = load_passage_meta(index_dir)
    print(f"[INFO] Loaded metadata: {len(meta_df)} passages")
    if not FAISS_OK:
        raise RuntimeError("faiss not available")
    faiss_index = load_faiss_index(index_dir)
    print(f"[INFO] FAISS loaded: faiss.index")

    encoder = SentenceTransformer(args.embed_model)
    print(f"[INFO] Query encoder ready: {args.embed_model}")

    doc_pf = load_doc_prefilter(Path(args.doc_index_dir))
    print(f"[INFO] Doc prefilter ready: n={doc_pf[0].ntotal}, dim={doc_pf[0].d}")

    subsets = load_queries_labeled(args.queries)
    for key in subsets:
        print(f"[INFO] Subset '{key}': {len(subsets[key])} queries")

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

    # init rewrite (optional)
    rewriter = None
    if args.rewrite_cfg and Rewriter and load_rw_config:
        try:
            rw_cfg = load_rw_config(args.rewrite_cfg)
            rewriter = Rewriter(rw_cfg)
            print(f"[INFO] Rewrite enabled (cfg: {args.rewrite_cfg})")
        except Exception as e:
            print(f"[WARN] failed to init rewrite: {e}")

    engine = StratifiedSearch(meta_df, faiss_index, encoder, doc_pf)

    dump_root = Path(args.dump_eval_dir) if args.dump_eval_dir else None
    if dump_root: ensure_dir(dump_root)

    summary: Dict[str, Dict[str, float]] = {}
    for subset_name, qlist in subsets.items():
        prec_list: List[float] = []
        rec_list:  List[float] = []
        ndcg_list: List[float] = []
        subset_dump = dump_root / subset_name if dump_root else None
        if subset_dump: ensure_dir(subset_dump)

        for qi, item in enumerate(tqdm(qlist, desc=f"Eval[{subset_name}]")):
            q_raw = item["query"]
            gold_ids = set(item.get("gold_doc_ids", []))
            rewrite_info = None
            q_used = q_raw
            if rewriter:
                ri = rewriter.rewrite(q_raw)
                q_used = ri.get("query_rewritten") or q_raw
                rewrite_info = ri

            run = engine.search_one(q_used, cfg)

            # dedup по doc_id перед підрахунком
            final = run["final"]
            seen: set[int] = set()
            final_doc_ids: List[int] = []
            final_rows = []
            for pid, sc in final:
                d = int(engine.pass_doc_ids[pid])
                if d in seen: 
                    continue
                seen.add(d)
                final_doc_ids.append(d)
                final_rows.append((pid, sc, d))

            hits = [1 if d in gold_ids else 0 for d in final_doc_ids]
            P = precision_k(hits, args.k)
            R = recall_k(hits, len(gold_ids), args.k)
            N = ndcg_hits_k(hits, len(gold_ids), args.k)
            prec_list.append(P); rec_list.append(R); ndcg_list.append(N)

            if subset_dump:
                dump = {
                    "query_raw": q_raw,
                    "query_used": q_used,
                    "gold_doc_ids": sorted(list(gold_ids)),
                    "rewrite": rewrite_info,
                    "doc_prefilter": {"topN": args.doc_topN},
                    "final@k": [
                        {"pid": int(pid), "score": float(sc), "doc_id": int(doc_id),
                         "section": engine.pass_sections[pid], "hit": 1 if doc_id in gold_ids else 0}
                        for (pid, sc, doc_id) in final_rows
                    ],
                }
                with (subset_dump / f"q{qi:04d}.json").open("w", encoding="utf-8") as f:
                    json.dump(dump, f, ensure_ascii=False, indent=2)

        summary[subset_name] = {
            "Precision@k": float(np.mean(prec_list) if prec_list else 0.0),
            "Recall@k":    float(np.mean(rec_list) if rec_list else 0.0),
            "nDCG@k":      float(np.mean(ndcg_list) if ndcg_list else 0.0),
            "n_queries":   int(len(prec_list)),
        }

    print("\n=== STRATIFIED EVAL (macro-averaged) ===")
    for name, m in summary.items():
        print(f"[{name}] Precision@{args.k}: {m['Precision@k']:.4f} | Recall@{args.k}: {m['Recall@k']:.4f} | nDCG@{args.k}: {m['nDCG@k']:.4f} | n={m['n_queries']}")

    if dump_root:
        with (dump_root / "summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    sys.exit(main())
