# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Iterable, List, Tuple

import faiss
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ------------------- utils -------------------

def ts_now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def _safe_to_list(x) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        return list(x)
    return [x]

def _safe_to_int_list(x) -> List[int]:
    out = []
    for t in _safe_to_list(x):
        if isinstance(t, int):
            out.append(t)
        elif isinstance(t, str) and t.isdigit():
            out.append(int(t))
    return out

def load_gold_ids(item: Dict[str, Any]) -> List[int]:
    # підтримаємо кілька можливих ключів
    CAND_KEYS = ["gold_doc_ids", "gold_ids", "gold_id", "golds", "doc_gold_ids", "doc_ids"]
    for k in CAND_KEYS:
        if k in item:
            g = _safe_to_int_list(item.get(k))
            if g:
                return g
    return []

def pick_query_text(item: Dict[str, Any]) -> str:
    for k in ("query", "q", "text", "question"):
        if k in item and isinstance(item[k], str):
            return item[k]
    # fallback — дамп усякого, але краще, щоби був "query"
    return str(item)

# ------------------- metrics -------------------

def precision_at_k(pred_ids: List[int], gold_ids: List[int], k: int) -> float:
    if not gold_ids:
        return 0.0
    pred = pred_ids[:k]
    inter = len(set(pred) & set(gold_ids))
    return inter / float(k)

def recall_at_k(rank_all: List[int], gold_ids: List[int], k: int) -> float:
    if not gold_ids:
        return 0.0
    pred = rank_all[:k]
    inter = len(set(pred) & set(gold_ids))
    return inter / float(len(gold_ids))

def ndcg_at_k(rank_all: List[int], gold_ids: List[int], k: int) -> float:
    if not gold_ids:
        return 0.0
    gold = set(gold_ids)
    dcg = 0.0
    for r, pid in enumerate(rank_all[:k]):
        if pid in gold:
            dcg += 1.0 / np.log2(r + 2.0)  # r is 0-based
    ideal = min(len(gold_ids), k)
    idcg = sum(1.0 / np.log2(i + 2.0) for i in range(ideal))
    return float(dcg / (idcg + 1e-9))

# ------------------- RRF / WRRF -------------------

try:
    from search.wrrf_fusion import rrf, weighted_rrf  # твій модуль (вже є в репо)
except Exception:
    from math import inf
    def rrf(ranks_a: Dict[int, int], ranks_b: Dict[int, int], alpha: float = 60.0):
        ids = set(ranks_a) | set(ranks_b)
        out = {}
        for d in ids:
            ra = ranks_a.get(d, inf)
            rb = ranks_b.get(d, inf)
            sa = 0.0 if ra is inf else 1.0 / (alpha + ra)
            sb = 0.0 if rb is inf else 1.0 / (alpha + rb)
            out[d] = sa + sb
        return sorted(out.items(), key=lambda x: x[1], reverse=True)

    def weighted_rrf(ranks_bm25: Dict[int, int], ranks_dense: Dict[int, int],
                     alpha: float = 60.0, w_bm25: float = 1.0, w_dense: float = 1.0):
        ids = set(ranks_bm25) | set(ranks_dense)
        out = {}
        for d in ids:
            rb = ranks_bm25.get(d, inf)
            rd = ranks_dense.get(d, inf)
            sb = 0.0 if rb is inf else (w_bm25 / (alpha + rb))
            sd = 0.0 if rd is inf else (w_dense / (alpha + rd))
            out[d] = sb + sd
        return sorted(out.items(), key=lambda x: x[1], reverse=True)

# ------------------- CrossEncoder backends -------------------

_FlagReranker = None
try:
    from FlagEmbedding import FlagReranker as _FlagReranker
except Exception:
    pass

_CrossEncoder = None
try:
    from sentence_transformers import CrossEncoder as _CrossEncoder
except Exception:
    pass

def _softmax(x: List[float]) -> List[float]:
    a = np.array(x, dtype=np.float32)
    a -= a.max()
    e = np.exp(a)
    return (e / (e.sum() + 1e-9)).tolist()

# ------------------- Pipeline -------------------

class WRRFCEPipeline:
    """
    Dense(FAISS) + BM25 -> (W)RRF -> (optional) CE -> doc-level mix -> топ-K.
    Пише qNNNN.json із pred_doc_ids (канонічні) та results (id, score).
    """
    def __init__(
        self,
        index_dir: Path,
        faiss_index: faiss.Index,
        encode_fn,  # (List[str]) -> np.ndarray [n, dim]
        rrf_alpha: float = 90.0,
        w_bm25: float = 2.0,
        w_dense: float = 1.0,
        k_bm25: int = 100,
        k_dense: int = 100,
        fused_top: int = 300,
        k_out: int = 10,
        bm25_truncate_tokens: int = 64,
        ce_model_name: str = "",
        ce_top: int = 60,
        ce_weight: float = 0.7,
        ce_batch: int = 8,
    ):
        self.index_dir = Path(index_dir)
        self.faiss = faiss_index
        self.encode_fn = encode_fn
        self.rrf_alpha = float(rrf_alpha)
        self.w_bm25 = float(w_bm25)
        self.w_dense = float(w_dense)
        self.k_bm25 = int(k_bm25)
        self.k_dense = int(k_dense)
        self.fused_top = int(fused_top)
        self.k_out = int(k_out)
        self.bm25_trunc = int(bm25_truncate_tokens)
        self.ce_top = int(ce_top)
        self.ce_weight = float(ce_weight)
        self.ce_batch = int(ce_batch)
        self.ce = None
        self.ce_name = (ce_model_name or "").strip()

        # --------- load chunks & mappings ---------
        p_chunks = None
        for c in [self.index_dir / "chunks.parquet", self.index_dir.parent / "chunks.parquet"]:
            if c.exists():
                p_chunks = c
                break
        if p_chunks is None:
            raise FileNotFoundError("chunks.parquet not found near index_dir")

        pf = pq.ParquetFile(p_chunks)
        cols = set(pf.schema_arrow.names)
        text_col = "text" if "text" in cols else ("chunk_text" if "chunk_text" in cols else None)
        if text_col is None:
            raise RuntimeError(f"No text column in chunks.parquet; got: {sorted(cols)}")

        use_cols = [text_col]
        doc_col = None
        if "doc_id" in cols:
            doc_col = "doc_id"
            use_cols.append("doc_id")
        df = pd.read_parquet(p_chunks, columns=use_cols)
        self.pass_texts = df[text_col].astype(str).tolist()
        if doc_col is not None:
            self.pid2doc = df[doc_col].to_numpy()
            pidmap_src = "chunks:doc_id"
        else:
            npy = None
            for c in [self.index_dir / "doc_ids.npy", self.index_dir.parent / "doc_ids.npy"]:
                if c.exists():
                    npy = c
                    break
            if npy is None:
                raise FileNotFoundError("Neither chunks:doc_id nor doc_ids.npy found")
            self.pid2doc = np.load(npy)
            pidmap_src = "doc_ids.npy"
        del df

        print(f"[INFO] BM25 corpus: {len(self.pass_texts)} passages")
        # компактний BM25 через триммінг токенів
        if self.bm25_trunc > 0:
            corpus = [" ".join(t.split()[: self.bm25_trunc]) for t in self.pass_texts]
        else:
            corpus = self.pass_texts
        self._bm25_tok = [c.lower().split() for c in corpus]
        self.bm25 = BM25Okapi(self._bm25_tok)
        print(f"[INFO] BM25 ready (truncate={self.bm25_trunc})")
        print(f"[INFO] PID→DOC source: {pidmap_src}")

        # --------- CrossEncoder (optional) ---------
        cm = self.ce_name.lower()
        if cm and cm not in ("none", "off", "0") and self.ce_top > 0:
            if _FlagReranker is not None and ("bge-reranker" in cm or "baai/" in cm):
                self.ce = _FlagReranker(self.ce_name, use_fp16=True)
                self._ce_type = "flag"
            elif _CrossEncoder is not None:
                self.ce = _CrossEncoder(self.ce_name)
                self._ce_type = "st"
            else:
                self.ce = None
                self._ce_type = "none"
                print("[WARN] CE requested but backends are unavailable (FlagEmbedding / sentence-transformers).")
            if self.ce is not None:
                print(f"[INFO] CrossEncoder active: {self.ce_name} (type={self._ce_type})")
        else:
            self._ce_type = "none"
            print("[INFO] CrossEncoder disabled")

    # --------- low-level helpers ---------

    def _dense_topk(self, q_text: str, k: int) -> List[Tuple[int, float]]:
        qv = self.encode_fn([q_text])[0].astype(np.float32)[None, :]
        D, I = self.faiss.search(qv, k)
        return [(int(i), float(d)) for i, d in zip(I[0].tolist(), D[0].tolist()) if i >= 0]

    def _bm25_topk(self, q_text: str, k: int) -> List[Tuple[int, float]]:
        qtok = q_text.lower().split()
        scores = self.bm25.get_scores(qtok)
        idx = np.argpartition(scores, -k)[-k:]
        idx = idx[np.argsort(scores[idx])[::-1]]
        return [(int(i), float(scores[i])) for i in idx]

    @staticmethod
    def _ranks(id_and_score: List[Tuple[int, float]]) -> Dict[int, int]:
        return {int(i): (r + 1) for r, (i, _) in enumerate(id_and_score)}

    def _fuse_passages(self, q_text: str) -> List[int]:
        top_dense = self._dense_topk(q_text, self.k_dense)
        top_bm25 = self._bm25_topk(q_text, self.k_bm25)
        r_dense = self._ranks(top_dense)
        r_bm25 = self._ranks(top_bm25)
        if self.w_bm25 == 1.0 and self.w_dense == 1.0:
            fused = rrf(r_bm25, r_dense, alpha=self.rrf_alpha)
        else:
            fused = weighted_rrf(r_bm25, r_dense, alpha=self.rrf_alpha,
                                 w_bm25=self.w_bm25, w_dense=self.w_dense)
        pids = [pid for pid, _ in fused[: self.fused_top]]
        return pids

    def _ce_rescore(self, q_text: str, cand_pids: List[int]) -> Dict[int, float]:
        if self.ce is None or self._ce_type == "none" or self.ce_top <= 0:
            return {}
        keep = cand_pids[: self.ce_top]
        pairs = [(q_text, self.pass_texts[pid]) for pid in keep]
        scores: List[float] = []
        if self._ce_type == "flag":
            # FlagEmbedding: compute_score(..., normalize=True) -> [0,1]
            for i in range(0, len(pairs), self.ce_batch):
                chunk = pairs[i : i + self.ce_batch]
                s = self.ce.compute_score(chunk, normalize=True)
                scores.extend([float(x) for x in s])
        else:
            # sentence-transformers CrossEncoder: повертає логіти; зробимо softmax
            for i in range(0, len(pairs), self.ce_batch):
                chunk = pairs[i : i + self.ce_batch]
                s = self.ce.predict(chunk)
                if isinstance(s, list):
                    scores.extend([float(x) for x in s])
                else:
                    scores.extend([float(x) for x in s.tolist()])
            scores = _softmax(scores)
        return {pid: float(sc) for pid, sc in zip(keep, scores)}

    def _aggregate_to_docs(self, pid_list: List[int], pid2score: Dict[int, float] | None) -> List[Tuple[int, float]]:
        """
        Переносимо passage-level у doc-level: max-score per doc.
        Якщо pid2score=None — використовуємо зворотний ранг (вище → краще).
        Повертаємо відсортовану пару (doc_id, score в [0,1]).
        """
        if not pid_list:
            return []
        doc2score: Dict[int, float] = {}
        if pid2score is None:
            for r, pid in enumerate(pid_list):
                doc = int(self.pid2doc[pid])
                base = 1.0 / (r + 1)  # простий ранг-скор
                if doc not in doc2score or base > doc2score[doc]:
                    doc2score[doc] = base
        else:
            for pid, sc in pid2score.items():
                doc = int(self.pid2doc[pid])
                base = float(sc)
                if doc not in doc2score or base > doc2score[doc]:
                    doc2score[doc] = base

        docs, scores = zip(*doc2score.items())
        s = np.array(scores, dtype=np.float32)
        s = (s - s.min()) / (s.max() - s.min() + 1e-9)
        return sorted(zip([int(d) for d in docs], s.tolist()), key=lambda x: x[1], reverse=True)

    def run_once(self, q_text: str, run_dir: Path, q_idx: int, out_k: int | None = None) -> Tuple[List[int], List[int]]:
        out_k = out_k or self.k_out

        # 1) (W)RRF на passage-level
        fused_pids = self._fuse_passages(q_text)

        # 2) CE (опційно) поверх fused_pids[:ce_top]
        ce_scores = self._ce_rescore(q_text, fused_pids)

        # 3) doc-level агрегати
        doc_fusion = self._aggregate_to_docs(fused_pids, pid2score=None)                  # з позицій фʼюжну
        doc_ce     = self._aggregate_to_docs(list(ce_scores.keys()), pid2score=ce_scores) if ce_scores else []

        # 4) змішування каналів
        mix: Dict[int, float] = {}
        def _acc(docscores: List[Tuple[int,float]], w: float):
            for d, s in docscores:
                mix[d] = mix.get(d, 0.0) + w * s

        if doc_fusion:
            _acc(doc_fusion, 1.0 - self.ce_weight)
        if doc_ce:
            _acc(doc_ce, self.ce_weight)

        final_all = sorted(mix.items(), key=lambda x: x[1], reverse=True)
        final_top = final_all[: out_k]
        pred_doc_ids = [int(d) for d, _ in final_top]
        doc_rank_all = [int(d) for d, _ in final_all]  # для метрик@300

        # 5) qNNNN.json
        out = {
            "query": q_text,
            "k": out_k,
            "rrf_alpha": self.rrf_alpha,
            "w_bm25": self.w_bm25,
            "w_dense": self.w_dense,
            "k_bm25": self.k_bm25,
            "k_dense": self.k_dense,
            "fused_top": self.fused_top,
            "bm25_truncate_tokens": self.bm25_trunc,
            "ce_model": self.ce_name or "none",
            "ce_top": self.ce_top,
            "ce_weight": self.ce_weight,
            "pred_doc_ids": pred_doc_ids,
            "doc_rank_topN": doc_rank_all[:300],
            "results": [{"id": int(d), "score": float(s)} for d, s in final_top],
        }
        (run_dir / f"q{q_idx:04d}.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        return pred_doc_ids, doc_rank_all

# ------------------- main (CLI) -------------------

def load_faiss_index(index_dir: Path) -> faiss.Index:
    cand = [index_dir / "faiss.index", index_dir / "index" / "faiss.index"]
    path = next((p for p in cand if p.exists()), None)
    if not path:
        raise FileNotFoundError(f"FAISS index not found in: {cand}")
    index = faiss.read_index(str(path))
    return index

def main():
    ap = argparse.ArgumentParser("Assistant from Parquet — hybrid FAISS + (W)RRF + optional CE (doc-level eval)")
    ap.add_argument("--index_dir", required=True, type=str)
    ap.add_argument("--doc_index_dir", type=str, default="")  # зарезервовано, не обовʼязково
    ap.add_argument("--embed_model", type=str, default="intfloat/multilingual-e5-base")
    ap.add_argument("--queries", required=True, type=str)
    ap.add_argument("--intent_policy", type=str, default="")  # сумісність із твоїми скриптами
    ap.add_argument("--dump_eval_dir", type=str, required=True)

    # (W)RRF кноби
    ap.add_argument("--rrf_alpha", type=float, default=90.0)
    ap.add_argument("--use_weighted_rrf", action="store_true", default=True)
    ap.add_argument("--w_bm25", type=float, default=2.0)
    ap.add_argument("--w_dense", type=float, default=1.0)
    ap.add_argument("--k_bm25", type=int, default=100)
    ap.add_argument("--k_dense", type=int, default=100)
    ap.add_argument("--fused_top", type=int, default=300)
    ap.add_argument("--k", dest="k_out", type=int, default=10, help="top-K docs to output")
    ap.add_argument("--bm25_truncate_tokens", type=int, default=64)

    # CE кноби
    ap.add_argument("--ce_model", type=str, default="", help="BAAI/bge-reranker-v2-m3 | cross-encoder/ms-marco-... | 'none'")
    ap.add_argument("--ce_top", type=int, default=60)
    ap.add_argument("--ce_weight", type=float, default=0.7)
    ap.add_argument("--ce_batch", type=int, default=8)

    args = ap.parse_args()

    index_dir = Path(args.index_dir).expanduser().resolve()
    dump_dir = ensure_dir(Path(args.dump_eval_dir).expanduser().resolve())
    run_dir = ensure_dir(dump_dir / f"run_{ts_now()}")

    # ---------- FAISS + Encoder ----------
    index = load_faiss_index(index_dir)
    dim = index.d
    print(f"[INFO] FAISS loaded: faiss.index | dim={dim}")

    enc = SentenceTransformer(args.embed_model)
    print(f"[INFO] Query encoder ready: {args.embed_model}")

    def _encode_fn(texts: List[str]) -> np.ndarray:
        # нормалізація корисна, якщо індекс побудовано під cos/IP
        return enc.encode(texts, normalize_embeddings=True)

    # ---------- Pipeline ----------
    # якщо вважаєш за краще чистий RRF — можеш подати --w_bm25 1 --w_dense 1
    pipe = WRRFCEPipeline(
        index_dir=index_dir,
        faiss_index=index,
        encode_fn=_encode_fn,
        rrf_alpha=args.rrf_alpha,
        w_bm25=args.w_bm25,
        w_dense=args.w_dense,
        k_bm25=args.k_bm25,
        k_dense=args.k_dense,
        fused_top=args.fused_top,
        k_out=args.k_out,
        bm25_truncate_tokens=args.bm25_truncate_tokens,
        ce_model_name=args.ce_model,
        ce_top=args.ce_top,
        ce_weight=args.ce_weight,
        ce_batch=args.ce_batch,
    )

    # ---------- Queries ----------
    qpath = Path(args.queries).expanduser().resolve()
    queries = list(iter_jsonl(qpath))
    print(f"[INFO] Loaded queries: {len(queries)}")

    # ---------- Run ----------
    pred_rows = []
    n_with_gold = 0
    p10s, r300s, ndcg300s = [], [], []

    for qi, item in enumerate(tqdm(queries, desc="Eval", ncols=100), start=0):
        q_text = pick_query_text(item)
        topk_docs, rank_all = pipe.run_once(q_text=q_text, run_dir=run_dir, q_idx=qi, out_k=args.k_out)

        gold_ids = load_gold_ids(item)
        if gold_ids:
            n_with_gold += 1
            p10s.append(precision_at_k(topk_docs, gold_ids, 10))
            r300s.append(recall_at_k(rank_all, gold_ids, 300))
            ndcg300s.append(ndcg_at_k(rank_all, gold_ids, 300))

        # preds.csv рядок (компактно)
        row = {
            "qid": qi + 1,
            "query": q_text,
            "pred_doc_ids": json.dumps([int(x) for x in topk_docs], ensure_ascii=False),
            "gold_doc_ids": json.dumps([int(x) for x in gold_ids], ensure_ascii=False),
        }
        pred_rows.append(row)

    # ---------- Metrics dump ----------
    if n_with_gold > 0:
        P10 = float(np.mean(p10s)) if p10s else 0.0
        R300 = float(np.mean(r300s)) if r300s else 0.0
        NDCG300 = float(np.mean(ndcg300s)) if ndcg300s else 0.0
    else:
        P10 = R300 = NDCG300 = 0.0

    preds_path = run_dir / "preds.csv"
    pd.DataFrame(pred_rows).to_csv(preds_path, index=False)
    print(f"[OK] predictions -> {preds_path}")

    metrics = {
        "Precision@10": P10,
        "Recall@300": R300,
        "nDCG@300": NDCG300,
        "N": len(queries),
        "N_with_gold": n_with_gold,
        "subset": "full",
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n=== METRICS ===")
    print(f"Precision@10: {P10:.4f}")
    print(f"Recall@300:   {R300:.4f}")
    print(f"nDCG@300:     {NDCG300:.4f}")
    print(f"N: {len(queries)} (with gold: {n_with_gold})")
    print(f"run_dir: {run_dir}")

if __name__ == "__main__":
    main()
