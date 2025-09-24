# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import math
import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# BM25
from rank_bm25 import BM25Okapi

# FAISS bi-encoder
try:
    import faiss  # type: ignore
    FAISS_OK = True
except Exception:
    FAISS_OK = False

from sentence_transformers import SentenceTransformer, CrossEncoder

# ------------------------- утиліти -------------------------

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def chunk_text(txt: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    txt = clean_text(txt)
    if not txt:
        return []
    tokens = txt.split(" ")
    chunks = []
    i = 0
    while i < len(tokens):
        piece = " ".join(tokens[i:i+chunk_size])
        if piece.strip():
            chunks.append(piece.strip())
        i += max(1, chunk_size - overlap)
    return chunks

# ------------------------- індекс -------------------------

@dataclass
class SearchConfig:
    rrf_alpha: float = 0.35
    top_k: int = 120
    show: int = 120
    prefer_indications: bool = False
    ce_min: float = 0.15
    ce_weight: float = 0.8

class AssistantIndex:
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.passages: List[str] = []
        self.meta: List[Dict] = []

        # BM25
        self.bm25: Optional[BM25Okapi] = None

        # FAISS/bi-encoder
        self.encoder: Optional[SentenceTransformer] = None
        self.faiss_index = None
        self.vecs: Optional[np.ndarray] = None

        # CrossEncoder
        self.ce: Optional[CrossEncoder] = None
        self.ce_top = 100
        self.ce_min = 0.15
        self.ce_w = 0.8

    # ---- побудова ----
    def build_from_dataframe(
        self,
        df: pd.DataFrame,
        encoder_model: Optional[str] = None,
        prefer_indications: bool = False,
        chunk_size: int = 1200,
        overlap: int = 200,
        show_batches: bool = True,
    ) -> None:
        """
        Формуємо корпус пасажів із основних полів і будуємо BM25 (+ за наявності — FAISS).
        """
        self.df = df.copy()
        self.passages = []
        self.meta = []
        prefer_ind = prefer_indications

        # які поля беремо
        fields = [
            "Назва препарату",
            "Показання",
            "Протипоказання",
            "Спосіб застосування та дози",
            "Склад",
            "Фармакотерапевтична група",
            "Фармакологічні властивості",
        ]
        for f in list(fields):
            if f not in self.df.columns:
                fields.remove(f)

        rows = []
        for i, row in self.df.iterrows():
            name = clean_text(row.get("Назва препарату", ""))
            # формуємо “основний текст”
            blocks = []
            if "Показання" in fields and prefer_ind:
                blocks.append(clean_text(row.get("Показання", "")))
            for f in fields:
                if f == "Показання" and prefer_ind:
                    continue
                blocks.append(clean_text(row.get(f, "")))
            txt = " [SEP] ".join([b for b in blocks if b])

            # chunking
            chunks = chunk_text(txt, chunk_size=chunk_size, overlap=overlap)
            for ch in chunks:
                self.passages.append(ch)
                self.meta.append({
                    "doc_id": i,
                    "name": name
                })

        # Показати батчі (крок енкодера)
        if show_batches:
            total = len(self.passages)
            bs = 128
            batches = math.ceil(total / bs)
            print(f"Batches: 0/{batches}", end="", flush=True)

        # BM25
        tokenized = [p.split() for p in self.passages]
        self.bm25 = BM25Okapi(tokenized)

        # FAISS (за наявності моделі і faiss)
        if encoder_model and FAISS_OK:
            self.encoder = SentenceTransformer(encoder_model)
            # енкодимо з прогресом і одночасно друкуємо батчі
            vecs = []
            bs = 128
            total = len(self.passages)
            batches = math.ceil(total / bs)
            for bi in tqdm(range(batches), total=batches, desc="Encode", leave=False):
                s = bi * bs
                e = min(total, s + bs)
                emb = self.encoder.encode(self.passages[s:e], show_progress_bar=False, normalize_embeddings=True)
                vecs.append(emb.astype("float32"))
            self.vecs = np.vstack(vecs)
            d = self.vecs.shape[1]
            self.faiss_index = faiss.IndexFlatIP(d)
            self.faiss_index.add(self.vecs)
            print(f"[INFO] Побудовано пасажів: {len(self.passages)}  | FAISS dim={d}")
        else:
            print(f"[INFO] Побудовано пасажів: {len(self.passages)}  | FAISS: OFF")

    def enable_crossencoder(self, model_name: str, ce_top: int, ce_min: float, ce_weight: float) -> None:
        self.ce = CrossEncoder(model_name, max_length=512)
        self.ce_top = ce_top
        self.ce_min = ce_min
        self.ce_w = ce_weight
        print(f"[INFO] CrossEncoder активний: {model_name} (top={ce_top})")

    # ---- пошук ----
    def search(self, query: str, cfg: SearchConfig) -> List[Dict]:
        assert self.bm25 is not None
        # BM25
        scores_bm = self.bm25.get_scores(query.split())
        bm_idx = np.argsort(-scores_bm)[:cfg.top_k]
        bm = [(int(i), float(scores_bm[i])) for i in bm_idx]

        # FAISS
        vec = None
        fa = []
        if self.encoder is not None and self.faiss_index is not None:
            vec = self.encoder.encode([query], normalize_embeddings=True).astype("float32")
            D, I = self.faiss_index.search(vec, cfg.top_k)
            fa = [(int(i), float(D[0, j])) for j, i in enumerate(I[0])]

        # RRF
        def rrf(ranks: Dict[int, int], k: int, alpha: float) -> Dict[int, float]:
            sc = {}
            for pid, r in ranks.items():
                sc[pid] = sc.get(pid, 0.0) + 1.0 / (alpha + r)
            return sc

        r_bm = {pid: r+1 for r, (pid, _) in enumerate(bm)}
        r_fa = {pid: r+1 for r, (pid, _) in enumerate(fa)}
        all_ids = set(r_bm) | set(r_fa)
        fuse = {}
        for pid in all_ids:
            s = 0.0
            if pid in r_bm: s += 1.0 / (cfg.rrf_alpha + r_bm[pid])
            if pid in r_fa: s += 1.0 / (cfg.rrf_alpha + r_fa[pid])
            fuse[pid] = s

        fused = sorted(fuse.items(), key=lambda x: -x[1])[:cfg.top_k]
        cand = [{"pid": pid, "text": self.passages[pid], "doc_id": self.meta[pid]["doc_id"], "name": self.meta[pid]["name"], "rrf": sc} for pid, sc in fused]

        # CrossEncoder rerank
        if self.ce is not None and len(cand) > 0:
            pairs = [(query, c["text"]) for c in cand[:self.ce_top]]
            ce_scores = self.ce.predict(pairs).tolist()
            for i, s in enumerate(ce_scores):
                cand[i]["ce"] = float(s)
            # фінальне злиття
            for c in cand[:self.ce_top]:
                ce_s = max(0.0, c.get("ce", 0.0) - self.ce_min)
                c["score"] = (1.0 - self.ce_w) * c["rrf"] + self.ce_w * ce_s
            for c in cand[self.ce_top:]:
                c["score"] = c["rrf"]
            cand = sorted(cand, key=lambda x: -x["score"])

        return cand[:cfg.show]
