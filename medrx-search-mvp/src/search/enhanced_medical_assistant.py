# -*- coding: utf-8 -*-
from __future__ import annotations
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

# ---- Medical-aware chunking
from preprocessing.medical_chunker import MedicalChunker

# ---- Safety filter (handle file name typo gracefully)
UserProfile = None
MedicalSafetyFilter = None
try:
    from safety.medical_safety_filter import MedicalSafetyFilter as _MSF, UserProfile as _UP  # preferred
    MedicalSafetyFilter, UserProfile = _MSF, _UP
except Exception:
    try:
        from safety.medical_safery_filter import MedicalSafetyFilter as _MSF2, UserProfile as _UP2  # fallback
        MedicalSafetyFilter, UserProfile = _MSF2, _UP2
    except Exception:
        class _StubUP:  # type: ignore
            def __init__(self, **kwargs): pass
        class _StubMSF:  # type: ignore
            def assess_drug_safety(self, contraindications: str, user_profile=None):
                return {'risk_score': 0, 'risk_level': 'LOW', 'warnings': [], 'critical_warnings': [], 'safe_to_use': True}
        UserProfile, MedicalSafetyFilter = _StubUP, _StubMSF

# ------------------------- utils -------------------------

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def medical_chunk_text(txt: str, chunker: MedicalChunker, max_tokens: int = 256) -> List[str]:
    """Use MedicalChunker instead of primitive chunking."""
    txt = clean_text(txt).replace("[SEP]", ". ")  # важливо: робимо [SEP] sentence boundary
    if not txt:
        return []
    return chunker.smart_chunking(txt, max_tokens=max_tokens, overlap_tokens=32, min_chunk_chars=60)

def _bm25_tokens(s: str) -> List[str]:
    """Lightweight UA/EN tokenization for BM25 (keep letters/digits from Latin + Cyrillic)."""
    s = s.lower()
    s = re.sub(r"[^\w\u0400-\u04FF]+", " ", s)
    return s.split()

# ------------------------- index -------------------------

@dataclass
class SearchConfig:
    # RRF параметр (костанта k в 1/(k + rank)):
    rrf_alpha: float = 60.0          # практичні значення: 60, 90, 120
    top_k: int = 120
    show: int = 120
    prefer_indications: bool = True  # вмикаємо невеликий буст за замовчуванням
    indications_boost: float = 0.05  # величина буста для секції "Показання"
    ce_min: float = 0.15
    ce_weight: float = 0.8
    # new
    enable_safety_filter: bool = True
    medical_chunking: bool = True
    max_chunk_tokens: int = 256

class EnhancedMedicalAssistant:
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

        # Components
        self.medical_chunker = MedicalChunker(
            tokenizer_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.safety_filter = MedicalSafetyFilter() if MedicalSafetyFilter else None

    # ---- build ----
    def build_from_dataframe(
        self,
        df: pd.DataFrame,
        encoder_model: Optional[str] = None,
        prefer_indications: bool = True,
        medical_chunking: bool = True,
        max_chunk_tokens: int = 256,
        show_batches: bool = True,
    ) -> None:
        """
        Build corpus of passages from key fields and construct BM25 (+ FAISS if available).
        Uses MedicalChunker for structured, medical-aware chunking.
        """
        self.df = df.copy()
        self.passages = []
        self.meta = []

        fields = [
            # !!! НЕ включаємо "Назва препарату" в текст для індексації (лише в meta)
            "Показання",
            "Протипоказання",
            "Спосіб застосування та дози",
            "Склад",
            "Фармакотерапевтична група",
            "Фармакологічні властивості",
        ]
        fields = [f for f in fields if f in self.df.columns]

        print(f"[INFO] Building index with {'MEDICAL' if medical_chunking else 'STANDARD'} chunking...")

        for i, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Processing drugs"):
            name = clean_text(row.get("Назва препарату", ""))

            # Structured medical text with field labels (helps BM25/CE)
            blocks: List[str] = []
            for field in fields:
                content = clean_text(row.get(field, ""))
                if content:
                    blocks.append(f"{field}: {content}")

            full_text = " [SEP] ".join(blocks)

            # Chunking
            if medical_chunking:
                chunks = medical_chunk_text(full_text, self.medical_chunker, max_tokens=max_chunk_tokens)
            else:
                chunks = self._old_chunk_text(full_text, chunk_size=1200, overlap=200)

            indications = row.get("Показання", "") or ""
            contraindications = row.get("Протипоказання", "") or ""

            for ch in chunks:
                section = ch.split(":", 1)[0].strip() if ":" in ch else "Текст"
                self.passages.append(ch)
                self.meta.append({
                    "doc_id": int(i),
                    "name": name,
                    "section": section,
                    "indications": indications,
                    "contraindications": contraindications,
                })

        # BM25
        tokenized = [_bm25_tokens(p) for p in self.passages]
        self.bm25 = BM25Okapi(tokenized)

        # FAISS (optional)
        if encoder_model and FAISS_OK:
            self._build_faiss_index(encoder_model, show_batches=show_batches)
        else:
            print(f"[INFO] Passages: {len(self.passages)} | FAISS: OFF")

    def _build_faiss_index(self, encoder_model: str, show_batches: bool = True) -> None:
        """Build FAISS index with batched encoding."""
        self.encoder = SentenceTransformer(encoder_model)
        vecs = []
        bs = 128
        total = len(self.passages)
        batches = math.ceil(total / bs)
        for bi in tqdm(range(batches), total=batches, desc="Encoding", leave=False):
            s = bi * bs
            e = min(total, s + bs)
            emb = self.encoder.encode(
                self.passages[s:e],
                show_progress_bar=False,
                normalize_embeddings=True
            )
            vecs.append(np.asarray(emb, dtype="float32"))
        self.vecs = np.vstack(vecs) if vecs else None
        if self.vecs is None:
            print("[WARN] No vectors encoded; FAISS disabled.")
            return
        d = self.vecs.shape[1]
        self.faiss_index = faiss.IndexFlatIP(d)
        self.faiss_index.add(self.vecs)
        print(f"[INFO] Passages: {len(self.passages)} | FAISS dim={d}")

    def _old_chunk_text(self, txt: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
        tokens = txt.split(" ")
        chunks = []
        i = 0
        while i < len(tokens):
            piece = " ".join(tokens[i:i+chunk_size])
            if piece.strip():
                chunks.append(piece.strip())
            i += max(1, chunk_size - overlap)
        return chunks

    def enable_crossencoder(self, model_name: str, ce_top: int = 100, ce_min: float = 0.15, ce_weight: float = 0.8) -> None:
        self.ce = CrossEncoder(model_name, max_length=512)
        self.ce_top = ce_top
        self.ce_min = ce_min
        self.ce_w = ce_weight
        print(f"[INFO] CrossEncoder active: {model_name} (top={ce_top})")

    # ---- search ----
    def search(self, query: str, cfg: SearchConfig, user_profile: Optional[UserProfile] = None) -> List[Dict]:
        """Enhanced search with optional medical safety filtering."""
        assert self.bm25 is not None

        # BM25
        scores_bm = self.bm25.get_scores(_bm25_tokens(query))
        bm_idx = np.argsort(-scores_bm)[:cfg.top_k]
        bm = [(int(i), float(scores_bm[i])) for i in bm_idx]

        # FAISS
        fa: List[Tuple[int, float]] = []
        if self.encoder is not None and self.faiss_index is not None:
            vec = self.encoder.encode([query], normalize_embeddings=True).astype("float32")
            D, I = self.faiss_index.search(vec, cfg.top_k)
            fa = [(int(i), float(D[0, j])) for j, i in enumerate(I[0])]

        # RRF fusion (k = cfg.rrf_alpha)
        candidates = self._rrf_fusion(bm, fa, cfg)

        # If no CE, use normalized RRF + optional indications boost
        if self.ce is None and candidates:
            rr_vals = [c["rrf"] for c in candidates]
            rr_min, rr_max = min(rr_vals), max(rr_vals)
            def nrm(v: float) -> float:
                return (v - rr_min) / (rr_max - rr_min + 1e-9)
            for c in candidates:
                score = nrm(c["rrf"])
                if cfg.prefer_indications and c.get("section") == "Показання":
                    score += cfg.indications_boost
                c["score"] = score
            candidates = sorted(candidates, key=lambda x: -x["score"])

        # CrossEncoder reranking (with normalization + indications boost)
        if self.ce is not None and len(candidates) > 0:
            candidates = self._crossencoder_rerank(query, candidates, cfg)

        # Safety filtering (optional)
        if cfg.enable_safety_filter and user_profile and self.safety_filter:
            candidates = self._apply_safety_filter(candidates, user_profile)

        # Group by drug
        return self._group_and_format_results(candidates[:cfg.show])

    def _rrf_fusion(self, bm_results: List[Tuple[int, float]], fa_results: List[Tuple[int, float]], cfg: SearchConfig) -> List[Dict]:
        """RRF fusion of BM25 and FAISS candidate lists."""
        r_bm = {pid: r+1 for r, (pid, _) in enumerate(bm_results)}
        r_fa = {pid: r+1 for r, (pid, _) in enumerate(fa_results)}
        all_ids = set(r_bm) | set(r_fa)

        k = float(cfg.rrf_alpha)
        fuse: Dict[int, float] = {}
        for pid in all_ids:
            s = 0.0
            if pid in r_bm: s += 1.0 / (k + r_bm[pid])
            if pid in r_fa: s += 1.0 / (k + r_fa[pid])
            fuse[pid] = s

        fused = sorted(fuse.items(), key=lambda x: -x[1])[:cfg.top_k]
        out: List[Dict] = []
        for pid, sc in fused:
            meta = self.meta[pid]
            out.append({
                "pid": pid,
                "text": self.passages[pid],
                "rrf": float(sc),
                "doc_id": meta["doc_id"],
                "name": meta["name"],
                "section": meta.get("section", "Текст"),
                "indications": meta.get("indications", ""),
                "contraindications": meta.get("contraindications", ""),
            })
        return out

    def _crossencoder_rerank(self, query: str, candidates: List[Dict], cfg: SearchConfig) -> List[Dict]:
        """CrossEncoder reranking with batch normalization and small section-based boost."""
        topN = min(self.ce_top, len(candidates))
        pairs = [(query, candidates[i]["text"]) for i in range(topN)]
        ce_scores = self.ce.predict(pairs).tolist()  # type: ignore

        for i, s in enumerate(ce_scores):
            candidates[i]["ce"] = float(s)

        # Normalize RRF and CE to comparable 0..1 ranges
        rrf_vals = [c["rrf"] for c in candidates]
        rr_min, rr_max = min(rrf_vals), max(rrf_vals)
        ce_vals = [c.get("ce", 0.0) for c in candidates[:topN]]
        ce_raw_min, ce_raw_max = (min(ce_vals) if ce_vals else 0.0), (max(ce_vals) if ce_vals else 1.0)

        def nrm(v: float, a: float, b: float) -> float:
            return (v - a) / (b - a + 1e-9)

        for idx, c in enumerate(candidates):
            rrf_norm = nrm(c["rrf"], rr_min, rr_max)
            if idx < topN:
                ce_adj = max(0.0, c.get("ce", 0.0) - cfg.ce_min)
                ce_norm = nrm(ce_adj, max(0.0, ce_raw_min - cfg.ce_min), max(1e-9, ce_raw_max - cfg.ce_min))
                score = (1.0 - cfg.ce_weight) * rrf_norm + cfg.ce_weight * ce_norm
            else:
                score = rrf_norm
            if cfg.prefer_indications and c.get("section") == "Показання":
                score += cfg.indications_boost
            c["score"] = float(score)

        return sorted(candidates, key=lambda x: -x["score"])

    def _apply_safety_filter(self, candidates: List[Dict], user_profile: UserProfile) -> List[Dict]:
        """Apply medical safety filtering; safest first (lower risk_score), then higher model score."""
        if not self.safety_filter:
            return candidates
        for c in candidates:
            contraindications = c.get("contraindications", "") or ""
            report = self.safety_filter.assess_drug_safety(contraindications, user_profile)  # type: ignore
            c["safety_report"] = report
            c["risk_level"] = report.get("risk_level", "LOW")
            c["safe_to_use"] = report.get("safe_to_use", True)
            c["risk_score"] = report.get("risk_score", 0.0)
        return sorted(candidates, key=lambda x: (x.get("risk_score", 0.0), -x.get("score", 0.0)))

    def _group_and_format_results(self, candidates: List[Dict]) -> List[Dict]:
        """Group passages by drug and roll-up best score and safety signals."""
        by_drug: Dict[str, Dict] = {}
        for c in candidates:
            name = c["name"]
            g = by_drug.get(name)
            if not g:
                g = {
                    "drug_name": name,
                    "doc_id": c["doc_id"],
                    "best_score": float(c.get("score", c.get("rrf", 0.0))),
                    "passages": [],
                    "indications": c.get("indications", ""),
                    "contraindications": c.get("contraindications", ""),
                    "safety_report": c.get("safety_report", {}),
                    "risk_level": c.get("risk_level", "UNKNOWN"),
                    "safe_to_use": c.get("safe_to_use", True),
                    "risk_score": c.get("risk_score", 0.0),
                }
                by_drug[name] = g
            else:
                g["best_score"] = max(g["best_score"], float(c.get("score", c.get("rrf", 0.0))))
                if "risk_score" in c and c["risk_score"] > g.get("risk_score", 0.0):
                    g["risk_score"] = c["risk_score"]
                    g["safety_report"] = c.get("safety_report", g.get("safety_report", {}))
                    g["risk_level"] = c.get("risk_level", g.get("risk_level", "UNKNOWN"))
                    g["safe_to_use"] = c.get("safe_to_use", g.get("safe_to_use", True))
            g["passages"].append({
                "text": c["text"],
                "section": c.get("section", "Текст"),
                "score": float(c.get("score", c.get("rrf", 0.0))),
            })

        groups = list(by_drug.values())
        # final sorting: safest first by risk_score, then by best_score desc
        groups.sort(key=lambda x: (x.get("risk_score", 0.0), -x.get("best_score", 0.0)))
        return groups

# Backward-compatible alias
AdvancedMedicalSearchEngine = EnhancedMedicalAssistant
__all__ = ["EnhancedMedicalAssistant", "AdvancedMedicalSearchEngine", "SearchConfig", "UserProfile"]

# ---- smoke test ----
if __name__ == "__main__":
    print("=== ENHANCED MEDICAL ASSISTANT ===")
    print("Medical chunking + RRF(k) + (opt.) FAISS + (opt.) CE + (opt.) safety ready.")
