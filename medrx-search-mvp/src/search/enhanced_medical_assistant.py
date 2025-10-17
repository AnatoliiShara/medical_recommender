# -*- coding: utf-8 -*-
"""
EnhancedMedicalAssistant
------------------------
Гібридний пошук для медичного датасету:
BM25 + Dense (SentenceTransformer/FAISS-in-memory) -> Weighted-RRF -> (optional) CrossEncoder rerank
з можливістю gate за секціями (require | prefer).

Вимоги:
    pip install rank-bm25 sentence-transformers numpy pandas pyyaml tqdm

Примітки:
- Модуль не читає зовнішні індекси з диска: він будує все "в пам'яті" з переданого DataFrame.
- Для продакшн-шляху з FAISS на диску використовуйте assistant_from_parquet.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, Literal

import re
import math
import numpy as np

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # pandas не обов'язковий для імпорту модуля

from rank_bm25 import BM25Okapi  # type: ignore

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder  # type: ignore
except Exception as e:  # pragma: no cover
    SentenceTransformer = None  # type: ignore
    CrossEncoder = None  # type: ignore

# ---------- Weighted-RRF ------------------------------------------------------


def _weighted_rrf(
    ranks_bm25: Dict[int, int],
    ranks_dense: Dict[int, int],
    alpha: float = 60.0,
    w_bm25: float = 1.0,
    w_dense: float = 1.0,
) -> List[Tuple[int, float]]:
    """
    Weighted Reciprocal Rank Fusion
        score = w_bm25/(alpha + rank_bm25) + w_dense/(alpha + rank_dense)

    ranks_*: словники {pid -> rank}, rank починається з 1.
    """
    INF = 10**12
    doc_ids = set(ranks_bm25) | set(ranks_dense)
    out: Dict[int, float] = {}
    for d in doc_ids:
        rb = ranks_bm25.get(d, INF)
        rd = ranks_dense.get(d, INF)
        sb = 0.0 if rb == INF else (w_bm25 / (alpha + float(rb)))
        sd = 0.0 if rd == INF else (w_dense / (alpha + float(rd)))
        out[d] = sb + sd
    return sorted(out.items(), key=lambda x: x[1], reverse=True)


# ---------- Конфіг ------------------------------------------------------------


@dataclass
class SearchConfig:
    # Параметри відбору
    top_k: int = 20
    rrf_alpha: float = 60.0
    w_bm25: float = 1.0
    w_dense: float = 1.0

    # Гейтинг за секціями
    gate_mode: Literal["none", "prefer", "require"] = "none"
    gate_sections: List[str] = field(default_factory=list)
    prefer_boost: float = 1.10  # множник для prefer

    # Cross-Encoder (опціонально)
    ce_model: Optional[str] = None      # якщо задано — lazy load
    ce_top: int = 0                     # скільки топів після ф’южну переоцінювати (0 = вимк.)
    ce_weight: float = 0.70             # вага CE у final score (0..1)

    # Побудова індексів
    max_chunk_tokens: int = 128         # для слотів build_from_dataframe
    indications_boost: float = 0.0      # легкий буст для "Показання" (необов'язково)


# ---------- Утиліти -----------------------------------------------------------

_WS_RE = re.compile(r"\s+", flags=re.U)
_PUNCT_RE = re.compile(r"[^\w\u0400-\u04FF]+", flags=re.U)  # лат + кирилиця


def _normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = _WS_RE.sub(" ", s)
    return s


def _tokenize_ua(s: str) -> List[str]:
    """Дуже простий токенайзер під UA/латиницю (без важких залежностей)."""
    s = _normalize_text(s)
    s = _PUNCT_RE.sub(" ", s)
    toks = [t for t in s.split(" ") if t]
    return toks


def _split_by_tokens(text: str, max_tokens: int) -> List[str]:
    toks = _tokenize_ua(text)
    if not toks:
        return []
    out: List[str] = []
    for i in range(0, len(toks), max_tokens):
        out.append(" ".join(toks[i : i + max_tokens]))
    return out


# ---------- Клас пошуку -------------------------------------------------------


class EnhancedMedicalAssistant:
    """
    Простий «все-в-одному» індекс: BM25 + dense + (optional) CE.
    Будується з DataFrame (кожний рядок = препарат). Поля секцій беруться зі стовпців.
    """

    # Типові імена секцій (українською)
    DEFAULT_SECTIONS = [
        "Показання",
        "Протипоказання",
        "Побічні реакції",
        "Спосіб застосування та дози",
        "Взаємодія з іншими лікарськими засобами та інші види взаємодій",
        "Фармакологічні властивості",
        "Склад",
    ]

    NAME_COL = "Назва препарату"

    def __init__(self) -> None:
        # Дані
        self.passages: List[str] = []
        self.meta: List[Dict[str, Any]] = []

        # BM25
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_tokens: Optional[List[List[str]]] = None

        # Dense
        self._encoder_name: Optional[str] = None
        self._encoder: Optional[SentenceTransformer] = None  # type: ignore
        self._emb: Optional[np.ndarray] = None  # (N, D), l2-normalized

        # CE
        self._ce_name: Optional[str] = None
        self._ce: Optional[CrossEncoder] = None  # type: ignore

    # ------------------------------------------------------------------ build

    def build_from_dataframe(
        self,
        df: "pd.DataFrame",
        encoder_model: str = "intfloat/multilingual-e5-base",
        medical_chunking: bool = True,
        max_chunk_tokens: int = 128,
        sections: Optional[List[str]] = None,
    ) -> "EnhancedMedicalAssistant":
        """
        Створює корпус пасажів із колонок секцій та будує індекси BM25 + dense.
        """
        assert pd is not None, "pandas is required to build from dataframe"
        assert SentenceTransformer is not None, "sentence-transformers is required"

        secs = sections or list(self.DEFAULT_SECTIONS)
        name_col = self.NAME_COL if self.NAME_COL in df.columns else df.columns[0]

        passages: List[str] = []
        meta: List[Dict[str, Any]] = []

        for _, row in df.iterrows():
            name = str(row.get(name_col, "")).strip()
            for sec in secs:
                txt = str(row.get(sec, "") or "").strip()
                if not txt:
                    continue
                if medical_chunking:
                    chunks = _split_by_tokens(txt, max_chunk_tokens)
                    for ch in chunks:
                        passages.append(ch)
                        meta.append(
                            {
                                "name": name,
                                "section": sec,
                                "indications": str(row.get("Показання", "") or ""),
                                "contraindications": str(row.get("Протипоказання", "") or ""),
                            }
                        )
                else:
                    passages.append(txt)
                    meta.append(
                        {
                            "name": name,
                            "section": sec,
                            "indications": str(row.get("Показання", "") or ""),
                            "contraindications": str(row.get("Протипоказання", "") or ""),
                        }
                    )

        self.passages = passages
        self.meta = meta

        print(f"[INFO] Built passages: {len(self.passages):,}")

        # --- BM25
        tok_corpus = [_tokenize_ua(p) for p in self.passages]
        self._bm25_tokens = tok_corpus
        self._bm25 = BM25Okapi(tok_corpus)
        print(f"[INFO] BM25 ready: {len(tok_corpus):,} chunks")

        # --- Dense
        self._encoder_name = encoder_model
        self._encoder = SentenceTransformer(encoder_model)
        emb = self._encoder.encode(self.passages, batch_size=128, show_progress_bar=True, normalize_embeddings=True)
        self._emb = emb.astype(np.float32)
        print(f"[INFO] Encoder loaded: {encoder_model} | dim={self._emb.shape[1]}")

        return self

    # ------------------------------------------------------------------ CE

    def _ensure_ce(self, model_name: str) -> None:
        if CrossEncoder is None:
            raise RuntimeError("CrossEncoder is not available. Install sentence-transformers.")
        if self._ce is None or self._ce_name != model_name:
            print(f"[INFO] Loading CrossEncoder: {model_name}")
            self._ce = CrossEncoder(model_name)
            self._ce_name = model_name

    # ------------------------------------------------------------------ search

    def _bm25_topn(self, query: str, n: int) -> List[Tuple[int, float]]:
        assert self._bm25 is not None and self._bm25_tokens is not None
        q_tokens = _tokenize_ua(query)
        scores = self._bm25.get_scores(q_tokens)
        # top-n
        idx = np.argpartition(scores, -n)[-n:]
        pairs = sorted(((int(i), float(scores[i])) for i in idx), key=lambda x: x[1], reverse=True)
        return pairs

    def _dense_topn(self, query: str, n: int) -> List[Tuple[int, float]]:
        assert self._encoder is not None and self._emb is not None
        q = self._encoder.encode([query], normalize_embeddings=True)[0].astype(np.float32)
        sims = (self._emb @ q)  # cosine (бо обидві нормалізовані)
        idx = np.argpartition(sims, -n)[-n:]
        pairs = sorted(((int(i), float(sims[i])) for i in idx), key=lambda x: x[1], reverse=True)
        return pairs

    def _apply_gate(self, items: List[Tuple[int, float]], cfg: SearchConfig) -> List[Tuple[int, float]]:
        if cfg.gate_mode == "none" or not cfg.gate_sections:
            return items
        allowed = set(cfg.gate_sections)
        out: List[Tuple[int, float]] = []
        if cfg.gate_mode == "require":
            for pid, sc in items:
                if self.meta[pid].get("section") in allowed:
                    out.append((pid, sc))
            return out
        # prefer
        for pid, sc in items:
            if self.meta[pid].get("section") in allowed:
                sc *= float(cfg.prefer_boost)
            out.append((pid, sc))
        return out

    def _minmax_norm(self, arr: np.ndarray) -> np.ndarray:
        if arr.size == 0:
            return arr
        mn, mx = float(np.min(arr)), float(np.max(arr))
        if not math.isfinite(mn) or not math.isfinite(mx) or mx <= mn:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    def search(self, query: str, cfg: Optional[SearchConfig] = None) -> List[Dict[str, Any]]:
        """
        Повертає список "груп" (по назві препарату) з best_score і деталями хітів.
        """
        cfg = cfg or SearchConfig()

        if not self.passages:
            return []

        # Кандидати від кожного ретрівера
        n_bm25 = max(cfg.top_k * 10, 200)
        n_dense = max(cfg.top_k * 10, 200)

        bm25_pairs = self._bm25_topn(query, min(n_bm25, len(self.passages)))
        bm25_pairs = self._apply_gate(bm25_pairs, cfg)

        dense_pairs = self._dense_topn(query, min(n_dense, len(self.passages)))
        dense_pairs = self._apply_gate(dense_pairs, cfg)

        if not bm25_pairs and not dense_pairs:
            return []

        # Ранги для RRF
        r_bm = {pid: r + 1 for r, (pid, _) in enumerate(sorted(bm25_pairs, key=lambda x: x[1], reverse=True))}
        r_de = {pid: r + 1 for r, (pid, _) in enumerate(sorted(dense_pairs, key=lambda x: x[1], reverse=True))}

        fused = _weighted_rrf(
            r_bm, r_de, alpha=float(cfg.rrf_alpha), w_bm25=float(cfg.w_bm25), w_dense=float(cfg.w_dense)
        )

        # ТОП перед CE
        fused = fused[: max(cfg.top_k * 10, 300)]

        # CE rerank (опційно)
        ce_scores: Dict[int, float] = {}
        if cfg.ce_model and cfg.ce_top > 0:
            self._ensure_ce(cfg.ce_model)
            ce_top = min(cfg.ce_top, len(fused))
            cand_ids = [pid for pid, _ in fused[:ce_top]]
            pairs = [(query, self.passages[pid]) for pid in cand_ids]
            scores = np.asarray(self._ce.predict(pairs), dtype=np.float32)  # type: ignore
            # Нормалізуємо і комбінуємо з RRF (також нормалізуємо)
            fused_scores = np.asarray([s for _, s in fused[:ce_top]], dtype=np.float32)
            ce_n = self._minmax_norm(scores)
            fr_n = self._minmax_norm(fused_scores)
            final = (1.0 - float(cfg.ce_weight)) * fr_n + float(cfg.ce_weight) * ce_n
            for pid, sc in zip(cand_ids, final.tolist()):
                ce_scores[pid] = float(sc)

        # Формуємо фінальний список пасажів з метаданими
        results: List[Dict[str, Any]] = []
        for pid, rrf_sc in fused:
            meta = self.meta[pid]
            sec = meta.get("section", "")
            sc = float(ce_scores.get(pid, rrf_sc))  # якщо CE переоцінював — беремо комбінований
            if cfg.indications_boost and sec == "Показання":
                sc *= (1.0 + float(cfg.indications_boost))
            results.append(
                {
                    "pid": pid,
                    "score": sc,
                    "name": meta.get("name", ""),
                    "text": self.passages[pid],
                    "section": sec,
                    "rrf": float(rrf_sc),
                }
            )

        # Агрегуємо по препарату (name)
        groups: Dict[str, Dict[str, Any]] = {}
        for item in results:
            nm = item["name"]
            if nm not in groups:
                groups[nm] = {
                    "name": nm,
                    "best_score": item["score"],
                    "best_section": item["section"],
                    "items": [item],
                }
            else:
                groups[nm]["items"].append(item)
                if item["score"] > groups[nm]["best_score"]:
                    groups[nm]["best_score"] = item["score"]
                    groups[nm]["best_section"] = item["section"]

        ranked_groups = sorted(groups.values(), key=lambda g: g["best_score"], reverse=True)
        return ranked_groups[: cfg.top_k]
