# -*- coding: utf-8 -*-
"""
Clinical priors: lightweight symptom → (INN/brands) hints.

- Формат джерела: JSONL, кожний рядок — об'єкт з полями:
  id, lang, intent, triggers[], neg_triggers[], synonyms[],
  inn[], brands_opt[], sections_prefer[], boost (float), notes (opt)

- Основна ідея:
  1) Виявити, чи запит «схожий» на один з частих consumer-style кейсів.
  2) Якщо так — повернути набір термінів для розширення запиту (INN/бренди),
     бажані секції та маленький буст.
  3) Інтегрується у пайплайн перед BM25/FAISS (додаємо терміни до тексту запиту)
     та у фінальному скорингу (малий add-on до score).

Безпека:
- Це лише «хинт». Буст тримаємо малим (0.02..0.05) і застосовуємо обережно.
- Якщо виявлено neg_triggers (вагітн/дитин/кров/висока температура) — не матчимо.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Tuple, Set
from pathlib import Path
import json
import re

# локальний нормалізатор
from search.query_reform.normalizer import Normalizer


@dataclass
class PriorEntry:
    id: str
    lang: str = "uk"
    intent: Optional[str] = None
    triggers: List[str] = None
    neg_triggers: List[str] = None
    synonyms: List[str] = None
    inn: List[str] = None
    brands_opt: List[str] = None
    sections_prefer: List[str] = None
    boost: float = 0.03
    notes: str = ""

    def all_match_phrases(self) -> List[str]:
        out = []
        if self.triggers: out.extend(self.triggers)
        if self.synonyms: out.extend(self.synonyms)
        # унікалізація збереженням порядку
        seen = set()
        uniq = []
        for t in out:
            if t not in seen:
                uniq.append(t)
                seen.add(t)
        return uniq


@dataclass
class PriorMatch:
    prior_id: str
    intent: Optional[str]
    matched_phrases: List[str]
    expansions: List[str]        # INN + brands_opt
    sections_prefer: List[str]
    boost: float


class ClinicalPriors:
    def __init__(self, jsonl_path: str, lang: str = "uk"):
        self.path = str(jsonl_path)
        self.lang = lang
        self.entries: List[PriorEntry] = []
        self.norm = Normalizer()
        self._load()

    def _load(self):
        p = Path(self.path)
        if not p.exists():
            raise FileNotFoundError(f"[ClinicalPriors] Not found: {self.path}")
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                obj = json.loads(line)
                entry = PriorEntry(
                    id=obj.get("id",""),
                    lang=obj.get("lang","uk"),
                    intent=obj.get("intent"),
                    triggers=self._norm_list(obj.get("triggers",[])),
                    neg_triggers=self._norm_list(obj.get("neg_triggers",[])),
                    synonyms=self._norm_list(obj.get("synonyms",[])),
                    inn=self._norm_list(obj.get("inn",[])),
                    brands_opt=self._norm_list(obj.get("brands_opt",[])),
                    sections_prefer=list(obj.get("sections_prefer",[]) or []),
                    boost=float(obj.get("boost", 0.03)),
                    notes=obj.get("notes","")
                )
                # фільтруємо мовою, якщо задано
                if entry.lang and entry.lang != self.lang:
                    continue
                self.entries.append(entry)

    def _norm_list(self, xs: List[str]) -> List[str]:
        out = []
        for x in xs:
            nx = self.norm.normalize(x)
            if nx:
                out.append(nx)
        # унікалізація
        seen = set()
        uniq = []
        for t in out:
            if t not in seen:
                uniq.append(t)
                seen.add(t)
        return uniq

    def match(self, query_norm: str, intent: Optional[str]) -> Optional[PriorMatch]:
        """
        Простий матчинг: якщо query містить будь-який тригер/синонім (словосполучення),
        і не містить жодного з neg_triggers → повертаємо PriorMatch.
        Якщо в entry задано intent і він не збігається з intent запиту — пропускаємо.
        """
        q = query_norm
        for e in self.entries:
            if e.intent and intent and e.intent != intent:
                continue
            if self._has_neg_trigger(q, e.neg_triggers):
                continue
            matched = self._has_any_phrase(q, e.all_match_phrases())
            if matched:
                expansions = self._gather_expansions(e)
                return PriorMatch(
                    prior_id=e.id,
                    intent=e.intent,
                    matched_phrases=matched,
                    expansions=expansions,
                    sections_prefer=e.sections_prefer or [],
                    boost=float(e.boost)
                )
        return None

    def _has_neg_trigger(self, q: str, negs: Optional[List[str]]) -> bool:
        if not negs: return False
        return any(x in q for x in negs)

    def _has_any_phrase(self, q: str, phrases: List[str]) -> List[str]:
        hits = []
        for p in phrases or []:
            if p and p in q:
                hits.append(p)
        return hits

    def _gather_expansions(self, e: PriorEntry) -> List[str]:
        xs = []
        if e.inn: xs.extend(e.inn)
        if e.brands_opt: xs.extend(e.brands_opt)
        # унікалізація
        seen = set()
        uniq = []
        for t in xs:
            if t not in seen:
                uniq.append(t)
                seen.add(t)
        return uniq
