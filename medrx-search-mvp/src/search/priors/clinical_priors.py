# src/search/priors/clinical_priors.py
from __future__ import annotations
import json, re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

_STEM_FLAGS = re.IGNORECASE | re.UNICODE

def _looks_like_regex(s: str) -> bool:
    return bool(re.search(r'[().|\\\[\]\?\+\*\{\}]', s))

def _make_pattern(tok: str) -> re.Pattern:
    tok = tok.strip()
    if _looks_like_regex(tok):
        return re.compile(tok, _STEM_FLAGS)
    # якщо просте слово/стем → матчим від слова і далі будь-які закінчення
    esc = re.escape(tok)
    return re.compile(rf'\b{esc}\w*', _STEM_FLAGS)

@dataclass
class PriorEntry:
    id: str
    intent: str
    boost: float
    inn: List[str]
    brands_opt: List[str]
    sections_prefer: List[str]
    triggers_raw: List[str]
    neg_raw: List[str]
    triggers: List[re.Pattern]
    neg_triggers: List[re.Pattern]

class ClinicalPriors:
    def __init__(self, jsonl_path: str, allow_intents: Tuple[str, ...] = ("indication", "unknown")):
        self.allow_intents = set(allow_intents)
        self.entries: List[PriorEntry] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                trig = item.get("triggers", []) or []
                neg = item.get("neg_triggers", []) or []
                entry = PriorEntry(
                    id=item.get("id", ""),
                    intent=item.get("intent", "unknown"),
                    boost=float(item.get("boost", 0.03)),
                    inn=[t.strip() for t in item.get("inn", []) if t.strip()],
                    brands_opt=[t.strip() for t in item.get("brands_opt", []) if t.strip()],
                    sections_prefer=item.get("sections_prefer", []) or [],
                    triggers_raw=trig,
                    neg_raw=neg,
                    triggers=[_make_pattern(t) for t in trig],
                    neg_triggers=[_make_pattern(t) for t in neg],
                )
                self.entries.append(entry)

    def match(self, normalized_query: str, intent: str) -> Optional[Dict[str, Any]]:
        # гейт по intent
        if intent not in self.allow_intents:
            return None
        q = normalized_query or ""
        for e in self.entries:
            # перестраховка: якщо entry має інший intent, але ми дозволяємо unknown — все одно матчимо,
            # або матчимо по суворому збігу intent, якщо він у дозволених
            if e.intent not in self.allow_intents and intent != "unknown":
                continue
            if any(p.search(q) for p in e.neg_triggers):
                continue
            m = None
            for p in e.triggers:
                m = p.search(q)
                if m:
                    break
            if not m:
                continue
            terms = list(dict.fromkeys([*e.inn, *e.brands_opt]))  # INN перші, бренди — далі
            return {
                "matched_prior": True,
                "matched_id": e.id,
                "matched_trigger": m.group(0),
                "terms": terms,
                "boost": e.boost,
                "sections_prefer": e.sections_prefer,
            }
        return None
