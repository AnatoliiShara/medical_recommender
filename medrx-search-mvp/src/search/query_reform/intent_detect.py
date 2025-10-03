# -*- coding: utf-8 -*-
from __future__ import annotations

from search.query_reform.normalizer import Normalizer

_norm = Normalizer()

# Ключові підказки (дуже легкий rule-based). Порядок важливий.
_INTENT_RULES = [
    ("dosage", [
        "доза", "скільки приймати", "як приймати", "інструкц", "дозуван",
        "скільки разів", "перед їжею", "після їжі", "курс", "скільки днів",
    ]),
    ("contraindication", [
        "протипоказ", "не можна", "вагітн", "лактац", "дітям не",
        "печінк", "нирк", "протипоказання",
    ]),
    ("interactions", [
        "взаємод", "разом з", "поєднувати", "алкогол", "запивати",
    ]),
    ("side_effects", [
        "побіч", "побочка", "небажан", "реакц", "безпека", "ризик",
    ]),
    # default: indication
]

def detect_intent(query_norm: str) -> str:
    q = _norm.normalize(query_norm)
    for label, keys in _INTENT_RULES:
        if any(k in q for k in keys):
            return label
    return "indication"
