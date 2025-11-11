# src/search/clinical/detector.py
# -*- coding: utf-8 -*-
"""
Лематизований детектор клінічних станів + нормалізація позитивів/пенальті.

Інтерфейс:
- load_clinical_dicts(dict_root) -> CompiledClinicalDicts
- detect_conditions(query, compiled) -> list[str]
- find_doc_bias_keys(doc_norm_text, compiled, matched_conditions) -> (pos_hits, pen_hits)

'compiled' містить як сирі (human-readable) рядки, так і нормалізовані ключі.
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Set, Tuple

from .normalizer import (
    normalize_phrase,
    normalize_for_match,
)

@dataclass
class ConditionDict:
    name: str
    triggers_raw: List[str]
    positives_raw: List[str]
    penalties_raw: List[str]

    triggers_norm: Set[str]
    positives_norm: Set[str]
    penalties_norm: Set[str]

@dataclass
class CompiledClinicalDicts:
    by_condition: Dict[str, ConditionDict]
    # Глобальний penalty (не прив’язаний до стану)
    global_penalty_raw: Set[str]
    global_penalty_norm: Set[str]

def _read_lines(fp: str) -> List[str]:
    if not os.path.exists(fp):
        return []
    with open(fp, "r", encoding="utf-8") as f:
        lines = []
        for ln in f:
            s = ln.strip()
            if not s or s.startswith("#"):
                continue
            lines.append(s)
        return lines

def _normalize_list(lines: List[str]) -> Set[str]:
    out = set()
    for s in lines:
        norm = normalize_phrase(s)
        if norm:
            out.add(norm)
    return out

def _load_condition_dir(cond_dir: str, cond_name: str) -> ConditionDict:
    trig = _read_lines(os.path.join(cond_dir, f"{cond_name}_trigger.txt"))
    pos  = _read_lines(os.path.join(cond_dir, f"{cond_name}_positive.txt"))
    pen  = _read_lines(os.path.join(cond_dir, f"{cond_name}_penalty.txt"))
    return ConditionDict(
        name=cond_name,
        triggers_raw=trig,
        positives_raw=pos,
        penalties_raw=pen,
        triggers_norm=_normalize_list(trig),
        positives_norm=_normalize_list(pos),
        penalties_norm=_normalize_list(pen),
    )

def _iter_condition_names(dict_root: str) -> List[str]:
    names = []
    for entry in os.listdir(dict_root):
        p = os.path.join(dict_root, entry)
        if not os.path.isdir(p):
            continue
        # усередині мають бути три файли з однаковим префіксом
        if (os.path.exists(os.path.join(p, f"{entry}_trigger.txt")) and
            os.path.exists(os.path.join(p, f"{entry}_positive.txt")) and
            os.path.exists(os.path.join(p, f"{entry}_penalty.txt"))):
            names.append(entry)
    names.sort()
    return names

@lru_cache(maxsize=4)
def load_clinical_dicts(dict_root: str) -> CompiledClinicalDicts:
    conds: Dict[str, ConditionDict] = {}
    for name in _iter_condition_names(dict_root):
        conds[name] = _load_condition_dir(os.path.join(dict_root, name), name)

    # Глобальний penalty
    gp_raw = set(_read_lines(os.path.join(dict_root, "penalty_general.txt")))
    gp_norm = _normalize_list(list(gp_raw))
    return CompiledClinicalDicts(
        by_condition=conds,
        global_penalty_raw=gp_raw,
        global_penalty_norm=gp_norm
    )

def detect_conditions(query: str, compiled: CompiledClinicalDicts) -> List[str]:
    """
    Лематизований матч по n-грамам (1..5) запиту проти нормалізованих тригерів.
    """
    qkeys = set(normalize_for_match(query, n_min=1, n_max=5))
    matched = []
    for cname, cd in compiled.by_condition.items():
        if cd.triggers_norm & qkeys:
            matched.append(cname)
    return matched

def find_doc_bias_keys(
    doc_norm_text: str,
    compiled: CompiledClinicalDicts,
    matched_conditions: List[str]
) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Повертає три множини нормалізованих збігів:
    - global_pen_hits
    - pos_hits (по matched_conditions)
    - pen_hits (по matched_conditions)
    """
    # нормалізуємо весь документ у n-грамні ключі для легкого пошуку фраз
    dkeys = set(doc_norm_text.split())  # оптимізація: очікуємо, що doc_norm_text = "..." (уже нормований)
    # якщо doc_norm_text зберігається як нор-фрази через пробіл — працюємо по токенах n=1
    # для фразових позитивів/пенальті спробуємо прямий include перевіркою рядків нижче

    # Глобальний penalty: перевірка підрядка на нормалізовані фрази
    global_pen_hits = set()
    for gph in compiled.global_penalty_norm:
        if gph and gph in doc_norm_text:
            global_pen_hits.add(gph)

    pos_hits, pen_hits = set(), set()
    for cname in matched_conditions:
        cd = compiled.by_condition.get(cname)
        if not cd:
            continue
        for ph in cd.positives_norm:
            if ph and ph in doc_norm_text:
                pos_hits.add(ph)
        for ph in cd.penalties_norm:
            if ph and ph in doc_norm_text:
                pen_hits.add(ph)
    return global_pen_hits, pos_hits, pen_hits
