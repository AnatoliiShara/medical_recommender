# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Set, Iterable, Optional, Tuple

from .normalizer import (
    compile_phrase_list,
    normalize_text,
    normalize_for_match,
    phrase_to_regex,
)

# ---------------------------------------------------------------------
# Структури даних
# ---------------------------------------------------------------------

@dataclass
class CEPatterns:
    # для raw-пошуку
    positive: List[re.Pattern]
    penalty: List[re.Pattern]
    # для normalized-пошуку
    positive_norm: Set[str]
    penalty_norm: Set[str]


@dataclass
class ClinicalLexicon:
    # тригери по умовам (нормалізовані фрази)
    triggers_by_cond: Dict[str, Set[str]]
    # позитиви/пенальті по умовам (raw та нормалізовані)
    positive_raw_by_cond: Dict[str, List[str]]
    penalty_raw_by_cond: Dict[str, List[str]]
    positive_norm_by_cond: Dict[str, Set[str]]
    penalty_norm_by_cond: Dict[str, Set[str]]


# ---------------------------------------------------------------------
# Завантаження словників із data/dicts/clinical/<cond>/
# Ім'я файлів: <cond>_trigger.txt, <cond>_positive.txt, <cond>_penalty.txt
# ---------------------------------------------------------------------

def _clean_line_basic(s: str) -> str:
    s = s.strip().strip('"').strip("'")
    return s

def _read_lines(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]

def _load_condition(base_dir: str, cond: str) -> Tuple[Set[str], List[str], List[str], Set[str], Set[str]]:
    """
    Повертає:
      triggers_norm, positive_raw, penalty_raw, positive_norm, penalty_norm
    """
    cdir = os.path.join(base_dir, cond)
    trig_p = os.path.join(cdir, f"{cond}_trigger.txt")
    pos_p  = os.path.join(cdir, f"{cond}_positive.txt")
    pen_p  = os.path.join(cdir, f"{cond}_penalty.txt")

    # тригери → нормалізована множина
    triggers_norm = compile_phrase_list(_read_lines(trig_p))

    # позитив/пенальті — і raw, і нормалізовані
    pos_lines = [ln for ln in _read_lines(pos_p) if ln and not ln.lstrip().startswith("#")]
    pen_lines = [ln for ln in _read_lines(pen_p) if ln and not ln.lstrip().startswith("#")]

    positive_raw = [_clean_line_basic(ln) for ln in pos_lines if _clean_line_basic(ln)]
    penalty_raw  = [_clean_line_basic(ln) for ln in pen_lines if _clean_line_basic(ln)]

    positive_norm = compile_phrase_list(positive_raw)
    penalty_norm  = compile_phrase_list(penalty_raw)

    return triggers_norm, positive_raw, penalty_raw, positive_norm, penalty_norm

def _list_conditions(base_dir: str) -> List[str]:
    if not os.path.isdir(base_dir):
        return []
    out = []
    for name in sorted(os.listdir(base_dir)):
        cdir = os.path.join(base_dir, name)
        if os.path.isdir(cdir):
            stem = os.path.join(cdir, f"{name}_")
            if any(os.path.exists(stem + sufx + ".txt") for sufx in ("trigger", "positive", "penalty")):
                out.append(name)
    return out

def load_clinical_lexicon(base_dir: str) -> ClinicalLexicon:
    triggers_by_cond: Dict[str, Set[str]] = {}
    positive_raw_by_cond: Dict[str, List[str]] = {}
    penalty_raw_by_cond: Dict[str, List[str]] = {}
    positive_norm_by_cond: Dict[str, Set[str]] = {}
    penalty_norm_by_cond: Dict[str, Set[str]] = {}

    for cond in _list_conditions(base_dir):
        trigs, pos_raw, pen_raw, pos_norm, pen_norm = _load_condition(base_dir, cond)
        if trigs:
            triggers_by_cond[cond] = trigs
        if pos_raw:
            positive_raw_by_cond[cond] = pos_raw
            positive_norm_by_cond[cond] = pos_norm
        if pen_raw:
            penalty_raw_by_cond[cond] = pen_raw
            penalty_norm_by_cond[cond] = pen_norm

    return ClinicalLexicon(
        triggers_by_cond=triggers_by_cond,
        positive_raw_by_cond=positive_raw_by_cond,
        penalty_raw_by_cond=penalty_raw_by_cond,
        positive_norm_by_cond=positive_norm_by_cond,
        penalty_norm_by_cond=penalty_norm_by_cond,
    )

# ---------------------------------------------------------------------
# Детекція умов у запиті
# ---------------------------------------------------------------------

def detect_conditions(query_text: str, assets_or_lex, n_min: int = 1, n_max: int = 5) -> List[str]:
    """
    Повертає список умов (cond), які тригеряться запитом.
    Працює з assets (dict) або ClinicalLexicon.
    """
    if isinstance(assets_or_lex, dict):
        lex = assets_or_lex.get("lex")
    else:
        lex = assets_or_lex
    if lex is None:
        return []

    qkeys = set(normalize_for_match(query_text, n_min=n_min, n_max=n_max))
    matched: List[str] = []
    for cond, trigset in lex.triggers_by_cond.items():
        if qkeys & trigset:
            matched.append(cond)
    return matched

# ---------------------------------------------------------------------
# Побудова CE-патернів для умов
# ---------------------------------------------------------------------

def build_ce_patterns(lex: ClinicalLexicon, active_conditions: Optional[Iterable[str]] = None) -> CEPatterns:
    """
    Компілює:
      - positive/penalty → regex-патерни (RAW пошук),
      - positive_norm/penalty_norm → нормалізовані множини (швидка перевірка).
    Якщо active_conditions=None — береться повний пул.
    """
    if active_conditions is None:
        active_conditions = lex.triggers_by_cond.keys()

    pos_raw: List[str] = []
    pen_raw: List[str] = []
    pos_norm: Set[str] = set()
    pen_norm: Set[str] = set()

    for cond in active_conditions:
        pos_raw += lex.positive_raw_by_cond.get(cond, [])
        pen_raw += lex.penalty_raw_by_cond.get(cond, [])
        pos_norm |= lex.positive_norm_by_cond.get(cond, set())
        pen_norm |= lex.penalty_norm_by_cond.get(cond, set())

    # унікалізація + компіляція
    pos_patterns = [phrase_to_regex(s) for s in sorted(set(pos_raw)) if s]
    pen_patterns = [phrase_to_regex(s) for s in sorted(set(pen_raw)) if s]

    return CEPatterns(
        positive=pos_patterns,
        penalty=pen_patterns,
        positive_norm=pos_norm,
        penalty_norm=pen_norm,
    )

# ---------------------------------------------------------------------
# CE-біас документа
# ---------------------------------------------------------------------

def ce_bias_for_doc(
    doc_text: str,
    assets: dict,
    matched_conditions: Iterable[str],
    bias_pos: float = 0.15,
    bias_pen: float = 0.20,
    n_min: int = 1,
    n_max: int = 5,
) -> float:
    """
    Алгоритм:
      1) Формуємо normalized n-grams з doc_text і шукаємо перетин із positive_norm/penalty_norm для matched_conditions.
      2) Якщо 1) не спрацювало — робимо RAW-regex пошук.
      3) Повертаємо +bias_pos при позитиві, −bias_pen при пенальті (можливе одночасне застосування).
    """
    if not matched_conditions:
        return 0.0

    lex: ClinicalLexicon = assets.get("lex")
    pats_by_cond: Dict[str, CEPatterns] = assets.get("patterns_by_cond", {})

    norm_keys = set(normalize_for_match(doc_text, n_min=n_min, n_max=n_max))

    pos_hit = False
    pen_hit = False

    for cond in matched_conditions:
        pats = pats_by_cond.get(cond)
        if not pats:
            # на випадок, якщо precompile не містить cond
            pats = build_ce_patterns(lex, [cond])

        if not pos_hit and pats.positive_norm and (norm_keys & pats.positive_norm):
            pos_hit = True
        if not pen_hit and pats.penalty_norm and (norm_keys & pats.penalty_norm):
            pen_hit = True

        # fallback RAW-regex
        if not pos_hit and pats.positive:
            for rgx in pats.positive:
                if rgx.search(doc_text):
                    pos_hit = True
                    break
        if not pen_hit and pats.penalty:
            for rgx in pats.penalty:
                if rgx.search(doc_text):
                    pen_hit = True
                    break

        if pos_hit and pen_hit:
            break

    delta = 0.0
    if pos_hit:
        delta += float(bias_pos)
    if pen_hit:
        delta -= float(bias_pen)
    return delta

# ---------------------------------------------------------------------
# Побудова інвертованого індексу (NORM-логіка, швидкий fallback)
# ---------------------------------------------------------------------

def build_clinical_index(doc_texts_norm: List[str], lex: ClinicalLexicon) -> Dict[str, List[int]]:
    """
    Для кожного cond збираємо doc_id, у яких normalized(title+text) містить хоч один тригер cond.
    """
    index: Dict[str, List[int]] = {c: [] for c in lex.triggers_by_cond.keys()}
    for i, norm in enumerate(doc_texts_norm):
        keys = set(norm.split()) | set(normalize_for_match(norm, n_min=2, n_max=5))
        for cond, trigset in lex.triggers_by_cond.items():
            if keys & trigset:
                index[cond].append(i)
    return index

# ---------------------------------------------------------------------
# UNION кандидатів
# ---------------------------------------------------------------------

def union_candidates_for_conditions(
    assets: dict,
    matched_conditions: Iterable[str],
    bm25_top_ids: Optional[Set[int]] = None,
    union_cap: int = 10,
    force_include: bool = False,
) -> List[int]:
    """
    Повертає до union_cap doc_id з інвертованого індексу за умовами (уніон).
    Якщо force_include=False — пропускаємо лише ті, що вже є у BM25-top (gate).
    """
    inv_index: Dict[str, List[int]] = assets.get("inv_index", {})
    seen: Set[int] = set()
    out: List[int] = []

    def _extend(c: str):
        nonlocal out
        cands = inv_index.get(c, [])
        for did in cands:
            if did in seen:
                continue
            if (not force_include) and (bm25_top_ids is not None) and (did not in bm25_top_ids):
                continue
            seen.add(did)
            out.append(int(did))
            if len(out) >= union_cap:
                return True
        return False

    for cond in matched_conditions or []:
        if _extend(cond):
            break

    return out[:union_cap]

# ---------------------------------------------------------------------
# Побудова clinical assets
# ---------------------------------------------------------------------

def build_clinical_assets(doc_texts_raw: List[str], dict_root: str) -> dict:
    """
    Повертає словник:
      {
        "lex": ClinicalLexicon,
        "doc_texts_raw": [...],
        "doc_texts_norm": [...],
        "inv_index": {cond -> [doc_id, ...]},
        "patterns_by_cond": {cond -> CEPatterns}
      }
    """
    doc_texts_norm: List[str] = [normalize_text(t) for t in doc_texts_raw]
    lex = load_clinical_lexicon(dict_root)

    inv_index = build_clinical_index(doc_texts_norm, lex)

    # precompile patterns per condition (для швидшого ce_bias_for_doc)
    patterns_by_cond: Dict[str, CEPatterns] = {}
    for cond in lex.triggers_by_cond.keys():
        patterns_by_cond[cond] = build_ce_patterns(lex, [cond])

    return {
        "lex": lex,
        "doc_texts_raw": doc_texts_raw,
        "doc_texts_norm": doc_texts_norm,
        "inv_index": inv_index,
        "patterns_by_cond": patterns_by_cond,
    }
