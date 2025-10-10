# -*- coding: utf-8 -*-
from __future__ import annotations
import re
import unicodedata
from typing import Dict

_TRADE = r"[®™©]"
_QUOTES = r"[’‘´`“”«»]"
_PUNCT  = r"[(){}\[\];,]"

def nfkc_lower(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = s.replace("\u00A0", " ")
    s = s.lower()
    s = re.sub(_TRADE, " ", s)
    s = re.sub(_QUOTES, " ", s)
    s = re.sub(_PUNCT,  " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_numbers(s: str) -> str:
    # 1,5 -> 1.5 ;  1 . 5 -> 1.5
    s = re.sub(r"(\d)\s*[,]\s*(\d)", r"\1.\2", s)
    s = re.sub(r"(\d)\s*[.]\s*(\d)", r"\1.\2", s)
    return s

def normalize_units(s: str, unit_map: Dict[str, str]) -> str:
    # заміни токенів із межами слів
    if not unit_map:
        return s
    # сортуємо за спаданням довжини, щоб «мг.» зловити до «мг»
    keys = sorted(unit_map.keys(), key=len, reverse=True)
    for k in keys:
        v = unit_map[k]
        # межі слова або декорації на кшталт крапок
        s = re.sub(rf"(?<!\w){re.escape(k)}(?!\w)", v, s)
    return s

def basic_norm(s: str, unit_map: Dict[str, str]) -> str:
    s = nfkc_lower(s)
    s = normalize_numbers(s)
    s = normalize_units(s, unit_map)
    s = re.sub(r"\s+", " ", s).strip()
    return s
