# -*- coding: utf-8 -*-
"""
UA/RU текстова нормалізація + легка лематизація для детектора клінічних станів.
Мета: стабільні ключі порівняння для тригерів/позитивів/пенальті та запитів.

Експортує (back-compat + нове):
- tokenize(text) -> list[str]
- tokens(text) -> list[str]                     # АЛІАС для сумісності з існуючими імпортами
- lemma_key(word) -> str
- normalize_tokens(tokens) -> list[str]
- normalize_text(text) -> str
- normalize_phrase(text) -> str
- normalized_ngrams(tokens, n_min=1, n_max=5) -> list[str]
- normalize_for_match(text, n_min=1, n_max=5) -> list[str]
- compile_phrase_list(lines) -> set[str]
- any_phrase_in(text, compiled_phrases, n_min=1, n_max=5) -> set[str]
- phrase_to_regex(phrase: str) -> re.Pattern   # НОВЕ: очікується clinical/index.py

Примітка:
- Це не повна морфологія; це безпечний "стемер" для помірного recall.
- Лемматизацію/нормалізацію застосовуємо до детектора; назви ЛЗ/брендів у корпусі не лампується.
"""

from __future__ import annotations
import re
import unicodedata
from functools import lru_cache
from typing import Iterable, Set, List, Pattern

# ---------- Базова юнікод-нормалізація ----------

def _basic_unicode_norm(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = unicodedata.normalize("NFKC", s)
    # Уніфікація лапок/дефісів
    s = s.replace("’", "'").replace("`", "'").replace("“", '"').replace("”", '"')
    s = s.replace("—", "-").replace("–", "-").replace("−", "-")
    # Зайві пробіли
    s = re.sub(r"\s+", " ", s)
    return s.strip()

# ---------- Токенізація ----------

# Слова/цифри (латиниця + розширена латиниця + кирилиця) + цифри
_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿĀ-žΑ-ωЁёІіЇїЄєҐґ\u0400-\u04FF0-9]+", re.UNICODE)

def tokenize(s: str) -> List[str]:
    s = _basic_unicode_norm(s).lower()
    return _TOKEN_RE.findall(s)

# АЛІАС для сумісності з існуючими імпортами в інших модулях
def tokens(s: str) -> List[str]:
    return tokenize(s)

# ---------- Дуже легкий лематизатор/стемер ----------

# Увага: включені суфікси типу "-єю", "-ією" для інструментального відмінка.
_UA_RU_SUFFIXES = tuple(sorted([
    # укр (sg/pl, прикметники, відмінки)
    "ями","ами","ові","еві","ому","ему","ими","іми","ого","его","ях","ах","ям","ам",
    "ою","ею","єю","ією",     # інстр. відмінок: терапією/інфекцією/діареєю
    "ій","ий","им","ім","их","іх","ої",
    "а","я","у","ю","е","є","і","ї","о","и",
    # рос варіанти
    "ой","ей","ый","ий","ое","ее","ая","яя","ом","ем","ев","ов",
], key=len, reverse=True))

def _cheap_lemma_word(w: str) -> str:
    if len(w) <= 3:
        return w
    base = w
    for suf in _UA_RU_SUFFIXES:
        if base.endswith(suf) and len(base) - len(suf) >= 4:
            base = base[: -len(suf)]
            break
    return base

@lru_cache(maxsize=10000)
def lemma_key(w: str) -> str:
    return _cheap_lemma_word(w.lower())

def normalize_tokens(tokens_iter: Iterable[str]) -> List[str]:
    return [lemma_key(t) for t in tokens_iter if t]

def normalize_text(s: str) -> str:
    """Повна нормалізація рядка -> токени у "базових" формах, з'єднані пробілом."""
    toks = tokenize(s)
    toks = normalize_tokens(toks)
    return " ".join(toks)

def normalize_phrase(s: str) -> str:
    """Нормалізація фрази (для фраз-словників)."""
    toks = tokenize(s)
    toks = normalize_tokens(toks)
    return " ".join(toks)

# ---------- N-грамні ключі ----------

def normalized_ngrams(tokens_list: List[str], n_min: int = 1, n_max: int = 5) -> List[str]:
    out: List[str] = []
    n_max = max(n_max, n_min)
    L = len(tokens_list)
    for n in range(n_min, n_max + 1):
        for i in range(0, L - n + 1):
            out.append(" ".join(tokens_list[i:i+n]))
    return out

def normalize_for_match(text: str, n_min: int = 1, n_max: int = 5) -> List[str]:
    """Отримати нормалізовані n-грамні ключі для звіряння з фразами з clinical-словників."""
    toks = normalize_tokens(tokenize(text))
    return normalized_ngrams(toks, n_min=n_min, n_max=n_max)

# ---------- Утиліти для словників ----------

def _clean_line(line: str) -> str:
    s = _basic_unicode_norm(line)
    s = s.strip().strip('"').strip("'")
    return s

def compile_phrase_list(lines: Iterable[str]) -> Set[str]:
    """
    Приймає сирі рядки словника (можуть бути з коментарями/порожніми).
    Повертає МНОЖИНУ нормалізованих фраз.
    """
    phrases: Set[str] = set()
    for raw in lines:
        if not raw:
            continue
        s = _clean_line(raw)
        if not s or s.startswith("#"):
            continue
        norm = normalize_phrase(s)
        if norm:
            phrases.add(norm)
    return phrases

def any_phrase_in(text: str, compiled_phrases: Iterable[str], n_min: int = 1, n_max: int = 5) -> Set[str]:
    """
    Перевіряє наявність будь-якої з фраз у нормалізованих n-граммах тексту.
    Повертає множину збігів (порожня множина == збігів немає).
    """
    if not compiled_phrases:
        return set()
    keys = set(normalize_for_match(text, n_min=n_min, n_max=n_max))
    dict_keys = set(compiled_phrases)
    return keys.intersection(dict_keys)

# ---------- Regex-матчинг для "сирого" тексту (очікується clinical/index.py) ----------

def phrase_to_regex(phrase: str) -> Pattern[str]:
    """
    Будує толерантний до розділових/пробілів regex для пошуку фрази у НЕнормалізованому тексті.
    Логіка:
      - фраза → токени (tokenize), без леми (бо regex працює по raw-тексту);
      - між токенами дозволяємо будь-які розділові/пробільні послідовності: \W+;
      - по краях використовуємо "межі слова" (?<!\\w) ... (?!\\w) замість \\b для коректної Юнікод-поведінки.
    Приклад: "гостра діарея" → (?<!\w)гостра\W+діарея(?!\w)
    """
    toks = tokenize(phrase)
    if not toks:
        # Регекс, що ніколи не матчиться
        return re.compile(r"(?!x)x")
    core = r"\W+".join(re.escape(t) for t in toks)
    pattern = rf"(?<!\w){core}(?!\w)"
    return re.compile(pattern, flags=re.IGNORECASE | re.UNICODE)

__all__ = [
    "tokenize",
    "tokens",                # важливо для імпортів у clinical/index.py
    "lemma_key",
    "normalize_tokens",
    "normalize_text",
    "normalize_phrase",
    "normalized_ngrams",
    "normalize_for_match",
    "compile_phrase_list",
    "any_phrase_in",
    "phrase_to_regex",       # важливо для імпортів у clinical/index.py
]
