# -*- coding: utf-8 -*-
"""
Уніфікація назв препаратів + витяг МНН (INN) з корпусу.
- canon_name: зводить рядок до канонічного вигляду (регістри/символи/пропуски/лат-укр).
- extract_inn_from_row: намагається витягнути INN із рядка "Склад" і/або назви.
- Brand2INNIndex: будує мапінг name -> множина INN-токенів.
- match_logic: набір правил збігу predicted ↔ gold (canon, INN, синоніми).
"""

import re
import unicodedata
from typing import Dict, List, Set, Tuple, Optional

# --- Базові таблиці транслітерацій/синонімів (можна розширювати) ---
LAT_TO_UA = {
    "ibuprofen": "ібупрофен",
    "paracetamol": "парацетамол",
    "acetaminophen": "парацетамол",
    "omeprazole": "омепразол",
    "esomeprazole": "езомепразол",
    "pantoprazole": "пантопразол",
    "rabeprazole": "рабепразол",
    "loperamide": "лоперамід",
    "metformin": "метформін",
    "gliclazide": "гліклазид",
    "sitagliptin": "ситагліптин",
    "dapagliflozin": "дапагліфлозин",
    "losartan": "лозартан",
    "valsartan": "валсартан",
    "amlodipine": "амлодипін",
    "telmisartan": "телмісартан",
    "sildenafil": "силденафіл",
    "levothyroxine": "левотироксин",
    "methimazole": "тіамазол",
    "propylthiouracil": "пропілтіоурацил",
    "loratadine": "лоратадин",
    "cetirizine": "цетиризин",
    "diosmectite": "діосмектит",
    "nifuroxazide": "ніфуроксазид",
}

SIMPLE_SYNONYMS = {
    # укорочення/варіанти тієї ж молекули
    "ібупром": "ібупрофен",
    "нурофен": "ібупрофен",
    "панадол": "парацетамол",
    "колдфлю": "парацетамол",
    "імодіум": "лоперамід",
    "мезим": "панкреатин",
}

# з назв часто прибираємо «хвости»
DROP_TOKENS = set("""
mr xr sr forte rapid max ретард суспензія сироп краплі табл таблетки капсули капс ін'єкцій 
для інфузій гранули мазь крем гель спрей р-н розчин супозиторії шипучі forte/форте форте
""".split())

BRACKETS_RE = re.compile(r"[\(\)\[\]{}]+")
MULTISPACE_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[^\w\s\-’']")
DIGIT_RE = re.compile(r"\d+([.,]\d+)?")

def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def canon_name(s: str) -> str:
    """звести назву до канону: нижній регістр, прибрати цифри/форми/дози/дужки, уніф. апострофи, лат→укр."""
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    s = _strip_accents(s)
    s = s.replace("’", "'").replace("`", "'").replace("ʼ", "'")
    s = BRACKETS_RE.sub(" ", s)
    s = PUNCT_RE.sub(" ", s)
    s = DIGIT_RE.sub(" ", s)

    # груба лат→укр заміна по словнику
    for lat, ua in LAT_TO_UA.items():
        s = re.sub(rf"\b{re.escape(lat)}\b", ua, s)

    # прості синоніми
    for k, v in SIMPLE_SYNONYMS.items():
        s = re.sub(rf"\b{re.escape(k)}\b", v, s)

    toks = [t for t in MULTISPACE_RE.sub(" ", s).split() if t and t not in DROP_TOKENS]
    return " ".join(toks)

def _split_by_commas_and_plus(s: str) -> List[str]:
    parts = re.split(r"[;,/+\u2212\-]", s)
    out = []
    for p in parts:
        p = p.strip()
        if p:
            out.append(p)
    return out

ACTIVE_HINTS = [
    r"діюча речовина",
    r"містить",
    r"активн(а|і) речовин(а|и)",
    r"субстанц(ія|ії)",
]

def extract_inn_from_row(name: str, composition: str) -> Set[str]:
    """
    Жадібно витягуємо INN із 'Склад' і з 'Назви'.
    Повертаємо множину канонічних токенів (кілька речовин -> кілька елементів).
    """
    inns: Set[str] = set()

    def _pull(text: str):
        if not text:
            return
        # грубо ріжемо по розділювачах і чистимо
        for chunk in _split_by_commas_and_plus(text):
            c = canon_name(chunk)
            # фільтр прикладних слів/форм
            if not c:
                continue
            inns.add(c)

    # з "Склад"
    _pull(composition)

    # інколи INN є в назві у дужках або латиницею
    _pull(name)

    # прогонимо через базову таблицю лат→укр ще раз у випадку довгих рядків
    more = set()
    for i in list(inns):
        for lat, ua in LAT_TO_UA.items():
            if lat in i and ua not in i:
                more.add(canon_name(ua))
    inns |= more

    # пост-фільтр: дуже короткі чи службові токени відкидаємо
    inns = {i for i in inns if len(i) >= 4 and i not in DROP_TOKENS}
    return inns

class Brand2INNIndex:
    """Індекс: канонічна назва бренду -> множина INN-варіантів (канонічних)."""
    def __init__(self):
        self.name2inn: Dict[str, Set[str]] = {}

    def build(self, df, name_col: str, composition_col: Optional[str] = None):
        comp_col = composition_col or "Склад"
        if name_col not in df.columns:
            raise ValueError(f"Column '{name_col}' not in dataframe")
        if comp_col not in df.columns:
            df[comp_col] = ""

        for _, row in df.iterrows():
            raw_name = str(row.get(name_col, "") or "")
            raw_comp = str(row.get(comp_col, "") or "")
            cname = canon_name(raw_name)
            inns = extract_inn_from_row(raw_name, raw_comp)
            if not cname:
                continue
            if cname not in self.name2inn:
                self.name2inn[cname] = set()
            self.name2inn[cname] |= inns

    def get_inn(self, canon_brand: str) -> Set[str]:
        return self.name2inn.get(canon_brand, set())

# --- Головна функція збігу ---
def match_pred_to_gold(
    pred_name: str,
    gold_names: Set[str],
    index: Brand2INNIndex
) -> bool:
    """
    True, якщо предикт збігається з будь-яким елементом gold_names:
      1) exact по канону,
      2) перетин INN-множин (предикт-INN ∩ gold-INN != ∅),
      3) золотова назва є підрядком канону або навпаки (після нормалізації).
    """
    p = canon_name(pred_name)
    if not p:
        return False

    # 1) exact
    if p in gold_names:
        return True

    # 2) INN-перетин
    p_inn = index.get_inn(p)
    if p_inn:
        for g in gold_names:
            if p_inn & index.get_inn(g):
                return True

    # 3) підрядки
    for g in gold_names:
        if g in p or p in g:
            return True

    return False

def canonize_set(names: List[str]) -> Set[str]:
    return {canon_name(x) for x in names if isinstance(x, str) and canon_name(x)}
