# -*- coding: utf-8 -*-
import re
import csv
from typing import Dict, Tuple, Optional
from .ua_normalizer import normalize_text

ROUTE_HINTS = [
    ("очні краплі", "ophthalmic"),
    ("вушні краплі", "otic"),
    ("назальний спрей", "nasal"),
    ("спрей", "spray"),
    ("сироп", "syrup"),
    ("таблет", "tablet"),
    ("капсул", "capsule"),
    ("розчин для ін'єкцій", "injection"),
    ("для ін'єкцій", "injection"),
    ("мазь", "ointment"),
    ("крем", "cream"),
    ("гел", "gel"),
    ("суспензія", "suspension"),
    ("супозитор", "suppository"),
]

def load_alias_map(path: str) -> Dict[str, str]:
    amap: Dict[str, str] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            rd = csv.DictReader(f)
            for row in rd:
                a = normalize_text(row.get("alias", ""))
                t = normalize_text(row.get("target", ""))
                if a and t:
                    amap[a] = t
    except Exception:
        pass
    return amap

def guess_route(name: str, form: str = "", pharm_group: str = "") -> Optional[str]:
    txt = " ".join([name or "", form or "", pharm_group or ""]).lower()
    for key, route in ROUTE_HINTS:
        if key in txt:
            return route
    return None

def canonical_inn(name: str, composition: str, alias_map: Dict[str, str]) -> Optional[str]:
    """
    1) спробуємо знайти INN за точним/приблизним ключем у name;
    2) якщо ні — дістаємо перше слово/фрагмент зі Складу; 
    3) застосовуємо alias_map (brand→inn, inn→inn).
    """
    cand = normalize_text(name)
    if cand in alias_map:
        return alias_map[cand]

    # іноді INN уже в назві (латинка/укр)
    # спростимо: візьмемо перший токен зі 'Склад' (composition), якщо схожий
    comp = normalize_text(composition)
    if comp:
        # грубо: інгредієнт до першої коми/крапки/дужки
        m = re.match(r"([a-zA-Zа-яіїєґ\-]+)", comp)
        if m:
            token = m.group(1).strip()
            if token in alias_map:
                return alias_map[token]
            # якщо не в alias_map, але виглядає як латинський INN
            if re.match(r"[a-z]{3,}", token):
                return token

    # fallback: слова з name (на випадок "омепразол-дарниця")
    parts = re.split(r"[\s\-–—/]+", cand)
    for p in parts:
        if p in alias_map:
            return alias_map[p]
        if re.match(r"[a-z]{3,}", p):
            return p
    return None

def canonical_key(
    name: str,
    composition: str,
    form: str,
    pharm_group: str,
    alias_map: Dict[str, str],
    mode: str = "inn_route",  # 'doc' | 'inn' | 'inn_route'
) -> str:
    """
    Повертає канонічний ключ для дедуплікації/агрегації.
    """
    inn = canonical_inn(name or "", composition or "", alias_map) or normalize_text(name or "")
    if mode == "inn":
        return inn
    if mode == "doc":
        return normalize_text(name or "")
    # inn_route
    route = guess_route(name or "", form or "", pharm_group or "") or "generic"
    return f"{inn}|{route}"
