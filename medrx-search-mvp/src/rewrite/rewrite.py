# src/rewrite/rewrite.py
from __future__ import annotations

import csv
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None

# =========================
#  Debug / logging helpers
# =========================
_DBG = bool(os.environ.get("REWRITE_DEBUG"))

def _dbg(msg: str) -> None:
    if _DBG:
        print(f"[rewrite][dbg] {msg}", file=sys.stderr)

# =========================================
#  Unicode-aware normalization & boundaries
# =========================================

# «Символи слова»: латиниця (включно з розширеннями), кирилиця, цифри, апострофи та дефіс.
# Це використовується у кастомних слово-межах.
_WORD_CHARS_CLASS = (
    r"A-Za-z"
    r"\u00C0-\u024F"     # Latin Extended (À-ž)
    r"\u0400-\u04FF"     # Cyrillic
    r"\u0500-\u052F"     # Cyrillic Supplement
    r"0-9"
    r"\u0027\u2019\u02BC"  # '  ’  ʼ (апострофи)
    r"\-"                 # дефіс
)

# Шаблони кастомних меж слова (з урахуванням кирилиці/латини/апострофів/дефіса)
# (?<![WORD])  ...  (?![WORD])
def _wrap_word_boundaries(token_escaped: str) -> str:
    return rf"(?<![{_WORD_CHARS_CLASS}]){token_escaped}(?![{_WORD_CHARS_CLASS}])"

# Нормалізація UA/EN рядків: нижній регістр, прибрати зайві знаки, уніфікувати апострофи, стиснути пробіли
_NORMALIZE_KEEP = re.compile(rf"[{_WORD_CHARS_CLASS}\s]", re.UNICODE)

def _ua_norm(text: str) -> str:
    if not text:
        return ""
    t = text.lower()
    # уніфікуємо різні апострофи
    t = t.replace("\u2019", "'").replace("\u02BC", "'")
    # прибираємо «спеціалки» (™®, тощо), лишаємо тільки «легальні» символи з _NORMALIZE_KEEP
    t = "".join(ch if _NORMALIZE_KEEP.match(ch) else " " for ch in t)
    # стабілізуємо дефіси як дефіси, не міняємо їх на пробіли
    # вже вище «зайві» прибрали, дефіс входить у _WORD_CHARS_CLASS
    # зайві пробіли -> один пробіл
    t = re.sub(r"\s+", " ", t).strip()
    return t

# ==================
#  Config structure
# ==================
@dataclass
class RewriteConfig:
    aliases_csv: Optional[str] = None
    symptoms_csv: Optional[str] = None
    units_csv: Optional[str] = None
    # Додавати INN наприкінці, якщо знайдено brand/alias (не для type="inn")
    append_inn_for_brand_or_alias: bool = True
    # Максимальна кількість INN для додавання (щоб не «засмічувати» запит)
    max_appended_inn: int = 5

# ============================
#  CSV loader util (robust)
# ============================
def _resolve_path(base_cfg: Path, p: Optional[str]) -> Optional[Path]:
    if not p:
        return None
    pp = Path(p)
    if not pp.is_absolute():
        pp = (base_cfg.parent / pp).resolve()
    return pp

def load_config(yaml_path: str | Path) -> RewriteConfig:
    yaml_path = Path(yaml_path).resolve()
    if yaml is None:
        raise RuntimeError("PyYAML is required to load config.")
    with yaml_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = RewriteConfig(
        aliases_csv = str(_resolve_path(yaml_path, raw.get("aliases_csv"))) if raw.get("aliases_csv") else None,
        symptoms_csv= str(_resolve_path(yaml_path, raw.get("symptoms_csv"))) if raw.get("symptoms_csv") else None,
        units_csv   = str(_resolve_path(yaml_path, raw.get("units_csv"))) if raw.get("units_csv") else None,
        append_inn_for_brand_or_alias = bool(raw.get("append_inn_for_brand_or_alias", True)),
        max_appended_inn = int(raw.get("max_appended_inn", 5)),
    )
    _dbg(f"CFG loaded from {str(yaml_path)}")
    _dbg(f"  aliases_csv = {cfg.aliases_csv or '—'}")
    _dbg(f"  symptoms_csv= {cfg.symptoms_csv or '—'}")
    _dbg(f"  units_csv   = {cfg.units_csv or '—'}")
    return cfg

def _read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # захист від None та пробілів/кавічок
            cleaned = { (k or "").strip().lower(): (v or "").strip() for k, v in row.items() }
            # пропускаємо порожні рядки
            if any(cleaned.values()):
                rows.append(cleaned)
    return rows

# ============================
#  Loaders for dictionaries
# ============================
def _normalize_alias_type(t: str) -> str:
    t = (t or "").strip().lower()
    # різні джерела можуть мати brand2inn/typo/alias/inn тощо
    if t in {"brand2inn", "brand"}:
        return "brand"
    if t in {"typo", "alias", "syn"}:
        return "alias"
    if t in {"inn"}:
        return "inn"
    # дефолт
    return "alias"

def load_aliases(path_like: str | Path) -> Dict[str, Dict[str, str]]:
    """
    Повертає dict:
      { normalized_alias: {"canon": <inn>, "type": "inn|brand|alias"} }
    """
    path = Path(path_like)
    rows = _read_csv_dicts(path)
    alias_map: Dict[str, Dict[str, str]] = {}
    kept, skipped = 0, 0
    for r in rows:
        alias_raw = r.get("alias", "")
        target_raw = r.get("target", "")
        type_raw = r.get("type", "")

        alias_n = _ua_norm(alias_raw)
        target_n = _ua_norm(target_raw)
        t_norm = _normalize_alias_type(type_raw)

        if not alias_n or not target_n:
            skipped += 1
            continue

        # Якщо alias == таргет → це INN
        if alias_n == target_n:
            t_norm = "inn"

        # зберігаємо першу зустріч — цього досить; конфлікти зазвичай не критичні
        if alias_n not in alias_map:
            alias_map[alias_n] = {"canon": target_n, "type": t_norm}
            kept += 1

    _dbg(f"aliases loaded: {kept} (schema=alias/target/type)")
    return alias_map

def load_symptoms(path_like: str | Path) -> List[Dict[str, str]]:
    """
    Очікувані колонки: canon_symptom, aliases, intent_hint
    'aliases' — список, розділений '|'
    """
    path = Path(path_like)
    rows = _read_csv_dicts(path)
    out: List[Dict[str, str]] = []
    for r in rows:
        canon = _ua_norm(r.get("canon_symptom", ""))
        aliases = [ _ua_norm(x) for x in (r.get("aliases", "")).split("|") if _ua_norm(x) ]
        intent = (r.get("intent_hint") or "").strip().lower()
        if canon and aliases:
            out.append({"canon": canon, "aliases": "|".join(aliases), "intent": intent})
    _dbg(f"symptoms loaded: {len(out)}")
    return out

def load_unit_map(path_like: str | Path) -> Dict[str, str]:
    """
    Очікувані колонки: raw, canon
    """
    path = Path(path_like)
    rows = _read_csv_dicts(path)
    mp: Dict[str, str] = {}
    total, skipped_empty_norm = 0, 0
    for r in rows:
        raw = _ua_norm(r.get("raw", ""))
        canon = _ua_norm(r.get("canon", ""))
        total += 1
        if not raw or not canon:
            skipped_empty_norm += 1
            continue
        mp[raw] = canon
    _dbg(f"unit rules loaded: {len(mp)} (skipped_empty_norm={skipped_empty_norm}, total_rows={total})")
    return mp

# ====================================
#  Rewriter with precompiled patterns
# ====================================
class Rewriter:
    def __init__(self, cfg: RewriteConfig):
        self.cfg = cfg
        self.aliases: Dict[str, Dict[str, str]] = load_aliases(cfg.aliases_csv) if cfg.aliases_csv else {}
        self.symptom_rows: List[Dict[str, str]] = load_symptoms(cfg.symptoms_csv) if cfg.symptoms_csv else []
        self.unit_map: Dict[str, str] = load_unit_map(cfg.units_csv) if cfg.units_csv else {}

        # ---- precompile alias patterns
        self._alias_patterns: List[Tuple[re.Pattern, str, str, str]] = []
        # (compiled_regex, alias_norm, canon_inn, type)
        for alias_norm, meta in self.aliases.items():
            # ескейпимо нормалізований alias і додаємо кастомні межі слова
            patt = _wrap_word_boundaries(re.escape(alias_norm))
            self._alias_patterns.append((
                re.compile(patt, re.IGNORECASE | re.UNICODE),
                alias_norm,
                meta["canon"],
                meta["type"],
            ))

        # ---- precompile symptom patterns
        # мапа <alias_term_norm> -> (canon_symptom, full_aliases_list)
        self._symptom_alias_to_row: Dict[str, Tuple[str, List[str]]] = {}
        for row in self.symptom_rows:
            canon = row["canon"]           # вже нормалізований
            aliases = [a for a in row["aliases"].split("|") if a]
            for a in aliases:
                self._symptom_alias_to_row[a] = (canon, aliases)

        self._symptom_patterns: List[Tuple[re.Pattern, str]] = []
        # (compiled_regex, alias_term_norm)
        for alias_term in self._symptom_alias_to_row.keys():
            patt = _wrap_word_boundaries(re.escape(alias_term))
            self._symptom_patterns.append((
                re.compile(patt, re.IGNORECASE | re.UNICODE),
                alias_term
            ))

    # -----------------------------
    #  Public helpers for scripts
    # -----------------------------
    def _find_alias_hits(self, text: str) -> List[Dict[str, str]]:
        """
        Повертає список попадань alias’ів у тексті (пошук ведемо по НОРМАЛІЗОВАНОМУ тексту).
        Кожен елемент: {"alias": <alias_norm>, "canon": <inn>, "type": "inn|brand|alias"}
        """
        norm = _ua_norm(text)
        hits: List[Dict[str, str]] = []
        # щоб не дублювати однакові alias по декількох входженнях
        seen_aliases: set[str] = set()
        for patt, alias_norm, canon, typ in self._alias_patterns:
            if patt.search(norm):
                if alias_norm in seen_aliases:
                    continue
                seen_aliases.add(alias_norm)
                hits.append({"alias": alias_norm, "canon": canon, "type": typ})
        return hits

    def _find_symptom_hits(self, text: str) -> Tuple[List[str], List[str]]:
        """
        Повертає:
          - symptom_hits: список термінів-аліасів (нормалізованих), які знайшли
          - symptoms_applied: «розширення» (всі варіанти з того ж ряду), для прев’ю
        """
        norm = _ua_norm(text)
        found_aliases: List[str] = []
        expanded: List[str] = []
        seen_rows: set[str] = set()  # canon to avoid duplicate expansion
        for patt, alias_term in self._symptom_patterns:
            if patt.search(norm):
                found_aliases.append(alias_term)
                canon, all_aliases = self._symptom_alias_to_row.get(alias_term, ("", []))
                if canon and canon not in seen_rows:
                    seen_rows.add(canon)
                    # додаємо усі аліаси цього симптома (як у твоєму прев’ю)
                    expanded.extend(all_aliases)
        # унікалізація зі збереженням порядку
        def _uniq(xs: Iterable[str]) -> List[str]:
            out, seen = [], set()
            for x in xs:
                if x not in seen:
                    seen.add(x); out.append(x)
            return out
        return _uniq(found_aliases), _uniq(expanded)

    def _apply_units(self, text: str) -> Tuple[str, List[Tuple[str, str]]]:
        """
        Дуже консервативно нормалізуємо одиниці: токенізуємо нормалізований текст,
        мапимо рівно ті токени, що є у довіднику. Нічого порожнього не повертаємо.
        """
        norm = _ua_norm(text)
        tokens = norm.split()
        hits: List[Tuple[str, str]] = []
        # Підміна робиться лише для прев’ю (повертаємо список), не змінюємо вихідний текст
        for t in tokens:
            if t in self.unit_map:
                hits.append((t, self.unit_map[t]))
        return text, hits

    # -------------
    #  Main method
    # -------------
    def rewrite(self, text: str) -> Dict[str, object]:
        """
        Повертає словник з ключами:
          - raw
          - rewritten
          - alias_hits, aliases_applied      (щоб ALIASES+ точно відображався)
          - symptom_hits, symptoms_applied   (для прев’ю)
          - unit_hits, units_applied         (сумісність зі старими скриптами)
        """
        raw = text

        # 1) alias-и
        alias_hits = self._find_alias_hits(raw)

        # 2) симптоми
        symptom_hits, symptoms_applied = self._find_symptom_hits(raw)

        # 3) одиниці
        _, unit_hits = self._apply_units(raw)

        # 4) формуємо «ALIASES+» (INN до додавання)
        aliases_applied: List[str] = []
        if alias_hits:
            # беремо лише ті, що не type="inn"
            for h in alias_hits:
                if h.get("type") != "inn":
                    inn = h.get("canon", "")
                    if inn and inn not in aliases_applied:
                        aliases_applied.append(inn)

        # 5) (опційно) додаємо INN у кінець запиту
        rewritten = raw
        if self.cfg.append_inn_for_brand_or_alias and aliases_applied:
            # уникаємо додавання INN, які і так уже в запиті
            norm = _ua_norm(raw)
            final_to_add: List[str] = []
            for inn in aliases_applied:
                if inn not in norm:
                    final_to_add.append(inn)
                if len(final_to_add) >= max(1, int(self.cfg.max_appended_inn)):
                    break
            if final_to_add:
                # додамо одним «хвостом», щоб не ламати оригінальний текст
                rewritten = f"{raw} " + " ".join(final_to_add)

        # 6) формуємо результат з «дублями» ключів для сумісності з різними скриптами
        res = {
            "raw": raw,
            "rewritten": rewritten,

            # aliases
            "alias_hits": alias_hits,
            "aliases_applied": aliases_applied,
            "aliases": aliases_applied,          # інколи скрипти читають саме це

            # symptoms
            "symptom_hits": symptom_hits,
            "symptoms_applied": symptoms_applied,

            # units
            "unit_hits": unit_hits,
            "units_applied": unit_hits,
        }
        return res
