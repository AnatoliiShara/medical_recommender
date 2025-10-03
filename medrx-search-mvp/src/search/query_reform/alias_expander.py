# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Set, List, Tuple
from pathlib import Path
import csv

from search.query_reform.normalizer import Normalizer

class AliasExpander:
    """
    Зчитує CSV зі стовпцями:
      alias,target,type
    де target — канонічний INN/група, alias — синонім/бренд/опечатка.
    Завдання: якщо у запиті знайдено alias — додати target (і, за бажанням, кілька споріднених термінів).
    """

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.norm = Normalizer()
        self.alias2canon: Dict[str, str] = {}
        self.canon2aliases: Dict[str, Set[str]] = {}
        self._load()

    def _load(self):
        p = Path(self.csv_path)
        if not p.exists():
            raise FileNotFoundError(f"[AliasExpander] CSV not found: {self.csv_path}")
        with p.open("r", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            if not {"alias","target","type"}.issubset(set(rdr.fieldnames or [])):
                raise ValueError("[AliasExpander] CSV must have columns: alias,target,type")
            for row in rdr:
                alias_raw = (row.get("alias") or "").strip()
                target_raw = (row.get("target") or "").strip()
                if not alias_raw or not target_raw:
                    continue
                alias = self.norm.normalize(alias_raw)
                target = self.norm.normalize(target_raw)
                if not alias or not target:
                    continue
                self.alias2canon[alias] = target
                self.canon2aliases.setdefault(target, set()).add(alias)
                self.canon2aliases[target].add(target)  # включаємо і сам канон

    def expand(self, query_norm: str, max_terms: int = 5) -> List[str]:
        """
        Повертає список канонічних термінів (переважно INN), відповідних знайденим у запиті alias.
        Якщо у запиті зустрічаються кілька alias — об'єднуємо канони.
        """
        if not query_norm:
            return []
        hits: Set[str] = set()
        for alias, canon in self.alias2canon.items():
            if alias in query_norm:
                hits.add(canon)
        # обмеження на кількість, стабільний порядок
        out = sorted(hits)[:max_terms]
        return out
