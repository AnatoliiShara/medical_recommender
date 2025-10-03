# -*- coding: utf-8 -*-
from __future__ import annotations
import unicodedata
import re

class Normalizer:
    """
    Легка нормалізація користувацького запиту:
    - Unicode NFKC
    - нижній регістр
    - уніфікація лапок/дефісів/символів ™®©
    - видалення зайвої пунктуації
    - стискання пробілів
    """
    def __init__(self):
        # символи, які часто псують індексацію
        self._trash = "®™©·•✓”’“`'«»()[]{}，、—–--/\\|"
        self._trash_re = re.compile(rf"[{re.escape(self._trash)}]+")
        # все не-слово і не кирилиця → пробіл
        self._nonword_re = re.compile(r"[^\w\u0400-\u04FF]+")

    def normalize(self, text: str) -> str:
        if not text:
            return ""
        s = unicodedata.normalize("NFKC", str(text))
        s = s.lower()
        s = self._trash_re.sub(" ", s)
        s = s.replace("\u00A0", " ")
        # залишаємо букви/цифри (латиниця + кирилиця); інше у пробіли
        s = self._nonword_re.sub(" ", s)
        # злиплі літери типу "loperaмide" уже окей як токени
        s = " ".join(s.split())
        return s
