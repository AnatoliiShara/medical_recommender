# -*- coding: utf-8 -*-
import re
from typing import List, Dict, Optional, Tuple

try:
    # Опційно: для токен-орієнтованих меж чанків
    from transformers import AutoTokenizer  # type: ignore
except Exception:
    AutoTokenizer = None  # type: ignore


class MedicalChunker:
    """
    Медично-обізнаний chunker:
    - не рве складені терміни (regex + word boundaries, case-insensitive)
    - розпізнає та префіксує секції (Показання, Протипоказання, Дозування, тощо)
    - видаляє заголовок секції з контенту (щоб не було дублю)
    - підтримує токен-орієнтований чанкінг (якщо передано модель токенайзера)
    - має overlap між чанками; фолбек — символьний чанкінг
    """

    def __init__(self,
                 medical_phrases: Optional[List[str]] = None,
                 tokenizer_model: Optional[str] = None):
        # Медичні фрази, які слід зберігати як цілі спани
        self.medical_phrases = medical_phrases or [
            'артеріальна гіпертензія', 'серцева недостатність', 'стенокардія',
            'гіпертонічна хвороба', 'ішемічна хвороба серця', 'інфаркт міокарда',
            'цукровий діабет', 'хвороба крона', 'виразкова хвороба', 'гастрит',
            'печінкова недостатність', 'ниркова недостатність'
        ]
        # Довші спочатку, щоб уникати часткових перетинів
        self.medical_phrases.sort(key=len, reverse=True)
        self._phrase_regexes = [
            re.compile(rf"\b{re.escape(p)}\b", flags=re.IGNORECASE) for p in self.medical_phrases
        ]

        # Секції: для кожної тримаємо 2 регекси — detect (виявити у реченні) і strip (вирізати заголовок з початку речення)
        self._sections_patterns: List[Tuple[str, re.Pattern, re.Pattern]] = [
            # Показання / Показання до застосування:
            (
                'Показання',
                re.compile(r'\bпоказан(?:ня|і)\b', flags=re.IGNORECASE),
                re.compile(r'^\s*показан(?:ня|і)(?:[^:]{0,40})?:\s*', flags=re.IGNORECASE),
            ),
            # Протипоказання:
            (
                'Протипоказання',
                re.compile(r'\bпротипоказан(?:ня|і)\b', flags=re.IGNORECASE),
                re.compile(r'^\s*протипоказан(?:ня|і)(?:[^:]{0,40})?:\s*', flags=re.IGNORECASE),
            ),
            # Дозування / Дозування та спосіб застосування / Спосіб застосування та дози:
            (
                'Дозування',
                re.compile(r'\b(дозуван(?:ня|і)|спосіб застосування(?: та дози)?)\b', flags=re.IGNORECASE),
                re.compile(
                    r'^\s*(?:дозуван(?:ня|і)(?:[^:]{0,40})?|спосіб застосування(?: та дози)?)\s*:\s*',
                    flags=re.IGNORECASE
                ),
            ),
            # Побічні реакції / Небажані реакції:
            (
                'Побічні реакції',
                re.compile(r'\b(побічн(?:і(?: реакції)?)|небажан(?:і(?: реакції)?))\b', flags=re.IGNORECASE),
                re.compile(r'^\s*(?:побічн(?:і(?: реакції)?)|небажан(?:і(?: реакції)?))\s*:\s*', flags=re.IGNORECASE),
            ),
            # Взаємодії:
            (
                'Взаємодії',
                re.compile(r'\bвзаємоді(?:я|ї)\b', flags=re.IGNORECASE),
                re.compile(r'^\s*взаємоді(?:я|ї)\s*:\s*', flags=re.IGNORECASE),
            ),
            # Особливості / Особливості застосування / Застереження:
            (
                'Особливості',
                re.compile(r'\b(особливост(?:і|і застосування)|застереженн(?:я))\b', flags=re.IGNORECASE),
                re.compile(r'^\s*(?:особливост(?:і|і застосування)|застереженн(?:я))\s*:\s*', flags=re.IGNORECASE),
            ),
        ]

        self._tokenizer = None
        if tokenizer_model and AutoTokenizer is not None:
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
            except Exception:
                self._tokenizer = None  # graceful fallback, працюємо по символах

    # ------------------- ПУБЛІЧНИЙ API -------------------

    def smart_chunking(self,
                       text: str,
                       *,
                       max_chunk_size: int = 800,     # fallback у символах
                       max_tokens: Optional[int] = 128,  # дефолт, який у твоєму прикладі дає ~30–45 токенів
                       overlap_tokens: int = 16,
                       min_chunk_chars: int = 60) -> List[str]:
        """
        Повертає список чанків у форматі:
        "Показання: ...", "Протипоказання: ...", ...
        """
        text = self._normalize_spaces(text)
        if not text:
            return []

        # 1) Розрізати текст на секції
        sections = self._split_by_sections(text)
        if not sections:  # fallback: все як одна секція
            sections = [("Текст", text)]

        # 2) Усередині секцій — захистити фрази та порізати
        chunks: List[str] = []
        for sec_name, sec_text in sections:
            protected, restore_map = self._protect_phrases(sec_text)

            # Або токен-орієнтоване різання, або символьне
            if self._tokenizer and max_tokens:
                subchunks = self._token_chunk(protected, max_tokens=max_tokens, overlap=overlap_tokens)
            else:
                subchunks = self._char_chunk(protected, max_chars=max_chunk_size)

            for sc in subchunks:
                restored = self._restore_phrases(sc, restore_map)
                out = f"{sec_name}: {restored}".strip()
                if len(out) >= min_chunk_chars:
                    chunks.append(out)

        return chunks

    # ------------------- ДОПОМІЖНІ МЕТОДИ -------------------

    @staticmethod
    def _normalize_spaces(s: str) -> str:
        s = s.replace("\u00A0", " ")
        return re.sub(r"\s+", " ", s).strip()

    def _split_by_sections(self, text: str) -> List[Tuple[str, str]]:
        """
        Грубо розбиває текст на секції за реченнями.
        Якщо у реченні детектиться заголовок секції — починаємо нову секцію,
        а сам заголовок вирізаємо зі змісту (щоб не було дублю при префіксації).
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)
        accumulator: List[Tuple[str, List[str]]] = []  # [(sec_name, [sentences...])]

        def start_section(name: str):
            accumulator.append((name, []))

        for sent in sentences:
            s = sent.strip()
            if not s:
                continue

            matched_name: Optional[str] = None
            strip_rx: Optional[re.Pattern] = None

            # Шукаємо секцію
            for name, detect_rx, srx in self._sections_patterns:
                if detect_rx.search(s):
                    matched_name = name
                    strip_rx = srx
                    break

            if matched_name:
                # нова секція
                start_section(matched_name)
                # Вирізати заголовок секції з початку речення (до двокрапки, якщо вона є)
                content = strip_rx.sub("", s, count=1).strip() if strip_rx else s
                if content:
                    accumulator[-1][1].append(content)
            else:
                # Якщо секції ще не було — створимо безіменну "Текст"
                if not accumulator:
                    start_section("Текст")
                accumulator[-1][1].append(s)

        # Зібрати пару (назва_секції, контент)
        out: List[Tuple[str, str]] = []
        for name, sents in accumulator:
            joined = " ".join(sents).strip()
            if joined:
                out.append((name, joined))
        return out

    def _protect_phrases(self, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Захист фраз: підміна на плейсхолдери з урахуванням word-boundary + ignorecase.
        Зберігаємо оригінальний регістр знайдених збігів у restore_map.
        """
        restore_map: Dict[str, str] = {}
        protected = text
        for i, rx in enumerate(self._phrase_regexes):
            placeholder = f"__MEDICAL_PHRASE_{i}__"

            def repl(m: re.Match) -> str:
                restore_map[placeholder] = m.group(0)  # оригінал з регістром
                return placeholder

            protected = rx.sub(repl, protected)
        return protected, restore_map

    @staticmethod
    def _restore_phrases(text: str, restore_map: Dict[str, str]) -> str:
        for ph, orig in restore_map.items():
            text = text.replace(ph, orig)
        return text

    @staticmethod
    def _char_chunk(text: str, max_chars: int) -> List[str]:
        if len(text) <= max_chars:
            return [text]
        chunks: List[str] = []
        i = 0
        n = len(text)
        while i < n:
            j = min(n, i + max_chars)
            # Намагаймось різати по пробілу/крапці
            k = text.rfind(' ', i, j)
            if k == -1 or (j - i) < 0.6 * max_chars:
                k = j
            chunk = text[i:k].strip()
            if chunk:
                chunks.append(chunk)
            i = k
        return chunks

    def _token_chunk(self, text: str, *, max_tokens: int, overlap: int) -> List[str]:
        assert self._tokenizer is not None
        toks = self._tokenizer.encode(text, add_special_tokens=False)
        if len(toks) <= max_tokens:
            return [text]

        out: List[str] = []
        start = 0
        n = len(toks)
        while start < n:
            end = min(n, start + max_tokens)
            piece_ids = toks[start:end]
            piece = self._tokenizer.decode(piece_ids, skip_special_tokens=True).strip()
            if piece:
                out.append(piece)
            if end == n:
                break
            start = max(0, end - overlap)
        return out


# ------------------- ЛОКАЛЬНИЙ ТЕСТ -------------------
if __name__ == "__main__":
    chunker = MedicalChunker(tokenizer_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    test_text = '''
    Показання до застосування: артеріальна гіпертензія легкого та помірного ступеня, серцева недостатність, стенокардія напруги.
    Протипоказання: діти до 18 років, вагітність, період лактації, гостра серцева недостатність, печінкова недостатність тяжкого ступеня.
    Дозування та спосіб застосування: по 1 таблетці двічі на день після їжі, курс лікування визначається лікарем.
    '''

    print("=== ТЕСТ ADVANCED MEDICAL CHUNKER ===")
    chunks = chunker.smart_chunking(test_text, max_tokens=128, overlap_tokens=16)
    for i, ch in enumerate(chunks):
        print(f"{i}: \"{ch}\"")
        print(f"   Довжина: {len(ch)} символів")
        if chunker._tokenizer:
            toks = len(chunker._tokenizer.encode(ch, add_special_tokens=False))
            print(f"   Токенів: {toks}")
        print()
