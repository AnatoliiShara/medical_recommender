# src/search/query_reform/normalizer.py
from __future__ import annotations
import re
import unicodedata

__all__ = ["normalize_query", "Normalizer"]

_QUOTE_MAP = {
    "’": "'",
    "‘": "'",
    "`": "'",
    "ʼ": "'",
    "“": '"',
    "”": '"',
    "«": '"',
    "»": '"',
}

_WS_RE = re.compile(r"\s+")
_PUNCT_SPACES_RE = re.compile(r"\s*([,.;:!?()])\s*")

def _norm_unicode(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    for k, v in _QUOTE_MAP.items():
        s = s.replace(k, v)
    s = _PUNCT_SPACES_RE.sub(r" \1 ", s)
    s = _WS_RE.sub(" ", s)
    return s.strip()

def normalize_query(text: str | None) -> str:
    if not text:
        return ""
    s = str(text)
    s = _norm_unicode(s)
    s = s.lower()
    s = unicodedata.normalize("NFKC", s)
    s = _WS_RE.sub(" ", s).strip()
    return s

# 🔧 compat shim для старого коду
class Normalizer:
    @staticmethod
    def normalize(text: str | None) -> str:
        return normalize_query(text)
