# -*- coding: utf-8 -*-
"""
P0 — Simple token-based inverted index with phrase verification for clinical dictionaries.

Design:
- Build an inverted index of tokens → doc_ids over normalized documents.
- For a multi-token phrase, we preselect candidate docs by intersecting token posting lists,
  then verify with a compiled phrase regex (handles word boundaries & flexible whitespace).
- This drastically reduces the need to scan all documents for every phrase.

No external dependencies.
"""

from __future__ import annotations
import re
from typing import Dict, Iterable, List, Pattern, Sequence, Set

from .normalizer import normalize_text, tokens, phrase_to_regex


class ClinicalInvertedIndex:
    def __init__(self, docs: Sequence[str]) -> None:
        """docs: sequence of raw document texts (strings)."""
        self.num_docs = len(docs)
        # Store normalized documents for verification regex search
        self._docs_norm: List[str] = [normalize_text(d or "") for d in docs]
        # Build token → doc_ids
        self._postings: Dict[str, Set[int]] = {}
        for doc_id, text in enumerate(self._docs_norm):
            for tok in tokens(text):
                self._postings.setdefault(tok, set()).add(doc_id)

    def _candidate_docs_for_phrase(self, phrase: str) -> Set[int]:
        toks = tokens(phrase)
        if not toks:
            return set()
        # Intersect postings for all tokens. If a token is unseen → empty set.
        it = None
        for t in toks:
            ids = self._postings.get(t, set())
            it = ids if it is None else (it & ids)
            if not it:
                return set()
        return it or set()

    def docs_matching_phrase(self, phrase: str) -> Set[int]:
        """Return doc_ids where the phrase matches with proper boundaries."""
        cand = self._candidate_docs_for_phrase(phrase)
        if not cand:
            return set()
        rex: Pattern = phrase_to_regex(phrase)
        out: Set[int] = set()
        for doc_id in cand:
            if rex.search(self._docs_norm[doc_id]):
                out.add(doc_id)
        return out

    def find_docs_by_terms(self, phrases: Iterable[str], cap: int | None = None) -> Set[int]:
        """Union of documents that match ANY of the phrases (after verification).
        cap: if provided, early-stop once we collected this many doc_ids (useful for UNION-cap).
        """
        seen: Set[int] = set()
        for ph in phrases:
            for doc_id in self.docs_matching_phrase(ph):
                seen.add(doc_id)
                if cap is not None and len(seen) >= cap:
                    return seen
        return seen

    # Optional: expose the normalized document for debugging/explain
    def doc_text(self, doc_id: int) -> str:
        return self._docs_norm[doc_id]
