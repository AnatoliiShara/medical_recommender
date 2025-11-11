# -*- coding: utf-8 -*-
from src.search.clinical.index import ClinicalInvertedIndex

DOCS = [
    "Препарат застосовують при гострій діареї. Склад: ...",
    "Засіб від кашлю, бронхіт.",
    "Лікарський засіб для лікування запору (констипації).",
    "Допоміжний засіб з магнієм; не для діареї.",
]

def test_phrase_matching_and_candidates():
    idx = ClinicalInvertedIndex(DOCS)
    # Should match doc 0 and 3 for single token 'діареї', but phrase 'гостра діарея' only doc 0
    m1 = idx.find_docs_by_terms(["діареї"])
    assert m1  # some match
    m2 = idx.find_docs_by_terms(["гостра діарея"])
    assert 0 in m2 and 3 not in m2

def test_union_cap():
    idx = ClinicalInvertedIndex(DOCS)
    m = idx.find_docs_by_terms(["діареї", "кашлю", "запору"], cap=2)
    assert len(m) == 2
