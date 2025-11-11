# -*- coding: utf-8 -*-
import re
from src.search.clinical.normalizer import normalize_text, phrase_to_regex, any_phrase_in

def test_normalize_basic():
    s = "“Діарея”  —  ШЛУНКОВО-кишковий   розлад ’ "
    out = normalize_text(s)
    assert "діарея" in out and "шлунково" in out
    assert "  " not in out

def test_phrase_boundaries():
    text = "препарат від діареї призначають дорослим"
    r = phrase_to_regex("діареї")
    assert r.search(text.lower())
    r2 = phrase_to_regex("діарея")
    assert r2.search(text.lower())

def test_any_phrase_in():
    text = "страждаю на гостру діарею"
    res = [phrase_to_regex("гостра діарея"), phrase_to_regex("кишковий розлад")]
    assert any_phrase_in(text, res) is True
