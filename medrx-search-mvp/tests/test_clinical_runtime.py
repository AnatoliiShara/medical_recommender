# -*- coding: utf-8 -*-
from src.search.clinical.p0_runtime import build_clinical_assets, detect_conditions, union_candidates_for_conditions, ce_bias_for_doc

DOCS = [
    "Препарат застосовують при гострій діареї. Склад: ...",
    "Засіб від кашлю, бронхіт.",
    "Лікарський засіб для лікування запору (констипації).",
]

def test_runtime_flow(tmp_path):
    # Build temporary dict structure
    root = tmp_path / "clinical"
    cond = "diarrhea"
    (root / cond).mkdir(parents=True, exist_ok=True)
    (root / cond / f"{cond}_trigger.txt").write_text("діарея\nгостра діарея\n", encoding="utf-8")
    (root / cond / f"{cond}_positive.txt").write_text("гостра діарея\n", encoding="utf-8")
    (root / cond / f"{cond}_penalty.txt").write_text("кашель\n", encoding="utf-8")

    assets = build_clinical_assets(DOCS, str(root))

    matched = detect_conditions("страждаю на гостру діарею", assets)
    assert "diarrhea" in matched

    bm25_top_ids = {0, 2}
    union = union_candidates_for_conditions(assets, matched, bm25_top_ids, union_cap=5)
    assert 0 in union and 2 not in union

    delta = ce_bias_for_doc(DOCS[0], assets, matched, bias_pos=0.2, bias_pen=0.3)
    assert delta > 0
