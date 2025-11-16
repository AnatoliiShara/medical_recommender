#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, re
from pathlib import Path
import pandas as pd

# --- Патерни (розширювані) ---
# Diarrhea – лишаємо як у тебе (припускаю, що вже є)
DIARRHEA_PATTERNS = {
    "ORS": [r"регідрат", r"оральн.? регідрат", r"oral rehydration"],
    "DIOSMECTITE": [r"діосмект", r"smecta", r"diosmect"],
    "LOPERAMIDE": [r"лоперамід", r"loperamide"],
    "RACECADOTRIL": [r"рацекадотрил", r"racecadotril"],
    "RIFAXIMIN": [r"ріфаксимін", r"rifaximin"],
    "C_DIFF": [r"c\.?\s*difficile", r"клостриді(ум|альн)"]
}

# Common cold / ГРВІ
COLD_PATTERNS = {
    "PARACETAMOL": [r"парацетамол", r"\bacetaminophen\b"],
    "IBUPROFEN": [r"ібупрофен", r"\bibuprofen\b"],
    "OXYMETAZOLINE": [r"оксиметазолін", r"\boxymetazoline\b"],
    "PHENYLEPHRINE": [r"фенілефрин", r"\bphenylephrine\b"],
    "MOMETASONE": [r"мометазон", r"\bmometasone\b"],
}

# Hypertension (ACEi, ARB, CCB, thiazides)
HYP_PATTERNS = {
    "ACEI":  [r"еналаприл", r"лізіноприл", r"периндоприл", r"раміприл", r"каптоприл", r"\btrandolapril\b"],
    "ARB":   [r"лозартан", r"валсартан", r"кандесартан", r"телмісартан", r"олмесартан"],
    "CCB":   [r"амлодипін", r"ніфедипін", r"фелодипін"],
    "THIAZ": [r"гідрохлортіазид", r"індапамід", r"\bchlorthalidone\b"],
}

CORPUS_FIELDS = [
    "Назва препарату","Назва","Name","Склад","Показання","Показання до застосування",
    "Фармакотерапевтична група","Особливості застосування","Текст"
]

def compile_re(patterns):
    return re.compile("|".join(patterns), re.IGNORECASE|re.UNICODE)

def find_doc_ids(df: pd.DataFrame, compiled_re) -> list[int]:
    hits=set()
    for col in CORPUS_FIELDS:
        if col in df.columns:
            s = df[col].fillna("").astype(str)
            idx = s.str.contains(compiled_re, regex=True)
            hits.update(df.index[idx].tolist())
    return sorted(map(int, hits))

def diarrhea_gold_for_text(t: str, dia_map: dict) -> set[int]:
    t=t.lower(); rel=set()
    if any(k in t for k in ["діаре", "пронос", "ibsd", "спк з діаре"]):
        for key in ["ORS","DIOSMECTITE","LOPERAMIDE","RACECADOTRIL","RIFAXIMIN"]:
            rel.update(dia_map.get(key, []))
        if ("c. diff" in t) or ("клострид" in t):
            rel.update(dia_map.get("C_DIFF", []))
    return rel

def cold_gold_for_text(t: str, cold_map: dict) -> set[int]:
    t=t.lower(); rel=set()
    if any(k in t for k in ["застуд", "грві", "нежит", "горл", "кашл", "common cold"]):
        for key in ["PARACETAMOL","IBUPROFEN","OXYMETAZOLINE","PHENYLEPHRINE","MOMETASONE"]:
            rel.update(cold_map.get(key, []))
    return rel

def hyp_gold_for_text(t: str, hyp_map: dict) -> set[int]:
    t=t.lower(); rel=set()
    if ("гіпертон" in t) or ("тиск" in t):
        for key in ["ACEI","ARB","CCB","THIAZ"]:
            rel.update(hyp_map.get(key, []))
        if "амлодипін" in t: rel.update(hyp_map.get("CCB", []))
        if ("лозартан" in t) or ("валсартан" in t): rel.update(hyp_map.get("ARB", []))
    return rel

def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--queries_in", required=True)
    ap.add_argument("--queries_out", required=True)
    args=ap.parse_args()

    df = pd.read_parquet(args.corpus).reset_index(drop=True)

    # Побудова мап для кожного набору патернів
    dia_map  = {k: find_doc_ids(df, compile_re(v)) for k,v in DIARRHEA_PATTERNS.items()}
    cold_map = {k: find_doc_ids(df, compile_re(v)) for k,v in COLD_PATTERNS.items()}
    hyp_map  = {k: find_doc_ids(df, compile_re(v)) for k,v in HYP_PATTERNS.items()}

    out=[]
    covered=0; cnt_gold=0
    for line in Path(args.queries_in).read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        q = json.loads(line); text = q["query"]
        gid=set()
        gid |= diarrhea_gold_for_text(text, dia_map)
        gid |= cold_gold_for_text(text, cold_map)
        gid |= hyp_gold_for_text(text, hyp_map)
        if gid:
            q["gold_ids"] = sorted(map(int, gid))
            covered += 1
            cnt_gold += len(gid)
        out.append(q)

    Path(args.queries_out).write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in out)+"\n", encoding="utf-8")
    avg = (cnt_gold/covered) if covered else 0.0
    print(f"✅ Saved: {args.queries_out}")
    print(f"coverage_queries_with_gold = {covered}/{len(out)}")
    print(f"avg_gold_per_labeled_query = {avg:.1f}")

if __name__ == "__main__":
    main()
