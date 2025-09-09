#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, re, argparse
from pathlib import Path
import pandas as pd, numpy as np

print("[BOOT] starting eda_deep_diagnostics")

HTML_TAG_RE = re.compile(r"<[^>]+>")
MULTISPACE_RE = re.compile(r"\s+")
TRADEMARK_RE = re.compile(r"[®™©]+")
CLEAN_CROSSREF_PATTERNS = [
    r"див(\.|іться)?\s+розділ[^.;\n]*",
    r"розділ\s+«[^»]+»",
    r"особливості застосування",
    r"особливі заходи безпеки",
    r"спосіб застосування( та доз[иів]*)?",
    r"протипоказання",
    r"побічні реакції",
    r"передозування",
    r"фармакологічні властивості",
    r"взаємодія з іншими лікарськими засобами та інші види взаємодій",
    r"застосування у період вагітності[^.;\n]*",
    r"здатність впливати на швидкість реакції[^.;\n]*",
]
FOOD_CONTAM_PATTERNS = [
    r"\bвуглевод[иів]\b", r"\bжири?\b", r"\bбілк[иів]\b", r"\bкалорійн[аості]\b", r"\b\d+[.,]\d+\s*(г|ккал)\b"
]
TRIGGERS = {
    "dietary":[r"дієтичн[аії] добавк", r"харчов(ий|і) продукт", r"біологічно активн", r"\bбад\b", r"вітамінн(ий|а) комплекс", r"добавк[аи] до їж"],
    "cosmetic":[r"косметичн", r"шампун", r"бальзам( для губ|)", r"лосьйон", r"крем(?!\s*від)", r"гель для душ", r"маска для волос", r"тонік для облич", r"гігієнічн", r"скраб", r"пінка"],
    "device":[r"медичн(ий|і) виріб", r"виріб медичн(ого|ого призначення)", r"термометр", r"пульсоксиметр", r"катетер", r"шприц", r"рукавичк", r"халат", r"маска (?:медичн|для обличчя)", r"пластир (?:катушков|рулон)"],
}

def clean_text(s:str)->str:
    if not isinstance(s,str): return ""
    s = HTML_TAG_RE.sub(" ", s)
    s = TRADEMARK_RE.sub("", s).replace("\xa0"," ")
    return MULTISPACE_RE.sub(" ", s).strip()

def remove_patterns(s:str, patterns):
    t = s
    for pat in patterns: t = re.sub(pat, " ", t, flags=re.IGNORECASE)
    return MULTISPACE_RE.sub(" ", t).strip(" ;,.:—-")

def detect_product_type(name, group):
    text = f"{name} {group}".lower()
    def any_match(pats): return any(re.search(p, text) for p in pats)
    if any_match(TRIGGERS["device"]):   return "device"
    if any_match(TRIGGERS["cosmetic"]): return "cosmetic"
    if any_match(TRIGGERS["dietary"]):  return "dietary_supplement"
    return "medicine"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--outdir", default="data/interim/eda2")
    ap.add_argument("--probes", default="")
    args = ap.parse_args()

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] reading parquet: {args.dataset}")
    df = pd.read_parquet(args.dataset)
    print(f"[INFO] shape={df.shape}")

    COL = {"name":"Назва препарату","form":"Лікарська форма","group":"Фармакотерапевтична група",
           "ind":"Показання","con":"Протипоказання","url":"url"}
    for v in COL.values():
        if v not in df.columns: raise SystemExit(f"[ERR] column missing: {v}")

    # nulls (empty strings -> NaN)
    work = df.copy()
    for c in work.columns:
        work[c] = work[c].apply(lambda x: np.nan if (isinstance(x,str) and x.strip()=="") or (not isinstance(x,str) and pd.isna(x)) else x)
    nulls = work.isna().sum().reset_index().rename(columns={"index":"column",0:"nulls"})
    nulls["rows"]=len(work); nulls["null_rate"]=(nulls["nulls"]/nulls["rows"]).round(4)
    nulls.to_csv(out/"nulls_true.csv", index=False); print("[OK] nulls_true.csv")

    # clean fields
    for c in [COL["name"],COL["form"],COL["group"],COL["ind"],COL["con"]]:
        df[c]=df[c].map(clean_text)

    # product types
    df["product_type"] = [detect_product_type(n,g) for n,g in zip(df[COL["name"]], df[COL["group"]])]
    df[["product_type"]].value_counts().rename("count").reset_index().to_csv(out/"product_type_counts.csv", index=False)
    print("[OK] product_type_counts.csv")

    # clean indications/contraindications
    df["Показання_clean"] = df[COL["ind"]].map(lambda s: remove_patterns(s, CLEAN_CROSSREF_PATTERNS))
    df["Протипоказання_clean"] = df[COL["con"]].map(lambda s: remove_patterns(s, CLEAN_CROSSREF_PATTERNS))
    df["Протипоказання_clean"] = df["Протипоказання_clean"].map(lambda s: remove_patterns(s, FOOD_CONTAM_PATTERNS))
    df[[COL["name"],"Показання_clean"]].head(500).to_csv(out/"indications_clean_sample.csv", index=False)
    df[[COL["name"],"Протипоказання_clean"]].head(500).to_csv(out/"contraind_clean_sample.csv", index=False)
    print("[OK] saved clean samples")

    # probes (regex), only medicines
    meds = df[df["product_type"]=="medicine"].copy()
    rows=[]
    if args.probes.strip():
        for expr in [p.strip() for p in args.probes.split(";") if p.strip()]:
            pat = re.compile(expr, flags=re.IGNORECASE)
            mask = meds["Показання_clean"].str.contains(pat, na=False)
            hits = meds.loc[mask, [COL["name"],COL["form"],COL["group"],"Показання_clean"]]
            rows.append({"probe_regex":expr,"hits":int(mask.sum())})
            hits.head(30).to_csv(out/f"probe_samples_{re.sub(r'[^a-zA-Zа-яА-Я0-9]+','_',expr)}.csv", index=False)
    pd.DataFrame(rows or []).to_csv(out/"probe_hits_clean.csv", index=False)
    print("[OK] probe files")

    print(f"[DONE] artifacts in {out.resolve()}")

if __name__ == "__main__":
    main()
