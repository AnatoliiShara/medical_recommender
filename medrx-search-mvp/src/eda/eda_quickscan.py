#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
import textwrap
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearn.feature_extraction.text import CountVectorizer
except Exception:
    CountVectorizer = None

try:
    from dotenv import load_dotenv
except Exception:
    # не критично, просто не підхопимо .env
    def load_dotenv(*args, **kwargs):
        return False


UA_STOPWORDS = set("""
і й та але або що щоб чи як коли також уже вжеж такожже також-то бо ж то це ця цей ці цих цими цих
у в на до при по над під між з зі із від для без про надто дуже тільки той такий така такі таких
або/чи або-або через також-бо однак проте адже тобто наприклад наприклад, тобто, тощо тому тому-то
""".split())

# Колонки з твого скріншоту (підтримує варіанти)
CANDIDATE_TEXT_COLS = [
    "Назва препарату", "Показання", "Протипоказання", "Фармакотерапевтична група",
    "Склад", "Лікарська форма", "Побічні реакції", "Спосіб застосування та дозування",
    "Спосіб застосування та до...", "Спосіб застосування та д...",  # на випадок обрізаних заголовків
    "Фармакологічні властивості", "Застосування у період вагітності або годув...",
    "Здатність впливати на швидкість реакції при керуванні автотранспортом або іншими механізмами",
    "Виробник", "Умови зберігання", "Термін придатності", "Упаковка", "url",
]

PHRASE_SPLIT_RE = re.compile(r"[.;•\n\r—\-]+")  # розділювачі для фраз у показаннях/протипоказаннях
HTML_TAG_RE = re.compile(r"<[^>]+>")
MULTISPACE_RE = re.compile(r"\s+")


def clean_text(x: str) -> str:
    if not isinstance(x, str):
        return ""
    x = HTML_TAG_RE.sub(" ", x)
    x = x.replace("\xa0", " ")
    x = MULTISPACE_RE.sub(" ", x).strip()
    return x


def normalize_lower(x: str) -> str:
    return clean_text(x).lower()


def safe_col(df: pd.DataFrame, name_variants):
    for c in name_variants if isinstance(name_variants, list) else [name_variants]:
        if c in df.columns:
            return c
    return None


def top_phrases(series: pd.Series, top_k=200) -> pd.DataFrame:
    phrases = []
    for txt in series.fillna(""):
        txt = normalize_lower(txt)
        if not txt:
            continue
        for ph in PHRASE_SPLIT_RE.split(txt):
            ph = ph.strip(" :;,-–.()[]{}")
            if len(ph) < 3:
                continue
            phrases.append(ph)
    cnt = Counter(phrases)
    rows = [{"phrase": p, "count": n} for p, n in cnt.most_common(top_k)]
    return pd.DataFrame(rows)


def lengths_stats(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    stats = []
    for c in cols:
        if c not in df.columns:
            continue
        lens = df[c].fillna("").map(lambda s: len(clean_text(str(s))))
        if lens.empty:
            continue
        stats.append({
            "column": c,
            "count": int(lens.shape[0]),
            "non_null": int(df[c].notna().sum()),
            "nulls": int(df[c].isna().sum()),
            "min": int(lens.min()),
            "mean": float(lens.mean()),
            "median": float(lens.median()),
            "p95": float(np.percentile(lens, 95)),
            "p99": float(np.percentile(lens, 99)),
            "max": int(lens.max()),
        })
    return pd.DataFrame(stats).sort_values("p95", ascending=False)


def ngrams_top(series: pd.Series, ngram_range=(1,3), top_k=200):
    if CountVectorizer is None:
        return pd.DataFrame(columns=["ngram", "freq"])

    docs = series.fillna("").map(normalize_lower).tolist()
    # простий токенайзер: слова з букв і цифр, довше 2-х
    token_pattern = r"(?u)\b[а-щьюяєіїґa-z0-9][а-щьюяєіїґa-z0-9\-]{1,}\b"
    cv = CountVectorizer(
        analyzer="word",
        token_pattern=token_pattern,
        ngram_range=ngram_range,
        min_df=5,
        stop_words=list(UA_STOPWORDS)
    )
    try:
        X = cv.fit_transform(docs)
    except ValueError:
        return pd.DataFrame(columns=["ngram", "freq"])

    freqs = np.asarray(X.sum(axis=0)).ravel()
    vocab = np.array(sorted(cv.vocabulary_.items(), key=lambda kv: kv[1]))[:,0]
    order = np.argsort(freqs)[::-1][:top_k]
    return pd.DataFrame({"ngram": vocab[order], "freq": freqs[order]})


def probe_diagnoses(df: pd.DataFrame, indications_col: str, names_col: str, probes: list) -> pd.DataFrame:
    rows = []
    ind = df[indications_col].fillna("").map(normalize_lower)
    for diag in probes:
        d = normalize_lower(diag)
        mask = ind.str.contains(re.escape(d))
        hits = df.loc[mask]
        rows.append({
            "probe": diag,
            "hits": int(mask.sum()),
            "example_names": "; ".join(hits[names_col].astype(str).head(15).tolist())
        })
    return pd.DataFrame(rows)


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Quick EDA for Compendium parquet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Приклад:
              python src/eda/eda_quickscan.py --dataset "/path/compendium_all.parquet" --probe "хвороба крона, стенокардія"
        """)
    )
    parser.add_argument("--dataset", type=str, default=os.getenv("DATASET_PATH", "data/raw/compendium_all.parquet"))
    parser.add_argument("--outdir", type=str, default="data/interim/eda")
    parser.add_argument("--probe", type=str, default="")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading parquet: {args.dataset}")
    df = pd.read_parquet(args.dataset)
    print(f"[INFO] Shape: {df.shape}; columns: {len(df.columns)}")
    print(f"[INFO] Columns: {list(df.columns)}\n")

    # виберемо корисні колонки, які справді існують
    cols_present = [c for c in CANDIDATE_TEXT_COLS if c in df.columns]
    if not cols_present:
        print("[WARN] Жодна з очікуваних колонок не знайдена. Збережу лише загальний опис і вийду.")
        (outdir / "columns_present.txt").write_text("\n".join(df.columns), encoding="utf-8")
        return

    # Санітарка: nulls
    nulls = df[cols_present].isna().sum().reset_index()
    nulls.columns = ["column", "nulls"]
    nulls["rows"] = df.shape[0]
    nulls["null_rate"] = (nulls["nulls"] / nulls["rows"]).round(4)
    nulls.to_csv(outdir / "nulls.csv", index=False)

    # Дублікати по Назві + Лікарська форма (як є)
    name_col = safe_col(df, "Назва препарату")
    form_col = safe_col(df, "Лікарська форма")
    if name_col:
        dup_cols = [name_col] + ([form_col] if form_col else [])
        dup_counts = df.groupby(dup_cols).size().reset_index(name="n")
        dup_counts = dup_counts[dup_counts["n"] > 1].sort_values("n", ascending=False)
        dup_counts.to_csv(outdir / "dup_counts.csv", index=False)
    else:
        dup_counts = pd.DataFrame()

    # Довжини тексту
    lens = lengths_stats(df, cols_present)
    lens.to_csv(outdir / "lengths.csv", index=False)

    # Топ фраз з Показань/Протипоказань
    ind_col = safe_col(df, "Показання")
    contra_col = safe_col(df, "Протипоказання")
    if ind_col:
        ind_ph = top_phrases(df[ind_col], top_k=400)
        ind_ph.to_csv(outdir / "indications_top_phrases.csv", index=False)
    else:
        ind_ph = pd.DataFrame()

    if contra_col:
        con_ph = top_phrases(df[contra_col], top_k=400)
        con_ph.to_csv(outdir / "contraindications_top_phrases.csv", index=False)
    else:
        con_ph = pd.DataFrame()

    # N-грами (якщо доступний sklearn)
    if CountVectorizer is not None and ind_col:
        ngrams_ind = ngrams_top(df[ind_col], (1,3), top_k=300)
        ngrams_ind.to_csv(outdir / "indications_ngrams.csv", index=False)
    if CountVectorizer is not None and contra_col:
        ngrams_con = ngrams_top(df[contra_col], (1,3), top_k=300)
        ngrams_con.to_csv(outdir / "contraindications_ngrams.csv", index=False)

    # Проби діагнозів
    if args.probe.strip() and ind_col and name_col:
        probes = [p.strip() for p in args.probe.split(",") if p.strip()]
        probe_df = probe_diagnoses(df, ind_col, name_col, probes)
        probe_df.to_csv(outdir / "probe_hits.csv", index=False)
    else:
        probe_df = pd.DataFrame()

    # Консольний підсумок
    print("\n===== SUMMARY =====")
    print(f"Rows: {df.shape[0]}")
    print("\n[Nulls per column] (top 10):")
    print(nulls.sort_values("null_rate", ascending=False).head(10).to_string(index=False))

    if not lens.empty:
        print("\n[Text length stats] (top by p95):")
        print(lens.head(8).to_string(index=False))

    if not dup_counts.empty:
        print("\n[Top duplicates by Name(+Form)]:")
        print(dup_counts.head(10).to_string(index=False))

    if not ind_ph.empty:
        print("\n[Top indication phrases]:")
        print(ind_ph.head(20).to_string(index=False))

    if not con_ph.empty:
        print("\n[Top contraindication phrases]:")
        print(con_ph.head(20).to_string(index=False))

    if not probe_df.empty:
        print("\n[Probe hits]:")
        print(probe_df.to_string(index=False))

    print(f"\n[OK] Artifacts saved to: {outdir.resolve()}")
    print("Files: nulls.csv, lengths.csv, dup_counts.csv, indications_top_phrases.csv, contraindications_top_phrases.csv, "
          "indications_ngrams.csv, contraindications_ngrams.csv, probe_hits.csv")


if __name__ == "__main__":
    main()
