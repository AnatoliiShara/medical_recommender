#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Будує "slim" FAISS індекс: рівно 1 ембеддінг на 1 документ.
Полегшена схема: беремо змістовні поля (назва/INN/показання/фармгрупа),
нормалізуємо вектори (cosine via inner product), IndexFlatIP для стабільності на CPU.

Вихід:
  {out_dir}/faiss.index
  {out_dir}/doc_ids.npy
  {out_dir}/meta.json
"""

import argparse, json, os, sys
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import faiss  # type: ignore
except Exception as e:
    print("[ERR] faiss not available:", e)
    sys.exit(1)

from sentence_transformers import SentenceTransformer

CAND_NAME_COLS = [
    "Назва препарату", "Назва", "name", "brand", "brand_name"
]
CAND_INN_COLS = [
    "Діюча речовина", "Діюча речовина (INN)", "inn", "active_ingredient"
]
CAND_INDICATIONS_COLS = [
    "Показання", "показання", "indications", "Indications"
]
CAND_GROUP_COLS = [
    "Фармакотерапевтична група", "фармгрупа", "pharm_group", "Pharmacotherapeutic group"
]

def pick_first(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def make_doc_text(row, name_c, inn_c, ind_c, grp_c):
    bits = []
    if name_c: bits.append(str(row.get(name_c, "")))
    if inn_c:  bits.append(str(row.get(inn_c, "")))
    if ind_c:  bits.append(str(row.get(ind_c, "")))
    if grp_c:  bits.append(str(row.get(grp_c, "")))
    text = " ".join([t for t in bits if isinstance(t, str) and t.strip()])
    text = text.replace("\n", " ").replace("\r", " ").strip()
    return text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Parquet із повним датасетом")
    ap.add_argument("--out_dir", required=True, help="Куди писати faiss.index/doc_ids.npy")
    ap.add_argument("--model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max_rows", type=int, default=0, help="0=усі")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_parquet(args.dataset)
    n = len(df) if args.max_rows <= 0 else min(args.max_rows, len(df))
    df = df.iloc[:n].copy()
    print(f"[INFO] loaded rows={len(df)}")

    name_c = pick_first(df, CAND_NAME_COLS)
    inn_c  = pick_first(df, CAND_INN_COLS)
    ind_c  = pick_first(df, CAND_INDICATIONS_COLS)
    grp_c  = pick_first(df, CAND_GROUP_COLS)
    print(f"[INFO] columns: name={name_c} inn={inn_c} indications={ind_c} group={grp_c}")

    texts = [make_doc_text(df.iloc[i], name_c, inn_c, ind_c, grp_c) for i in range(len(df))]
    # Якщо рядок порожній — підставимо назву, щоб не втратити документ
    if name_c:
        for i, t in enumerate(texts):
            if not t:
                texts[i] = str(df.iloc[i][name_c])

    # E5: для документів краще prefix "passage: "
    texts = [("passage: " + t) if not t.startswith("passage: ") else t for t in texts]

    model = SentenceTransformer(args.model)
    model.max_seq_length = 256  # компактніше і достатньо для наших полів

    # Ембеддимо батчами
    embs = []
    for i in tqdm(range(0, len(texts), args.batch), total=(len(texts)+args.batch-1)//args.batch, desc="[EMB]"):
        batch = texts[i:i+args.batch]
        m = model.encode(batch, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        embs.append(m.astype("float32"))
    X = np.vstack(embs)
    assert X.shape[0] == len(df), f"emb rows {X.shape[0]} != df {len(df)}"
    d = X.shape[1]
    print(f"[INFO] emb shape={X.shape} (d={d})")

    # IndexFlatIP + нормовані ембеддинги → косинус через inner product
    index = faiss.IndexFlatIP(d)
    index.add(X)
    ntotal = index.ntotal
    print(f"[INFO] faiss.ntotal={ntotal}")

    # doc_ids — 1:1 до рядків датасету (індекси датафрейму)
    doc_ids = df.index.values
    if doc_ids.dtype != np.int64 and doc_ids.dtype != np.int32:
        # перестраховка: зробити щільні id [0..N-1]
        doc_ids = np.arange(len(df), dtype=np.int64)

    np.save(os.path.join(args.out_dir, "doc_ids.npy"), doc_ids)
    faiss.write_index(index, os.path.join(args.out_dir, "faiss.index"))
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "dataset": os.path.abspath(args.dataset),
            "rows": int(len(df)),
            "dim": int(d),
            "model": args.model,
            "normalized": True,
            "index_type": "IndexFlatIP"
        }, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote -> {args.out_dir}/faiss.index ; {args.out_dir}/doc_ids.npy")

if __name__ == "__main__":
    main()
