#!/usr/bin/env python3
import argparse, random, json, math
from pathlib import Path
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faiss", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--docs_meta", required=True)
    ap.add_argument("--text_col", default="drug_name")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    df = pd.read_parquet(args.docs_meta, columns=["doc_id", args.text_col])
    df = df.dropna()
    rows = list(zip(df["doc_id"].astype(int).tolist(), df[args.text_col].astype(str).tolist()))
    random.Random(args.seed).shuffle(rows)
    rows = rows[:min(args.n, len(rows))]

    idx = faiss.read_index(args.faiss)
    model = SentenceTransformer(args.model, device=args.device)

    prefixes = ["", "query: ", "passage: "]
    maxK = 10

    print("index ntotal=", idx.ntotal, "dim=", idx.d, "metric=", idx.metric_type)
    print("sample size =", len(rows), "text_col=", args.text_col)

    for pref in prefixes:
        hit1=hit5=hit10=0
        for did, text in rows:
            q = pref + text
            v = model.encode([q], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
            D,I = idx.search(v, maxK)
            top = I[0].tolist()
            if did == top[0]: hit1 += 1
            if did in top[:5]: hit5 += 1
            if did in top[:10]: hit10 += 1
        n = len(rows)
        print(f"\nPREFIX='{pref}' self-hit@1={hit1/n:.3f} self-hit@5={hit5/n:.3f} self-hit@10={hit10/n:.3f}")

if __name__ == "__main__":
    main()
