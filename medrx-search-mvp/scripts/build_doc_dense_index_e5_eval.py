#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.compute as pc

import faiss  # type: ignore
from sentence_transformers import SentenceTransformer


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs_parquet", required=True, help="docs.parquet with doc_id + text_used")
    ap.add_argument("--model", required=True, help="HF model id or local path")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--text_col", default="text_used")
    ap.add_argument("--name_col", default="drug_name")
    ap.add_argument("--doc_id_col", default="doc_id")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--doc_prefix", default="passage: ", help="E5 style prefix; keep same for baseline+FT")
    ap.add_argument("--no_prefix", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Reading docs_parquet: {args.docs_parquet}")
    pf = pq.ParquetFile(args.docs_parquet)
    print(f"[INFO] Rows: {pf.metadata.num_rows}")
    print(f"[INFO] Columns: {pf.schema.names}")

    need = [args.doc_id_col, args.text_col]
    if args.name_col:
        need.append(args.name_col)
    for c in need:
        if c not in pf.schema.names:
            raise ValueError(f"Очікую стовпець '{c}' у docs_parquet, але його нема. Є: {pf.schema.names}")

    tbl = pq.read_table(args.docs_parquet, columns=need)
    # basic text sanity
    txt = tbl[args.text_col]
    is_empty = pc.or_(pc.is_null(txt), pc.equal(pc.utf8_length(pc.utf8_trim_whitespace(txt)), 0))
    empty_cnt = int(pc.sum(is_empty).as_py())
    if empty_cnt:
        print(f"[WARN] Empty texts in {args.text_col}: {empty_cnt}")

    doc_ids = tbl[args.doc_id_col].to_pylist()
    texts = txt.to_pylist()
    names = tbl[args.name_col].to_pylist() if args.name_col in tbl.column_names else None

    # stringify ids to avoid type mismatches later
    doc_ids = [str(x) for x in doc_ids]
    # replace None with ""
    texts = [("" if t is None else str(t)) for t in texts]

    model = SentenceTransformer(args.model, device=args.device)
    try:
        model.max_seq_length = args.max_length
    except Exception:
        pass
    print(f"[INFO] Model loaded: {args.model} | device={args.device} | max_length={args.max_length}")

    embs = []
    bs = args.batch_size
    prefix = "" if args.no_prefix else (args.doc_prefix or "")
    n = len(texts)

    for i in range(0, n, bs):
        batch = texts[i:i+bs]
        if prefix:
            batch = [prefix + t for t in batch]
        # normalize ourselves to be deterministic across ST versions
        e = model.encode(
            batch,
            batch_size=bs,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype("float32")
        e = l2_normalize(e)
        embs.append(e)
        if (i // bs) % 20 == 0:
            print(f"[INFO] Encoded {min(i+bs, n)}/{n}")

    X = np.vstack(embs).astype("float32")
    dim = X.shape[1]
    print(f"[INFO] Embeddings: shape={X.shape}, dim={dim}, norms~{X[:5].sum():.3f} (sanity)")

    # Build FAISS IP on normalized vectors => cosine similarity
    index = faiss.IndexFlatIP(dim)
    index.add(X)
    print(f"[INFO] FAISS: ntotal={index.ntotal}, dim={dim}")

    # Persist artifacts (order is important!)
    np.save(out_dir / "doc_ids.npy", np.array(doc_ids, dtype=object), allow_pickle=True)
    faiss.write_index(index, str(out_dir / "faiss.index"))

    if names is not None:
        meta = pa.table({
            "doc_id": doc_ids,
            "drug_name": [("" if x is None else str(x)) for x in names],
        })
    else:
        meta = pa.table({"doc_id": doc_ids})
    pq.write_table(meta, out_dir / "docs_meta.parquet")

    with open(out_dir / "build_info.json", "w", encoding="utf-8") as f:
        json.dump({
            "docs_parquet": args.docs_parquet,
            "model": args.model,
            "device": args.device,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "doc_prefix": prefix,
            "rows": int(n),
            "dim": int(dim),
        }, f, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote: {out_dir/'faiss.index'}")
    print(f"[OK] Wrote: {out_dir/'doc_ids.npy'}")
    print(f"[OK] Wrote: {out_dir/'docs_meta.parquet'}")


if __name__ == "__main__":
    main()
