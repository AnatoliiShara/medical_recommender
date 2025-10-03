# -*- coding: utf-8 -*-
"""
Build FAISS index for DOC-level embeddings.

Inputs:
  --docs_parquet data/processed/embeddings/<model>/docs.parquet
Outputs (in --out_dir, usually the same folder):
  - doc.index          (FAISS IndexFlatIP with normalized vectors)
  - doc_ids.npy        (np.int64 mapping: faiss_idx -> doc_id)
  - docs_meta.parquet  (doc_id, drug_name)

"""
import argparse
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa

try:
    import faiss  # type: ignore
except Exception as e:
    raise RuntimeError("FAISS is required. Please install faiss-cpu or faiss-gpu.") from e

def main():
    ap = argparse.ArgumentParser("Build DOC-level FAISS index")
    ap.add_argument("--docs_parquet", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Reading docs parquet: {args.docs_parquet}")
    # read columns separately (arrow)
    tbl = pq.read_table(args.docs_parquet, columns=["doc_id", "drug_name", "embedding_doc"])
    n = tbl.num_rows
    print(f"[INFO] Rows: {n:,}")

    # extract embeddings into contiguous float32 array
    emb_col = tbl.column("embedding_doc")
    # first row to get dim
    sample = np.array(emb_col[0].as_py(), dtype="float32")
    dim = int(sample.shape[0])
    X = np.empty((n, dim), dtype="float32")
    for i in range(n):
        X[i] = np.array(emb_col[i].as_py(), dtype="float32")

    # FAISS index (cosine via IP on normalized vectors; we assume they are normalized already)
    index = faiss.IndexFlatIP(dim)
    index.add(X)
    faiss_path = out_dir / "doc.index"
    faiss.write_index(index, str(faiss_path))
    print(f"[OK] FAISS saved: {faiss_path} (n={n}, dim={dim})")

    # doc_id map
    doc_ids = np.array(tbl.column("doc_id").to_numpy(), dtype="int64").reshape(-1)
    npy_path = out_dir / "doc_ids.npy"
    np.save(npy_path, doc_ids)
    print(f"[OK] doc_ids.npy saved: {npy_path}")

    # meta (lightweight)
    meta_tbl = pa.table({
        "doc_id": tbl.column("doc_id"),
        "drug_name": tbl.column("drug_name"),
    })
    meta_path = out_dir / "docs_meta.parquet"
    pq.write_table(meta_tbl, meta_path, compression="zstd", version="2.6")
    print(f"[OK] docs_meta.parquet saved: {meta_path}")

if __name__ == "__main__":
    main()
