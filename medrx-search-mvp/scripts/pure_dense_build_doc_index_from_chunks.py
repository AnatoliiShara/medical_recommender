#!/usr/bin/env python3
"""
Build a PURE dense doc-level FAISS index by:
- reading an existing chunks.parquet (doc_id, text)
- encoding each chunk with a sentence-transformers bi-encoder
- L2-normalizing chunk embeddings
- mean-pooling chunk embeddings per doc_id
- L2-normalizing doc vectors
- writing faiss.index + doc_ids.npy (+ build_report.json)

This avoids writing chunk-embeddings to disk and guarantees a clean, reproducible doc universe
(doc_ids present in chunks.parquet).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from tqdm import tqdm

try:
    import faiss  # type: ignore
except Exception as e:
    raise SystemExit("faiss import failed. Install faiss-cpu (or faiss-gpu).") from e

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise SystemExit("sentence-transformers import failed. Is it installed in your venv?") from e


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _infer_columns(chunks_path: str, doc_id_col: Optional[str], text_col: Optional[str]) -> Tuple[str, str]:
    pf = pq.ParquetFile(chunks_path)
    cols = pf.schema.names

    if doc_id_col is None:
        for c in ("doc_id", "docid", "docID"):
            if c in cols:
                doc_id_col = c
                break
    if doc_id_col is None:
        raise SystemExit(f"Could not infer doc_id column in {chunks_path}. Columns: {cols}")

    if text_col is None:
        for c in ("text", "chunk", "passage", "content"):
            if c in cols:
                text_col = c
                break
    if text_col is None:
        raise SystemExit(f"Could not infer text column in {chunks_path}. Columns: {cols}")

    return doc_id_col, text_col


def _read_distinct_doc_ids(chunks_path: str, doc_id_col: str) -> List[str]:
    t = pq.read_table(chunks_path, columns=[doc_id_col])
    uniq = t[doc_id_col].unique().to_pylist()
    uniq = [str(x) for x in uniq if x is not None]
    uniq.sort()
    return uniq


def _iter_record_batches(chunks_path: str, columns: List[str], parquet_batch_rows: int) -> Iterable[Tuple[List[str], List[str]]]:
    dataset = ds.dataset(chunks_path, format="parquet")
    scanner = dataset.scanner(columns=columns, batch_size=parquet_batch_rows)
    for rb in scanner.to_batches():
        dcol = rb.column(0).to_pylist()
        tcol = rb.column(1).to_pylist()
        doc_ids = [str(x) if x is not None else "" for x in dcol]
        texts = [x if isinstance(x, str) else "" for x in tcol]
        yield doc_ids, texts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True, help="Path to chunks.parquet")
    ap.add_argument("--docs_meta", default=None, help="Optional docs_meta.parquet to copy+filter")
    ap.add_argument("--out", required=True, help="Output directory for the dense doc index")
    ap.add_argument("--model", required=True, help="SentenceTransformer model name/path")
    ap.add_argument("--device", default="cpu", help="cpu / cuda / mps ...")
    ap.add_argument("--doc_id_col", default=None)
    ap.add_argument("--text_col", default=None)
    ap.add_argument("--passage_prefix", default="passage: ", help='E5-style doc prefix. Use "" to disable.')
    ap.add_argument("--batch_size", type=int, default=128, help="ST encode() batch size")
    ap.add_argument("--parquet_batch_rows", type=int, default=2048, help="Rows read per Arrow batch")
    ap.add_argument("--max_seq_length", type=int, default=512)
    ap.add_argument("--limit_chunks", type=int, default=0, help="Debug: stop after N chunks (0 = all)")
    args = ap.parse_args()

    _ensure_dir(args.out)

    doc_id_col, text_col = _infer_columns(args.chunks, args.doc_id_col, args.text_col)

    pf = pq.ParquetFile(args.chunks)
    n_chunks = pf.metadata.num_rows
    print(f"[INFO] chunks: {args.chunks}")
    print(f"[INFO] chunks rows: {n_chunks:,} | columns: {pf.schema.names}")
    print(f"[INFO] using columns: doc_id={doc_id_col} text={text_col}")

    doc_ids = _read_distinct_doc_ids(args.chunks, doc_id_col)
    n_docs = len(doc_ids)
    print(f"[INFO] doc_ids in chunks: {n_docs:,}")

    doc2i: Dict[str, int] = {d: i for i, d in enumerate(doc_ids)}

    print(f"[INFO] loading model: {args.model} (device={args.device})")
    model = SentenceTransformer(args.model, device=args.device)
    model.max_seq_length = args.max_seq_length

    probe = model.encode([args.passage_prefix + "probe"], batch_size=1, convert_to_numpy=True, normalize_embeddings=False)
    dim = int(probe.shape[1])
    print(f"[INFO] embedding dim: {dim}")

    sums = np.zeros((n_docs, dim), dtype=np.float32)
    counts = np.zeros((n_docs,), dtype=np.int32)

    total_seen = 0
    empty_text = 0
    unknown_doc = 0

    target_total = n_chunks if args.limit_chunks <= 0 else min(n_chunks, args.limit_chunks)
    pbar = tqdm(total=target_total, desc="encode+pool", unit="chunk")

    for batch_doc_ids, batch_texts in _iter_record_batches(args.chunks, [doc_id_col, text_col], args.parquet_batch_rows):
        if args.limit_chunks > 0 and total_seen >= args.limit_chunks:
            break

        if args.limit_chunks > 0:
            take = min(len(batch_doc_ids), args.limit_chunks - total_seen)
            batch_doc_ids = batch_doc_ids[:take]
            batch_texts = batch_texts[:take]

        texts: List[str] = []
        idxs: List[int] = []
        for d, t in zip(batch_doc_ids, batch_texts):
            total_seen += 1
            if not t or not t.strip():
                empty_text += 1
                pbar.update(1)
                continue
            i = doc2i.get(d)
            if i is None:
                unknown_doc += 1
                pbar.update(1)
                continue
            idxs.append(i)
            texts.append(args.passage_prefix + t)
            pbar.update(1)

        if not texts:
            continue

        emb = model.encode(
            texts,
            batch_size=args.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        ).astype(np.float32, copy=False)

        # normalize per-chunk to equalize weights
        faiss.normalize_L2(emb)

        idx_arr = np.asarray(idxs, dtype=np.int32)
        np.add.at(sums, idx_arr, emb)
        np.add.at(counts, idx_arr, 1)

    pbar.close()

    print(f"[INFO] seen chunks: {total_seen:,} | empty_text_skipped: {empty_text:,} | unknown_doc_skipped: {unknown_doc:,}")
    zero_docs = int(np.sum(counts == 0))
    print(f"[INFO] docs with 0 chunks (after skip empties): {zero_docs:,} / {n_docs:,}")

    safe_counts = np.maximum(counts, 1).astype(np.float32)
    doc_vecs = (sums / safe_counts[:, None]).astype(np.float32, copy=False)
    faiss.normalize_L2(doc_vecs)

    norms = np.linalg.norm(doc_vecs, axis=1)
    zero_vecs = int(np.sum(norms == 0.0))
    print(f"[INFO] zero vectors after pooling+norm: {zero_vecs:,} / {n_docs:,}")

    out_index = os.path.join(args.out, "faiss.index")
    out_ids = os.path.join(args.out, "doc_ids.npy")

    index = faiss.IndexFlatIP(dim)
    index.add(doc_vecs)
    faiss.write_index(index, out_index)
    np.save(out_ids, np.asarray(doc_ids, dtype=object), allow_pickle=True)

    print(f"[OK] wrote: {out_index}")
    print(f"[OK] wrote: {out_ids} (ntotal={index.ntotal:,})")

    # Optional: filtered docs_meta copy for convenience (small, safe to do via pandas)
    if args.docs_meta and os.path.exists(args.docs_meta):
        try:
            import pandas as pd
            tmeta = pq.read_table(args.docs_meta)
            df = tmeta.to_pandas()
            col = "doc_id" if "doc_id" in df.columns else df.columns[0]
            keep = set(doc_ids)
            df[col] = df[col].astype(str)
            df2 = df[df[col].isin(keep)]
            pq.write_table(pq.Table.from_pandas(df2, preserve_index=False), os.path.join(args.out, "docs_meta.parquet"))
            print(f"[OK] wrote: {os.path.join(args.out,'docs_meta.parquet')} rows={len(df2):,}")
        except Exception as e:
            print(f"[WARN] failed to copy/filter docs_meta: {type(e).__name__}: {e}", file=sys.stderr)

    report = {
        "chunks_path": args.chunks,
        "docs_meta": args.docs_meta,
        "model": args.model,
        "device": args.device,
        "doc_id_col": doc_id_col,
        "text_col": text_col,
        "passage_prefix": args.passage_prefix,
        "n_chunks_reported": int(n_chunks),
        "n_chunks_seen": int(total_seen),
        "n_docs": int(n_docs),
        "embedding_dim": int(dim),
        "empty_text_skipped": int(empty_text),
        "unknown_doc_skipped": int(unknown_doc),
        "docs_with_zero_chunks_after_skip": int(zero_docs),
        "zero_vectors_after_norm": int(zero_vecs),
        "vector_norm_min": float(norms.min()) if norms.size else None,
        "vector_norm_max": float(norms.max()) if norms.size else None,
    }
    with open(os.path.join(args.out, "build_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[OK] wrote: {os.path.join(args.out,'build_report.json')}")


if __name__ == "__main__":
    main()
