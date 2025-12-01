#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np

def read_doc_ids_jsonl(path: Path):
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                j = json.loads(line)
                out.append(str(j.get("doc_id")))
    return out

def iter_chunk_doc_ids(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                j = json.loads(line)
                yield str(j.get("doc_id"))

def l2_normalize(x: np.ndarray, eps: float = 1e-12):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return x / n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk_index_dir", type=Path, required=True)
    ap.add_argument("--base_doc_ids", type=Path, required=True)
    ap.add_argument("--out_index_dir", type=Path, required=True)
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--block", type=int, default=50000)
    args = ap.parse_args()

    import faiss
    from tqdm import tqdm

    chunk_dir = args.chunk_index_dir
    emb_path = chunk_dir / "embeddings.f32"
    chunk_docids_path = chunk_dir / "doc_ids.jsonl"

    if not emb_path.exists():
        raise FileNotFoundError(f"missing {emb_path}")
    if not chunk_docids_path.exists():
        raise FileNotFoundError(f"missing {chunk_docids_path}")
    if not args.base_doc_ids.exists():
        raise FileNotFoundError(f"missing {args.base_doc_ids}")

    out_dir = args.out_index_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) baseline doc order
    base_doc_ids = read_doc_ids_jsonl(args.base_doc_ids)
    ndocs = len(base_doc_ids)
    doc2i = {d:i for i,d in enumerate(base_doc_ids)}

    # 2) chunk doc ids -> doc indices
    chunk_doc_ids = list(iter_chunk_doc_ids(chunk_docids_path))
    nchunks = len(chunk_doc_ids)

    doc_idx = np.full((nchunks,), -1, dtype=np.int32)
    missing = 0
    for i, did in enumerate(chunk_doc_ids):
        j = doc2i.get(did, -1)
        doc_idx[i] = j
        if j < 0:
            missing += 1

    # 3) memmap embeddings
    dim = args.dim
    expect = nchunks * dim
    got = emb_path.stat().st_size // 4
    if got != expect:
        raise ValueError(f"embeddings.f32 size mismatch: got {got} floats, expected {expect}")

    emb = np.memmap(emb_path, dtype=np.float32, mode="r", shape=(nchunks, dim))

    sums = np.zeros((ndocs, dim), dtype=np.float32)
    cnts = np.zeros((ndocs,), dtype=np.int32)

    block = args.block
    for s in tqdm(range(0, nchunks, block), desc="pool chunks -> docs"):
        e = min(s + block, nchunks)
        idx = doc_idx[s:e]
        m = idx >= 0
        if not np.any(m):
            continue
        idx = idx[m]
        x = np.asarray(emb[s:e])[m]
        np.add.at(sums, idx, x)
        cnts += np.bincount(idx, minlength=ndocs).astype(np.int32)

    nonzero = cnts > 0
    doc_emb = np.zeros_like(sums, dtype=np.float32)
    doc_emb[nonzero] = sums[nonzero] / cnts[nonzero, None]
    doc_emb = l2_normalize(doc_emb)

    index = faiss.IndexFlatIP(dim)
    index.add(doc_emb)

    faiss_path = out_dir / "faiss.index"
    faiss.write_index(index, str(faiss_path))

    out_doc_ids = out_dir / "doc_ids.jsonl"
    with out_doc_ids.open("w", encoding="utf-8") as f:
        for i, did in enumerate(base_doc_ids):
            f.write(json.dumps({"row_id": i, "doc_id": did}, ensure_ascii=False) + "\n")

    meta = {
        "built_from_chunk_index_dir": str(chunk_dir),
        "base_doc_ids": str(args.base_doc_ids),
        "nchunks": int(nchunks),
        "ndocs": int(ndocs),
        "missing_doc_ids_in_base_order": int(missing),
        "dim": int(dim),
        "pooling": "mean_then_l2norm",
        "faiss_type": "IndexFlatIP",
        "normalized": True,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] wrote {faiss_path} ntotal={index.ntotal} dim={dim}")
    print(f"[OK] wrote {out_doc_ids}")
    print(f"[OK] wrote {out_dir/'meta.json'}")

if __name__ == "__main__":
    main()
