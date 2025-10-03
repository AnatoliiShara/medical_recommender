# -*- coding: utf-8 -*-
"""
Export DOC-level embeddings -> Parquet (with checkpoints & robust merge)
Output: data/processed/embeddings/{model_slug}/docs.parquet
Schema:
  - doc_id: int64 (row index у parquet-джерелі)
  - drug_name: str
  - text_used: str (який конкат текст інкапсулювали у embedding)
  - embedding_doc: list<float32> (нормалізований)
  - norm: float32
"""
import os, re, gc, json, math, argparse
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

SECTIONS_DEFAULT = [
    "Фармакотерапевтична група",
    "Фармакологічні властивості", 
    "Показання",
    "Протипоказання",
    "Взаємодія з іншими лікарськими засобами та інші види взаємодій",
    "Особливості застосування",
    "Застосування у період вагітності або годування груддю",
    "Здатність впливати на швидкість реакції при керування автотранспортом або іншими механізмами",
    "Спосіб застосування та дози",
    "Передозування",
    "Побічні реакції",
]

def model_to_slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", s.strip()).strip("-").lower()

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def arrow_schema() -> pa.schema:
    return pa.schema([
        pa.field("doc_id", pa.int64()),
        pa.field("drug_name", pa.string()),
        pa.field("text_used", pa.string()),
        pa.field("embedding_doc", pa.list_(pa.float32())),
        pa.field("norm", pa.float32()),
    ])

def concat_doc_text(row: pd.Series, sections: List[str], max_chars: int) -> str:
    name = str(row.get("Назва препарату", "") or "").strip()
    blocks = []
    if name:
        blocks.append(f"Назва: {name}")
    for sec in sections:
        v = str(row.get(sec, "") or "").strip()
        if v:
            blocks.append(f"{sec}: {v}")
    txt = " [SEP] ".join(blocks)
    if len(txt) > max_chars:
        txt = txt[:max_chars] + "..."
    return txt

def write_part(records: List[Dict[str, Any]], out_part: Path) -> None:
    if not records:
        return
    tbl = pa.table({
        "doc_id": pa.array([r["doc_id"] for r in records], type=pa.int64()),
        "drug_name": pa.array([r["drug_name"] for r in records], type=pa.string()),
        "text_used": pa.array([r["text_used"] for r in records], type=pa.string()),
        "embedding_doc": pa.array([r["embedding_doc"] for r in records], type=pa.list_(pa.float32())),
        "norm": pa.array([r["norm"] for r in records], type=pa.float32()),
    }, schema=arrow_schema())
    pq.write_table(tbl, out_part, compression="zstd", version="2.6")

def merge_parts(parts_dir: Path, final_path: Path) -> None:
    files = sorted(parts_dir.glob("part_*.parquet"))
    if not files:
        # create empty file
        empty = pa.table({n: pa.array([], type=f.type) for n, f in zip(arrow_schema().names, arrow_schema())}, schema=arrow_schema())
        pq.write_table(empty, final_path, compression="zstd", version="2.6")
        return
    tables = []
    for f in tqdm(files, desc="Merging parts", unit="file", leave=False):
        tables.append(pq.read_table(f))
    merged = pa.concat_tables(tables, promote=True)
    pq.write_table(merged, final_path, compression="zstd", version="2.6")
    for f in files:
        try: f.unlink()
        except: pass
    try: parts_dir.rmdir()
    except: pass

def main():
    ap = argparse.ArgumentParser("DOC-level embeddings → Parquet")
    ap.add_argument("--parquet", required=True, help="compendium_all.parquet")
    ap.add_argument("--out_base", default="data/processed/embeddings")
    ap.add_argument("--model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--rows", type=int, default=0)
    ap.add_argument("--sections", default=None, help="comma-separated")
    ap.add_argument("--max_chars", type=int, default=8000, help="truncate doc text to at most N chars")
    ap.add_argument("--batch_docs", type=int, default=512)
    ap.add_argument("--encode_bs", type=int, default=128)
    ap.add_argument("--workers", type=int, default=0, help="(kept for symmetry; encode() single-process)")
    ap.add_argument("--force_restart", action="store_true")
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    if args.rows > 0:
        df = df.head(args.rows)
    df = df.reset_index(drop=True)
    df["__doc_id__"] = df.index.astype("int64")

    sections = [s.strip() for s in (args.sections.split(",") if args.sections else SECTIONS_DEFAULT) if s.strip()]

    model_slug = model_to_slug(args.model)
    out_dir = Path(args.out_base) / model_slug
    parts_dir = out_dir / "docs_tmp"
    final_path = out_dir / "docs.parquet"
    ckpt_path = out_dir / "docs.ckpt.json"
    ensure_dir(out_dir); ensure_dir(parts_dir)

    ckpt = {"done_ids": [], "next_part_id": 0}
    if ckpt_path.exists() and not args.force_restart:
        try:
            ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
            print(f"[INFO] Resuming: {len(ckpt['done_ids'])} docs already done")
        except Exception as e:
            print(f"[WARN] Failed to load checkpoint: {e}")

    done = set(ckpt.get("done_ids", []))
    part_id = int(ckpt.get("next_part_id", 0))

    print(f"[INFO] Loading encoder: {args.model}")
    enc = SentenceTransformer(args.model)
    dim = enc.get_sentence_embedding_dimension()
    # set max_seq_length defensively (ST will truncate)
    try:
        enc.max_seq_length = 512
    except Exception:
        pass
    print(f"[INFO] Model dim={dim}")

    todo = df[~df["__doc_id__"].isin(done)]
    total = len(todo)
    print(f"[INFO] Docs to process: {total}")

    pbar = tqdm(total=total, desc="Docs", unit="doc")
    if done:
        pbar.update(len(done))

    for start in range(0, total, args.batch_docs):
        batch = todo.iloc[start:start + args.batch_docs]
        if batch.empty:
            continue

        texts = []
        metas: List[Dict[str, Any]] = []
        for _, row in batch.iterrows():
            doc_id = int(row["__doc_id__"])
            name = str(row.get("Назва препарату", "") or "")
            text = concat_doc_text(row, sections, args.max_chars)
            texts.append(text)
            metas.append({"doc_id": doc_id, "drug_name": name, "text_used": text})

        # encode single-process (robust on CPU; avoids multi-proc overhead on laptops)
        embs = enc.encode(texts, batch_size=args.encode_bs, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        embs = np.asarray(embs, dtype="float32")
        norms = np.linalg.norm(embs, axis=1).astype("float32")

        records = []
        for i, meta in enumerate(metas):
            row = {
                "doc_id": meta["doc_id"],
                "drug_name": meta["drug_name"],
                "text_used": meta["text_used"],
                "embedding_doc": embs[i].tolist(),
                "norm": float(norms[i]),
            }
            records.append(row)

        out_part = parts_dir / f"part_{part_id:06d}.parquet"
        write_part(records, out_part)
        part_id += 1

        done.update(int(x) for x in batch["__doc_id__"])
        pbar.update(len(batch))

        ckpt["done_ids"] = sorted(list(done))
        ckpt["next_part_id"] = part_id
        ckpt_path.write_text(json.dumps(ckpt, ensure_ascii=False, indent=2), encoding="utf-8")

        # free mem
        del texts, metas, embs, norms, records
        gc.collect()

    pbar.close()
    print("\n[INFO] Merging parts into final docs.parquet ...")
    merge_parts(parts_dir, final_path)
    # cleanup checkpoint (complete)
    try: ckpt_path.unlink()
    except: pass

    tbl = pq.read_table(final_path)
    print(f"[OK] {final_path} | rows={tbl.num_rows:,} | dim={dim} | size={tbl.nbytes/1024**2:.1f} MB")

if __name__ == "__main__":
    main()
