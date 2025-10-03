# -*- coding: utf-8 -*-
"""
Будує FAISS IndexFlatIP з parquet-файлу з колонкою `embedding` (list<float32>)
та зберігає разом спрощений meta-файл без ембеддингів.

Вхід: data/processed/embeddings/<model-slug>/chunks.parquet
Вихід:
  - data/processed/embeddings/<model-slug>/faiss.index
  - data/processed/embeddings/<model-slug>/chunks_meta.parquet
"""
import os
import argparse
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

try:
    import faiss  # type: ignore
except Exception as e:
    raise RuntimeError("faiss не встановлено або недоступне") from e


META_COLS = ["doc_id", "drug_name", "section", "passage_id", "text"]


def infer_dim_from_parquet(pf: pq.ParquetFile) -> int:
    """Зчитує перший елемент embedding, щоб визначити розмірність."""
    if pf.num_row_groups == 0:
        # fallback: прочитати малий зріз таблиці
        tbl = pq.read_table(pf.path, columns=["embedding"]).slice(0, 1)
        first = tbl.column("embedding")[0].as_py()
        return len(first)

    # читаємо перший row group і перший елемент
    rg0 = pf.read_row_group(0, columns=["embedding"])
    if rg0.num_rows == 0:
        # якщо перший порожній — шукаємо наступний непорожній
        for rg_idx in range(1, pf.num_row_groups):
            rg = pf.read_row_group(rg_idx, columns=["embedding"])
            if rg.num_rows > 0:
                first = rg.column("embedding")[0].as_py()
                return len(first)
        raise RuntimeError("У parquet немає жодного рядка з embedding")
    first = rg0.column("embedding")[0].as_py()
    return len(first)


def write_meta_only(src_parquet: str, out_meta_path: str):
    """Записуємо полегшений meta parquet без embedding."""
    table = pq.read_table(src_parquet, columns=META_COLS)
    pq.write_table(table, out_meta_path, compression="zstd")
    return table.num_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, help="path to chunks.parquet (з embedding)")
    ap.add_argument("--out_dir", required=True, help="куди зберегти faiss.index та chunks_meta.parquet")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    meta_out = os.path.join(args.out_dir, "chunks_meta.parquet")
    faiss_out = os.path.join(args.out_dir, "faiss.index")

    # 1) meta без embedding
    rows_meta = write_meta_only(args.parquet, meta_out)
    print(f"[INFO] Meta saved: {meta_out}  (rows={rows_meta:,})")

    # 2) створюємо FAISS
    pf = pq.ParquetFile(args.parquet)
    dim = infer_dim_from_parquet(pf)
    print(f"[INFO] Embedding dim={dim}")

    index = faiss.IndexFlatIP(dim)  # під нормалізовані вектори

    # 3) додаємо embedding по row group-ах (ефективно за пам’яттю)
    if pf.num_row_groups == 0:
        # файл без row groups — читаємо всю колонку (може бути важко для RAM)
        print("[WARN] Parquet has 0 row groups; reading entire 'embedding' column at once.")
        tbl = pq.read_table(args.parquet, columns=["embedding"])
        col = tbl.column("embedding")
        embs = np.vstack([np.asarray(col[i].as_py(), dtype="float32", order="C") for i in range(len(col))])
        index.add(embs)
    else:
        for rg_idx in tqdm(range(pf.num_row_groups), desc="FAISS add (row groups)", unit="rg"):
            rg = pf.read_row_group(rg_idx, columns=["embedding"])
            if rg.num_rows == 0:
                continue
            col = rg.column("embedding")
            # збираємо батчем у щільний np.ndarray
            embs = np.vstack([np.asarray(col[i].as_py(), dtype="float32", order="C") for i in range(len(col))])
            index.add(embs)

    faiss.write_index(index, faiss_out)
    print(f"[OK] FAISS index saved: {faiss_out}")


if __name__ == "__main__":
    main()
