import argparse
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_parquet", required=True)
    ap.add_argument("--out_parquet", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--prefix", default="passage: ")
    ap.add_argument("--text_col", default="text")
    args = ap.parse_args()

    pf = pq.ParquetFile(args.in_parquet)
    # беремо потрібні колонки — як у baseline chunks.parquet
    cols = ["doc_id", "drug_name", "section", "passage_id", args.text_col]
    schema = pa.schema([
        ("doc_id", pa.int64()),
        ("drug_name", pa.string()),
        ("section", pa.string()),
        ("passage_id", pa.int32()),
        (args.text_col, pa.string()),
        ("embedding", pa.fixed_size_list(pa.float32(), 768)),
        ("norm", pa.float32()),
    ])

    model = SentenceTransformer(args.model, device=args.device)

    writer = pq.ParquetWriter(args.out_parquet, schema=schema, compression="zstd")
    total = pf.metadata.num_rows
    pbar = tqdm(total=total, desc="re-encode", unit="rows")

    for batch in pf.iter_batches(batch_size=2048, columns=cols):
        d = batch.to_pydict()

        texts = d[args.text_col]
        texts = [("" if t is None else (args.prefix + t)) for t in texts]

        emb = model.encode(
            texts,
            batch_size=args.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")

        norms = np.linalg.norm(emb, axis=1).astype("float32")

        # Arrow fixed-size list for embeddings
        flat = pa.array(emb.reshape(-1), type=pa.float32())
        emb_arr = pa.FixedSizeListArray.from_arrays(flat, 768)

        out = pa.record_batch(
            [
                pa.array(d["doc_id"], type=pa.int64()),
                pa.array(d["drug_name"], type=pa.string()),
                pa.array(d["section"], type=pa.string()),
                pa.array(d["passage_id"], type=pa.int32()),
                pa.array(d[args.text_col], type=pa.string()),
                emb_arr,
                pa.array(norms, type=pa.float32()),
            ],
            schema=schema,
        )
        writer.write_batch(out)
        pbar.update(len(texts))

    writer.close()
    pbar.close()
    print("[OK] wrote", args.out_parquet, "rows=", total)

if __name__ == "__main__":
    main()
