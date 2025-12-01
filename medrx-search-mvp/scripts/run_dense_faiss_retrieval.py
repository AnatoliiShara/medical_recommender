#!/usr/bin/env python3
from __future__ import annotations

import argparse, json
from pathlib import Path
import numpy as np
import faiss  # type: ignore
from sentence_transformers import SentenceTransformer


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", required=True, help="dir with faiss.index + doc_ids.npy")
    ap.add_argument("--model", required=True, help="SentenceTransformer model id/path")
    ap.add_argument("--queries", required=True, help="jsonl with qid/id + query (+ optional gold_doc_ids)")
    ap.add_argument("--out", required=True, help="predictions.jsonl output")
    ap.add_argument("--top_k", type=int, default=2000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max_seq_length", type=int, default=256)
    ap.add_argument("--query_prefix", default="query: ")
    args = ap.parse_args()

    index_dir = Path(args.index_dir)
    index = faiss.read_index(str(index_dir / "faiss.index"))
    doc_ids = np.load(index_dir / "doc_ids.npy", allow_pickle=True).astype(object).tolist()
    doc_ids = [str(x) for x in doc_ids]

    model = SentenceTransformer(args.model, device=args.device)
    try:
        model.max_seq_length = args.max_seq_length
    except Exception:
        pass

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    with open(outp, "w", encoding="utf-8") as w:
        for o in iter_jsonl(args.queries):
            qid = str(o.get("qid", o.get("id")))
            query = str(o["query"])
            gold = o.get("gold_doc_ids", o.get("gold_ids", [])) or []
            gold = [str(x) for x in gold]

            q = model.encode([args.query_prefix + query], convert_to_numpy=True, show_progress_bar=False).astype("float32")
            q = l2_normalize(q)

            D, I = index.search(q, args.top_k)  # cosine via IP on normalized vecs
            hits = []
            for rank, (idx, score) in enumerate(zip(I[0].tolist(), D[0].tolist()), 1):
                if idx < 0 or idx >= len(doc_ids):
                    continue
                hits.append({"rank": rank, "doc_id": doc_ids[idx], "score": float(score)})

            rec = {"qid": qid, "query": query, "gold_doc_ids": gold, "hits": hits}
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("[OK] wrote:", outp)


if __name__ == "__main__":
    main()
