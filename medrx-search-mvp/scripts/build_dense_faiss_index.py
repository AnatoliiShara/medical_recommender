#!/usr/bin/env python3
import argparse, json, os, time
from pathlib import Path

import numpy as np


# -------------------------
# helpers
# -------------------------
def atomic_write_json(path: Path, obj: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

def pick_first_existing(base: Path, names):
    for n in names:
        p = base / n
        if p.exists():
            return p
    return None

def detect_docs_parquet(doc_index_dir: Path) -> Path:
    preferred = [
        "docs.parquet", "documents.parquet", "doc_index.parquet",
        "chunks.parquet", "passages.parquet", "corpus.parquet",
    ]
    p = pick_first_existing(doc_index_dir, preferred)
    if p:
        return p

    candidates = sorted(doc_index_dir.glob("*.parquet"))
    candidates = [c for c in candidates if "emb" not in c.name.lower()]
    if not candidates:
        raise FileNotFoundError(
            f"Cannot find docs parquet in {doc_index_dir}. "
            f"Tried: {preferred} and *.parquet (excluding *emb*)."
        )
    return candidates[0]

def detect_cols(df_cols):
    cols = set(df_cols)
    id_candidates = ["doc_id", "id", "chunk_id", "passage_id", "pid", "product_id"]
    text_candidates = ["text", "chunk_text", "content", "body", "passage", "chunk", "doc_text"]
    title_candidates = ["title", "name", "heading"]

    id_col = next((c for c in id_candidates if c in cols), None)
    text_col = next((c for c in text_candidates if c in cols), None)
    title_col = next((c for c in title_candidates if c in cols), None)
    return id_col, text_col, title_col

def file_nlines(path: Path) -> int:
    if not path.exists():
        return 0
    n = 0
    with path.open("rb") as f:
        for _ in f:
            n += 1
    return n


# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc_index_dir", type=Path, required=True, help="де лежить docs parquet (baseline index dir)")
    ap.add_argument("--out_index_dir", type=Path, required=True, help="куди зберегти finetuned dense artifacts")
    ap.add_argument("--model", type=str, required=True, help="HF id або локальний шлях до embedding model")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_chars", type=int, default=6500)
    ap.add_argument("--use_title", action="store_true", help="якщо є title/name — додати до тексту")

    # checkpointing / resume
    ap.add_argument("--checkpoint_rows", type=int, default=5000, help="як часто фіксувати прогрес (рядків)")
    ap.add_argument("--resume", action="store_true", help="продовжити з чекпойнта (default: True)", default=True)
    ap.add_argument("--overwrite", action="store_true", help="почати з нуля (видалить старі артефакти)")

    # performance knobs
    ap.add_argument("--faiss_add_batch", type=int, default=50000, help="чанк при побудові FAISS з memmap")
    args = ap.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import pandas as pd
    import faiss
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm

    doc_index_dir = args.doc_index_dir
    out_dir = args.out_index_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # output paths
    state_path = out_dir / "state.json"
    memmap_path = out_dir / "embeddings.f32"      # raw float32 buffer
    meta_path = out_dir / "meta.json"
    faiss_path = out_dir / "faiss.index"
    map_path = out_dir / "doc_ids.jsonl"

    # overwrite behavior
    if args.overwrite:
        for p in [state_path, memmap_path, meta_path, faiss_path, map_path]:
            if p.exists():
                p.unlink()
        print("[OVERWRITE] removed previous artifacts in", out_dir)

    docs_parquet = detect_docs_parquet(doc_index_dir)

    # read only needed columns (avoid loading whole parquet)
    df0 = pd.read_parquet(docs_parquet)
    id_col, text_col, title_col = detect_cols(df0.columns)
    if text_col is None:
        raise ValueError(f"Cannot detect text column in {docs_parquet}. Have columns: {list(df0.columns)}")

    cols = [text_col]
    if id_col: cols.append(id_col)
    if args.use_title and title_col: cols.append(title_col)
    df = pd.read_parquet(docs_parquet, columns=cols)

    total = len(df)
    model_name = args.model.lower()
    is_e5 = ("e5" in model_name)

    print(f"[DOCS] parquet={docs_parquet}")
    print(f"[DOCS] rows={total} id_col={id_col} text_col={text_col} title_col={(title_col if args.use_title else None)}")
    print(f"[MODEL] {args.model} device={args.device} batch_size={args.batch_size} e5_passage_prefix={is_e5}")
    print(f"[OUT]   {out_dir}")

    model = SentenceTransformer(args.model, device=args.device)
    d = model.get_sentence_embedding_dimension()

    # ---- load/resume state
    done = 0
    phase = "encoding"  # encoding -> faiss_build -> done
    if args.resume and state_path.exists() and memmap_path.exists():
        st = json.loads(state_path.read_text(encoding="utf-8"))
        # strict-ish compatibility checks
        if st.get("docs_parquet") != str(docs_parquet) or st.get("model") != args.model or st.get("dim") != d:
            raise RuntimeError(
                "Found existing state.json, but it does not match current run.\n"
                f"state.docs_parquet={st.get('docs_parquet')}\n"
                f"curr.docs_parquet ={str(docs_parquet)}\n"
                f"state.model={st.get('model')}\n"
                f"curr.model ={args.model}\n"
                f"state.dim={st.get('dim')} curr.dim={d}\n"
                "Use --overwrite to start from scratch, or keep args identical."
            )
        done = int(st.get("done", 0))
        phase = st.get("phase", phase)
        print(f"[RESUME] phase={phase} done={done}/{total}")

    # ---- ensure memmap exists with correct size
    expected_bytes = total * d * 4
    if memmap_path.exists():
        actual_bytes = memmap_path.stat().st_size
        if actual_bytes != expected_bytes:
            raise RuntimeError(
                f"memmap size mismatch: {memmap_path} bytes={actual_bytes}, expected={expected_bytes}. "
                f"Use --overwrite to rebuild."
            )
        mm = np.memmap(memmap_path, dtype="float32", mode="r+", shape=(total, d))
    else:
        mm = np.memmap(memmap_path, dtype="float32", mode="w+", shape=(total, d))
        mm.flush()

    # ---- helper(s) to build batch texts
    text_s = df[text_col]
    title_s = df[title_col] if (args.use_title and title_col and title_col in df.columns) else None

    def build_text_batch(i0: int, i1: int):
        tx = text_s.iloc[i0:i1].astype(str).tolist()

        if args.max_chars and args.max_chars > 0:
            tx = [t[:args.max_chars] for t in tx]

        if title_s is not None:
            ti = title_s.iloc[i0:i1].astype(str).tolist()
            tx = [f"{t}\n{x}" if (t and t != "nan") else x for t, x in zip(ti, tx)]

        if is_e5:
            tx = [("passage: " + x) if not x.startswith("passage:") else x for x in tx]

        return tx

    # ---- write/ensure doc_ids mapping (optional but very useful)
    # We'll create it once, idempotent: if exists + correct line count => skip
    need_map = True
    if map_path.exists():
        n = file_nlines(map_path)
        if n == total:
            need_map = False
            print(f"[MAP] exists OK: {map_path} (lines={n})")

    if need_map:
        print(f"[MAP] writing: {map_path}")
        t0 = time.time()
        with map_path.open("w", encoding="utf-8") as f:
            if id_col is None:
                for i in range(total):
                    f.write(json.dumps({"row_id": i, "doc_id": str(i)}, ensure_ascii=False) + "\n")
            else:
                ids = df[id_col].astype(str).tolist()
                for i, did in enumerate(ids):
                    f.write(json.dumps({"row_id": i, "doc_id": did}, ensure_ascii=False) + "\n")
        print(f"[MAP] done in {time.time()-t0:.1f}s")

    # ---- Phase A: encoding (resumable)
    if phase in ("encoding", "faiss_build") and done < total:
        print("[PHASE] encoding -> memmap (resumable)")
        start_time = time.time()

        # initial state write
        st = {
            "phase": "encoding",
            "done": done,
            "total": total,
            "doc_index_dir": str(doc_index_dir),
            "docs_parquet": str(docs_parquet),
            "id_col": id_col,
            "text_col": text_col,
            "title_col": (title_col if args.use_title else None),
            "model": args.model,
            "device": args.device,
            "batch_size": int(args.batch_size),
            "max_chars": int(args.max_chars),
            "use_title": bool(args.use_title),
            "dim": int(d),
            "normalized": True,
            "e5_passage_prefix": bool(is_e5),
            "checkpoint_rows": int(args.checkpoint_rows),
            "created_at": st.get("created_at") if (args.resume and state_path.exists()) else time.strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        atomic_write_json(state_path, st)

        bs = args.batch_size
        next_ckpt = max(done + args.checkpoint_rows, done)

        pbar = tqdm(total=total, initial=done, unit="emb", dynamic_ncols=True, desc="encode")
        try:
            i = done
            while i < total:
                j = min(i + bs, total)
                batch = build_text_batch(i, j)

                emb = model.encode(
                    batch,
                    batch_size=bs,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ).astype(np.float32)

                mm[i:j, :] = emb  # write to disk-backed buffer
                i = j
                done = i
                pbar.update(len(batch))

                # checkpoint
                if done >= next_ckpt or done == total:
                    mm.flush()
                    st["done"] = done
                    st["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    atomic_write_json(state_path, st)
                    next_ckpt = done + args.checkpoint_rows

            pbar.close()

        except KeyboardInterrupt:
            pbar.close()
            mm.flush()
            st["done"] = done
            st["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            atomic_write_json(state_path, st)
            elapsed = time.time() - start_time
            print(f"\n[INTERRUPTED] saved checkpoint at done={done}/{total} elapsed={elapsed:.1f}s")
            print("Re-run the same command to continue (resume).")
            return

        # mark phase switch
        st["phase"] = "faiss_build"
        st["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        atomic_write_json(state_path, st)
        phase = "faiss_build"
        print("[PHASE] encoding complete")

    # ---- Phase B: build FAISS from memmap (no re-embedding)
    if phase == "faiss_build" and not faiss_path.exists():
        import faiss
        from tqdm import tqdm

        print("[PHASE] build FAISS from memmap (no re-embed)")
        index = faiss.IndexFlatIP(d)

        add_bs = max(1, int(args.faiss_add_batch))
        pbar = tqdm(total=total, unit="vec", dynamic_ncols=True, desc="faiss.add")
        for i in range(0, total, add_bs):
            j = min(i + add_bs, total)
            chunk = np.asarray(mm[i:j, :], dtype=np.float32)  # ensure contiguous enough
            index.add(chunk)
            pbar.update(j - i)
        pbar.close()

        faiss.write_index(index, str(faiss_path))
        print(f"[OK] wrote: {faiss_path} (ntotal={index.ntotal}, d={d})")

        # write meta (final)
        meta = {
            "doc_index_dir": str(doc_index_dir),
            "docs_parquet": str(docs_parquet),
            "id_col": id_col,
            "text_col": text_col,
            "title_col": (title_col if args.use_title else None),
            "model": args.model,
            "device": args.device,
            "batch_size": int(args.batch_size),
            "max_chars": int(args.max_chars),
            "faiss_type": "IndexFlatIP",
            "dim": int(d),
            "ntotal": int(index.ntotal),
            "normalized": True,
            "e5_passage_prefix": bool(is_e5),
            "embeddings_memmap": str(memmap_path),
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] wrote: {meta_path}")

        # update state
        st = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
        st["phase"] = "done"
        st["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        atomic_write_json(state_path, st)

    else:
        if faiss_path.exists():
            print(f"[SKIP] faiss exists: {faiss_path}")

    print("[DONE] You can now use:", faiss_path)


if __name__ == "__main__":
    main()
