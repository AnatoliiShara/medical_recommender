# -*- coding: utf-8 -*-
"""
Export chunk-level embeddings у parquet з урахуванням:
- Автономного медичного чанкінгу (EnhancedMedicalChunker) з суворим лімітом токенів
- Безпечного паралелізму через encoder.encode(num_workers=...)
- Checkpoint + part_*.parquet + фінальне злиття в один файл
- Прогрес-барів (tqdm)
- Контролю пам'яті, retry на OOM зі зменшенням batch_size
- Float32 з нормою для cosine/IP
"""
import os
import sys
import json
import math
import argparse
import re
import gc
import warnings
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Наш базовий чанкер (буде використано лише як транспорт для токенайзера/словників; логіку не викликаємо)
sys.path.append('src')
from preprocessing.medical_chunker import MedicalChunker

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

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

# -------- Utils --------
def model_to_slug(model_name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9]+', '-', model_name.strip()).strip('-').lower()

def ensure_dirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def arrow_schema() -> pa.schema:
    return pa.schema([
        pa.field("doc_id", pa.int64()),
        pa.field("drug_name", pa.string()),
        pa.field("section", pa.string()),
        pa.field("passage_id", pa.int64()),
        pa.field("text", pa.string()),
        pa.field("embedding", pa.list_(pa.float32())),
        pa.field("norm", pa.float32()),
    ])

# -------- Chunker (повністю автономний) --------
_SENT_SPLIT = re.compile(r'(?:\.\s+|!\s+|\?\s+|;\s+|\n+|\s+\[SEP\]\s+)', re.UNICODE)

class EnhancedMedicalChunker(MedicalChunker):
    """Автономний смарт-чанкер з жорстким контролем токенів та оверлапом."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # для токенайзера, якщо він є
        # Якщо базовий клас не створює токенайзер, зробимо легкий lazy
        if not hasattr(self, "_tokenizer") or self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            except Exception:
                self._tokenizer = None

    def _len_tokens(self, text: str) -> int:
        if self._tokenizer is None:
            # дуже груба оцінка: ~1 токен ≈ 4 символи
            return max(1, len(text) // 4)
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        if self._tokenizer is None:
            approx_chars = max_tokens * 4
            return text[:approx_chars]
        ids = self._tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= max_tokens:
            return text
        return self._tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)

    def _take_tail_tokens(self, text: str, tail_tokens: int) -> str:
        """Бере хвіст text у tail_tokens токенах (для оверлапу)."""
        if self._tokenizer is None:
            approx_chars = tail_tokens * 4
            return text[-approx_chars:]
        ids = self._tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= tail_tokens:
            return text
        return self._tokenizer.decode(ids[-tail_tokens:], skip_special_tokens=True)

    def smart_chunking(self, text: str, max_tokens: int = 256,
                       overlap_tokens: int = 32, min_chunk_chars: int = 60) -> List[str]:
        """
        Повністю автономний чанкінг:
        1) Спліт на речення/квазі-речення
        2) Грідне пакування до max_tokens
        3) Оверлап хвостом попереднього чанку (в токенах)
        4) Жорстке обрізання до max_tokens
        """
        txt = (text or "").strip()
        if not txt:
            return []

        # 1) розбивка
        sents = [s.strip() for s in _SENT_SPLIT.split(txt) if s and s.strip()]
        if not sents:
            sents = [txt]

        chunks: List[str] = []
        cur: List[str] = []
        cur_len = 0

        for s in sents:
            s_len = self._len_tokens(s)
            if not cur:
                if s_len <= max_tokens:
                    cur = [s]; cur_len = s_len
                else:
                    # дуже довге речення — рубимо жорстко
                    sl = self._truncate_to_tokens(s, max_tokens)
                    cur = [sl]; cur_len = self._len_tokens(sl)
                continue

            if cur_len + s_len <= max_tokens:
                cur.append(s); cur_len += s_len
            else:
                chunk_text = " ".join(cur).strip()
                if len(chunk_text) >= min_chunk_chars:
                    # 2) оверлап
                    if overlap_tokens > 0 and chunk_text:
                        tail = self._take_tail_tokens(chunk_text, overlap_tokens)
                    else:
                        tail = ""
                    chunks.append(self._truncate_to_tokens(chunk_text, max_tokens))
                    # новий буфер стартує з overlap-хвоста (якщо є)
                    if tail:
                        cur = [tail, s]
                        cur_len = self._len_tokens(tail) + s_len
                        if cur_len > max_tokens:
                            # якщо навіть з tail перебор — зберігаємо тільки s (обрізавши)
                            s2 = self._truncate_to_tokens(s, max_tokens)
                            cur = [s2]; cur_len = self._len_tokens(s2)
                    else:
                        if s_len <= max_tokens:
                            cur = [s]; cur_len = s_len
                        else:
                            s2 = self._truncate_to_tokens(s, max_tokens)
                            cur = [s2]; cur_len = self._len_tokens(s2)

        # фінальний флаш
        if cur:
            last = self._truncate_to_tokens(" ".join(cur).strip(), max_tokens)
            if len(last) >= min_chunk_chars:
                chunks.append(last)

        return chunks

# -------- Побудова пасажів --------
def build_passages_for_row(
    row: pd.Series,
    sections: List[str],
    chunker: EnhancedMedicalChunker,
    max_chunk_tokens: int,
    overlap_tokens: int,
    min_chunk_chars: int,
    per_doc_cap: int = 100,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    drug_name = str(row.get("Назва препарату", "") or "").strip() or f"Unknown_{row.get('__doc_id__')}"
    pid = 0

    for sec in sections:
        content = str(row.get(sec, "") or "").strip()
        if not content or len(content) < 10:
            continue
        if len(content) > 50000:
            content = content[:50000] + "..."

        text = f"{sec}: {content}"
        try:
            chunks = chunker.smart_chunking(
                text,
                max_tokens=max_chunk_tokens,
                overlap_tokens=overlap_tokens,
                min_chunk_chars=min_chunk_chars
            )
        except Exception as e:
            print(f"[warn] chunking failed for '{drug_name}' / '{sec}': {e}")
            continue

        for ch in chunks:
            m = re.match(r"^\s*([^:]{2,80})\s*:\s*(.+)$", ch, re.DOTALL)
            sec_name = (m.group(1).strip() if m else sec)
            body = (m.group(2).strip() if m else ch.strip())
            if len(body) < min_chunk_chars:
                continue

            out.append({
                "doc_id": int(row["__doc_id__"]),
                "drug_name": drug_name,
                "section": sec_name,
                "passage_id": pid,
                "text": body,
            })
            pid += 1
            if pid >= per_doc_cap:
                break

    return out

# -------- Енкодинг --------
def safe_encode(encoder: SentenceTransformer, texts: List[str],
                batch_size: int, workers: int = 0) -> np.ndarray:
    if not texts:
        dim = encoder.get_sentence_embedding_dimension()
        return np.zeros((0, dim), dtype=np.float32)

    encode_kwargs = dict(
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    try:
        if workers and workers > 1:
            embs = encoder.encode(texts, num_workers=workers, **encode_kwargs)
        else:
            embs = encoder.encode(texts, **encode_kwargs)
        return np.asarray(embs, dtype=np.float32)
    except TypeError:
        embs = encoder.encode(texts, **encode_kwargs)
        return np.asarray(embs, dtype=np.float32)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            smaller = max(1, batch_size // 2)
            print(f"[OOM] retry with batch_size={smaller}")
            return safe_encode(encoder, texts, smaller, workers=0)
        raise
    except Exception as e:
        print(f"[error] encode failed: {e}")
        dim = encoder.get_sentence_embedding_dimension()
        return np.zeros((len(texts), dim), dtype=np.float32)

# -------- Arrow I/O --------
def make_table(records: List[Dict[str, Any]]) -> pa.Table:
    if not records:
        schema = arrow_schema()
        return pa.Table.from_arrays(
            [pa.array([], type=f.type) for f in schema],
            names=schema.names,
            schema=schema
        )
    return pa.table({
        "doc_id": pa.array([r["doc_id"] for r in records], type=pa.int64()),
        "drug_name": pa.array([r["drug_name"] for r in records], type=pa.string()),
        "section": pa.array([r["section"] for r in records], type=pa.string()),
        "passage_id": pa.array([r["passage_id"] for r in records], type=pa.int64()),
        "text": pa.array([r["text"] for r in records], type=pa.string()),
        "embedding": pa.array([r["embedding"] for r in records], type=pa.list_(pa.float32())),
        "norm": pa.array([r["norm"] for r in records], type=pa.float32()),
    }, schema=arrow_schema())

def merge_parquet_parts(parts_dir: str, final_path: str) -> None:
    files = sorted([f for f in os.listdir(parts_dir) if f.endswith(".parquet")])
    if not files:
        pq.write_table(make_table([]), final_path, compression="zstd", version="2.6")
        return

    print(f"Merging {len(files)} parts into final file...")
    tables = []
    for f in tqdm(files, desc="Reading parts", unit="file", leave=False):
        try:
            tables.append(pq.read_table(os.path.join(parts_dir, f)))
        except Exception as e:
            print(f"[warn] failed to read {f}: {e}")

    if tables:
        merged = pa.concat_tables(tables, promote=True)
        pq.write_table(merged, final_path, compression="zstd", version="2.6")
        print(f"Final table: {merged.num_rows:,} rows, {merged.nbytes/1024**2:.1f} MB")
    else:
        pq.write_table(make_table([]), final_path, compression="zstd", version="2.6")

    for f in files:
        try: os.remove(os.path.join(parts_dir, f))
        except: pass
    try: os.rmdir(parts_dir)
    except: pass

def estimate_memory_usage(num_passages: int, embedding_dim: int) -> float:
    return (num_passages * embedding_dim * 4) / (1024**2) + num_passages * 0.001

# -------- Main --------
def main() -> int:
    ap = argparse.ArgumentParser(description="Export chunk-level embeddings to a single parquet with checkpointing")
    ap.add_argument("--parquet", required=True, help="Шлях до compendium_all.parquet")
    ap.add_argument("--out_base", default="data/processed/embeddings", help="Базова директорія")
    ap.add_argument("--model", default="intfloat/multilingual-e5-base")
    ap.add_argument("--rows", type=int, default=0, help="Ліміт рядків (0 = всі)")
    ap.add_argument("--batch_docs", type=int, default=16, help="Документів за батч")
    ap.add_argument("--encode_bs", type=int, default=32, help="Batch size в encode")
    ap.add_argument("--workers", type=int, default=0, help="num_workers для encode")
    ap.add_argument("--max_chunk_tokens", type=int, default=80)
    ap.add_argument("--overlap_tokens", type=int, default=8)
    ap.add_argument("--min_chunk_chars", type=int, default=60)
    ap.add_argument("--sections", default=None, help="Кома-сепарований список секцій")
    ap.add_argument("--force_restart", action="store_true", help="Ігнорувати checkpoint")
    args = ap.parse_args()

    sections = [s.strip() for s in (args.sections.split(",") if args.sections else SECTIONS_DEFAULT) if s.strip()]

    print("Loading data...")
    df = pd.read_parquet(args.parquet)
    if args.rows > 0:
        df = df.head(args.rows).copy()
    df = df.reset_index(drop=True)
    df["__doc_id__"] = df.index.astype(int)
    print(f"Processing {len(df):,} documents")

    model_slug = model_to_slug(args.model)
    out_dir = os.path.join(args.out_base, model_slug)
    parts_dir = os.path.join(out_dir, "chunks_tmp")
    final_path = os.path.join(out_dir, "chunks.parquet")
    ckpt_path = os.path.join(out_dir, "chunks.ckpt.json")
    ensure_dirs(out_dir); ensure_dirs(parts_dir)

    ckpt = {"done_ids": [], "next_part_id": 0, "total_passages": 0}
    if not args.force_restart and os.path.isfile(ckpt_path):
        try:
            ckpt = json.load(open(ckpt_path, "r", encoding="utf-8"))
            print(f"Resuming from checkpoint: {len(ckpt.get('done_ids', [])):,} docs done")
        except Exception as e:
            print(f"[warn] checkpoint load failed: {e}. Starting fresh.")

    done = set(ckpt.get("done_ids", []))
    part_id = int(ckpt.get("next_part_id", 0))
    total_passages_written = int(ckpt.get("total_passages", 0))

    print(f"Loading model: {args.model}")
    encoder = SentenceTransformer(args.model)
    emb_dim = encoder.get_sentence_embedding_dimension()
    print(f"Model OK. dim={emb_dim}")
    chunker = EnhancedMedicalChunker()

    docs_to_process = df[~df["__doc_id__"].isin(done)]
    docs_remaining = len(docs_to_process)
    doc_batches = math.ceil(docs_remaining / args.batch_docs)
    print(f"Documents remaining: {docs_remaining:,} | Batches: {doc_batches:,}")

    docs_pbar = tqdm(total=docs_remaining, desc="Documents", unit="doc", position=0)
    passages_pbar = tqdm(total=total_passages_written, desc="Passages (total)", unit="pass", position=1)

    try:
        batch_idx = 0
        for s in range(0, len(docs_to_process), args.batch_docs):
            batch_df = docs_to_process.iloc[s:s+args.batch_docs]
            if batch_df.empty:
                continue
            batch_idx += 1

            passages: List[Dict[str, Any]] = []

            for _, row in tqdm(batch_df.iterrows(),
                               total=len(batch_df),
                               desc=f"Batch {batch_idx}/{doc_batches} chunking",
                               leave=False, position=2):
                try:
                    pas = build_passages_for_row(
                        row, sections, chunker,
                        max_chunk_tokens=args.max_chunk_tokens,
                        overlap_tokens=args.overlap_tokens,
                        min_chunk_chars=args.min_chunk_chars,
                        per_doc_cap=100
                    )
                    passages.extend(pas)
                except Exception as e:
                    print(f"[warn] {row.get('Назва препарату','Unknown')}: {e}")

                est_mb = estimate_memory_usage(len(passages), emb_dim)
                if est_mb > 1000:
                    print(f"[mem] ~{est_mb:.1f} MB passages buffered → early encode")
                    break

            if not passages:
                done.update(batch_df["__doc_id__"].tolist())
                docs_pbar.update(len(batch_df))
                continue

            texts = [p["text"] for p in passages]
            embs = safe_encode(encoder, texts, args.encode_bs, args.workers)
            norms = np.linalg.norm(embs, axis=1).astype(np.float32)

            for i, p in enumerate(passages):
                p["embedding"] = embs[i].tolist()
                p["norm"] = float(norms[i])

            table = make_table(passages)
            part_file = os.path.join(parts_dir, f"part_{part_id:06d}.parquet")
            pq.write_table(table, part_file, compression="zstd", version="2.6")
            part_id += 1
            total_passages_written += len(passages)
            passages_pbar.total = total_passages_written
            passages_pbar.update(len(passages))
            passages_pbar.refresh()

            done.update(batch_df["__doc_id__"].tolist())
            docs_pbar.update(len(batch_df))
            with open(ckpt_path, "w", encoding="utf-8") as f:
                json.dump({
                    "done_ids": sorted(list(done)),
                    "next_part_id": part_id,
                    "total_passages": total_passages_written
                }, f, ensure_ascii=False, indent=2)

            del passages, texts, embs, norms, table
            gc.collect()

        docs_pbar.close(); passages_pbar.close()

        print("\nMerging parts into final file...")
        merge_parquet_parts(parts_dir, final_path)

        if os.path.exists(final_path):
            final_table = pq.read_table(final_path)
            print(f"\nSuccess!")
            print(f"  File: {final_path}")
            print(f"  Rows: {final_table.num_rows:,}")
            print(f"  Size: {final_table.nbytes/1024**2:.1f} MB")
            print(f"  Cols: {final_table.column_names}")

        try: os.remove(ckpt_path)
        except: pass
        return 0

    except KeyboardInterrupt:
        docs_pbar.close(); passages_pbar.close()
        print("\nInterrupted. Progress saved to checkpoint. Re-run to resume.")
        return 1
    except Exception as e:
        docs_pbar.close(); passages_pbar.close()
        print(f"\n[fatal] {e}")
        import traceback; traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
