# src/utils/preflight.py
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = [
    "Назва препарату","Склад","Лікарська форма","Фармакотерапевтична група",
    "Фармакологічні властивості","Показання","Взаємодія з іншими лікарськими засобами та інші види взаємодій",
    "Особливості застосування","Застосування у період вагітності або годування груддю",
    "Здатність впливати на швидкість реакції при керуванні автотранспортом або іншими механізмами",
    "Спосіб застосування та дози","Передозування","Побічні реакції","Термін придатності",
    "Умови зберігання","Упаковка","Виробник","url","Протипоказання",
]

def assert_file(p: Path, msg: str):
    if not p.exists():
        raise FileNotFoundError(f"[PRECHECK] Missing {msg}: {p}")
    print(f"[OK] {msg}: {p}")

def check_dataset(parquet_path: Path):
    assert_file(parquet_path, "dataset parquet")
    df = pd.read_parquet(parquet_path)
    miss = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if miss:
        raise ValueError(f"[PRECHECK] Missing required columns: {miss}")
    key_nulls = df["Назва препарату"].isna().sum() + df["Показання"].isna().sum()
    print(f"[OK] columns present ({len(df.columns)}). Rows={len(df)}. Key nulls={key_nulls}")
    dups = df["Назва препарату"].str.lower().value_counts()
    print(f"[INFO] unique brands={dups.size}, possible duplicates={int((dups>1).sum())}")
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--faiss", type=Path, required=True,
                    help="Path to faiss.index")
    ap.add_argument("--bm25_index", type=Path, required=False,
                    help="Optional BM25 index file if ви зберігаєте його на диск")
    args = ap.parse_args()

    df = check_dataset(args.dataset)
    assert_file(args.faiss, "FAISS index")

    if args.bm25_index:
        assert_file(args.bm25_index, "BM25 index")

    # міні-семпл для smoke-тестів (не коммітимо повні дані)
    sample_dir = Path("data/sample")
    sample_dir.mkdir(parents=True, exist_ok=True)
    out = sample_dir / "compendium_sample.parquet"
    if not out.exists():
        df.head(200).to_parquet(out, index=False)
        print(f"[OK] wrote sample -> {out}")
    print("[PRECHECK] all good.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)
