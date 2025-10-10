import pandas as pd
from pathlib import Path
import csv, re, argparse, sys

def norm(s: str) -> str:
    s = (s or "").replace("\u00A0"," ").lower()
    s = re.sub(r"[’‘`']", "", s)
    s = re.sub(r"[-‐-‒–—]", " ", s)  # unify dashes
    s = re.sub(r"[®™©“”«»]", " ", s)
    s = re.sub(r"\s+"," ", s).strip()
    return s

def main(index_dir: str, out_csv: str):
    idx = Path(index_dir)
    chunks = idx / "chunks.parquet"
    docs_meta = idx / "docs_meta.parquet"

    if chunks.exists():
        dfc = pd.read_parquet(chunks, columns=["doc_id","drug_name","section"])
        df = dfc[["doc_id","drug_name"]].dropna().copy()
        # рахуємо частоту по кількості пасажів цього бренду
        freq = (df.assign(drug_name_norm=df["drug_name"].astype(str).map(norm))
                  .groupby("drug_name_norm")["doc_id"].count()
                  .sort_values(ascending=False)
                  .reset_index()
                  .rename(columns={"doc_id":"freq"}))
    elif docs_meta.exists():
        dmd = pd.read_parquet(docs_meta)
        if "drug_name" not in dmd.columns:
            print("[ERROR] docs_meta.parquet does not contain 'drug_name'", file=sys.stderr)
            sys.exit(2)
        freq = (dmd.assign(drug_name_norm=dmd["drug_name"].astype(str).map(norm))
                  .groupby("drug_name_norm")["drug_name"].count()
                  .sort_values(ascending=False)
                  .reset_index()
                  .rename(columns={"drug_name":"freq"}))
    else:
        print("[ERROR] neither chunks.parquet nor docs_meta.parquet found", file=sys.stderr)
        sys.exit(1)

    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["alias","target","type","freq","note"])
        for _, row in freq.iterrows():
            alias = row["drug_name_norm"]
            wr.writerow([alias, "", "brand2inn", int(row["freq"]), "TODO: fill INN"])

    print(f"[DONE] wrote candidates: {out} ({len(freq)} rows) — sorted by freq desc")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()
    main(args.index_dir, args.out_csv)
