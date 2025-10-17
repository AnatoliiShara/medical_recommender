import sys, argparse
from pathlib import Path
import pandas as pd
sys.path.append('src')

from src.search.enhanced_medical_assistant import EnhancedMedicalAssistant, SearchConfig

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/raw/compendium_all.parquet")
    ap.add_argument("--rows", type=int, default=500)
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet).head(args.rows).copy()
    print(f"[INFO] Loaded: {args.parquet}  rows={len(df):,}")

    assistant = EnhancedMedicalAssistant()
    assistant.build_from_dataframe(
        df, encoder_model="intfloat/multilingual-e5-base",
        medical_chunking=True, max_chunk_tokens=128
    )
    print(f"[OK] passages={len(assistant.passages):,}")

    cfg = SearchConfig(rrf_alpha=60.0, w_bm25=1.0, w_dense=1.0)

    for q in ["кашель у дитини", "біль у горлі", "підвищена температура"]:
        groups = assistant.search(q, cfg)
        print(f"\nQ: {q}")
        for g in groups[:5]:
            print(f"- {g['name']}  best={g['best_score']:.3f}")

if __name__ == "__main__":
    main()
