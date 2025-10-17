import sys, argparse
from pathlib import Path
import pandas as pd
sys.path.append('src')

from src.search.enhanced_medical_assistant import EnhancedMedicalAssistant, SearchConfig

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/raw/compendium_all.parquet")
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    print(f"[INFO] Loaded: {args.parquet}  rows={len(df):,}")

    assistant = EnhancedMedicalAssistant()
    assistant.build_from_dataframe(
        df, encoder_model="intfloat/multilingual-e5-base",
        medical_chunking=True, max_chunk_tokens=128
    )
    cfg = SearchConfig(rrf_alpha=60.0, w_bm25=1.0, w_dense=1.0, enable_safety_filter=False)

    print("\n[SMOKE] query='гіпертензія'...")
    groups = assistant.search("гіпертензія", cfg)
    for g in groups[:5]:
        print(f"- {g['name']} / best_score={g['best_score']:.3f}")

if __name__ == "__main__":
    main()
