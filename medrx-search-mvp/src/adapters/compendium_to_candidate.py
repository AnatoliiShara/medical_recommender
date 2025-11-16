"""
Adapter: Compendium Parquet → DrugCandidate (for Gemini reranker).
Витягує потрібні поля з твого compendium_all.parquet для LLM-аналізу.
"""
from __future__ import annotations

from typing import Any, Dict
import pandas as pd

# Assuming llm.types is already available
import sys
sys.path.insert(0, "/home/anatolii-shara/Documents/medrx-search-mvp/src")

from llm.types import DrugCandidate


def extract_field(row: pd.Series, field_name: str, max_chars: int = 500) -> str:
    """
    Safely extract field from parquet row, with truncation.
    
    Args:
        row: Pandas Series (one row from parquet)
        field_name: Column name to extract
        max_chars: Maximum characters to return (for LLM token economy)
    
    Returns:
        Cleaned string, truncated if needed
    """
    value = row.get(field_name)
    
    # Handle None/NaN
    if pd.isna(value) or value is None:
        return ""
    
    # Convert to string
    text = str(value).strip()
    
    # Truncate if too long
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    
    return text


def compendium_row_to_drug_candidate(
    row: pd.Series,
    ce_score: float = 0.0,
    drug_id: str | None = None,
) -> DrugCandidate:
    """
    Converts one row from compendium_all.parquet to DrugCandidate.
    
    Args:
        row: Single row from parquet (pd.Series)
        ce_score: CrossEncoder score (if available)
        drug_id: Optional explicit drug_id (otherwise uses index)
    
    Returns:
        DrugCandidate ready for Gemini reranking
    """
    # [1] Drug ID
    if drug_id is None:
        # Use row index or URL hash as fallback
        drug_id = str(row.name) if row.name is not None else hash(row.get("url", "unknown"))
    
    # [2] Brand name (Назва препарату)
    brand_name = extract_field(row, "Назва препарату", max_chars=200)
    if not brand_name:
        brand_name = "Невідомий препарат"
    
    # [3] INN name - часто в складі або фармакотерапевтичній групі
    # Спробуємо витягти з "Склад" або "Фармакотерапевтична група"
    composition = extract_field(row, "Склад", max_chars=300)
    pharma_group = extract_field(row, "Фармакотерапевтична група", max_chars=200)
    
    # Простий heuristic: перше слово з великої літери в складі
    inn_name = None
    if composition:
        # Спроба витягти МНН (перші 50 символів складу)
        inn_candidate = composition.split(".")[0].strip()
        if len(inn_candidate) < 100:
            inn_name = inn_candidate
    
    # [4] Indications (Показання)
    indications = extract_field(row, "Показання", max_chars=500)
    if not indications:
        indications = "Показання не вказані"
    
    # [5] Contraindications (Протипоказання) - КРИТИЧНО для safety!
    contraindications = extract_field(row, "Протипоказання", max_chars=500)
    if not contraindications:
        contraindications = "Протипоказання не вказані"
    
    # [6] Dosage form (Лікарська форма)
    dosage_form = extract_field(row, "Лікарська форма", max_chars=100)
    
    return DrugCandidate(
        drug_id=str(drug_id),
        brand_name=brand_name,
        inn_name=inn_name,
        indications=indications,
        contraindications=contraindications,
        dosage_form=dosage_form,
        ce_score=ce_score,
    )


def batch_convert_to_candidates(
    df: pd.DataFrame,
    ce_scores: Dict[str, float] | None = None,
) -> list[DrugCandidate]:
    """
    Convert multiple rows from compendium to DrugCandidates.
    
    Args:
        df: DataFrame with compendium data
        ce_scores: Optional dict mapping drug_id → ce_score
    
    Returns:
        List of DrugCandidate objects
    """
    candidates = []
    
    for idx, row in df.iterrows():
        drug_id = str(idx)
        ce_score = ce_scores.get(drug_id, 0.0) if ce_scores else 0.0
        
        candidate = compendium_row_to_drug_candidate(
            row=row,
            ce_score=ce_score,
            drug_id=drug_id,
        )
        candidates.append(candidate)
    
    return candidates


# ============ CLI для тестування ============
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test adapter: compendium → DrugCandidate"
    )
    parser.add_argument(
        "--parquet",
        type=str,
        default="data/raw/compendium_all.parquet",
        help="Path to compendium parquet file",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Number of rows to test",
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading {args.parquet}...")
    df = pd.read_parquet(args.parquet)
    print(f"Loaded {len(df)} drugs")
    
    # Convert first N rows
    print(f"\nConverting first {args.n} rows to DrugCandidate...")
    test_df = df.head(args.n)
    candidates = batch_convert_to_candidates(test_df)
    
    # Display results
    for i, cand in enumerate(candidates, 1):
        print(f"\n{'='*60}")
        print(f"Drug #{i}")
        print(f"{'='*60}")
        print(f"ID: {cand.drug_id}")
        print(f"Brand: {cand.brand_name}")
        print(f"INN: {cand.inn_name or 'N/A'}")
        print(f"Form: {cand.dosage_form or 'N/A'}")
        print(f"\nIndications (truncated):")
        print(f"  {cand.indications[:200]}...")
        print(f"\nContraindications (truncated):")
        print(f"  {cand.contraindications[:200]}...")