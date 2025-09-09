#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clean Medical Dataset Creator
Filters and prepares medical drugs data for ML training
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def clean_text(text):
    """Basic text cleaning"""
    if not isinstance(text, str):
        return ""
    
    text = text.replace('\xa0', ' ').replace('\n', ' ').replace('\r', ' ')
    text = ' '.join(text.split())  # Remove extra whitespace
    return text.strip()

def filter_medical_records(df_main, df_classification):
    """Filter only medical drugs from classification"""
    print(f"Original dataset: {len(df_main):,} records")
    
    # Merge with classification
    df_merged = df_main.merge(
        df_classification[['index', 'is_medicine', 'is_cosmetic', 'is_medical_device', 'is_dietary_supplement']], 
        left_index=True, 
        right_on='index',
        how='left'
    )
    
    # Filter only medicines
    medical_mask = df_merged['is_medicine'] == True
    df_medical = df_merged[medical_mask].copy()
    
    print(f"Medical drugs only: {len(df_medical):,} records ({len(df_medical)/len(df_main)*100:.1f}%)")
    return df_medical

def analyze_field_quality(df, field_name):
    """Analyze quality of specific field"""
    if field_name not in df.columns:
        return {"exists": False}
    
    series = df[field_name].fillna('')
    non_empty = series.astype(str).str.strip().str.len() > 10
    
    return {
        "exists": True,
        "total_records": len(series),
        "non_empty_count": non_empty.sum(),
        "non_empty_percentage": non_empty.mean() * 100,
        "avg_length": series[non_empty].astype(str).str.len().mean() if non_empty.any() else 0
    }

def apply_truncation_strategy(df):
    """Apply smart truncation to text fields"""
    text_fields_limits = {
        'Показання': 1200,          # Most important - 1200 chars
        'Протипоказання': 800,      # Safety info - 800 chars  
        'Фармакотерапевтична група': 400,  # Group info - 400 chars
        'Склад': 600,               # Composition - 600 chars
        'Фармакологічні властивості': 1000  # Pharmacology - 1000 chars
    }
    
    df_truncated = df.copy()
    truncation_stats = {}
    
    for field, max_chars in text_fields_limits.items():
        if field in df_truncated.columns:
            original_lengths = df_truncated[field].astype(str).str.len()
            
            # Apply truncation
            df_truncated[field] = df_truncated[field].astype(str).str[:max_chars]
            
            # Calculate stats
            truncated_count = (original_lengths > max_chars).sum()
            truncation_stats[field] = {
                "max_chars": max_chars,
                "truncated_records": truncated_count,
                "truncated_percentage": truncated_count / len(df_truncated) * 100
            }
            
            print(f"{field}: truncated {truncated_count} records ({truncated_count/len(df_truncated)*100:.1f}%)")
    
    return df_truncated, truncation_stats

def main():
    parser = argparse.ArgumentParser(description="Clean medical dataset for training")
    parser.add_argument("--raw_data", default="data/raw/compendium_all.parquet")
    parser.add_argument("--classification", default="data/interim/comprehensive_eda/product_classification_20250902_200653.csv")
    parser.add_argument("--output", default="data/processed/clean_medical.parquet")
    
    args = parser.parse_args()
    
    print("=== MEDICAL DATASET CLEANING ===")
    
    # Load data
    print("Loading datasets...")
    df_main = pd.read_parquet(args.raw_data)
    df_classification = pd.read_csv(args.classification)
    
    # Filter medical records
    df_medical = filter_medical_records(df_main, df_classification)
    
    # Analyze key fields quality
    print("\n=== FIELD QUALITY ANALYSIS ===")
    key_fields = ['Назва препарату', 'Показання', 'Протипоказання', 'Фармакотерапевтична група']
    
    field_stats = {}
    for field in key_fields:
        stats = analyze_field_quality(df_medical, field)
        field_stats[field] = stats
        
        if stats["exists"]:
            print(f"{field}: {stats['non_empty_count']:,} records ({stats['non_empty_percentage']:.1f}%) with content")
        else:
            print(f"{field}: FIELD NOT FOUND")
    
    # Filter records with minimal required fields (Name + Indications)
    name_col = 'Назва препарату'
    indications_col = 'Показання'
    
    if name_col in df_medical.columns and indications_col in df_medical.columns:
        quality_mask = (
            (df_medical[name_col].astype(str).str.strip().str.len() > 0) &
            (df_medical[indications_col].astype(str).str.strip().str.len() > 10)  # At least 10 chars for indications
        )
        
        df_quality = df_medical[quality_mask].copy()
        print(f"\nHigh quality records (Name + Indications): {len(df_quality):,} ({len(df_quality)/len(df_medical)*100:.1f}%)")
    else:
        print(f"\nERROR: Required fields not found!")
        return
    
    # Apply truncation strategy
    print("\n=== APPLYING TRUNCATION STRATEGY ===")
    df_final, truncation_stats = apply_truncation_strategy(df_quality)
    
    # Clean text fields
    print("\n=== CLEANING TEXT FIELDS ===")
    text_columns = ['Назва препарату', 'Показання', 'Протипоказання', 'Фармакотерапевтична група', 'Склад']
    
    for col in text_columns:
        if col in df_final.columns:
            df_final[col] = df_final[col].apply(clean_text)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df_final.to_parquet(output_path, index=False)
    print(f"\n=== CLEANING COMPLETED ===")
    print(f"Final dataset: {len(df_final):,} high-quality medical records")
    print(f"Saved to: {output_path}")
    
    # Save quality report
    quality_report = {
        "original_records": len(df_main),
        "medical_records": len(df_medical),
        "high_quality_records": len(df_final),
        "field_stats": field_stats,
        "truncation_stats": truncation_stats
    }
    
    import json
    report_path = output_path.parent / "cleaning_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(quality_report, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"Quality report saved to: {report_path}")

if __name__ == "__main__":
    main()
