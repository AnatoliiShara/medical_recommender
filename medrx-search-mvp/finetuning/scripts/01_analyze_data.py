"""
EDA для MedRx compendium data - аналіз потенціалу для файнтюнінгу
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
from collections import Counter
import re

def analyze_compendium():
    """Аналіз compendium dataset"""
    
    print("="*80)
    print("📊 АНАЛІЗ COMPENDIUM DATASET ДЛЯ ФАЙНТЮНІНГУ")
    print("="*80)
    
    # Load data
    parquet_path = "data/raw/compendium_all.parquet"
    print(f"\n📂 Завантаження: {parquet_path}")
    
    df = pd.read_parquet(parquet_path)
    print(f"✅ Завантажено {len(df):,} записів")
    
    # Basic info
    print(f"\n📋 Структура даних:")
    print(f"   Колонок: {len(df.columns)}")
    print(f"   Пам'ять: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Check key columns
    key_columns = ['Назва препарату', 'Показання', 'Протипоказання', 
                   'Фармакотерапевтична група', 'Склад']
    
    print(f"\n🔍 Ключові поля (для training data):")
    for col in key_columns:
        if col in df.columns:
            non_null = df[col].notna().sum()
            pct = (non_null / len(df)) * 100
            
            if non_null > 0:
                avg_len = df[col].dropna().astype(str).str.len().mean()
                print(f"   {col:35s}: {non_null:5,} ({pct:5.1f}%) | avg {avg_len:4.0f} chars")
            else:
                print(f"   {col:35s}: {non_null:5,} ({pct:5.1f}%)")
    
    # Analyze Показання (main source for queries)
    if 'Показання' in df.columns:
        indications = df['Показання'].dropna().astype(str)
        
        print(f"\n💊 Аналіз поля 'Показання' (джерело для queries):")
        print(f"   Препаратів з показаннями: {len(indications):,}")
        print(f"   Середня довжина: {indications.str.len().mean():.0f} символів")
        print(f"   Медіана довжини: {indications.str.len().median():.0f} символів")
        
        # Sample indication
        sample = indications.iloc[0]
        print(f"\n   Приклад показання:")
        print(f"   {sample[:200]}...")
    
    # Training data potential
    drugs_with_indications = df['Показання'].notna().sum()
    queries_per_drug = 7  # Будемо генерувати 7 queries на препарат
    
    print(f"\n📈 ПОТЕНЦІАЛ ДЛЯ TRAINING DATA:")
    print(f"   Препаратів з показаннями: {drugs_with_indications:,}")
    print(f"   Queries на препарат: {queries_per_drug}")
    print(f"   Очікувана кількість queries: {drugs_with_indications * queries_per_drug:,}")
    print(f"   Training pairs (з negatives): ~{drugs_with_indications * queries_per_drug:,}")
    
    # Therapeutic groups distribution
    if 'Фармакотерапевтична група' in df.columns:
        groups = df['Фармакотерапевтична група'].dropna()
        top_groups = groups.value_counts().head(10)
        
        print(f"\n🏥 Топ-10 терапевтичних груп:")
        for i, (group, count) in enumerate(top_groups.items(), 1):
            pct = (count / len(df)) * 100
            print(f"   {i:2d}. {group[:50]:50s} | {count:4d} ({pct:.1f}%)")
    
    # Save summary
    summary = {
        "total_drugs": len(df),
        "drugs_with_indications": int(drugs_with_indications),
        "estimated_training_queries": int(drugs_with_indications * queries_per_drug),
        "columns": list(df.columns)
    }
    
    output_path = "data/interim/eda/compendium_analysis.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Звіт збережено: {output_path}")
    
    print("\n" + "="*80)
    print("✅ АНАЛІЗ ЗАВЕРШЕНО")
    print("="*80)
    print(f"\n🎯 Висновок: Можемо згенерувати ~{drugs_with_indications * queries_per_drug:,} training pairs")
    print(f"   Це достатньо для якісного файнтюнінгу!")

if __name__ == "__main__":
    analyze_compendium()
