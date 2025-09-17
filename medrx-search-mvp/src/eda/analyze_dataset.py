#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Аналіз медичного датасету - перевірка протипоказань та структури даних
"""

import pandas as pd
import os
from pathlib import Path
import sys

def analyze_dataset():
    """Аналізує структуру медичного датасету"""
    
    print("🔍 АНАЛІЗ МЕДИЧНОГО ДАТАСЕТУ")
    print("=" * 50)
    
    # Можливі шляхи до датасету
    data_paths = [
        "data/processed/clean_medical.parquet",
        "data/raw/compendium_all.parquet",
        "data/processed/clean_medical_demo.parquet",
        "../data/processed/clean_medical.parquet",
        "../data/raw/compendium_all.parquet"
    ]
    
    found_datasets = []
    
    # Пошук датасетів
    for path in data_paths:
        if os.path.exists(path):
            found_datasets.append(path)
            print(f"✅ Знайдено: {path}")
        else:
            print(f"❌ Не знайдено: {path}")
    
    if not found_datasets:
        print("\n❌ ДАТАСЕТИ НЕ ЗНАЙДЕНО!")
        print("Перевірте чи правильно вказані шляхи до файлів")
        
        # Спробуємо знайти всі .parquet файли
        print("\n🔍 Пошук всіх .parquet файлів:")
        for root, dirs, files in os.walk("."):
            for file in files:
                if file.endswith(".parquet"):
                    full_path = os.path.join(root, file)
                    file_size = os.path.getsize(full_path) / 1024 / 1024  # MB
                    print(f"   📁 {full_path} ({file_size:.1f} MB)")
        
        return
    
    # Аналіз кожного знайденого датасету
    for i, path in enumerate(found_datasets):
        print(f"\n{'='*60}")
        print(f"📊 АНАЛІЗ ДАТАСЕТУ #{i+1}: {path}")
        print(f"{'='*60}")
        
        try:
            # Завантаження датасету
            print("🔄 Завантажую датасет...")
            df = pd.read_parquet(path)
            
            # Загальна інформація
            print(f"📈 Загальна кількість записів: {len(df):,}")
            print(f"📋 Кількість колонок: {len(df.columns)}")
            
            file_size = os.path.getsize(path) / 1024 / 1024  # MB
            print(f"💾 Розмір файлу: {file_size:.1f} MB")
            
            # Показати всі колонки
            print(f"\n📋 ВСІ КОЛОНКИ ({len(df.columns)}):")
            for j, col in enumerate(df.columns):
                non_null = df[col].notna().sum()
                percentage = (non_null / len(df)) * 100
                print(f"   {j:2d}. {col:<30} | Заповнено: {non_null:>6,} ({percentage:5.1f}%)")
            
            # Пошук колонок з протипоказаннями
            print(f"\n🔍 ПОШУК ПРОТИПОКАЗАНЬ:")
            contraindication_keywords = [
                'протипоказ', 'contraind', 'побічн', 'side_effect', 
                'warning', 'попереджен', 'обмежен', 'restriction'
            ]
            
            contraindication_cols = []
            for col in df.columns:
                col_lower = col.lower()
                for keyword in contraindication_keywords:
                    if keyword in col_lower:
                        contraindication_cols.append(col)
                        break
            
            if contraindication_cols:
                print(f"✅ Знайдено {len(contraindication_cols)} колонки з протипоказаннями:")
                
                for col in contraindication_cols:
                    non_null = df[col].notna().sum()
                    percentage = (non_null / len(df)) * 100
                    print(f"\n   📋 Колонка: '{col}'")
                    print(f"   📊 Заповнено: {non_null:,} записів ({percentage:.1f}%)")
                    
                    # Показати приклади даних
                    sample_data = df[col].dropna().head(3)
                    if len(sample_data) > 0:
                        print(f"   📝 ПРИКЛАДИ ДАНИХ:")
                        for idx, text in enumerate(sample_data.values):
                            text_preview = str(text)[:200].replace('\n', ' ')
                            if len(str(text)) > 200:
                                text_preview += "..."
                            print(f"      {idx+1}. {text_preview}")
                    
                    # Статистика довжини тексту
                    text_lengths = df[col].dropna().str.len()
                    if len(text_lengths) > 0:
                        print(f"   📏 Статистика довжини тексту:")
                        print(f"      Мін: {text_lengths.min()} символів")
                        print(f"      Макс: {text_lengths.max()} символів")
                        print(f"      Середня: {text_lengths.mean():.0f} символів")
                        print(f"      Медіана: {text_lengths.median():.0f} символів")
            else:
                print("❌ Колонки з протипоказаннями НЕ ЗНАЙДЕНО!")
                print("🔍 Можливо протипоказання в інших колонках:")
                
                # Показати перші 5 записів для мануального пошуку
                print("\n📝 Перші 5 записів для ручного аналізу:")
                sample_df = df.head(5)
                for col in df.columns:
                    print(f"\n   Колонка: {col}")
                    for idx, val in enumerate(sample_df[col]):
                        if pd.notna(val):
                            val_str = str(val)[:100]
                            if len(str(val)) > 100:
                                val_str += "..."
                            print(f"      Запис {idx+1}: {val_str}")
                        else:
                            print(f"      Запис {idx+1}: [ПУСТО]")
                        if idx >= 2:  # Показати тільки перші 3
                            break
            
            # Пошук основних колонок (назва препарату, показання, тощо)
            print(f"\n🏥 ПОШУК ОСНОВНИХ МЕДИЧНИХ КОЛОНОК:")
            medical_keywords = {
                'назва препарату': ['назва', 'препарат', 'drug', 'name', 'medicine'],
                'показання': ['показання', 'indication', 'використання', 'застосування'],
                'дозування': ['доза', 'дозування', 'dosage', 'dose'],
                'фармакологічна група': ['фармако', 'група', 'group', 'класифікація']
            }
            
            found_medical_cols = {}
            for category, keywords in medical_keywords.items():
                for col in df.columns:
                    col_lower = col.lower()
                    for keyword in keywords:
                        if keyword in col_lower:
                            found_medical_cols[category] = col
                            break
                    if category in found_medical_cols:
                        break
            
            for category, col in found_medical_cols.items():
                non_null = df[col].notna().sum()
                percentage = (non_null / len(df)) * 100
                print(f"   ✅ {category}: '{col}' ({non_null:,} записів, {percentage:.1f}%)")
            
        except Exception as e:
            print(f"❌ Помилка при аналізі {path}: {e}")
    
    print(f"\n{'='*60}")
    print("✅ АНАЛІЗ ЗАВЕРШЕНО!")
    print(f"{'='*60}")

if __name__ == "__main__":
    analyze_dataset()