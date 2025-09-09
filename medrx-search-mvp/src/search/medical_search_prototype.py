#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Medical Search Prototype with Fine-tuned Model
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from typing import List, Dict
import argparse

class MedicalSearchEngine:
    def __init__(self, model_path: str, data_path: str):
        print("Завантаження моделі та даних...")
        self.model = SentenceTransformer(model_path)
        self.df = pd.read_parquet(data_path)
        
        # Створити документи для пошуку  
        self.documents = []
        self.doc_embeddings = None
        self._prepare_search_index()
    
    def _prepare_search_index(self):
        """Підготувати індекс для пошуку"""
        print(f"Підготовка індексу для {len(self.df)} препаратів...")
        
        for _, row in self.df.iterrows():
            parts = []
            if pd.notna(row['Назва препарату']):
                parts.append(str(row['Назва препарату']))
            if 'Фармакотерапевтична група' in row and pd.notna(row['Фармакотерапевтична група']):
                parts.append(str(row['Фармакотерапевтична група'])[:200])
            if pd.notna(row['Показання']):
                parts.append(str(row['Показання'])[:800])
            
            doc_text = ' [SEP] '.join(parts)
            self.documents.append({
                'text': doc_text,
                'name': row['Назва препарату'],
                'indications': row['Показання']
            })
        
        # Генерувати embeddings
        print("Генерація embeddings...")
        doc_texts = [doc['text'] for doc in self.documents]
        self.doc_embeddings = self.model.encode(doc_texts, show_progress_bar=True)
        
        print("Індекс готовий!")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Пошук релевантних препаратів"""
        # Encode query
        query_embedding = self.model.encode(query)
        
        # Compute similarities
        similarities = cos_sim(query_embedding, self.doc_embeddings)[0]
        
        # Get top results
        top_indices = similarities.argsort(descending=True)[:top_k]
        
        results = []
        for idx in top_indices:
            idx = idx.item()
            result = {
                'drug_name': self.documents[idx]['name'],
                'similarity': similarities[idx].item(),
                'indications': self.documents[idx]['indications'][:300] + '...'
            }
            results.append(result)
        
        return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/medical-search-ua-full")
    parser.add_argument("--data", default="data/processed/clean_medical.parquet")
    
    args = parser.parse_args()
    
    # Ініціалізувати пошуковик
    search_engine = MedicalSearchEngine(args.model, args.data)
    
    # Інтерактивний пошук
    print("\n=== МЕДИЧНИЙ ПОШУКОВИК (введіть 'exit' для виходу) ===")
    
    while True:
        query = input("\nВведіть запит: ").strip()
        
        if query.lower() in ['exit', 'quit', 'вихід']:
            break
        
        if not query:
            continue
        
        results = search_engine.search(query, top_k=5)
        
        print(f"\nРезультати для '{query}':")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result['drug_name']}")
            print(f"   Релевантність: {result['similarity']:.3f}")
            print(f"   Показання: {result['indications']}")

if __name__ == "__main__":
    main()
