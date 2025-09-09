#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test fine-tuned medical model quality
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import pandas as pd

def test_model_quality():
    print("=== ТЕСТУВАННЯ ЯКОСТІ FINE-TUNED МОДЕЛІ ===")
    
    # Load both models
    print("Завантаження моделей...")
    base_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    finetuned_model = SentenceTransformer('models/medical-search-ua-fixed')
    
    # Test queries
    test_queries = [
        "ліки від головного болю",
        "препарати при високому тиску",
        "що приймати при інфекції",
        "засоби від алергії",
        "лікування серцевих захворювань"
    ]
    
    # Sample documents from our training data
    with open('data/processed/training/improved_training_pairs.jsonl', 'r') as f:
        sample_pairs = [json.loads(line) for line in f][:20]
    
    print(f"Тестування на {len(test_queries)} запитах та {len(sample_pairs)} документах\n")
    
    for query in test_queries:
        print(f"ЗАПИТ: '{query}'")
        
        # Encode query with both models
        base_query_emb = base_model.encode(query)
        ft_query_emb = finetuned_model.encode(query)
        
        # Find best matches in sample documents
        base_scores = []
        ft_scores = []
        
        for pair in sample_pairs[:10]:  # Test with 10 documents
            doc = pair['positive']
            
            base_doc_emb = base_model.encode(doc)
            ft_doc_emb = finetuned_model.encode(doc)
            
            base_score = cos_sim(base_query_emb, base_doc_emb).item()
            ft_score = cos_sim(ft_query_emb, ft_doc_emb).item()
            
            base_scores.append((base_score, pair['metadata']['drug_name']))
            ft_scores.append((ft_score, pair['metadata']['drug_name']))
        
        # Sort by similarity
        base_scores.sort(reverse=True)
        ft_scores.sort(reverse=True)
        
        print("  Базова модель - топ 3:")
        for score, drug in base_scores[:3]:
            print(f"    {score:.3f}: {drug}")
        
        print("  Fine-tuned модель - топ 3:")
        for score, drug in ft_scores[:3]:
            print(f"    {score:.3f}: {drug}")
        
        print()

if __name__ == "__main__":
    test_model_quality()
