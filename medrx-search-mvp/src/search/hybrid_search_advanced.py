#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Hybrid Medical Search Engine
BM25 + Vector Search + RRF + CrossEncoder для медичного пошуку
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from sentence_transformers.util import cos_sim
from rank_bm25 import BM25Okapi
from typing import List, Dict, Tuple, Optional
import re
import argparse
from pathlib import Path
import json
import time
from collections import defaultdict

class AdvancedMedicalSearchEngine:
    """
    Hybrid search з BM25 + Vector + RRF + CrossEncoder
    """
    
    def __init__(self, model_path: str, data_path: str, crossencoder_model: str = None):
        print("🔄 Ініціалізація Advanced Medical Search Engine...")
        
        # Load models
        self.sentence_model = SentenceTransformer(model_path)
        self.crossencoder = CrossEncoder(
            crossencoder_model or 'cross-encoder/ms-marco-MiniLM-L-2-v2'
        ) if crossencoder_model != "none" else None
        
        # Load data
        self.df = pd.read_parquet(data_path)
        print(f"📊 Завантажено {len(self.df)} медичних записів")
        
        # Initialize search components
        self.documents = []
        self.doc_embeddings = None
        self.bm25_index = None
        self.contraindications_index = {}
        
        # Search statistics
        self.search_stats = {
            'total_searches': 0,
            'avg_bm25_time': 0,
            'avg_vector_time': 0,
            'avg_crossencoder_time': 0
        }
        
        self._prepare_search_indices()
    
    def _clean_text_for_bm25(self, text: str) -> List[str]:
        """Очистка тексту для BM25 tokenization"""
        if not isinstance(text, str):
            return []
        
        # Видалити [SEP] токени та очистити
        text = text.replace('[SEP]', ' ')
        # Lowercase та видалити зайві символи
        text = re.sub(r'[^\w\s\-]', ' ', text.lower())
        # Tokenize
        tokens = text.split()
        # Видалити дуже короткі токени
        tokens = [token for token in tokens if len(token) > 2]
        
        return tokens
    
    def _prepare_search_indices(self):
        """Підготовка всіх індексів для пошуку"""
        print("🏗️  Підготовка індексів...")
        
        bm25_docs = []
        
        for idx, row in self.df.iterrows():
            # Створити документ для пошуку
            doc_parts = []
            contraindications = ""
            
            if pd.notna(row['Назва препарату']):
                doc_parts.append(str(row['Назва препарату']))
            
            if 'Фармакотерапевтична група' in row and pd.notna(row['Фармакотерапевтична група']):
                group_text = str(row['Фармакотерапевтична група'])[:200]
                doc_parts.append(group_text)
            
            if pd.notna(row['Показання']):
                indications = str(row['Показання'])[:800]
                doc_parts.append(indications)
            
            if 'Протипоказання' in row and pd.notna(row['Протипоказання']):
                contraindications = str(row['Протипоказання'])[:500]
            
            # Документ для vector search
            doc_text = ' [SEP] '.join(doc_parts)
            
            # Документ для BM25 (без [SEP], більш природний)
            bm25_text = ' '.join(doc_parts)
            bm25_tokens = self._clean_text_for_bm25(bm25_text)
            
            self.documents.append({
                'id': idx,
                'text': doc_text,
                'bm25_text': bm25_text,
                'name': row['Назва препарату'],
                'indications': row['Показання'] if pd.notna(row['Показання']) else "",
                'contraindications': contraindications,
                'therapeutic_group': row.get('Фармакотерапевтична група', ""),
                'original_row': row
            })
            
            bm25_docs.append(bm25_tokens)
            self.contraindications_index[idx] = contraindications.lower()
        
        # Створити BM25 індекс
        print("🔍 Створення BM25 індексу...")
        self.bm25_index = BM25Okapi(bm25_docs)
        
        # Створити vector індекс
        print("🧠 Генерація embeddings...")
        doc_texts = [doc['text'] for doc in self.documents]
        self.doc_embeddings = self.sentence_model.encode(
            doc_texts, 
            show_progress_bar=True,
            batch_size=32
        )
        
        print("✅ Всі індекси готові!")
    
    def _bm25_search(self, query: str, top_k: int = 50) -> List[Tuple[int, float]]:
        """BM25 пошук"""
        start_time = time.time()
        
        query_tokens = self._clean_text_for_bm25(query)
        if not query_tokens:
            return []
        
        # BM25 scoring
        scores = self.bm25_index.get_scores(query_tokens)
        
        # Нормалізувати scores (0-1)
        if scores.max() > 0:
            scores = scores / scores.max()
        
        # Топ результати
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]
        
        # Статистика
        elapsed = time.time() - start_time
        self.search_stats['avg_bm25_time'] = (
            self.search_stats['avg_bm25_time'] * self.search_stats['total_searches'] + elapsed
        ) / (self.search_stats['total_searches'] + 1)
        
        return results
    
    def _vector_search(self, query: str, top_k: int = 50) -> List[Tuple[int, float]]:
        """Vector пошук"""
        start_time = time.time()
        
        # Encode query
        query_embedding = self.sentence_model.encode(query)
        
        # Compute similarities
        similarities = cos_sim(query_embedding, self.doc_embeddings)[0]
        
        # Топ результати
        top_indices = similarities.argsort(descending=True)[:top_k]
        results = [(int(idx), float(similarities[idx])) for idx in top_indices]
        
        # Статистика
        elapsed = time.time() - start_time
        self.search_stats['avg_vector_time'] = (
            self.search_stats['avg_vector_time'] * self.search_stats['total_searches'] + elapsed
        ) / (self.search_stats['total_searches'] + 1)
        
        return results
    
    def _reciprocal_rank_fusion(self, bm25_results: List[Tuple[int, float]], 
                              vector_results: List[Tuple[int, float]], 
                              k: int = 60) -> List[Tuple[int, float]]:
        """Reciprocal Rank Fusion для об'єднання результатів"""
        
        # Створити ранкінги
        bm25_ranks = {doc_id: rank + 1 for rank, (doc_id, score) in enumerate(bm25_results)}
        vector_ranks = {doc_id: rank + 1 for rank, (doc_id, score) in enumerate(vector_results)}
        
        # Всі унікальні документи
        all_docs = set(bm25_ranks.keys()) | set(vector_ranks.keys())
        
        # RRF scoring
        rrf_scores = {}
        for doc_id in all_docs:
            rrf_score = 0
            if doc_id in bm25_ranks:
                rrf_score += 1 / (k + bm25_ranks[doc_id])
            if doc_id in vector_ranks:
                rrf_score += 1 / (k + vector_ranks[doc_id])
            
            rrf_scores[doc_id] = rrf_score
        
        # Сортувати за RRF score
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_results
    
    def _crossencoder_rerank(self, query: str, candidate_results: List[Tuple[int, float]], 
                           top_k: int = 10) -> List[Tuple[int, float]]:
        """CrossEncoder re-ranking"""
        if not self.crossencoder or len(candidate_results) == 0:
            return candidate_results[:top_k]
        
        start_time = time.time()
        
        # Підготувати пари query-document
        query_doc_pairs = []
        doc_ids = []
        
        for doc_id, _ in candidate_results[:20]:  # Обмежити для швидкості
            doc_text = self.documents[doc_id]['bm25_text'][:500]  # Truncate for CrossEncoder
            query_doc_pairs.append([query, doc_text])
            doc_ids.append(doc_id)
        
        if not query_doc_pairs:
            return []
        
        # CrossEncoder scoring
        ce_scores = self.crossencoder.predict(query_doc_pairs)
        
        # Об'єднати з doc_ids та сортувати
        reranked = list(zip(doc_ids, ce_scores))
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        # Статистика
        elapsed = time.time() - start_time
        self.search_stats['avg_crossencoder_time'] = (
            self.search_stats['avg_crossencoder_time'] * self.search_stats['total_searches'] + elapsed
        ) / (self.search_stats['total_searches'] + 1)
        
        return reranked[:top_k]
    
    def _check_contraindications(self, query: str, doc_id: int) -> Dict[str, any]:
        """Перевірка протипоказань"""
        contraindications = self.contraindications_index.get(doc_id, "")
        
        # Прості heuristics для попереджень
        warning_keywords = [
            'вагітність', 'годування груддю', 'діти', 'печінкова недостатність',
            'ниркова недостатність', 'серцева недостатність', 'алергія'
        ]
        
        warnings = []
        for keyword in warning_keywords:
            if keyword in contraindications:
                warnings.append(keyword)
        
        return {
            'has_contraindications': len(contraindications) > 0,
            'warnings': warnings,
            'contraindications_text': contraindications[:200] + "..." if len(contraindications) > 200 else contraindications
        }
    
    def search(self, query: str, top_k: int = 5, use_crossencoder: bool = True,
               include_safety: bool = True) -> Dict:
        """Головний метод пошуку"""
        
        search_start = time.time()
        self.search_stats['total_searches'] += 1
        
        print(f"🔍 Пошук: '{query}'")
        
        # Stage 1: BM25 Search
        print("   📝 BM25 пошук...")
        bm25_results = self._bm25_search(query, top_k=50)
        
        # Stage 2: Vector Search  
        print("   🧠 Vector пошук...")
        vector_results = self._vector_search(query, top_k=50)
        
        # Stage 3: Reciprocal Rank Fusion
        print("   🔄 RRF об'єднання...")
        rrf_results = self._reciprocal_rank_fusion(bm25_results, vector_results)
        
        # Stage 4: CrossEncoder Re-ranking
        if use_crossencoder and self.crossencoder:
            print("   🎯 CrossEncoder re-ranking...")
            final_results = self._crossencoder_rerank(query, rrf_results, top_k=top_k)
        else:
            final_results = rrf_results[:top_k]
        
        # Stage 5: Prepare results with safety info
        search_results = []
        for doc_id, score in final_results:
            doc = self.documents[doc_id]
            
            result = {
                'rank': len(search_results) + 1,
                'drug_name': doc['name'],
                'score': float(score),
                'indications': doc['indications'][:300] + "..." if len(doc['indications']) > 300 else doc['indications'],
                'therapeutic_group': doc['therapeutic_group'][:100] if doc['therapeutic_group'] else ""
            }
            
            # Додати safety information
            if include_safety:
                safety_info = self._check_contraindications(query, doc_id)
                result['safety'] = safety_info
            
            search_results.append(result)
        
        total_time = time.time() - search_start
        
        return {
            'query': query,
            'results': search_results,
            'search_info': {
                'total_time': f"{total_time:.3f}s",
                'bm25_candidates': len(bm25_results),
                'vector_candidates': len(vector_results),
                'rrf_candidates': len(rrf_results),
                'final_results': len(final_results),
                'used_crossencoder': use_crossencoder and self.crossencoder is not None
            },
            'disclaimer': "⚠️  Завжди консультуйтесь з лікарем перед прийомом ліків"
        }
    
    def get_search_statistics(self) -> Dict:
        """Статистика пошуку"""
        return {
            'total_searches': self.search_stats['total_searches'],
            'avg_times': {
                'bm25': f"{self.search_stats['avg_bm25_time']:.4f}s",
                'vector': f"{self.search_stats['avg_vector_time']:.4f}s", 
                'crossencoder': f"{self.search_stats['avg_crossencoder_time']:.4f}s"
            },
            'index_info': {
                'total_documents': len(self.documents),
                'embedding_dimension': self.doc_embeddings.shape[1] if self.doc_embeddings is not None else 0,
                'has_crossencoder': self.crossencoder is not None
            }
        }

def print_search_results(search_response: Dict):
    """Красиво вивести результати пошуку"""
    results = search_response['results']
    search_info = search_response['search_info']
    
    print(f"\n{'='*60}")
    print(f"🔍 Результати для: '{search_response['query']}'")
    print(f"⏱️  Час пошуку: {search_info['total_time']}")
    print(f"📊 Кандидатів: BM25={search_info['bm25_candidates']}, Vector={search_info['vector_candidates']}")
    print(f"🎯 CrossEncoder: {'Так' if search_info['used_crossencoder'] else 'Ні'}")
    print(f"{'='*60}")
    
    if not results:
        print("❌ Результатів не знайдено")
        return
    
    for result in results:
        print(f"\n{result['rank']}. 💊 {result['drug_name']}")
        print(f"   📈 Релевантність: {result['score']:.3f}")
        print(f"   🏥 Показання: {result['indications']}")
        
        if result.get('therapeutic_group'):
            print(f"   📋 Група: {result['therapeutic_group']}")
        
        # Safety warnings
        if 'safety' in result:
            safety = result['safety']
            if safety['warnings']:
                print(f"   ⚠️  Попередження: {', '.join(safety['warnings'])}")
            if safety['contraindications_text']:
                print(f"   🚫 Протипоказання: {safety['contraindications_text']}")
    
    print(f"\n{search_response['disclaimer']}")

def main():
    parser = argparse.ArgumentParser(description="Advanced Hybrid Medical Search")
    parser.add_argument("--model", default="models/medical-search-ua-full", 
                       help="Path to fine-tuned sentence transformer")
    parser.add_argument("--data", default="data/processed/clean_medical.parquet",
                       help="Path to medical data")
    parser.add_argument("--crossencoder", default="cross-encoder/ms-marco-MiniLM-L-2-v2",
                       help="CrossEncoder model (use 'none' to disable)")
    parser.add_argument("--stats", action="store_true", 
                       help="Show search statistics")
    
    args = parser.parse_args()
    
    # Ініціалізувати пошуковик
    search_engine = AdvancedMedicalSearchEngine(
        model_path=args.model,
        data_path=args.data, 
        crossencoder_model=args.crossencoder
    )
    
    if args.stats:
        stats = search_engine.get_search_statistics()
        print("\n📊 СТАТИСТИКА ПОШУКОВИКА:")
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        return
    
    # Інтерактивний пошук
    print("\n🏥 ADVANCED MEDICAL SEARCH ENGINE")
    print("Введіть 'exit' для виходу, 'stats' для статистики")
    print("Команди: --no-ce (без CrossEncoder), --no-safety (без safety check)")
    
    while True:
        user_input = input("\n🔍 Введіть запит: ").strip()
        
        if user_input.lower() in ['exit', 'quit', 'вихід']:
            break
        
        if user_input.lower() == 'stats':
            stats = search_engine.get_search_statistics()
            print(json.dumps(stats, indent=2, ensure_ascii=False))
            continue
        
        if not user_input:
            continue
        
        # Парсинг команд
        use_crossencoder = '--no-ce' not in user_input
        include_safety = '--no-safety' not in user_input
        query = user_input.replace('--no-ce', '').replace('--no-safety', '').strip()
        
        # Пошук
        try:
            search_response = search_engine.search(
                query, 
                top_k=5,
                use_crossencoder=use_crossencoder,
                include_safety=include_safety
            )
            print_search_results(search_response)
            
        except Exception as e:
            print(f"❌ Помилка пошуку: {e}")

if __name__ == "__main__":
    main()