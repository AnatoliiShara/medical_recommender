#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrated Medical Assistant with Gemini AI
Базується на вашому відмінному hybrid search engine + додає Gemini природне спілкування
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from sentence_transformers.util import cos_sim
from rank_bm25 import BM25Okapi
from typing import List, Dict, Tuple, Optional, Any
import re
import argparse
from pathlib import Path
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
import os
import sys

import google.generativeai as genai
from dotenv import load_dotenv

@dataclass
class ConversationMessage:
    """Повідомлення в розмові"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime
    search_data: Optional[Dict] = None

class AdvancedMedicalSearchEngine:
    """
    Hybrid search з BM25 + Vector + RRF + CrossEncoder
    (Ваш перевірений та надійний search engine)
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
                'side_effects': row.get('Побічні реакції', "") if pd.notna(row.get('Побічні реакції', "")) else "",
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
               include_safety: bool = True, verbose: bool = False) -> Dict:
        """Головний метод пошуку"""
        
        search_start = time.time()
        self.search_stats['total_searches'] += 1
        
        if verbose:
            print(f"🔍 Пошук: '{query}'")
        
        # Stage 1: BM25 Search
        if verbose:
            print("   📝 BM25 пошук...")
        bm25_results = self._bm25_search(query, top_k=50)
        
        # Stage 2: Vector Search  
        if verbose:
            print("   🧠 Vector пошук...")
        vector_results = self._vector_search(query, top_k=50)
        
        # Stage 3: Reciprocal Rank Fusion
        if verbose:
            print("   🔄 RRF об'єднання...")
        rrf_results = self._reciprocal_rank_fusion(bm25_results, vector_results)
        
        # Stage 4: CrossEncoder Re-ranking
        if use_crossencoder and self.crossencoder:
            if verbose:
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
                'therapeutic_group': doc['therapeutic_group'][:100] if doc['therapeutic_group'] else "",
                'side_effects': doc['side_effects'][:300] + "..." if len(doc['side_effects']) > 300 else doc['side_effects']
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

class GeminiMedicalAssistant:
    """
    Gemini AI Assistant поверх вашого надійного hybrid search engine
    """
    
    def __init__(self, gemini_api_key: str, search_engine: AdvancedMedicalSearchEngine):
        """
        Ініціалізація асистента
        
        Args:
            gemini_api_key: API ключ для Gemini
            search_engine: Ваш надійний hybrid search engine
        """
        print("🤖 Ініціалізація Gemini Medical Assistant...")
        
        # Configure Gemini
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Search engine
        self.search_engine = search_engine
        
        # Conversation history
        self.conversation_history: List[ConversationMessage] = []
        
        print("✅ Gemini Medical Assistant готовий!")
    
    def enhance_user_query(self, user_query: str) -> str:
        """
        Покращити запит користувача для кращого пошуку
        """
        enhancement_prompt = f"""
Ти - медичний експерт. Перетвори запит користувача в оптимальний пошуковий запит для знаходження релевантних медичних препаратів.

Правила:
1. Витягни головні медичні терміни (захворювання, симптоми)
2. Додай синоніми та медичні терміни
3. Видали зайві слова ("порадьте", "скажіть", "допоможіть")
4. Залиши лише ключові медичні концепти
5. Пиши українською

Приклади:
"Порадьте препарат від діареї" → "діарея лікування протидіарейні засоби"
"Що приймати від високого тиску?" → "артеріальна гіпертензія гіпертонія лікування"
"Допоможіть з болем в животі" → "біль живот шлунок спазм"
"Чим лікувати застуду?" → "застуда ГРВІ вірусна інфекція лікування"

Запит користувача: "{user_query}"

Покращений пошуковий запит (тільки ключові слова):"""

        try:
            response = self.model.generate_content(enhancement_prompt)
            enhanced_query = response.text.strip()
            
            print(f"🔍 Оригінальний запит: '{user_query}'")
            print(f"🎯 Покращений запит: '{enhanced_query}'")
            
            return enhanced_query
        except Exception as e:
            print(f"⚠️  Помилка покращення запиту: {e}")
            return user_query
    
    def generate_medical_response(self, user_query: str, search_results: Dict, 
                                user_profile: Optional[Dict] = None) -> str:
        """
        Генерувати природномовну відповідь на основі результатів пошуку
        """
        
        # Підготувати дані про знайдені препарати
        medications_info = ""
        if search_results['results']:
            medications_info = "ЗНАЙДЕНІ ПРЕПАРАТИ:\n"
            for i, result in enumerate(search_results['results'], 1):
                medications_info += f"""
{i}. ПРЕПАРАТ: {result['drug_name']}
   ПОКАЗАННЯ: {result['indications']}
   ФАРМГРУПА: {result.get('therapeutic_group', 'Не вказано')}"""
                
                if 'safety' in result:
                    safety = result['safety']
                    if safety['contraindications_text']:
                        medications_info += f"\n   ПРОТИПОКАЗАННЯ: {safety['contraindications_text']}"
                    if safety['warnings']:
                        medications_info += f"\n   ПОПЕРЕДЖЕННЯ: {', '.join(safety['warnings'])}"
                
                if result.get('side_effects'):
                    medications_info += f"\n   ПОБІЧНІ ЕФЕКТИ: {result['side_effects']}"
                
                medications_info += "\n"
        else:
            medications_info = "НЕ ЗНАЙДЕНО РЕЛЕВАНТНИХ ПРЕПАРАТІВ"
        
        # Інформація про профіль користувача
        profile_info = ""
        if user_profile:
            profile_info = f"""
ПРОФІЛЬ КОРИСТУВАЧА:
- Хронічні захворювання: {user_profile.get('chronic_conditions', 'Не вказано')}
- Алергії: {user_profile.get('allergies', 'Не вказано')}
- Вік: {user_profile.get('age', 'Не вказано')}
- Стать: {user_profile.get('gender', 'Не вказано')}
"""
        
        # Контекст попередніх повідомлень
        context = ""
        if self.conversation_history:
            context = "КОНТЕКСТ РОЗМОВИ:\n"
            for msg in self.conversation_history[-3:]:  # Останні 3 повідомлення
                context += f"- {msg.role}: {msg.content[:100]}...\n"
        
        # Головний промпт для Gemini
        medical_prompt = f"""Ти - досвідчений медичний консультант та фармацевт. Твоя задача - надати користувачеві зрозумілу, корисну та безпечну інформацію про медичні препарати.

{context}

{profile_info}

ЗАПИТ КОРИСТУВАЧА: "{user_query}"

{medications_info}

ІНСТРУКЦІЇ ДЛЯ ВІДПОВІДІ:

1. ПРИРОДНЕ СПІЛКУВАННЯ:
   - Спілкуйся тепло та з розумінням
   - Використовуй прості, зрозумілі терміни
   - Будь емпатичним до проблеми користувача

2. СТРУКТУРА ВІДПОВІДІ:
   - Почни з розуміння проблеми користувача
   - Поясни знайдені препарати простими словами
   - Обов'язково розкажи про протипоказання та побічні ефекти
   - Дай практичні поради

3. БЕЗПЕКА:
   - ЗАВЖДИ наголошуй на необхідності консультації з лікарем
   - Якщо є конфлікти з профілем користувача - обов'язково попереди
   - Не діагностуй та не призначай лікування
   - Пояснюй ризики та обмеження

4. ЯКЩО ПРЕПАРАТІВ НЕ ЗНАЙДЕНО:
   - Поясни можливі причини
   - Дай загальні рекомендації
   - Наголоси на консультації з лікарем
   - Запропонуй альтернативні дії

5. МОВА:
   - Пиши українською
   - Використовуй медичну термінологію коли потрібно, але з поясненнями
   - Будь конкретним та корисним

Створи відповідь, яка справді допоможе користувачеві та забезпечить його безпеку:"""

        try:
            response = self.model.generate_content(medical_prompt)
            return response.text.strip()
        except Exception as e:
            return f"❌ Вибачте, виникла помилка при генерації відповіді: {e}\n\nПроте я знайшов наступну інформацію:\n{self._format_basic_results(search_results)}"
    
    def _format_basic_results(self, search_results: Dict) -> str:
        """Базове форматування результатів якщо Gemini не працює"""
        if not search_results['results']:
            return "❌ На жаль, не знайдено релевантних препаратів для вашого запиту."
        
        formatted = "💊 ЗНАЙДЕНІ ПРЕПАРАТИ:\n\n"
        for i, result in enumerate(search_results['results'], 1):
            formatted += f"{i}. {result['drug_name']}\n"
            formatted += f"   📝 Показання: {result['indications']}\n"
            
            if 'safety' in result and result['safety']['contraindications_text']:
                formatted += f"   ⚠️  Протипоказання: {result['safety']['contraindications_text']}\n"
            
            formatted += "\n"
        
        formatted += "⚠️  ВАЖЛИВО: Обов'язково проконсультуйтеся з лікарем перед прийомом будь-яких ліків!"
        return formatted
    
    def chat(self, user_message: str, user_profile: Optional[Dict] = None) -> str:
        """
        Головний метод для спілкування з асистентом
        """
        print(f"\n🗣️  Користувач: {user_message}")
        
        try:
            # 1. Покращити запит
            enhanced_query = self.enhance_user_query(user_message)
            
            # 2. Пошук препаратів (з вашим надійним search engine!)
            print("🔍 Пошук релевантних препаратів...")
            search_results = self.search_engine.search(
                enhanced_query, 
                top_k=5,
                use_crossencoder=True,
                include_safety=True,
                verbose=False
            )
            
            # 3. Генерація природномовної відповіді
            print("🤖 Генерація відповіді...")
            response = self.generate_medical_response(
                user_message, 
                search_results, 
                user_profile
            )
            
            # 4. Зберегти в історії
            user_msg = ConversationMessage(
                role='user',
                content=user_message,
                timestamp=datetime.now()
            )
            
            assistant_msg = ConversationMessage(
                role='assistant',
                content=response,
                timestamp=datetime.now(),
                search_data=search_results
            )
            
            self.conversation_history.extend([user_msg, assistant_msg])
            
            print(f"🤖 Асистент відповів!")
            return response
            
        except Exception as e:
            error_response = f"❌ Вибачте, виникла помилка: {e}\n\n💡 Спробуйте переформулювати ваш запит або зверніться до лікаря."
            print(f"❌ Помилка: {e}")
            return error_response
    
    def get_conversation_summary(self) -> Dict:
        """Отримати підсумок розмови"""
        return {
            'total_messages': len(self.conversation_history),
            'conversation_start': self.conversation_history[0].timestamp.isoformat() if self.conversation_history else None,
            'last_message': self.conversation_history[-1].timestamp.isoformat() if self.conversation_history else None,
            'topics_discussed': len(set(msg.search_data['query'] for msg in self.conversation_history if msg.search_data))
        }

def main():
    """Головна функція"""
    parser = argparse.ArgumentParser(description="Integrated Medical Assistant with Gemini")
    parser.add_argument("--model", default="../../models/medical-search-ua-full", 
                       help="Path to fine-tuned model")
    parser.add_argument("--data", default="../../data/processed/clean_medical.parquet",
                       help="Path to medical data")
    parser.add_argument("--env", default="../../.env", 
                       help="Path to .env file")
    parser.add_argument("--mode", default="assistant", choices=["assistant", "search"],
                       help="Mode: assistant (with Gemini) or search (original)")
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv(args.env)
    
    # Initialize search engine (ваш надійний движок)
    print("🔄 Ініціалізація вашого надійного hybrid search engine...")
    search_engine = AdvancedMedicalSearchEngine(
        model_path=args.model,
        data_path=args.data,
        crossencoder_model=os.getenv('CROSSENCODER_MODEL', 'cross-encoder/ms-marco-MiniLM-L-2-v2')
    )
    
    if args.mode == "search":
        # Режим оригінального пошуку
        print("\n🔍 РЕЖИМ ОРИГІНАЛЬНОГО ПОШУКУ")
        print("Введіть 'exit' для виходу, 'stats' для статистики")
        
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
            
            # Пошук
            search_results = search_engine.search(user_input, verbose=True)
            
            # Показати результати
            print(f"\n{'='*60}")
            print(f"🔍 Результати для: '{search_results['query']}'")
            print(f"{'='*60}")
            
            for result in search_results['results']:
                print(f"\n{result['rank']}. 💊 {result['drug_name']}")
                print(f"   📈 Релевантність: {result['score']:.3f}")
                print(f"   🏥 Показання: {result['indications']}")
                
                if 'safety' in result and result['safety']['contraindications_text']:
                    print(f"   🚫 Протипоказання: {result['safety']['contraindications_text']}")
        
        return
    
    # Режим асистента з Gemini
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    
    if not gemini_api_key or gemini_api_key == 'your_actual_gemini_api_key_here':
        print("❌ Помилка: GEMINI_API_KEY не знайдено або не налаштовано в .env файлі")
        print("📝 Будь ласка, встановіть ваш справжній API ключ в .env файлі")
        return
    
    # Initialize assistant
    assistant = GeminiMedicalAssistant(gemini_api_key, search_engine)
    
    # Interactive chat
    print("\n" + "="*60)
    print("🏥 GEMINI MEDICAL ASSISTANT")
    print("💬 Розмовляйте природною мовою про медичні проблеми")
    print("📝 Введіть 'exit' для виходу, 'summary' для підсумку розмови")
    print("⚠️  ВАЖЛИВО: Це лише інформаційний асистент, не замінює консультацію лікаря!")
    print("="*60)
    
    # Optional user profile
    user_profile = None
    setup_profile = input("\n❓ Бажаєте налаштувати медичний профіль? (y/n): ").strip().lower()
    if setup_profile == 'y':
        user_profile = {}
        user_profile['age'] = input("🎂 Вік: ").strip() or "Не вказано"
        user_profile['gender'] = input("👤 Стать (М/Ж): ").strip() or "Не вказано" 
        user_profile['chronic_conditions'] = input("🏥 Хронічні захворювання (через кому): ").strip() or "Немає"
        user_profile['allergies'] = input("⚠️  Алергії (через кому): ").strip() or "Немає"
        print("✅ Профіль створено!")
    
    # Chat loop
    while True:
        try:
            user_input = input(f"\n💬 Ви: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'вихід']:
                print("👋 До зустрічі! Будьте здорові!")
                break
            
            if user_input.lower() in ['summary', 'підсумок']:
                summary = assistant.get_conversation_summary()
                print(f"\n📊 ПІДСУМОК РОЗМОВИ:")
                print(json.dumps(summary, indent=2, ensure_ascii=False))
                continue
            
            if not user_input:
                continue
            
            # Get response from assistant
            response = assistant.chat(user_input, user_profile)
            print(f"\n🤖 Медичний асистент:\n{response}")
            
        except KeyboardInterrupt:
            print("\n👋 До зустрічі! Будьте здорові!")
            break
        except Exception as e:
            print(f"\n❌ Помилка: {e}")

if __name__ == "__main__":
    main()