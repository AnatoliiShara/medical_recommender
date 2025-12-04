#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_quality_gold_v3.py — Генератор Gold Set з clinical dicts.

Використовує:
  - data/dicts/clinical/{topic}/{topic}_positive.txt — куровані INN/brands
  - data/dicts/clinical/{topic}/{topic}_trigger.txt — для генерації запитів
  - Пошук в parquet по Назві та Складі

Результат: Точний gold set з курованих списків.
"""

import json
import re
import argparse
import random
import hashlib
from pathlib import Path
from typing import List, Dict, Set, Optional
from collections import defaultdict

import pandas as pd


# ============================================================
# TOPIC CONFIGURATION
# ============================================================
TOPICS_CONFIG = {
    "diarrhea": {
        "dir": "diarrhea",
        "canonical": "diarrhea",
    },
    "common_cold_urti": {
        "dir": "common_cold_urti",
        "canonical": "cold_flu",
    },
    "cough_bronchitis": {
        "dir": "cough_bronchitis",
        "canonical": "bronchitis",
    },
    "headache_migraine": {
        "dir": "headache_migraine",
        "canonical": "pain",
    },
    "acid_reflux_heartburn": {
        "dir": "acid_reflux_heartburn",
        "canonical": "gerd",
    },
    "gastritis_acid": {
        "dir": "gastritis_acid",
        "canonical": "gerd",
    },
    "allergic_rhinitis": {
        "dir": "allergic_rhinitis",
        "canonical": "allergy",
    },
    "hypertension": {
        "dir": "hypertension",
        "canonical": "hypertension",
    },
    "constipation": {
        "dir": "constipation",
        "canonical": "constipation",
    },
    "uti_cystitis": {
        "dir": "uti_cystitis",
        "canonical": "cystitis",
    },
    "sore_throat_pharyngitis": {
        "dir": "sore_throat_pharyngitis", 
        "canonical": "cold_flu",
    },
    "traveler_diarrhea": {
        "dir": "traveler_diarrhea",
        "canonical": "diarrhea",
    },
    "bacterial_diarrhea": {
        "dir": "bacterial_diarrhea",
        "canonical": "diarrhea",
    },
    "oral_candidiasis": {
        "dir": "oral_candidiasis",
        "canonical": "candidiasis",
    },
}


def normalize_text(text: str) -> str:
    """Нормалізує текст для пошуку."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[®™©()«»"\'\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def load_positive_terms(filepath: Path) -> List[str]:
    """Завантажує positive terms з файлу."""
    terms = []
    if not filepath.exists():
        return terms
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Handle multiple terms separated by /
            if '/' in line:
                parts = line.split('/')
                for part in parts:
                    part = part.strip()
                    if part and len(part) >= 2:
                        terms.append(part)
            else:
                if len(line) >= 2:
                    terms.append(line)
    
    return list(set(terms))


def load_trigger_terms(filepath: Path) -> List[str]:
    """Завантажує trigger terms з файлу."""
    terms = []
    if not filepath.exists():
        return terms
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if len(line) >= 2:
                terms.append(line)
    
    return list(set(terms))


def search_docs_by_terms(
    df: pd.DataFrame,
    terms: List[str],
    search_columns: List[str] = ['Назва препарату', 'Склад'],
) -> Set[int]:
    """
    Шукає препарати по термінах в заданих колонках.
    """
    found_ids = set()
    
    # Prepare normalized columns
    norm_cols = {}
    for col in search_columns:
        if col in df.columns:
            norm_cols[col] = df[col].fillna('').apply(normalize_text)
    
    for term in terms:
        term_norm = normalize_text(term)
        if len(term_norm) < 2:
            continue
        
        for col, norm_series in norm_cols.items():
            # Use word boundary matching for short terms
            if len(term_norm) <= 4:
                # Exact word match
                pattern = r'\b' + re.escape(term_norm) + r'\b'
                try:
                    mask = norm_series.str.contains(pattern, regex=True, na=False)
                except:
                    mask = norm_series.str.contains(term_norm, regex=False, na=False)
            else:
                # Substring match for longer terms
                mask = norm_series.str.contains(term_norm, regex=False, na=False)
            
            matches = df[mask].index.tolist()
            found_ids.update(matches)
    
    return found_ids


def generate_query_variants(
    triggers: List[str],
    n_queries: int = 30,
) -> List[str]:
    """Генерує запити з triggers."""
    templates = [
        "{symptom}",
        "від {symptom}",
        "що допоможе від {symptom}",
        "ліки від {symptom}",
        "препарат від {symptom}",
        "засіб від {symptom}",
        "таблетки від {symptom}",
        "як лікувати {symptom}",
        "що прийняти при {symptom}",
        "порадьте щось від {symptom}",
        "сильний {symptom} що робити",
        "{symptom} у дорослого",
        "{symptom} не проходить",
        "швидко зняти {symptom}",
        "ефективне від {symptom}",
        "чим лікувати {symptom}",
    ]
    
    if not triggers:
        return []
    
    queries = []
    used = set()
    
    attempts = 0
    max_attempts = n_queries * 20
    
    while len(queries) < n_queries and attempts < max_attempts:
        attempts += 1
        
        template = random.choice(templates)
        symptom = random.choice(triggers)
        
        query = template.format(symptom=symptom)
        query_norm = normalize_text(query)
        
        if query_norm not in used and len(query) > 5:
            used.add(query_norm)
            queries.append(query)
    
    return queries[:n_queries]


def generate_qid(query: str) -> str:
    return hashlib.md5(query.encode('utf-8')).hexdigest()[:12]


def main():
    parser = argparse.ArgumentParser(description="Generate quality gold set v3 from clinical dicts")
    parser.add_argument("--dicts_root", type=str, default="data/dicts/clinical",
                        help="Root directory for clinical dicts")
    parser.add_argument("--parquet", type=str, required=True,
                        help="Path to compendium_all.parquet")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL path")
    parser.add_argument("--queries_per_topic", type=int, default=30,
                        help="Number of queries per canonical topic")
    parser.add_argument("--min_gold", type=int, default=3,
                        help="Minimum gold docs to include topic")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    dicts_root = Path(args.dicts_root)
    
    print(f"Loading parquet from: {args.parquet}")
    df = pd.read_parquet(args.parquet)
    print(f"  Loaded {len(df)} documents")
    
    # Group configs by canonical topic
    canonical_data = defaultdict(lambda: {
        "positive_terms": [],
        "trigger_terms": [],
        "source_dirs": [],
    })
    
    # Process each clinical dict
    print(f"\nProcessing clinical dicts from: {dicts_root}")
    
    for topic_key, config in TOPICS_CONFIG.items():
        topic_dir = dicts_root / config["dir"]
        canonical = config["canonical"]
        
        if not topic_dir.exists():
            print(f"  ⚠️  Directory not found: {topic_dir}")
            continue
        
        # Load positive terms
        positive_file = topic_dir / f"{config['dir']}_positive.txt"
        positives = load_positive_terms(positive_file)
        
        # Load trigger terms
        trigger_file = topic_dir / f"{config['dir']}_trigger.txt"
        triggers = load_trigger_terms(trigger_file)
        
        if positives:
            canonical_data[canonical]["positive_terms"].extend(positives)
            canonical_data[canonical]["source_dirs"].append(config["dir"])
        
        if triggers:
            canonical_data[canonical]["trigger_terms"].extend(triggers)
        
        print(f"  {config['dir']:30} → {canonical:15} | {len(positives):3} positives, {len(triggers):3} triggers")
    
    # Deduplicate terms
    for canonical in canonical_data:
        canonical_data[canonical]["positive_terms"] = list(set(canonical_data[canonical]["positive_terms"]))
        canonical_data[canonical]["trigger_terms"] = list(set(canonical_data[canonical]["trigger_terms"]))
    
    # Generate gold for each canonical topic
    all_queries = []
    topic_stats = {}
    
    print(f"\n{'='*70}")
    print("Generating gold sets")
    print(f"{'='*70}")
    
    for canonical, data in sorted(canonical_data.items()):
        print(f"\n=== {canonical.upper()} ===")
        print(f"  Sources: {data['source_dirs']}")
        print(f"  Positive terms ({len(data['positive_terms'])}): {data['positive_terms'][:5]}...")
        print(f"  Trigger terms ({len(data['trigger_terms'])}): {data['trigger_terms'][:5]}...")
        
        # Search for gold docs
        gold_ids = search_docs_by_terms(df, data['positive_terms'])
        gold_ids_list = sorted(list(gold_ids))
        
        print(f"  Found {len(gold_ids_list)} gold documents")
        
        if len(gold_ids_list) < args.min_gold:
            print(f"  ⚠️  Too few gold docs, skipping")
            continue
        
        # Show sample gold docs
        print(f"  Sample gold:")
        for doc_id in gold_ids_list[:5]:
            name = df.iloc[doc_id]['Назва препарату'][:50]
            print(f"    {doc_id}: {name}")
        
        # Generate queries
        queries = generate_query_variants(
            triggers=data['trigger_terms'],
            n_queries=args.queries_per_topic,
        )
        
        if not queries:
            print(f"  ⚠️  No queries generated, skipping")
            continue
        
        print(f"  Generated {len(queries)} queries")
        print(f"  Sample queries: {queries[:3]}")
        
        # Create query objects
        for i, query in enumerate(queries):
            qid = generate_qid(f"{canonical}_{i}_{query}")
            obj = {
                "id": len(all_queries),
                "qid": qid,
                "query": query,
                "topic": canonical,
                "intent": "indication",
                "lang": "uk",
                "gold_ids": gold_ids_list,
                "n_gold": len(gold_ids_list),
                "source": "clinical_dicts",
                "gold_mode": "positive_terms_search",
                "source_dirs": data['source_dirs'],
            }
            all_queries.append(obj)
        
        topic_stats[canonical] = {
            "n_queries": len(queries),
            "n_gold": len(gold_ids_list),
            "sources": data['source_dirs'],
        }
    
    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for obj in all_queries:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total queries: {len(all_queries)}")
    print(f"Output: {args.output}")
    print(f"\nPer-topic stats:")
    for topic, stats in sorted(topic_stats.items()):
        print(f"  {topic:15} | {stats['n_queries']:3} queries | {stats['n_gold']:4} gold | sources: {stats['sources']}")


if __name__ == "__main__":
    main()