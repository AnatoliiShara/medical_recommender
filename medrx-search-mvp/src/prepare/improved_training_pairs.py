#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved Training Pairs Generator with better quality control
"""

import pandas as pd
import numpy as np
import json
import re
import random
from pathlib import Path
from typing import List, Dict
from collections import defaultdict, Counter
import argparse

# Same medical patterns as before...
COMPREHENSIVE_MEDICAL_PATTERNS = {
    "hypertension": {
        "patterns": [r"гіпертензія", r"артеріальн.*тиск", r"високий тиск"],
        "symptoms": ["високий тиск", "гіпертензія"],
        "category": "cardiovascular"
    },
    "heart_disease": {
        "patterns": [r"серцев.*недостатність", r"стенокардія", r"аритмія"],
        "symptoms": ["біль в серці", "серцебиття"],
        "category": "cardiovascular"
    },
    "diabetes": {
        "patterns": [r"діабет", r"цукров.*діабет"],
        "symptoms": ["високий цукор", "спрага"],
        "category": "endocrine"
    },
    "pain": {
        "patterns": [r"біль", r"больов.*синдром"],
        "symptoms": ["біль", "больові відчуття"],
        "category": "pain"
    },
    "infection": {
        "patterns": [r"інфекц", r"бактеріальн.*інфекц"],
        "symptoms": ["температура", "гарячка"],
        "category": "infectious"
    },
    "respiratory_infection": {
        "patterns": [r"пневмонія", r"бронхіт", r"ангіна"],
        "symptoms": ["кашель", "біль в горлі"],
        "category": "respiratory"
    },
    "gastric": {
        "patterns": [r"гастрит", r"виразк.*хвороба", r"шлунк"],
        "symptoms": ["біль в шлунку", "печія"],
        "category": "gastroenterology"
    },
    "allergy": {
        "patterns": [r"алергія", r"алергічн.*реакц"],
        "symptoms": ["свербіж", "висип"],
        "category": "immunology"
    }
}

QUERY_TEMPLATES = {
    "direct": [
        "які ліки від {condition}",
        "препарати для лікування {condition}",
        "що приймати при {condition}",
        "медикаменти від {condition}"
    ],
    "symptom": [
        "що пити при {symptom}",
        "ліки від {symptom}",
        "як лікувати {symptom}",
        "препарати при {symptom}"
    ]
}

class ImprovedTrainingPairsGenerator:
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        # Pre-create negatives pool
        self.negatives_pool = self._create_negatives_pool()
        
    def _create_negatives_pool(self) -> Dict[str, List[str]]:
        """Create pool of real negative examples by category"""
        negatives = defaultdict(list)
        
        for _, row in self.df.iterrows():
            conditions = self.extract_conditions(row['Показання'])
            doc_text = self.create_clean_document(row)
            
            for condition in conditions:
                category = condition['category']
                negatives[category].append(doc_text)
        
        return dict(negatives)
    
    def extract_conditions(self, text: str) -> List[Dict]:
        """Extract conditions from text"""
        if not isinstance(text, str):
            return []
            
        text_lower = text.lower()
        found = []
        
        for name, data in COMPREHENSIVE_MEDICAL_PATTERNS.items():
            for pattern in data["patterns"]:
                if re.search(pattern, text_lower):
                    found.append({
                        "name": name,
                        "category": data["category"],
                        "symptoms": data["symptoms"]
                    })
                    break
        return found
    
    def create_clean_document(self, row: pd.Series) -> str:
        """Create clean, properly truncated document"""
        parts = []
        
        # Drug name
        if pd.notna(row['Назва препарату']):
            parts.append(str(row['Назва препарату']))
        
        # Therapeutic group (cleaned and truncated)
        if 'Фармакотерапевтична група' in row and pd.notna(row['Фармакотерапевтична група']):
            group_text = str(row['Фармакотерапевтична група'])
            # Remove duplicates and truncate
            group_clean = re.sub(r'(.{20,}?)\1+', r'\1', group_text)  # Remove repetitions
            group_clean = group_clean[:200]  # Truncate to 200 chars
            parts.append(group_clean)
        
        # Indications (clean and truncate)
        if pd.notna(row['Показання']):
            indications = str(row['Показання'])
            # Remove duplicates and truncate
            indications_clean = re.sub(r'(.{50,}?)\1+', r'\1', indications)
            indications_clean = indications_clean[:800]  # Truncate to 800 chars
            parts.append(indications_clean)
            
        return ' [SEP] '.join(parts)
    
    def generate_strategic_negatives(self, target_category: str, k: int = 3) -> List[str]:
        """Generate strategic negatives from different categories"""
        negatives = []
        
        # Get negatives from different categories
        other_categories = [cat for cat in self.negatives_pool.keys() if cat != target_category]
        
        for category in random.sample(other_categories, min(k, len(other_categories))):
            if self.negatives_pool[category]:
                neg = random.choice(self.negatives_pool[category])
                negatives.append(neg)
        
        # Fill remaining with random negatives if needed
        while len(negatives) < k:
            all_negatives = []
            for cat_negatives in self.negatives_pool.values():
                all_negatives.extend(cat_negatives)
            if all_negatives:
                negatives.append(random.choice(all_negatives))
            else:
                break
                
        return negatives[:k]
    
    def generate_training_pairs(self, target_pairs: int = 15000) -> List[Dict]:
        """Generate improved training pairs"""
        training_pairs = []
        
        print(f"Generating {target_pairs} improved training pairs...")
        
        for idx, row in self.df.iterrows():
            if len(training_pairs) >= target_pairs:
                break
                
            if idx % 500 == 0:
                print(f"Processed {idx}/{len(self.df)} drugs, generated {len(training_pairs)} pairs")
            
            # Extract conditions
            conditions = self.extract_conditions(row['Показання'])
            if not conditions:
                continue
            
            # Create clean document
            document_text = self.create_clean_document(row)
            
            # Generate queries for each condition (max 2 conditions per drug)
            for condition in conditions[:2]:
                if len(training_pairs) >= target_pairs:
                    break
                
                condition_name = condition["name"]
                category = condition["category"]
                
                # Map to Ukrainian
                condition_mapping = {
                    "diabetes": "діабету",
                    "hypertension": "високого тиску",
                    "heart_disease": "серцевих захворювань",
                    "infection": "інфекції",
                    "pain": "болю",
                    "respiratory_infection": "респіраторних інфекцій",
                    "gastric": "захворювань шлунка",
                    "allergy": "алергії"
                }
                
                condition_ua = condition_mapping.get(condition_name, condition_name)
                
                # Generate 2 queries per condition: direct + symptom
                queries = []
                
                # Direct condition query
                template = random.choice(QUERY_TEMPLATES["direct"])
                queries.append({
                    "query": template.format(condition=condition_ua),
                    "type": "direct_condition"
                })
                
                # Symptom query
                if condition["symptoms"]:
                    symptom = random.choice(condition["symptoms"])
                    template = random.choice(QUERY_TEMPLATES["symptom"])
                    queries.append({
                        "query": template.format(symptom=symptom),
                        "type": "symptom_search"
                    })
                
                # Generate negatives
                negatives = self.generate_strategic_negatives(category, k=3)
                
                # Create pairs
                for query_info in queries:
                    if len(training_pairs) >= target_pairs:
                        break
                        
                    training_pair = {
                        "query": query_info["query"],
                        "positive": document_text,
                        "negatives": negatives,
                        "metadata": {
                            "drug_name": row['Назва препарату'],
                            "condition": condition_name,
                            "category": category,
                            "query_type": query_info["type"]
                        }
                    }
                    training_pairs.append(training_pair)
        
        print(f"Generated {len(training_pairs)} training pairs")
        return training_pairs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/clean_medical.parquet")
    parser.add_argument("--output", default="data/processed/training/improved_training_pairs.jsonl")
    parser.add_argument("--target", type=int, default=15000)
    
    args = parser.parse_args()
    
    print("=== IMPROVED MEDICAL TRAINING PAIRS GENERATION ===")
    
    # Load data
    df = pd.read_parquet(args.input)
    print(f"Loaded {len(df)} medical records")
    
    # Generate
    generator = ImprovedTrainingPairsGenerator(df)
    pairs = generator.generate_training_pairs(args.target)
    
    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    main()
