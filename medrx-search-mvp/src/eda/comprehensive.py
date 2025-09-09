#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive EDA for Medical Drug Database
Analyzes dataset structure, quality, medical content, and ML readiness
"""

import os
import re
import argparse
import textwrap
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

# Optional imports with fallbacks
try:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Regular expressions for content analysis
HTML_TAG_RE = re.compile(r'<[^>]+>')
MULTISPACE_RE = re.compile(r'\s+')
TRADEMARK_RE = re.compile(r'[®™©]+')
SPECIAL_CHARS_RE = re.compile(r'[^\w\s\-.,;:()[\]{}«»"""\'']', re.UNICODE)
EMAIL_RE = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
URL_RE = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

# Medical patterns
DOSAGE_PATTERN = re.compile(r'\d+(?:[.,]\d+)?\s*(?:мг|г|мл|мкг|ІО|од|%)\b', re.IGNORECASE)
FREQUENCY_PATTERN = re.compile(r'(?:\d+\s*рази?\s*(?:на\s*день|добу)|щоденно|двічі|тричі|через|кожн)', re.IGNORECASE)
ICD_PATTERN = re.compile(r'\b[A-Z]\d{2}(?:\.\d{1,2})?\b')
AGE_PATTERN = re.compile(r'(?:від|до|після|понад|старше|молодше|віком)\s*\d+\s*(?:років?|місяців?|днів?|рок)', re.IGNORECASE)

# Product classification triggers
PRODUCT_TRIGGERS = {
    "dietary_supplement": [
        r"дієтичн[аії] добавк", r"харчов(ий|і) продукт", r"біологічно активн", r"БАД",
        r"вітамінн(ий|а|і) комплекс", r"добавк[аи] до їж", r"поживн[аії] речовин"
    ],
    "cosmetic": [
        r"косметичн", r"шампун", r"бальзам(?!\s+для\s+(?:серця|болю))", r"лосьйон", 
        r"гель для душ", r"маска для волос", r"тонік для облич", r"скраб", r"пінка",
        r"крем(?!\s*(?:від|для\s*лікування|проти))", r"гігієнічн[ий|а]"
    ],
    "medical_device": [
        r"медичн(ий|і) виріб", r"виріб медичн", r"термометр", r"пульсоксиметр",
        r"катетер", r"шприц", r"рукавичк", r"халат", r"маска(?!\s*для\s*волосся)",
        r"пластир(?:\s*(?:катушков|рулон|медичн))", r"бинт", r"вата"
    ],
    "prescription_drug": [
        r"рецептурн", r"за рецептом", r"строго за призначенням", r"під наглядом лікаря"
    ]
}

# Medical specialties mapping
MEDICAL_SPECIALTIES = {
    "кардіологія": [r"серц", r"кардіо", r"гіпертензія", r"стенокардія", r"аритмія", r"інфаркт"],
    "гастроентерологія": [r"шлунк", r"кишечник", r"печінк", r"гастрит", r"коліт", r"язв"],
    "неврологія": [r"нервов", r"головн.*біль", r"мігрень", r"епілепсія", r"інсульт"],
    "ендокринологія": [r"діабет", r"щитовидн", r"гормон", r"інсулін"],
    "респіраторна": [r"дихальн", r"легені", r"астма", r"бронхіт", r"пневмонія", r"кашель"],
    "імунологія": [r"імун", r"алергія", r"алергічн", r"антигістамін"],
    "дерматологія": [r"шкір", r"дерматит", r"екзема", r"псоріаз"],
    "урологія": [r"нирк", r"сечов", r"простат", r"цистит"],
    "гінекологія": [r"жіноч", r"вагітн", r"менструал", r"гормональн"],
    "ортопедія": [r"суглоб", r"кістк", r"артрит", r"артроз", r"ревматизм"]
}

class MedicalDatasetAnalyzer:
    """Comprehensive analyzer for medical pharmaceutical datasets"""
    
    def __init__(self, dataset_path: str, output_dir: str):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.df = None
        self.analysis_results = {}
        
    def load_data(self) -> bool:
        """Load dataset with error handling"""
        try:
            if self.dataset_path.suffix == '.parquet':
                self.df = pd.read_parquet(self.dataset_path)
            elif self.dataset_path.suffix == '.csv':
                self.df = pd.read_csv(self.dataset_path)
            else:
                print(f"Unsupported format: {self.dataset_path.suffix}")
                return False
            
            print(f"Dataset loaded: {self.df.shape[0]:,} rows, {self.df.shape[1]} columns")
            return True
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not isinstance(text, str):
            return ""
        
        text = HTML_TAG_RE.sub(" ", text)
        text = TRADEMARK_RE.sub("", text)
        text = text.replace("\xa0", " ").replace("\n", " ").replace("\r", " ")
        text = MULTISPACE_RE.sub(" ", text).strip()
        return text
    
    def analyze_basic_stats(self) -> Dict:
        """Basic dataset statistics"""
        stats = {
            'total_records': len(self.df),
            'total_columns': len(self.df.columns),
            'columns_list': list(self.df.columns),
            'memory_usage_mb': round(self.df.memory_usage(deep=True).sum() / 1024**2, 2),
            'duplicate_rows': self.df.duplicated().sum(),
        }
        
        # Data types analysis
        dtype_counts = self.df.dtypes.value_counts()
        stats['data_types'] = {str(dtype): int(count) for dtype, count in dtype_counts.items()}
        
        return stats
    
    def analyze_field_completeness(self) -> pd.DataFrame:
        """Comprehensive field completeness analysis"""
        completeness_data = []
        
        for col in self.df.columns:
            series = self.df[col]
            
            # Basic null analysis
            total_nulls = series.isna().sum()
            
            # String-specific analysis
            if series.dtype == 'object':
                # Empty strings
                empty_strings = series.astype(str).str.strip().eq('').sum()
                # Very short content (likely placeholders)
                very_short = series.astype(str).str.len().between(1, 5).sum()
                # Meaningful content
                meaningful = series.astype(str).str.len() > 10
                meaningful_count = meaningful.sum()
            else:
                empty_strings = 0
                very_short = 0
                meaningful_count = len(series) - total_nulls
            
            # Text quality metrics for text fields
            if series.dtype == 'object' and meaningful_count > 0:
                text_sample = series[meaningful].head(1000).astype(str)
                avg_length = text_sample.str.len().mean()
                has_html = text_sample.str.contains('<[^>]+>', na=False).mean()
                has_special_chars = text_sample.apply(lambda x: len(SPECIAL_CHARS_RE.findall(x)) / max(len(x), 1)).mean()
            else:
                avg_length = 0
                has_html = 0
                has_special_chars = 0
            
            completeness_data.append({
                'column': col,
                'total_records': len(series),
                'null_count': int(total_nulls),
                'null_percentage': round(total_nulls / len(series) * 100, 2),
                'empty_strings': int(empty_strings),
                'very_short_content': int(very_short),
                'meaningful_content': int(meaningful_count),
                'meaningful_percentage': round(meaningful_count / len(series) * 100, 2),
                'avg_text_length': round(avg_length, 1),
                'html_contamination_rate': round(has_html * 100, 2),
                'special_chars_density': round(has_special_chars * 100, 2)
            })
        
        return pd.DataFrame(completeness_data).sort_values('meaningful_percentage', ascending=False)
    
    def analyze_medical_content(self) -> Dict:
        """Analyze medical content and terminology"""
        medical_analysis = {}
        
        # Find medical text columns
        medical_columns = []
        for col in self.df.columns:
            col_lower = col.lower()
            if any(term in col_lower for term in ['показання', 'протипоказання', 'фармакол', 'застосування']):
                medical_columns.append(col)
        
        # Medical entities extraction
        all_medical_text = ""
        for col in medical_columns:
            if col in self.df.columns:
                text_series = self.df[col].fillna('').astype(str)
                all_medical_text += " ".join(text_series.head(5000))  # Sample for analysis
        
        all_medical_text = self.clean_text(all_medical_text.lower())
        
        # Extract medical patterns
        medical_analysis['dosage_mentions'] = len(DOSAGE_PATTERN.findall(all_medical_text))
        medical_analysis['frequency_mentions'] = len(FREQUENCY_PATTERN.findall(all_medical_text))
        medical_analysis['icd_codes'] = len(set(ICD_PATTERN.findall(all_medical_text)))
        medical_analysis['age_restrictions'] = len(AGE_PATTERN.findall(all_medical_text))
        
        # Analyze by specialty
        specialty_coverage = {}
        for specialty, patterns in MEDICAL_SPECIALTIES.items():
            combined_pattern = "|".join(patterns)
            matches = len(re.findall(combined_pattern, all_medical_text, re.IGNORECASE))
            specialty_coverage[specialty] = matches
        
        medical_analysis['specialty_coverage'] = specialty_coverage
        
        return medical_analysis
    
    def classify_products(self) -> pd.DataFrame:
        """Classify products into categories"""
        product_classifications = []
        
        # Get relevant fields for classification
        name_col = next((col for col in self.df.columns if 'назва' in col.lower()), None)
        group_col = next((col for col in self.df.columns if 'група' in col.lower()), None)
        
        if not name_col:
            return pd.DataFrame()
        
        for idx, row in self.df.iterrows():
            name = str(row[name_col]).lower()
            group = str(row[group_col]).lower() if group_col else ""
            combined_text = f"{name} {group}"
            
            classification = {
                'index': idx,
                'product_name': row[name_col],
                'is_dietary_supplement': False,
                'is_cosmetic': False,
                'is_medical_device': False,
                'is_prescription_drug': False,
                'is_medicine': True  # default
            }
            
            # Apply classification rules
            for category, patterns in PRODUCT_TRIGGERS.items():
                for pattern in patterns:
                    if re.search(pattern, combined_text, re.IGNORECASE):
                        classification[f'is_{category}'] = True
                        if category != 'prescription_drug':
                            classification['is_medicine'] = False
                        break
            
            product_classifications.append(classification)
        
        return pd.DataFrame(product_classifications)
    
    def analyze_training_potential(self) -> Dict:
        """Analyze potential for ML training data generation"""
        training_analysis = {}
        
        # Find key columns
        name_col = next((col for col in self.df.columns if 'назва' in col.lower()), None)
        indications_col = next((col for col in self.df.columns if 'показання' in col.lower()), None)
        
        if not name_col or not indications_col:
            return {'error': 'Required columns not found'}
        
        # Analyze field combinations for concatenation
        field_combinations = {
            'name_only': [name_col],
            'name_indications': [name_col, indications_col],
        }
        
        # Add other fields if available
        for col in self.df.columns:
            col_lower = col.lower()
            if 'група' in col_lower:
                field_combinations['name_group_indications'] = [name_col, col, indications_col]
            if 'протипоказання' in col_lower:
                field_combinations['name_indications_contraindications'] = [name_col, indications_col, col]
        
        combination_stats = {}
        for combo_name, fields in field_combinations.items():
            # Check completeness
            mask = pd.Series([True] * len(self.df))
            for field in fields:
                if field in self.df.columns:
                    mask &= self.df[field].astype(str).str.strip().str.len() > 10
            
            available_records = mask.sum()
            
            # Analyze concatenated length with truncation
            if available_records > 100:
                subset = self.df[mask].head(1000)  # Sample for analysis
                
                # Truncation strategy
                concat_parts = []
                for field in fields:
                    if field == name_col:
                        part = subset[field].astype(str)
                    elif 'показання' in field.lower():
                        part = subset[field].astype(str).str[:1200]  # 1200 chars
                    else:
                        part = subset[field].astype(str).str[:800]   # 800 chars
                    concat_parts.append(part)
                
                concatenated = ' [SEP] '.join(concat_parts)
                lengths = concatenated.str.len()
                
                combination_stats[combo_name] = {
                    'available_records': int(available_records),
                    'percentage': round(available_records / len(self.df) * 100, 2),
                    'avg_length_chars': round(lengths.mean(), 0),
                    'avg_length_tokens': round(lengths.mean() / 4, 0),
                    'p95_length_tokens': round(lengths.quantile(0.95) / 4, 0)
                }
        
        training_analysis['field_combinations'] = combination_stats
        
        # Medical condition coverage for query generation
        if indications_col:
            indications_text = self.df[indications_col].fillna('').str.lower()
            
            condition_patterns = {
                'діабет': r'діабет|цукрового діабету',
                'гіпертензія': r'гіпертензія|високий тиск|артеріальн.*тиск',
                'серцеві_захворювання': r'серцев|кардіо|стенокардія|інфаркт',
                'шлунково_кишкові': r'гастрит|язва|кишечник|шлунк',
                'респіраторні': r'астма|бронхіт|кашель|дихальн',
                'неврологічні': r'головн.*біль|мігрень|нервов',
                'алергічні': r'алергія|алергічн',
                'інфекційні': r'інфекц|бактеріальн|вірусн'
            }
            
            condition_coverage = {}
            for condition, pattern in condition_patterns.items():
                hits = indications_text.str.contains(pattern, na=False).sum()
                condition_coverage[condition] = {
                    'drug_count': int(hits),
                    'percentage': round(hits / len(self.df) * 100, 2),
                    'training_pairs_potential': hits * 4  # 4 query variations per drug
                }
            
            training_analysis['condition_coverage'] = condition_coverage
            
            total_training_potential = sum(
                data['training_pairs_potential'] for data in condition_coverage.values()
            )
            training_analysis['total_training_pairs_potential'] = total_training_potential
        
        return training_analysis
    
    def analyze_text_complexity(self) -> pd.DataFrame:
        """Analyze text complexity and processing requirements"""
        complexity_data = []
        
        text_columns = [col for col in self.df.columns if self.df[col].dtype == 'object']
        
        for col in text_columns:
            if self.df[col].astype(str).str.len().sum() == 0:
                continue
                
            text_series = self.df[col].fillna('').astype(str)
            non_empty = text_series[text_series.str.len() > 0]
            
            if len(non_empty) == 0:
                continue
            
            # Sample for analysis
            sample = non_empty.head(2000)
            lengths = sample.str.len()
            
            # Word count estimation
            word_counts = sample.str.split().str.len()
            
            # Sentence count estimation
            sentence_counts = sample.str.count('[.!?]+')
            
            # Language complexity
            avg_words_per_sentence = (word_counts / np.maximum(sentence_counts, 1)).mean()
            
            # Memory and processing estimates
            total_chars = lengths.sum()
            estimated_tokens = total_chars / 4  # rough estimate
            estimated_embeddings_size_mb = estimated_tokens * 768 * 4 / (1024**2)  # 768-dim float32
            
            complexity_data.append({
                'column': col,
                'records_with_content': len(non_empty),
                'avg_length_chars': round(lengths.mean(), 1),
                'median_length_chars': round(lengths.median(), 1),
                'p95_length_chars': round(lengths.quantile(0.95), 1),
                'max_length_chars': int(lengths.max()),
                'avg_words': round(word_counts.mean(), 1),
                'avg_sentences': round(sentence_counts.mean(), 1),
                'avg_words_per_sentence': round(avg_words_per_sentence, 1),
                'estimated_total_tokens': int(estimated_tokens),
                'estimated_embedding_size_mb': round(estimated_embeddings_size_mb, 2),
                'processing_complexity_score': round(lengths.mean() * len(non_empty) / 10000, 2)
            })
        
        return pd.DataFrame(complexity_data).sort_values('processing_complexity_score', ascending=False)
    
    def generate_recommendations(self) -> Dict:
        """Generate technical recommendations based on analysis"""
        recommendations = {
            'data_preprocessing': [],
            'ml_strategy': [],
            'architecture': [],
            'performance': []
        }
        
        # Based on completeness analysis
        completeness_df = self.analysis_results.get('field_completeness')
        if completeness_df is not None:
            high_quality_fields = completeness_df[
                completeness_df['meaningful_percentage'] > 50
            ]['column'].tolist()
            
            if len(high_quality_fields) < 3:
                recommendations['data_preprocessing'].append(
                    "WARNING: Very few fields have >50% meaningful content. Consider data augmentation."
                )
            
            recommendations['data_preprocessing'].append(
                f"Focus on {len(high_quality_fields)} high-quality fields: {', '.join(high_quality_fields[:5])}"
            )
        
        # Based on training potential
        training_data = self.analysis_results.get('training_potential', {})
        if 'total_training_pairs_potential' in training_data:
            total_pairs = training_data['total_training_pairs_potential']
            if total_pairs >= 20000:
                recommendations['ml_strategy'].append(
                    f"Sufficient training data potential: {total_pairs:,} pairs for MNRL fine-tuning"
                )
            else:
                recommendations['ml_strategy'].append(
                    f"Limited training data: {total_pairs:,} pairs. Consider synthetic data generation."
                )
        
        # Based on text complexity
        complexity_df = self.analysis_results.get('text_complexity')
        if complexity_df is not None:
            max_tokens = complexity_df['estimated_total_tokens'].sum()
            if max_tokens > 10**7:  # 10M tokens
                recommendations['performance'].append(
                    "High token volume detected. Consider batch processing and efficient tokenization."
                )
            
            heavy_fields = complexity_df[
                complexity_df['processing_complexity_score'] > 10
            ]['column'].tolist()
            
            if heavy_fields:
                recommendations['architecture'].append(
                    f"Heavy processing fields detected: {', '.join(heavy_fields)}. Consider field prioritization."
                )
        
        return recommendations
    
    def run_comprehensive_analysis(self) -> bool:
        """Run all analysis components"""
        print("Starting comprehensive medical dataset analysis...")
        
        if not self.load_data():
            return False
        
        # Run all analyses
        print("1. Basic statistics...")
        self.analysis_results['basic_stats'] = self.analyze_basic_stats()
        
        print("2. Field completeness...")
        self.analysis_results['field_completeness'] = self.analyze_field_completeness()
        
        print("3. Medical content analysis...")
        self.analysis_results['medical_content'] = self.analyze_medical_content()
        
        print("4. Product classification...")
        self.analysis_results['product_classification'] = self.classify_products()
        
        print("5. Training potential analysis...")
        self.analysis_results['training_potential'] = self.analyze_training_potential()
        
        print("6. Text complexity analysis...")
        self.analysis_results['text_complexity'] = self.analyze_text_complexity()
        
        print("7. Generating recommendations...")
        self.analysis_results['recommendations'] = self.generate_recommendations()
        
        return True
    
    def save_results(self) -> None:
        """Save all analysis results to CSV files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save DataFrames
        for name, data in self.analysis_results.items():
            if isinstance(data, pd.DataFrame):
                filename = f"{name}_{timestamp}.csv"
                filepath = self.output_dir / filename
                data.to_csv(filepath, index=False, encoding='utf-8')
                print(f"Saved: {filepath}")
        
        # Save dictionaries as structured CSV
        dict_results = {k: v for k, v in self.analysis_results.items() if isinstance(v, dict)}
        
        if dict_results:
            # Flatten nested dictionaries for CSV export
            flattened_data = []
            
            for category, data in dict_results.items():
                if category == 'basic_stats':
                    for key, value in data.items():
                        if isinstance(value, (dict, list)):
                            value = str(value)
                        flattened_data.append({
                            'category': category,
                            'metric': key,
                            'value': value,
                            'type': type(value).__name__
                        })
                
                elif category == 'medical_content':
                    for key, value in data.items():
                        if key == 'specialty_coverage' and isinstance(value, dict):
                            for specialty, count in value.items():
                                flattened_data.append({
                                    'category': f"{category}_specialty",
                                    'metric': specialty,
                                    'value': count,
                                    'type': 'int'
                                })
                        else:
                            flattened_data.append({
                                'category': category,
                                'metric': key,
                                'value': value,
                                'type': type(value).__name__
                            })
                
                elif category == 'training_potential':
                    # Handle nested structure
                    for key, value in data.items():
                        if isinstance(value, dict):
                            for subkey, subvalue in value.items():
                                if isinstance(subvalue, dict):
                                    for subsubkey, subsubvalue in subvalue.items():
                                        flattened_data.append({
                                            'category': f"{category}_{key}",
                                            'metric': f"{subkey}_{subsubkey}",
                                            'value': subsubvalue,
                                            'type': type(subsubvalue).__name__
                                        })
                                else:
                                    flattened_data.append({
                                        'category': f"{category}_{key}",
                                        'metric': subkey,
                                        'value': subvalue,
                                        'type': type(subvalue).__name__
                                    })
                        else:
                            flattened_data.append({
                                'category': category,
                                'metric': key,
                                'value': value,
                                'type': type(value).__name__
                            })
                
                elif category == 'recommendations':
                    for rec_type, rec_list in data.items():
                        for i, recommendation in enumerate(rec_list):
                            flattened_data.append({
                                'category': f"recommendation_{rec_type}",
                                'metric': f"item_{i+1}",
                                'value': recommendation,
                                'type': 'str'
                            })
            
            # Save flattened results
            if flattened_data:
                summary_df = pd.DataFrame(flattened_data)
                summary_path = self.output_dir / f"analysis_summary_{timestamp}.csv"
                summary_df.to_csv(summary_path, index=False, encoding='utf-8')
                print(f"Saved: {summary_path}")
        
        # Create executive summary
        self.create_executive_summary(timestamp)
    
    def create_executive_summary(self, timestamp: str) -> None:
        """Create executive summary report"""
        summary_lines = []
        summary_lines.append("MEDICAL DATASET COMPREHENSIVE EDA - EXECUTIVE SUMMARY")
        summary_lines.append("=" * 60)
        summary_lines.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary_lines.append(f"Dataset: {self.dataset_path}")
        summary_lines.append("")
        
        # Basic stats
        basic = self.analysis_results.get('basic_stats', {})
        summary_lines.append("DATASET OVERVIEW:")
        summary_lines.append(f"  Records: {basic.get('total_records', 'N/A'):,}")
        summary_lines.append(f"  Columns: {basic.get('total_columns', 'N/A')}")
        summary_lines.append(f"  Memory: {basic.get('memory_usage_mb', 'N/A')} MB")
        summary_lines.append(f"  Duplicates: {basic.get('duplicate_rows', 'N/A')}")
        summary_lines.append("")
        
        # Field completeness highlights
        completeness = self.analysis_results.get('field_completeness')
        if completeness is not None:
            summary_lines.append("FIELD COMPLETENESS (Top 5):")
            top_fields = completeness.head(5)
            for _, row in top_fields.iterrows():
                summary_lines.append(f"  {row['column']}: {row['meaningful_percentage']:.1f}%")
            summary_lines.append("")
        
        # Training potential
        training = self.analysis_results.get('training_potential', {})
        if 'total_training_pairs_potential' in training:
            summary_lines.append("ML TRAINING READINESS:")
            summary_lines.append(f"  Potential training pairs: {training['total_training_pairs_potential']:,}")
            
            field_combos = training.get('field_combinations', {})
            for combo, stats in field_combos.items():
                summary_lines.append(f"  {combo}: {stats.get('available_records', 0):,} records ({stats.get('percentage', 0):.1f}%)")
            summary_lines.append("")
        
        # Recommendations
        recommendations = self.analysis_results.get('recommendations', {})
        if recommendations:
            summary_lines.append("KEY RECOMMENDATIONS:")
            for category, recs in recommendations.items():
                summary_lines.append(f"  {category.upper()}:")
                for rec in recs[:3]:  # Top 3 per category
                    summary_lines.append(f"    - {rec}")
                summary_lines.append("")
        
        # Save summary
        summary_path = self.output_dir / f"executive_summary_{timestamp}.txt"
        summary_path.write_text("\n".join(summary_lines), encoding='utf-8')
        print(f"Saved: {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive EDA for Medical Pharmaceutical Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Example:
          python medical_eda.py --dataset "data/raw/compendium_all.parquet" --output "data/interim/comprehensive_eda"
        
        This script performs comprehensive analysis including:
        - Basic dataset statistics and structure
        - Field completeness and quality assessment  
        - Medical content and terminology analysis
        - Product classification (drugs vs supplements vs devices)
        - ML training data potential evaluation
        - Text complexity and processing requirements
        - Technical architecture recommendations
        
        All results are saved as CSV files and executive summary.
        """)
    )
    
    parser.add_argument("--dataset", required=True, help="Path to dataset file (.parquet or .csv)")
    parser.add_argument("--output", default="data/interim/comprehensive_eda", help="Output directory for results")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = MedicalDatasetAnalyzer(args.dataset, args.output)
    
    # Run comprehensive analysis
    if analyzer.run_comprehensive_analysis():
        print("\nSaving results...")
        analyzer.save_results()
        print(f"\nComprehensive EDA completed successfully!")
        print(f"Results saved to: {analyzer.output_dir.resolve()}")
        
        # Display quick summary
        print("\nQUICK SUMMARY:")
        basic_stats = analyzer.analysis_results.get('basic_stats', {})
        print(f"Dataset size: {basic_stats.get('total_records', 'N/A'):,} records")
        
        training_potential = analyzer.analysis_results.get('training_potential', {})
        if 'total_training_pairs_potential' in training_potential:
            print(f"Training pairs potential: {training_potential['total_training_pairs_potential']:,}")
        
        print(f"\nCheck '{analyzer.output_dir}/executive_summary_*.txt' for detailed insights.")
        
    else:
        print("Analysis failed. Check error messages above.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())