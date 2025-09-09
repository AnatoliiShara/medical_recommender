#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized Medical Training Pairs Generator for Local Development
Memory-efficient, resumable, parallelized generation for limited resources
"""

import pandas as pd
import numpy as np
import json
import re
import random
import multiprocessing as mp
import psutil
import gc
import time
import signal
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

# Configuration for resource management
@dataclass
class ResourceConfig:
    max_memory_gb: float = 6.0  # Max 6GB RAM usage
    chunk_size: int = 200       # Process 200 drugs at a time
    max_workers: int = 8        # Use 8 of 12 cores
    checkpoint_interval: int = 50  # Save every 50 chunks
    
    @classmethod
    def auto_detect(cls):
        """Auto-detect optimal settings based on system resources"""
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        cpu_count = psutil.cpu_count()
        
        return cls(
            max_memory_gb=min(available_memory_gb * 0.7, 6.0),  # Use 70% of available
            chunk_size=max(100, min(300, int(available_memory_gb * 30))),  # Scale with RAM
            max_workers=max(4, min(cpu_count - 2, 10)),  # Leave 2 cores free
            checkpoint_interval=max(20, min(100, int(available_memory_gb * 10)))
        )

# Global flag for graceful shutdown
INTERRUPTED = False

def signal_handler(signum, frame):
    global INTERRUPTED
    print(f"\n[INTERRUPT] Received signal {signum}. Saving progress and shutting down gracefully...")
    INTERRUPTED = True

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Medical patterns (same as before but moved to module level for multiprocessing)
COMPREHENSIVE_MEDICAL_PATTERNS = {
    "hypertension": {
        "patterns": [r"гіпертензія", r"артеріальн.*тиск", r"високий тиск", r"підвищений тиск"],
        "symptoms": ["високий тиск", "гіпертензія", "підвищений тиск"],
        "problems": ["підвищується тиск", "стрибає тиск", "високий тиск"],
        "category": "cardiovascular"
    },
    "heart_disease": {
        "patterns": [r"серцев.*недостатність", r"ішемічна хвороба", r"стенокардія", r"аритмія", r"інфаркт", r"серцев.*біль"],
        "symptoms": ["біль в серці", "серцебиття", "аритмія", "стенокардія"],
        "problems": ["болить серце", "неритмічно б'ється серце", "тиснуча біль в грудях"],
        "category": "cardiovascular"
    },
    "diabetes": {
        "patterns": [r"діабет", r"цукров.*діабет", r"підвищен.*цукор"],
        "symptoms": ["високий цукор", "спрага", "часте сечовипускання"],
        "problems": ["підвищений цукор в крові", "діабет", "порушення толерантності до глюкози"],
        "category": "endocrine"
    },
    "pain": {
        "patterns": [r"біль", r"больов.*синдром", r"анальгезія", r"болючий"],
        "symptoms": ["біль", "больові відчуття", "ниючий біль", "гострий біль"],
        "problems": ["болить", "сильний біль", "не проходить біль"],
        "category": "pain"
    },
    "headache": {
        "patterns": [r"головн.*біль", r"мігрень", r"цефалгія"],
        "symptoms": ["головний біль", "мігрень", "біль в голові"],
        "problems": ["болить голова", "мучить головний біль", "постійний головний біль"],
        "category": "neurological"
    },
    "infection": {
        "patterns": [r"інфекц", r"бактеріальн.*інфекц", r"запальн.*процес", r"сепсис"],
        "symptoms": ["температура", "гарячка", "озноб", "слабкість при інфекції"],
        "problems": ["підозра на інфекцію", "запальний процес", "бактеріальна інфекція"],
        "category": "infectious"
    },
    "respiratory_infection": {
        "patterns": [r"пневмонія", r"бронхіт", r"ангіна", r"ГРВІ", r"застуда"],
        "symptoms": ["кашель", "біль в горлі", "нежить", "утруднене дихання"],
        "problems": ["застудився", "болить горло", "мучить кашель"],
        "category": "respiratory"
    },
    "gastric": {
        "patterns": [r"гастрит", r"виразк.*хвороба", r"шлунк", r"пептичн.*виразк"],
        "symptoms": ["біль в шлунку", "печія", "нудота", "відрижка"],
        "problems": ["болить шлунок", "печія після їжі", "розлад травлення"],
        "category": "gastroenterology"
    },
    "diarrhea": {
        "patterns": [r"діарея", r"пронос", r"рідк.*стілець"],
        "symptoms": ["пронос", "діарея", "рідкий стілець"],
        "problems": ["розлад кишечника", "постійний пронос"],
        "category": "gastroenterology"
    },
    "asthma": {
        "patterns": [r"астма", r"бронхіальна астма", r"обструктивн.*хвороба"],
        "symptoms": ["задишка", "утруднене дихання", "напади астми"],
        "problems": ["важко дихати", "не вистачає повітря", "напад астми"],
        "category": "respiratory"
    },
    "cough": {
        "patterns": [r"кашель", r"кашльов.*рефлекс"],
        "symptoms": ["кашель", "сухий кашель", "вологий кашель"],
        "problems": ["мучить кашель", "не проходить кашель", "сильний кашель"],
        "category": "respiratory"
    },
    "allergy": {
        "patterns": [r"алергія", r"алергічн.*реакц", r"атопічн", r"кропив'янка"],
        "symptoms": ["свербіж", "висип", "алергічна реакція", "кропив'янка"],
        "problems": ["алергічна реакція", "свербить шкіра", "висип на шкірі"],
        "category": "immunology"
    },
    "depression": {
        "patterns": [r"депресія", r"депресивн.*розлад", r"пригніченість"],
        "symptoms": ["пригнічений настрій", "апатія", "безсоння"],
        "problems": ["депресія", "поганий настрій", "втрата інтересу"],
        "category": "psychiatry"
    },
    "anxiety": {
        "patterns": [r"тривожність", r"панічн.*атак", r"фобія"],
        "symptoms": ["тривожність", "безпокійство", "панічні атаки"],
        "problems": ["постійне хвилювання", "панічні атаки", "страх"],
        "category": "psychiatry"
    },
    "skin_problems": {
        "patterns": [r"дерматит", r"екзема", r"псоріаз", r"шкірн.*захворюв"],
        "symptoms": ["свербіж шкіри", "лущення", "червоні плями"],
        "problems": ["проблеми зі шкірою", "свербить шкіра", "висипання"],
        "category": "dermatology"
    }
}

MEDICAL_QUERY_TEMPLATES = {
    "direct_condition": [
        "які ліки від {condition}",
        "препарати для лікування {condition}",
        "що приймати при {condition}",
        "медикаменти від {condition}",
        "чим лікувати {condition}",
        "ефективні засоби від {condition}",
        "найкращі ліки від {condition}"
    ],
    "symptom_search": [
        "що пити при {symptom}",
        "ліки від {symptom}",
        "як лікувати {symptom}",
        "препарати при {symptom}",
        "що допомагає від {symptom}",
        "засоби від {symptom}",
        "чим зняти {symptom}"
    ],
    "problem_solving": [
        "у мене {problem}, що робити",
        "як позбутися {problem}",
        "що приймати коли {problem}",
        "ліки коли {problem}",
        "препарати якщо {problem}"
    ]
}

class MemoryMonitor:
    """Monitor memory usage and trigger cleanup when needed"""
    
    def __init__(self, max_memory_gb: float):
        self.max_memory_bytes = max_memory_gb * 1024**3
        
    def check_memory(self) -> bool:
        """Return True if memory usage is acceptable"""
        process = psutil.Process()
        memory_usage = process.memory_info().rss
        return memory_usage < self.max_memory_bytes
    
    def cleanup_if_needed(self) -> None:
        """Force garbage collection if memory usage is high"""
        if not self.check_memory():
            gc.collect()
            time.sleep(0.1)  # Brief pause for cleanup

class CheckpointManager:
    """Manage saving and loading of progress checkpoints"""
    
    def __init__(self, checkpoint_file: Path):
        self.checkpoint_file = checkpoint_file
        
    def save_checkpoint(self, processed_chunks: int, total_pairs: int, 
                       category_counts: Dict, output_files: List[str]) -> None:
        """Save current progress"""
        checkpoint_data = {
            "processed_chunks": processed_chunks,
            "total_pairs": total_pairs,
            "category_counts": category_counts,
            "output_files": output_files,
            "timestamp": time.time()
        }
        
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            
        print(f"[CHECKPOINT] Saved progress: {processed_chunks} chunks, {total_pairs} pairs")
    
    def load_checkpoint(self) -> Optional[Dict]:
        """Load previous progress if exists"""
        if not self.checkpoint_file.exists():
            return None
            
        try:
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARNING] Could not load checkpoint: {e}")
            return None

def extract_medical_conditions(text: str) -> List[Dict]:
    """Extract medical conditions - optimized for multiprocessing"""
    if not isinstance(text, str):
        return []
        
    text_lower = text.lower()
    found_conditions = []
    
    for condition_name, condition_data in COMPREHENSIVE_MEDICAL_PATTERNS.items():
        for pattern in condition_data["patterns"]:
            if re.search(pattern, text_lower):
                found_conditions.append({
                    "name": condition_name,
                    "category": condition_data["category"],
                    "symptoms": condition_data["symptoms"],
                    "problems": condition_data["problems"]
                })
                break
                
    return found_conditions

def process_drug_chunk(chunk_data: Tuple[List[Dict], int, int]) -> Dict:
    """Process a chunk of drugs in parallel - main worker function"""
    drugs_chunk, chunk_id, random_seed = chunk_data
    
    # Set random seed for reproducibility
    random.seed(random_seed + chunk_id)
    np.random.seed(random_seed + chunk_id)
    
    training_pairs = []
    category_counts = defaultdict(int)
    
    for drug_data in drugs_chunk:
        try:
            pairs = generate_drug_pairs(drug_data)
            for pair in pairs:
                training_pairs.append(pair)
                category_counts[pair["metadata"]["category"]] += 1
                
        except Exception as e:
            print(f"[ERROR] Failed to process drug {drug_data.get('Назва препарату', 'Unknown')}: {e}")
            continue
    
    return {
        "chunk_id": chunk_id,
        "training_pairs": training_pairs,
        "category_counts": dict(category_counts),
        "processed_drugs": len(drugs_chunk)
    }

def generate_drug_pairs(drug_data: Dict) -> List[Dict]:
    """Generate training pairs for a single drug"""
    drug_name = drug_data['Назва препарату']
    indications = drug_data['Показання']
    
    if not isinstance(indications, str) or len(indications.strip()) < 10:
        return []
    
    # Extract conditions
    conditions = extract_medical_conditions(indications)
    if not conditions:
        return []
    
    # Generate queries
    generated_queries = []
    
    for condition in conditions[:2]:  # Limit to 2 conditions per drug for memory
        condition_name = condition["name"]
        category = condition["category"]
        
        # Generate 2-3 queries per condition
        condition_ua = get_ukrainian_condition_name(condition_name)
        if condition_ua:
            template = random.choice(MEDICAL_QUERY_TEMPLATES["direct_condition"])
            query = template.format(condition=condition_ua)
            generated_queries.append({
                "query": query,
                "type": "direct_condition",
                "condition": condition_name,
                "category": category
            })
        
        # Add symptom query
        if condition["symptoms"]:
            symptom = random.choice(condition["symptoms"])
            template = random.choice(MEDICAL_QUERY_TEMPLATES["symptom_search"])
            query = template.format(symptom=symptom)
            generated_queries.append({
                "query": query,
                "type": "symptom_search",
                "condition": condition_name,
                "category": category
            })
    
    # Limit queries per drug
    if len(generated_queries) > 4:
        generated_queries = random.sample(generated_queries, 4)
    
    # Create document
    document_text = create_document_text(drug_data)
    
    # Generate simple negatives (for memory efficiency)
    negative_docs = generate_simple_negatives(drug_data, conditions)
    
    # Create training pairs
    training_pairs = []
    for query_info in generated_queries:
        training_pair = {
            "query": query_info["query"],
            "positive": document_text,
            "negatives": negative_docs,
            "metadata": {
                "drug_name": drug_name,
                "condition": query_info["condition"],
                "category": query_info["category"],
                "query_type": query_info["type"]
            }
        }
        training_pairs.append(training_pair)
    
    return training_pairs

def get_ukrainian_condition_name(condition_code: str) -> str:
    """Map condition codes to Ukrainian names"""
    mapping = {
        "diabetes": "діабету",
        "hypertension": "високого тиску",
        "heart_disease": "серцевих захворювань",
        "infection": "інфекції",
        "pain": "болю",
        "headache": "головного болю",
        "gastric": "захворювань шлунка",
        "asthma": "астми",
        "allergy": "алергії",
        "depression": "депресії",
        "anxiety": "тривожності",
        "respiratory_infection": "респіраторних інфекцій",
        "diarrhea": "діареї",
        "cough": "кашлю",
        "skin_problems": "шкірних проблем"
    }
    return mapping.get(condition_code, condition_code)

def create_document_text(drug_data: Dict) -> str:
    """Create document text for embedding"""
    parts = []
    
    if drug_data.get('Назва препарату'):
        parts.append(str(drug_data['Назва препарату']))
    
    if drug_data.get('Фармакотерапевтична група'):
        parts.append(str(drug_data['Фармакотерапевтична група'])[:300])
    
    if drug_data.get('Показання'):
        parts.append(str(drug_data['Показання']))
        
    return ' [SEP] '.join(parts)

def generate_simple_negatives(drug_data: Dict, conditions: List[Dict]) -> List[str]:
    """Generate simple negative examples"""
    # For memory efficiency, use predefined negative examples
    # In real implementation, you'd sample from other drugs
    negative_examples = [
        "Аспірин [SEP] Знеболюючі засоби [SEP] Біль та запалення",
        "Парацетамол [SEP] Жарознижуючі [SEP] Підвищена температура",
        "Ібупрофен [SEP] НПЗП [SEP] Запальні процеси",
        "Амоксицилін [SEP] Антибіотики [SEP] Бактеріальні інфекції",
        "Лоратадин [SEP] Антигістамінні [SEP] Алергічні реакції"
    ]
    return random.sample(negative_examples, min(3, len(negative_examples)))

class OptimizedTrainingPairsGenerator:
    """Memory-efficient, resumable training pairs generator"""
    
    def __init__(self, config: ResourceConfig):
        self.config = config
        self.memory_monitor = MemoryMonitor(config.max_memory_gb)
        
    def generate_training_pairs(self, input_file: Path, output_dir: Path, 
                              target_pairs: int = 20000) -> None:
        """Main generation method with checkpointing and parallelization"""
        
        # Setup checkpoint
        checkpoint_file = output_dir / "generation_checkpoint.json"
        checkpoint_manager = CheckpointManager(checkpoint_file)
        
        # Try to resume from checkpoint
        checkpoint = checkpoint_manager.load_checkpoint()
        if checkpoint:
            print(f"[RESUME] Found checkpoint with {checkpoint['processed_chunks']} chunks")
            processed_chunks = checkpoint['processed_chunks']
            total_pairs = checkpoint['total_pairs']
            category_counts = defaultdict(int, checkpoint['category_counts'])
            output_files = checkpoint['output_files']
        else:
            print("[START] Starting fresh generation")
            processed_chunks = 0
            total_pairs = 0
            category_counts = defaultdict(int)
            output_files = []
        
        # Load and chunk data
        print(f"[LOAD] Loading data from {input_file}")
        df = pd.read_parquet(input_file)
        
        # Convert to dict for multiprocessing efficiency
        drugs_data = df.to_dict('records')
        del df  # Free memory
        gc.collect()
        
        # Create chunks
        chunks = []
        for i in range(processed_chunks, len(drugs_data), self.config.chunk_size):
            if total_pairs >= target_pairs:
                break
            chunk = drugs_data[i:i + self.config.chunk_size]
            chunks.append((chunk, len(chunks) + processed_chunks, 42))  # 42 as base seed
        
        print(f"[PLAN] Will process {len(chunks)} chunks of {self.config.chunk_size} drugs each")
        print(f"[PLAN] Using {self.config.max_workers} workers, max {self.config.max_memory_gb}GB RAM")
        
        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            try:
                futures = {executor.submit(process_drug_chunk, chunk_data): i 
                          for i, chunk_data in enumerate(chunks)}
                
                for future in as_completed(futures):
                    if INTERRUPTED:
                        print("[INTERRUPT] Cancelling remaining tasks...")
                        break
                    
                    try:
                        result = future.result(timeout=300)  # 5 minute timeout per chunk
                        
                        # Save chunk results immediately
                        chunk_file = output_dir / f"chunk_{result['chunk_id']:04d}.jsonl"
                        with open(chunk_file, 'w', encoding='utf-8') as f:
                            for pair in result['training_pairs']:
                                f.write(json.dumps(pair, ensure_ascii=False) + '\n')
                        
                        output_files.append(str(chunk_file))
                        total_pairs += len(result['training_pairs'])
                        
                        # Update counts
                        for category, count in result['category_counts'].items():
                            category_counts[category] += count
                        
                        processed_chunks += 1
                        
                        print(f"[PROGRESS] Chunk {result['chunk_id']}: "
                              f"{len(result['training_pairs'])} pairs, "
                              f"total: {total_pairs:,}")
                        
                        # Save checkpoint periodically
                        if processed_chunks % self.config.checkpoint_interval == 0:
                            checkpoint_manager.save_checkpoint(
                                processed_chunks, total_pairs, 
                                dict(category_counts), output_files
                            )
                        
                        # Check memory and clean up
                        self.memory_monitor.cleanup_if_needed()
                        
                        # Stop if target reached
                        if total_pairs >= target_pairs:
                            print(f"[TARGET] Reached target of {target_pairs} pairs")
                            break
                            
                    except Exception as e:
                        print(f"[ERROR] Chunk failed: {e}")
                        continue
                        
            except KeyboardInterrupt:
                print("[INTERRUPT] Graceful shutdown...")
                
        # Final checkpoint
        checkpoint_manager.save_checkpoint(
            processed_chunks, total_pairs, dict(category_counts), output_files
        )
        
        # Merge all chunk files into final output
        self.merge_chunk_files(output_files, output_dir / "training_pairs.jsonl")
        
        # Generate final statistics
        self.generate_final_stats(dict(category_counts), total_pairs, 
                                output_dir / "training_stats.json")
        
        print(f"\n[COMPLETED] Generated {total_pairs:,} training pairs")
        print(f"[CLEANUP] Removing {len(output_files)} temporary chunk files...")
        
        # Clean up chunk files
        for chunk_file in output_files:
            try:
                Path(chunk_file).unlink()
            except:
                pass
    
    def merge_chunk_files(self, chunk_files: List[str], output_file: Path) -> None:
        """Merge all chunk files into final training pairs file"""
        print(f"[MERGE] Combining {len(chunk_files)} chunk files...")
        
        with open(output_file, 'w', encoding='utf-8') as outf:
            for chunk_file in chunk_files:
                try:
                    with open(chunk_file, 'r', encoding='utf-8') as inf:
                        for line in inf:
                            outf.write(line)
                except Exception as e:
                    print(f"[WARNING] Could not read chunk file {chunk_file}: {e}")
    
    def generate_final_stats(self, category_counts: Dict, total_pairs: int, 
                           stats_file: Path) -> None:
        """Generate final statistics"""
        stats = {
            "total_pairs": total_pairs,
            "category_distribution": category_counts,
            "generation_completed": True,
            "timestamp": time.time()
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser(description="Optimized training pairs generator")
    parser.add_argument("--input", default="data/processed/clean_medical.parquet")
    parser.add_argument("--output_dir", default="data/processed/training")
    parser.add_argument("--target_pairs", type=int, default=20000)
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    print("=== OPTIMIZED MEDICAL TRAINING PAIRS GENERATION ===")
    
    # Auto-detect optimal configuration
    config = ResourceConfig.auto_detect()
    print(f"[CONFIG] Max memory: {config.max_memory_gb:.1f}GB")
    print(f"[CONFIG] Chunk size: {config.chunk_size}")
    print(f"[CONFIG] Workers: {config.max_workers}")
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate training pairs
    generator = OptimizedTrainingPairsGenerator(config)
    
    try:
        generator.generate_training_pairs(
            Path(args.input), 
            output_dir, 
            args.target_pairs
        )
        print("[SUCCESS] Training pairs generation completed!")
        
    except Exception as e:
        print(f"[ERROR] Generation failed: {e}")
        print("Progress has been saved. Use --resume to continue.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
