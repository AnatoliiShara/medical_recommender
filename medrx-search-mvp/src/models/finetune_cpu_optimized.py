#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPU-Optimized Medical Sentence Transformer Fine-tuning
Designed for local laptops without GPU
"""

import json
import torch
import logging
import time
from pathlib import Path
from typing import List, Dict
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import argparse

# Setup logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

class MedicalTrainingDataLoader:
    def __init__(self, training_file: Path):
        self.training_file = training_file
        
    def load_training_examples(self, max_examples: int = None) -> List[InputExample]:
        """Load training examples with optional limit for testing"""
        examples = []
        
        with open(self.training_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if max_examples and len(examples) >= max_examples:
                    break
                    
                try:
                    data = json.loads(line)
                    texts = [data['query'], data['positive']] + data['negatives']
                    example = InputExample(texts=texts)
                    examples.append(example)
                    
                except json.JSONDecodeError as e:
                    logging.warning(f"Skipping line {line_num}: {e}")
                    continue
        
        logging.info(f"Loaded {len(examples)} training examples")
        return examples

def train_medical_model_cpu(training_file: Path, output_dir: Path, 
                           test_run: bool = False):
    """Train model optimized for CPU"""
    
    # CPU-optimized parameters
    if test_run:
        epochs = 1
        batch_size = 4
        max_examples = 100
        logging.info("=== TEST RUN MODE ===")
    else:
        epochs = 2  # Reduced from 4 for CPU
        batch_size = 8  # Reduced from 16 for CPU
        max_examples = None
    
    logging.info(f"CPU Training Configuration:")
    logging.info(f"  Epochs: {epochs}")
    logging.info(f"  Batch size: {batch_size}")
    logging.info(f"  Max examples: {max_examples or 'all'}")
    
    # Load model
    model_name = "paraphrase-multilingual-MiniLM-L12-v2"
    logging.info(f"Loading model: {model_name}")
    
    start_time = time.time()
    model = SentenceTransformer(model_name)
    
    # Force CPU usage
    model.to(torch.device('cpu'))
    
    logging.info(f"Model loaded in {time.time() - start_time:.1f}s")
    logging.info(f"Model dimension: {model.get_sentence_embedding_dimension()}")
    logging.info(f"Max sequence length: {model.max_seq_length}")
    
    # Load training data
    data_loader = MedicalTrainingDataLoader(training_file)
    training_examples = data_loader.load_training_examples(max_examples)
    
    if len(training_examples) == 0:
        raise ValueError("No training examples loaded!")
    
    # Create data loader
    train_dataloader = DataLoader(training_examples, shuffle=True, batch_size=batch_size)
    
    # Setup loss
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    # Calculate steps
    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * epochs
    warmup_steps = min(100, int(total_steps * 0.1))
    
    logging.info(f"Training steps: {total_steps} ({steps_per_epoch} per epoch)")
    logging.info(f"Warmup steps: {warmup_steps}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop with progress tracking
    logging.info("=== STARTING TRAINING ===")
    training_start = time.time()
    
    # Use simple fit without evaluation for speed
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=str(output_dir),
        save_best_model=True,
        show_progress_bar=True
    )
    
    training_time = time.time() - training_start
    logging.info(f"Training completed in {training_time:.1f} seconds ({training_time/60:.1f} minutes)")
    
    # Test the model
    logging.info("=== TESTING FINE-TUNED MODEL ===")
    test_queries = [
        "ліки від головного болю",
        "препарати при високому тиску",
        "що приймати при інфекції",
        "засоби від алергії",
        "лікування серцевих захворювань"
    ]
    
    for query in test_queries:
        start_encode = time.time()
        embedding = model.encode(query)
        encode_time = time.time() - start_encode
        logging.info(f"'{query}' -> {embedding.shape} in {encode_time:.3f}s")
    
    return str(output_dir)

def main():
    parser = argparse.ArgumentParser(description="CPU-optimized medical model training")
    parser.add_argument("--training_data", default="data/processed/training/improved_training_pairs.jsonl")
    parser.add_argument("--output_dir", default="models/medical-search-ua-cpu")
    parser.add_argument("--test_run", action="store_true", help="Quick test with 100 examples")
    
    args = parser.parse_args()
    
    print("=== CPU-OPTIMIZED MEDICAL MODEL FINE-TUNING ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CPU cores available: {torch.get_num_threads()}")
    print(f"Training data: {args.training_data}")
    print(f"Output directory: {args.output_dir}")
    
    try:
        model_path = train_medical_model_cpu(
            Path(args.training_data),
            Path(args.output_dir),
            test_run=args.test_run
        )
        
        print(f"\n=== SUCCESS ===")
        print(f"Model saved to: {model_path}")
        print(f"To use: SentenceTransformer('{model_path}')")
        
        return 0
        
    except Exception as e:
        print(f"ERROR: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
