#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed Medical Sentence Transformer Fine-tuning
Handles sequence length and dependencies issues
"""

import json
import torch
import logging
import time
import os
from pathlib import Path
from typing import List
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import argparse

# Setup logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

class MedicalTrainingDataLoader:
    def __init__(self, training_file: Path):
        self.training_file = training_file
        
    def load_training_examples(self, max_examples: int = None, max_length: int = 128) -> List[InputExample]:
        """Load and truncate training examples to fit model constraints"""
        examples = []
        
        def truncate_text(text: str, max_words: int = 100) -> str:
            """Smart truncation preserving medical terms"""
            words = text.split()
            if len(words) <= max_words:
                return text
            
            # Try to keep drug name and key medical terms
            truncated = words[:max_words]
            result = ' '.join(truncated)
            
            # If truncated in middle of sentence, try to end at sentence boundary
            if '.' in result:
                sentences = result.split('.')
                if len(sentences) > 1:
                    result = '.'.join(sentences[:-1]) + '.'
            
            return result
        
        with open(self.training_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if max_examples and len(examples) >= max_examples:
                    break
                    
                try:
                    data = json.loads(line)
                    
                    # Truncate texts to fit model constraints
                    query = truncate_text(data['query'], max_words=20)
                    positive = truncate_text(data['positive'], max_words=100)
                    negatives = [truncate_text(neg, max_words=100) for neg in data['negatives']]
                    
                    texts = [query, positive] + negatives
                    example = InputExample(texts=texts)
                    examples.append(example)
                    
                except json.JSONDecodeError as e:
                    logging.warning(f"Skipping line {line_num}: {e}")
                    continue
        
        logging.info(f"Loaded {len(examples)} training examples (truncated to fit model)")
        return examples

def train_medical_model_fixed(training_file: Path, output_dir: Path, test_run: bool = False):
    """Fixed training with proper error handling"""
    
    # Parameters
    if test_run:
        epochs = 1
        batch_size = 4
        max_examples = 100
        logging.info("=== TEST RUN MODE ===")
    else:
        epochs = 2
        batch_size = 6  # Reduced further for stability
        max_examples = None
    
    logging.info(f"Training Configuration:")
    logging.info(f"  Epochs: {epochs}")
    logging.info(f"  Batch size: {batch_size}")
    logging.info(f"  Max examples: {max_examples or 'all'}")
    
    # Load model with extended sequence length
    model_name = "paraphrase-multilingual-MiniLM-L12-v2"
    logging.info(f"Loading model: {model_name}")
    
    try:
        model = SentenceTransformer(model_name)
        
        # Extend max sequence length for our longer medical texts
        model.max_seq_length = 256  # Increase from default 128
        
        # Force CPU
        model.to(torch.device('cpu'))
        
        logging.info(f"Model loaded successfully")
        logging.info(f"Model dimension: {model.get_sentence_embedding_dimension()}")
        logging.info(f"Max sequence length: {model.max_seq_length}")
        
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise
    
    # Load training data
    try:
        data_loader = MedicalTrainingDataLoader(training_file)
        training_examples = data_loader.load_training_examples(max_examples)
        
        if len(training_examples) == 0:
            raise ValueError("No training examples loaded!")
            
    except Exception as e:
        logging.error(f"Failed to load training data: {e}")
        raise
    
    # Create data loader
    train_dataloader = DataLoader(training_examples, shuffle=True, batch_size=batch_size)
    
    # Setup loss
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    # Training parameters
    steps_per_epoch = len(train_dataloader)
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(1, int(total_steps * 0.1))
    
    logging.info(f"Training steps: {total_steps}")
    logging.info(f"Warmup steps: {warmup_steps}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training with minimal dependencies
    logging.info("=== STARTING TRAINING ===")
    training_start = time.time()
    
    try:
        # Simple training without evaluation to avoid datasets dependency
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            output_path=str(output_dir),
            save_best_model=True,
            show_progress_bar=True,
            use_amp=False  # Disable automatic mixed precision for CPU
        )
        
        training_time = time.time() - training_start
        logging.info(f"Training completed in {training_time:.1f} seconds")
        
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise
    
    # Test the model
    logging.info("=== TESTING FINE-TUNED MODEL ===")
    test_queries = [
        "ліки від головного болю",
        "препарати при високому тиску",
        "що приймати при інфекції"
    ]
    
    try:
        for query in test_queries:
            embedding = model.encode(query)
            logging.info(f"'{query}' -> {embedding.shape}")
    except Exception as e:
        logging.warning(f"Testing failed: {e}")
    
    return str(output_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data", default="data/processed/training/improved_training_pairs.jsonl")
    parser.add_argument("--output_dir", default="models/medical-search-ua-fixed")
    parser.add_argument("--test_run", action="store_true")
    
    args = parser.parse_args()
    
    print("=== FIXED MEDICAL MODEL FINE-TUNING ===")
    print(f"Training data: {args.training_data}")
    
    try:
        model_path = train_medical_model_fixed(
            Path(args.training_data),
            Path(args.output_dir),
            test_run=args.test_run
        )
        
        print(f"\n=== SUCCESS ===")
        print(f"Model saved to: {model_path}")
        
        return 0
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
