#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tune Sentence Transformer for Medical Search
Uses Multiple Negative Ranking Loss (MNRL) with Ukrainian medical data
"""

import json
import torch
import logging
from pathlib import Path
from typing import List, Dict
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from torch.utils.data import DataLoader
import argparse

# Setup logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

class MedicalTrainingDataLoader:
    """Load and prepare medical training data for MNRL"""
    
    def __init__(self, training_file: Path):
        self.training_file = training_file
        
    def load_training_examples(self) -> List[InputExample]:
        """Convert JSONL format to SentenceTransformers InputExample format"""
        examples = []
        
        with open(self.training_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    data = json.loads(line)
                    
                    # Create InputExample for MNRL
                    # Format: InputExample(texts=[query, positive, negative1, negative2, ...])
                    texts = [data['query'], data['positive']] + data['negatives']
                    
                    example = InputExample(texts=texts)
                    examples.append(example)
                    
                except json.JSONDecodeError as e:
                    logging.warning(f"Skipping invalid JSON on line {line_num}: {e}")
                    continue
        
        logging.info(f"Loaded {len(examples)} training examples")
        return examples
    
    def create_evaluation_set(self, examples: List[InputExample], eval_size: int = 500) -> Dict:
        """Create evaluation set for monitoring training progress"""
        import random
        
        # Sample evaluation examples
        eval_examples = random.sample(examples, min(eval_size, len(examples)))
        
        # Convert to IR evaluation format
        queries = {}
        corpus = {}
        relevant_docs = {}
        
        for i, example in enumerate(eval_examples):
            query_id = f"q_{i}"
            doc_id = f"doc_{i}"
            
            queries[query_id] = example.texts[0]  # query
            corpus[doc_id] = example.texts[1]     # positive document
            relevant_docs[query_id] = {doc_id: 1} # relevance score
        
        return {
            'queries': queries,
            'corpus': corpus, 
            'relevant_docs': relevant_docs
        }

class MedicalSentenceTransformerTrainer:
    """Fine-tune sentence transformer for medical search"""
    
    def __init__(self, base_model: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.base_model = base_model
        self.model = None
        
    def initialize_model(self) -> None:
        """Initialize base model"""
        logging.info(f"Loading base model: {self.base_model}")
        self.model = SentenceTransformer(self.base_model)
        
        # Print model info
        logging.info(f"Model max sequence length: {self.model.max_seq_length}")
        logging.info(f"Model embedding dimension: {self.model.get_sentence_embedding_dimension()}")
    
    def train(self, training_examples: List[InputExample], 
              output_path: Path,
              epochs: int = 4,
              batch_size: int = 16,
              warmup_steps: int = 100,
              evaluation_steps: int = 1000) -> None:
        """Train model with MNRL loss"""
        
        if self.model is None:
            self.initialize_model()
        
        # Create data loader
        train_dataloader = DataLoader(training_examples, shuffle=True, batch_size=batch_size)
        
        # Setup MNRL loss
        train_loss = losses.MultipleNegativesRankingLoss(self.model)
        
        # Create evaluation set
        data_loader = MedicalTrainingDataLoader(Path("dummy"))  # Just for eval method
        eval_data = data_loader.create_evaluation_set(training_examples, eval_size=200)
        
        evaluator = InformationRetrievalEvaluator(
            eval_data['queries'],
            eval_data['corpus'], 
            eval_data['relevant_docs'],
            name="medical_eval"
        )
        
        # Calculate warmup steps
        num_training_steps = len(train_dataloader) * epochs
        warmup_steps = min(warmup_steps, int(num_training_steps * 0.1))
        
        logging.info(f"Training configuration:")
        logging.info(f"  Training examples: {len(training_examples)}")
        logging.info(f"  Epochs: {epochs}")
        logging.info(f"  Batch size: {batch_size}")
        logging.info(f"  Training steps: {num_training_steps}")
        logging.info(f"  Warmup steps: {warmup_steps}")
        logging.info(f"  Evaluation steps: {evaluation_steps}")
        
        # Train model
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            output_path=str(output_path),
            evaluator=evaluator,
            evaluation_steps=evaluation_steps,
            save_best_model=True
        )
        
        logging.info(f"Training completed. Model saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Fine-tune sentence transformer for medical search")
    parser.add_argument("--training_data", default="data/processed/training/improved_training_pairs.jsonl")
    parser.add_argument("--model_name", default="paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--output_dir", default="models/medical-search-ua")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16) 
    parser.add_argument("--warmup_steps", type=int, default=100)
    
    args = parser.parse_args()
    
    print("=== MEDICAL SENTENCE TRANSFORMER FINE-TUNING ===")
    
    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load training data
    data_loader = MedicalTrainingDataLoader(Path(args.training_data))
    training_examples = data_loader.load_training_examples()
    
    if len(training_examples) == 0:
        print("ERROR: No training examples loaded!")
        return 1
    
    # Initialize trainer
    trainer = MedicalSentenceTransformerTrainer(args.model_name)
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Start training
    try:
        trainer.train(
            training_examples=training_examples,
            output_path=output_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            warmup_steps=args.warmup_steps
        )
        
        print(f"\n=== TRAINING COMPLETED SUCCESSFULLY ===")
        print(f"Fine-tuned model saved to: {output_path}")
        
        # Test model
        print(f"\n=== TESTING FINE-TUNED MODEL ===")
        model = SentenceTransformer(str(output_path))
        
        test_queries = [
            "ліки від головного болю",
            "препарати при високому тиску", 
            "що приймати при інфекції"
        ]
        
        for query in test_queries:
            embedding = model.encode(query)
            print(f"Query: '{query}' -> Embedding shape: {embedding.shape}")
        
        return 0
        
    except Exception as e:
        print(f"ERROR during training: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
