"""
Medical priors integration for boosting relevant drugs based on detected conditions.

Uses disease_med_priors.jsonl to map conditions → drugs and apply boosts.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
import pandas as pd


class MedicalPriorsEngine:
    """
    Loads disease priors and provides drug boosting based on matched conditions.
    """
    
    def __init__(self, priors_path: str, corpus_df: pd.DataFrame):
        """
        Initialize priors engine.
        
        Args:
            priors_path: Path to disease_med_priors.jsonl
            corpus_df: Full drug corpus (for INN/brand → doc_id mapping)
        """
        self.priors_path = Path(priors_path)
        self.corpus_df = corpus_df
        
        # Load priors
        self.priors: Dict[str, Dict] = {}
        self._load_priors()
        
        # Build INN/brand → doc_id index
        self.inn_to_docs: Dict[str, List[int]] = {}
        self.brand_to_docs: Dict[str, List[int]] = {}
        self._build_drug_index()
    
    def _load_priors(self):
        """Load disease priors from JSONL."""
        if not self.priors_path.exists():
            print(f"[WARN] Priors file not found: {self.priors_path}")
            return
        
        with open(self.priors_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    prior = json.loads(line)
                    prior_id = prior.get("id", "")
                    if prior_id:
                        self.priors[prior_id] = prior
                except json.JSONDecodeError:
                    continue
        
        print(f"[INFO] Loaded {len(self.priors)} medical priors")
    
    def _build_drug_index(self):
        """
        Build reverse index: INN/brand name → list of doc_ids.
        
        This allows us to quickly find which documents contain a specific INN or brand.
        """
        print("[INFO] Building drug index (INN/brand → doc_id)...")
        
        for idx, row in self.corpus_df.iterrows():
            doc_id = int(idx)
            
            # Extract brand name
            brand_name = row.get("Назва препарату", "")
            if brand_name and isinstance(brand_name, str):
                brand_clean = brand_name.lower().strip()
                if brand_clean:
                    if brand_clean not in self.brand_to_docs:
                        self.brand_to_docs[brand_clean] = []
                    self.brand_to_docs[brand_clean].append(doc_id)
            
            # Extract INN from composition (Склад)
            # Note: This is heuristic - may need refinement based on your data
            composition = row.get("Склад", "")
            if composition and isinstance(composition, str):
                comp_lower = composition.lower()
                # Simple extraction: look for known INNs in text
                # In production, you'd parse this more carefully
                for prior in self.priors.values():
                    for inn in prior.get("inn", []):
                        inn_lower = inn.lower().replace("_", " ")
                        if inn_lower in comp_lower:
                            if inn_lower not in self.inn_to_docs:
                                self.inn_to_docs[inn_lower] = []
                            if doc_id not in self.inn_to_docs[inn_lower]:
                                self.inn_to_docs[inn_lower].append(doc_id)
        
        print(f"[INFO] Indexed {len(self.brand_to_docs)} brands, {len(self.inn_to_docs)} INNs")
    
    def get_boosts(
        self,
        matched_conditions: List[str],
        query_text: str = ""
    ) -> Dict[int, float]:
        """
        Get drug boosts based on matched conditions.
        
        Args:
            matched_conditions: List of condition IDs (e.g., ["diarrhea", "fever"])
            query_text: Original user query (for neg_triggers check)
        
        Returns:
            Dict mapping doc_id → boost_score
        """
        boosts: Dict[int, float] = {}
        
        query_lower = query_text.lower()
        
        for condition_id in matched_conditions:
            # Map condition name to prior ID
            # Your clinical detection returns "diarrhea", prior ID is "diarrhea_basic"
            prior = self._find_prior_for_condition(condition_id)
            
            if not prior:
                continue
            
            # Check negative triggers
            neg_triggers = prior.get("neg_triggers", [])
            if any(neg in query_lower for neg in neg_triggers):
                # Don't boost if negative trigger found
                # Example: "діарея з кров'ю" has "кров" → skip boosting loperamide
                continue
            
            # Get boost value
            boost_value = float(prior.get("boost", 0.0))
            if boost_value <= 0:
                continue
            
            # Get drugs to boost (from INN and brands)
            drugs_to_boost = self._get_drugs_for_prior(prior)
            
            # Apply boost
            for doc_id in drugs_to_boost:
                # Additive boost (if multiple conditions match, boosts accumulate)
                boosts[doc_id] = boosts.get(doc_id, 0.0) + boost_value
        
        return boosts
    
    def _find_prior_for_condition(self, condition: str) -> Dict | None:
        """
        Find prior by condition name.
        
        Handles mapping like: "diarrhea" → "diarrhea_basic"
        """
        # Direct match
        if condition in self.priors:
            return self.priors[condition]
        
        # Try with suffix
        for suffix in ["_basic", "_general", "_symptoms"]:
            prior_id = f"{condition}{suffix}"
            if prior_id in self.priors:
                return self.priors[prior_id]
        
        # Partial match (e.g., "diarrhea" matches "diarrhea_basic")
        for prior_id, prior in self.priors.items():
            if condition in prior_id or prior_id.startswith(condition):
                return prior
        
        return None
    
    def _get_drugs_for_prior(self, prior: Dict) -> Set[int]:
        """
        Get all doc_ids that match this prior's INN or brands.
        """
        drugs: Set[int] = set()
        
        # Get from INNs
        for inn in prior.get("inn", []):
            inn_lower = inn.lower().replace("_", " ")
            if inn_lower in self.inn_to_docs:
                drugs.update(self.inn_to_docs[inn_lower])
        
        # Get from brand names
        for brand in prior.get("brands_opt", []):
            brand_lower = brand.lower().strip()
            
            # Exact match
            if brand_lower in self.brand_to_docs:
                drugs.update(self.brand_to_docs[brand_lower])
            
            # Partial match (brand name often has additional text)
            for brand_in_corpus, doc_ids in self.brand_to_docs.items():
                if brand_lower in brand_in_corpus or brand_in_corpus in brand_lower:
                    drugs.update(doc_ids)
        
        return drugs


# ============ CLI FOR TESTING ============
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test medical priors engine")
    parser.add_argument(
        "--corpus",
        type=str,
        default="data/raw/compendium_all.parquet",
        help="Path to corpus parquet"
    )
    parser.add_argument(
        "--priors",
        type=str,
        default="data/priors/disease_med_priors.jsonl",
        help="Path to priors JSONL"
    )
    
    args = parser.parse_args()
    
    # Load corpus
    print(f"Loading corpus: {args.corpus}")
    df = pd.read_parquet(args.corpus)
    
    # Initialize engine
    print(f"Loading priors: {args.priors}")
    engine = MedicalPriorsEngine(args.priors, df)
    
    # Test queries
    test_cases = [
        {
            "query": "діарея з кров'ю",
            "conditions": ["diarrhea"],
        },
        {
            "query": "легка діарея",
            "conditions": ["diarrhea"],
        },
        {
            "query": "головний біль",
            "conditions": ["headache_migraine"],
        },
    ]
    
    for test in test_cases:
        print(f"\n{'='*60}")
        print(f"Query: {test['query']}")
        print(f"Conditions: {test['conditions']}")
        print(f"{'='*60}")
        
        boosts = engine.get_boosts(test["conditions"], test["query"])
        
        if boosts:
            print(f"Found {len(boosts)} drugs to boost:")
            # Show top 10
            sorted_boosts = sorted(boosts.items(), key=lambda x: x[1], reverse=True)
            for doc_id, boost in sorted_boosts[:10]:
                drug_name = df.iloc[doc_id].get("Назва препарату", "Unknown")
                print(f"  {doc_id}: {drug_name} (boost: +{boost:.2f})")
        else:
            print("No boosts found")