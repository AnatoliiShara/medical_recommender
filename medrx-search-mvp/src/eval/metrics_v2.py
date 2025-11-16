# src/eval/metrics_v2.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Metrics Module for MedRx Search Evaluation
Comprehensive ranking metrics with stratified evaluation support
"""

import numpy as np
from typing import List, Dict, Set, Optional, Any
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')


class RankingMetrics:
    @staticmethod
    def precision_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
        if k <= 0 or not relevant:
            return 0.0
        retrieved_at_k = retrieved[:k]
        relevant_retrieved = sum(1 for doc_id in retrieved_at_k if doc_id in relevant)
        return relevant_retrieved / k

    @staticmethod
    def recall_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
        if k <= 0 or not relevant:
            return 0.0
        retrieved_at_k = set(retrieved[:k])
        relevant_retrieved = len(retrieved_at_k & relevant)
        return relevant_retrieved / len(relevant)

    @staticmethod
    def average_precision(retrieved: List[int], relevant: Set[int]) -> float:
        if not relevant:
            return 0.0
        relevant_found = 0
        sum_precisions = 0.0
        for i, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                relevant_found += 1
                sum_precisions += relevant_found / i
        return sum_precisions / len(relevant) if relevant_found > 0 else 0.0

    @staticmethod
    def ndcg_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
        if k <= 0 or not relevant:
            return 0.0
        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:k], start=1):
            if doc_id in relevant:
                dcg += 1.0 / np.log2(i + 1)
        idcg = sum(1.0 / np.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
        return dcg / idcg if idcg > 0 else 0.0

    @staticmethod
    def reciprocal_rank(retrieved: List[int], relevant: Set[int]) -> float:
        if not relevant:
            return 0.0
        for i, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                return 1.0 / i
        return 0.0

    @staticmethod
    def f1_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
        p = RankingMetrics.precision_at_k(retrieved, relevant, k)
        r = RankingMetrics.recall_at_k(retrieved, relevant, k)
        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)

    @staticmethod
    def hit_rate_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
        if k <= 0 or not relevant:
            return 0.0
        retrieved_at_k = set(retrieved[:k])
        return 1.0 if len(retrieved_at_k & relevant) > 0 else 0.0


class MetricsAggregator:
    def __init__(self):
        self.query_metrics: List[Dict[str, Any]] = []
        self.k_values = [5, 10, 20, 50, 100]

    def add_query_result(
        self,
        query_id: int,
        retrieved: List[int],
        relevant: Set[int],
        metadata: Optional[Dict[str, Any]] = None
    ):
        m: Dict[str, Any] = {
            'query_id': query_id,
            'num_retrieved': len(retrieved),
            'num_relevant': len(relevant)
        }
        if metadata:
            m.update(metadata)

        for k in self.k_values:
            m[f'p@{k}'] = RankingMetrics.precision_at_k(retrieved, relevant, k)
            m[f'r@{k}'] = RankingMetrics.recall_at_k(retrieved, relevant, k)
            m[f'ndcg@{k}'] = RankingMetrics.ndcg_at_k(retrieved, relevant, k)
            m[f'f1@{k}'] = RankingMetrics.f1_at_k(retrieved, relevant, k)
            m[f'hit@{k}'] = RankingMetrics.hit_rate_at_k(retrieved, relevant, k)

        m['ap'] = RankingMetrics.average_precision(retrieved, relevant)
        m['rr'] = RankingMetrics.reciprocal_rank(retrieved, relevant)
        self.query_metrics.append(m)

    def compute_aggregates(self) -> Dict[str, float]:
        if not self.query_metrics:
            return {}
        keys = [
            k for k in self.query_metrics[0].keys()
            if k not in {'query_id', 'num_retrieved', 'num_relevant', 'category', 'complexity', 'intent'}
        ]
        agg: Dict[str, float] = {}
        for key in keys:
            vals = [qm[key] for qm in self.query_metrics if key in qm]
            if vals:
                agg[f'{key}_mean'] = float(np.mean(vals))
                agg[f'{key}_std'] = float(np.std(vals))
                agg[f'{key}_median'] = float(np.median(vals))
        agg['map'] = float(np.mean([qm['ap'] for qm in self.query_metrics]))
        agg['mrr'] = float(np.mean([qm['rr'] for qm in self.query_metrics]))
        agg['total_queries'] = len(self.query_metrics)
        agg['avg_retrieved'] = float(np.mean([qm['num_retrieved'] for qm in self.query_metrics]))
        agg['avg_relevant'] = float(np.mean([qm['num_relevant'] for qm in self.query_metrics]))
        return agg

    def compute_stratified(self, stratify_by: str) -> Dict[str, Dict[str, float]]:
        if not self.query_metrics:
            return {}
        strata: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for qm in self.query_metrics:
            if stratify_by in qm:
                strata[str(qm[stratify_by])].append(qm)
        out: Dict[str, Dict[str, float]] = {}
        for s, lst in strata.items():
            tmp = MetricsAggregator()
            tmp.query_metrics = lst
            out[s] = tmp.compute_aggregates()
        return out

    def get_query_level_results(self) -> List[Dict[str, Any]]:
        return self.query_metrics


class MedicalRelevanceMetrics:
    @staticmethod
    def safety_score(retrieved: List[Dict[str, Any]], relevant: Set[int], k: int = 10) -> Dict[str, float]:
        if not retrieved:
            return {
                'dangerous_drugs_rate': 0.0,
                'evidence_A_rate': 0.0,
                'evidence_F_rate': 0.0,
                'avg_evidence_quality': 0.0,
            }
        top_k = retrieved[:k]
        dangerous = sum(1 for item in top_k if item.get('is_dangerous', False))
        ev_counts = defaultdict(int)
        for item in top_k:
            ev_counts[item.get('evidence_category', 'unknown')] += 1
        weight = {'A': 1.0, 'B': 0.75, 'C': 0.5, 'D': 0.3, 'F': 0.1}
        return {
            'dangerous_drugs_rate': dangerous / len(top_k),
            'evidence_A_rate': ev_counts.get('A', 0) / len(top_k),
            'evidence_F_rate': ev_counts.get('F', 0) / len(top_k),
            'avg_evidence_quality': sum(weight.get(item.get('evidence_category', 'F'), 0.1) for item in top_k) / len(top_k),
        }


def print_metrics_summary(aggregated: Dict[str, float], title: str = "Metrics Summary"):
    print(f"\n{'='*60}\n  {title}\n{'='*60}\n")
    core = [
        ('nDCG@5', 'ndcg@5_mean'),
        ('nDCG@10', 'ndcg@10_mean'),
        ('nDCG@20', 'ndcg@20_mean'),
        ('Precision@5', 'p@5_mean'),
        ('Precision@10', 'p@10_mean'),
        ('Recall@10', 'r@10_mean'),
        ('Recall@50', 'r@50_mean'),
        ('MRR', 'mrr'),
        ('MAP', 'map'),
    ]
    print(f"{'Metric':<20} {'Value':>10} {'Std':>10}")
    print(f"{'-'*42}")
    for label, key in core:
        if key in aggregated:
            val = aggregated[key]
            std = aggregated.get(key.replace('_mean', '_std'), 0.0)
            print(f"{label:<20} {val:>10.4f} {std:>10.4f}")
    print(f"\n{'Coverage':<20} {aggregated.get('total_queries', 0):>10.0f} queries")
    print(f"{'Avg Retrieved':<20} {aggregated.get('avg_retrieved', 0):>10.1f} docs")
    print(f"{'Avg Relevant':<20} {aggregated.get('avg_relevant', 0):>10.1f} docs")
    print(f"{'='*60}\n")
