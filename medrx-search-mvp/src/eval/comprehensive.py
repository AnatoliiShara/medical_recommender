# scripts/eval/comprehensive.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys, os, json, yaml, argparse, math, time, random
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from tqdm import tqdm
import pandas as pd

# add repo root /src
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'src'))

from eval.metrics_v2 import MetricsAggregator, print_metrics_summary
from qtrace.stage_logger import StageLogger

class EvaluationRunner:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.run_dir = self._setup_run_dir()
        print(f"\n🔍 MedRx Comprehensive Evaluation\n{'='*60}\nRun directory: {self.run_dir}\n{'='*60}\n")

    def _load_config(self, p: str) -> Dict:
        with open(p, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _setup_run_dir(self) -> Path:
        runs_dir = Path(self.config['output']['runs_dir'])
        run_name = self.config['output'].get('run_name') or f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.config['output']['run_name'] = run_name
        rd = runs_dir / run_name
        (rd / 'stage_logs').mkdir(parents=True, exist_ok=True)
        (rd / 'query_results').mkdir(exist_ok=True)
        return rd

    def load_queries(self, subset: str = 'all') -> List[Dict]:
        if subset == 'all':
            qp = self.config['data']['queries']
        else:
            qp = Path(self.config['data']['subsets_dir']) / f"{subset}.jsonl"
        if not Path(qp).exists():
            print(f"⚠️  Missing subset file: {qp}")
            return []
        out = []
        with open(qp, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def initialize_search_engine(self):
        # TODO: replace with real engine init
        self.search_engine = None

    def run_search(self, query: Dict, logger: Optional[StageLogger]) -> List[Dict]:
        # TODO: replace with your pipeline; this is MOCK for wiring
        bm25 = [{"id": random.randint(1, 100000), "bm25": random.random()} for _ in range(50)]
        dense = [{"id": random.randint(1, 100000), "dense": random.random()} for _ in range(50)]
        if logger:
            logger.log_stage('bm25', bm25, {'k': 50})
            logger.log_stage('dense', dense, {'k': 50})
        # naive RRF-like
        pool = {}
        for i, c in enumerate(bm25, start=1):
            pool.setdefault(c['id'], {}).update(c)
            pool[c['id']]['rrf'] = pool[c['id']].get('rrf', 0.0) + 1.0/(60.0+i)
        for i, c in enumerate(dense, start=1):
            pool.setdefault(c['id'], {}).update(c)
            pool[c['id']]['rrf'] = pool[c['id']].get('rrf', 0.0) + 1.0/(60.0+i)
        fused = [{"id": k, **v} for k, v in pool.items()]
        fused.sort(key=lambda x: x.get('rrf', 0.0), reverse=True)
        fused = fused[:20]
        if logger:
            logger.log_stage('rrf', fused, {'alpha': self.config['search']['rrf_alpha']})
        # fake CE
        for c in fused:
            c['ce'] = random.random()
            c['final'] = 0.7 * c.get('rrf', 0) + 0.3 * c['ce']
        fused.sort(key=lambda x: x['final'], reverse=True)
        if logger:
            logger.log_stage('crossencoder', fused[:10], {'top_k': 10})
        return fused[:self.config['search']['final_top_k']]

    def _gold_ids(self, q: Dict) -> set:
        gs = q.get('gold_ids') or q.get('gold') or []
        if isinstance(gs, list):
            return set(gs)
        if isinstance(gs, str):
            return set(int(x.strip()) for x in gs.split(',') if x.strip())
        return set()

    def evaluate_subset(self, name: str, queries: List[Dict]) -> Dict:
        if not queries:
            print(f"⚠️  No queries in subset '{name}'")
            return {}
        print(f"\n📊 Evaluating subset: {name} ({len(queries)} queries)")
        agg = MetricsAggregator()

        for q in tqdm(queries, desc=f"Processing {name}"):
            qid = q.get('qid') or q.get('id') or q.get('query_id') or 'NA'
            qtext = q['query']
            log_dir = self.run_dir / 'stage_logs' / name
            logger = StageLogger(str(qid), qtext, log_dir) if self.config['logging']['log_stages'] else None

            results = self.run_search(q, logger)
            if logger:
                logger.set_final(results)
                logger.save()

            retrieved_ids = [r['id'] for r in results]
            gold = self._gold_ids(q)
            meta = {
                'category': q.get('complexity'),
                'intent': q.get('intent')
            }
            if gold:
                agg.add_query_result(query_id=int(str(qid).strip().replace('Q','').replace('q','') or 0),
                                     retrieved=retrieved_ids,
                                     relevant=set(gold),
                                     metadata=meta)

            if self.config['output']['save_query_results']:
                out = {
                    'query_id': qid,
                    'query': qtext,
                    'results': results,
                    'gold_ids': list(gold)
                }
                p = self.run_dir / 'query_results' / name / f"query_{str(qid)}.json"
                p.parent.mkdir(parents=True, exist_ok=True)
                with open(p, 'w', encoding='utf-8') as f:
                    json.dump(out, f, ensure_ascii=False, indent=2)

        aggregated = agg.compute_aggregates()
        self._save_subset_metrics(name, aggregated, agg.get_query_level_results())
        if aggregated:
            print_metrics_summary(aggregated, f"Subset: {name}")
        return aggregated

    def _save_subset_metrics(self, name: str, aggregated: Dict, per_q: List[Dict]):
        with open(self.run_dir / f"metrics_{name}.json", 'w', encoding='utf-8') as f:
            json.dump(aggregated, f, indent=2, ensure_ascii=False)
        if per_q:
            pd.DataFrame(per_q).to_csv(self.run_dir / f"query_metrics_{name}.csv", index=False)

    def run_full(self):
        self.initialize_search_engine()
        results = {}
        for subset in self.config['evaluation']['subsets']:
            qs = self.load_queries(subset)
            if not qs:
                continue
            results[subset] = self.evaluate_subset(subset, qs)
        with open(self.run_dir / "evaluation_summary.json", 'w', encoding='utf-8') as f:
            json.dump({'timestamp': datetime.now().isoformat(),
                       'config': self.config,
                       'results_by_subset': results}, f, indent=2, ensure_ascii=False)
        # simple markdown
        with open(self.run_dir / "report.md", 'w', encoding='utf-8') as f:
            f.write("# MedRx Evaluation Report\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Run:** {self.run_dir.name}\n\n")
            f.write("| Subset | Queries | nDCG@10 | Recall@50 | MRR |\n|---|---:|---:|---:|---:|\n")
            for s, m in results.items():
                f.write(f"| {s} | {m.get('total_queries',0)} | {m.get('ndcg@10_mean',0):.4f} | {m.get('r@50_mean',0):.4f} | {m.get('mrr',0):.4f} |\n")
        print(f"\n✅ Evaluation complete!\n📁 {self.run_dir}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/eval_config.yaml')
    ap.add_argument('--subsets', nargs='+', default=None)
    ap.add_argument('--run-name', default=None)
    args = ap.parse_args()

    runner = EvaluationRunner(args.config)
    if args.subsets:
        runner.config['evaluation']['subsets'] = args.subsets
    if args.run_name:
        runner.config['output']['run_name'] = args.run_name
        runner.run_dir = runner._setup_run_dir()
    runner.run_full()


if __name__ == "__main__":
    main()
