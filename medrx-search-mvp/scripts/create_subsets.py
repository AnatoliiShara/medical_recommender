# scripts/create_subsets.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create Stratified Query Subsets for MedRx Evaluation
"""

import json
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

PEDIATRIC_PATTERNS = ["дитина", "дитині", "дітей", "дитячий", "малюк", "немовля", "років", "місяців"]
PREGNANCY_PATTERNS = ["вагітн", "вагітності", "вагітних", "при вагітності", "триместр", "годуванн"]
NO_ANTIBIOTICS_PATTERNS = ["без антибіотик", "не антибіотик", "без аб", "окрім антибіотик", "крім антибіотик"]
COLD_FLU_PATTERNS = ["застуд", "грві", "гРВІ", "нежит", "кашл", "горл", "температур", "лихоманка", "common cold"]
DIARRHEA_PATTERNS = ["діарея", "діареї", "діарею", "пронос", "розлад"]
HYPERTENSION_PATTERNS = ["гіпертон", "тиск", "артеріальн"]

SIMPLE_INDICATORS = ["головний біль", "нежить", "кашель", "температур", "застуд", "алергі", "від болю"]
COMPLEX_INDICATORS = ["c. difficile", "клостридіальн", "комбінован", "з кров'ю", "ускладнен", "диференціальн", "синдром", "хронічн"]


def load_queries(filepath: str) -> List[Dict]:
    out = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def matches(text: str, patterns: List[str]) -> bool:
    t = text.lower()
    return any(p.lower() in t for p in patterns)


def classify_query(q: Dict) -> Dict[str, object]:
    t = q['query']
    cls = {
        'pediatric': matches(t, PEDIATRIC_PATTERNS),
        'pregnancy': matches(t, PREGNANCY_PATTERNS),
        'no_antibiotics': matches(t, NO_ANTIBIOTICS_PATTERNS),
        'common_cold': matches(t, COLD_FLU_PATTERNS),
        'diarrhea': matches(t, DIARRHEA_PATTERNS),
        'hypertension': matches(t, HYPERTENSION_PATTERNS),
    }
    if matches(t, SIMPLE_INDICATORS):
        complexity = 'simple'
    elif matches(t, COMPLEX_INDICATORS):
        complexity = 'complex'
    else:
        complexity = 'moderate'
    cls['complexity'] = complexity
    return cls


def create_subsets(queries: List[Dict], out_dir: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    classified = [{**q, **classify_query(q)} for q in queries]

    subsets = {
        'all': classified,
        'pediatric': [q for q in classified if q['pediatric']],
        'pregnancy': [q for q in classified if q['pregnancy']],
        'no_antibiotics': [q for q in classified if q['no_antibiotics']],
        'common_cold': [q for q in classified if q['common_cold']],
        'diarrhea': [q for q in classified if q['diarrhea']],
        'hypertension': [q for q in classified if q['hypertension']],
        'simple': [q for q in classified if q['complexity'] == 'simple'],
        'moderate': [q for q in classified if q['complexity'] == 'moderate'],
        'complex': [q for q in classified if q['complexity'] == 'complex'],
    }

    stats: Dict[str, int] = {}
    for name, lst in subsets.items():
        if not lst:
            print(f"⚠️  Subset '{name}' is empty, skipping")
            continue
        p = out_dir / f"{name}.jsonl"
        with open(p, 'w', encoding='utf-8') as f:
            for q in lst:
                f.write(json.dumps(q, ensure_ascii=False) + '\n')
        stats[name] = len(lst)
        print(f"✅ Created {name}.jsonl: {len(lst)} queries")

    with open(out_dir / "subset_stats.json", 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    total = len(classified)
    print("\n" + "="*60)
    print("  SUBSET SUMMARY")
    print("="*60)
    print(f"{'Subset':<20} {'Count':>10} {'%':>10}")
    print("-"*42)
    for name, cnt in sorted(stats.items(), key=lambda x: -x[1]):
        pct = (cnt / total * 100) if total else 0
        print(f"{name:<20} {cnt:>10} {pct:>9.1f}%")
    print("="*60 + "\n")
    return stats


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Create stratified query subsets for evaluation")
    ap.add_argument('--input', required=True)
    ap.add_argument('--output-dir', default='data/eval/subsets')
    args = ap.parse_args()

    print("\n🔍 Creating Query Subsets")
    print(f"Input:  {args.input}")
    print(f"Output: {args.output_dir}\n")

    qs = load_queries(args.input)
    print(f"Loaded {len(qs)} queries\n")
    create_subsets(qs, args.output_dir)
    print("✅ Subset creation complete!\n")


if __name__ == "__main__":
    main()
