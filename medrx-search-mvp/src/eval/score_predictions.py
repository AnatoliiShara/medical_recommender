#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, argparse
from src.eval.metrics_v2 import MetricsAggregator, print_metrics_summary

def load_jsonl(p):
    with open(p,encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if line: yield json.loads(line)

if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--queries", required=True)  # з gold_ids
    ap.add_argument("--subset_jsonl", default=None)
    args=ap.parse_args()

    # gold за id (пріоритет) і, на всяк випадок, за текстом
    gold_by_id, gold_by_text = {}, {}
    for q in load_jsonl(args.queries):
        if "gold_ids" in q:
            if "id" in q: gold_by_id[q["id"]] = set(q["gold_ids"])
            gold_by_text[q["query"]] = set(q["gold_ids"])

    allowed_texts=None
    if args.subset_jsonl:
        allowed_texts=set(q["query"] for q in load_jsonl(args.subset_jsonl))

    agg=MetricsAggregator()
    for p in load_jsonl(args.predictions):
        if allowed_texts and p["query"] not in allowed_texts:
            continue
        rel = gold_by_id.get(p.get("id")) or gold_by_text.get(p["query"], set())
        retrieved=[t["doc_id"] for t in p.get("top",[])]
        agg.add_query_result(query_id=p.get("id",0), retrieved=retrieved, relevant=rel)

    overall=agg.compute_aggregates()
    print_metrics_summary(overall, "Overall (silver gold)")
