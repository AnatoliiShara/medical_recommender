# scripts/rewrite_preview.py
# -*- coding: utf-8 -*-
"""
Прев'ю перепису запитів (rewrite):
- читає кілька JSONL з ключем "query"
- застосовує Rewriter за configs/rewrite.yaml
- друкує перші N результатів (або всі, якщо --n не задано)
"""

from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

from rewrite.rewrite import load_config, Rewriter

def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception as e:
                print(f"[WARN] {path.name}:{ln} -> bad JSON: {e}")
                continue
            q = (obj.get("query") or "").strip()
            if not q:
                continue
            yield {"query": q, "raw": obj}

def main():
    ap = argparse.ArgumentParser("rewrite_preview")
    ap.add_argument("--cfg", required=True, help="configs/rewrite.yaml")
    ap.add_argument("--in", dest="inputs", nargs="+", required=True, help="JSONL files with 'query'")
    ap.add_argument("--n", type=int, default=20, help="How many examples to print (per all files combined)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_config(args.cfg)
    rw = Rewriter(cfg)

    # збираємо всі запити
    items = []
    for p in args.inputs:
        path = Path(p)
        if not path.exists():
            print(f"[ERR] file not found: {p}")
            continue
        items.extend(list(iter_jsonl(path)))

    total = len(items)
    print(f"[INFO] loaded queries: {total}")

    if total == 0:
        print("[WARN] no queries with key 'query' found in provided files.")
        return

    random.seed(args.seed)
    if args.n and args.n > 0 and args.n < total:
        sample = random.sample(items, args.n)
    else:
        sample = items

    for i, obj in enumerate(sample, start=1):
        q = obj["query"]
        res = rw.rewrite(q)
        print(f"[{i}] RAW:        {q}")
        print(f"    REWRITTEN:  {res.get('query_rewritten', q)}")
        print(f"    UNITS:      {res.get('units_applied', [])}")
        print(f"    ALIASES+:   {res.get('alias_hits', [])}")
        print(f"    SYMPTOMS+:  {res.get('symptom_exp', [])}")
        print()

if __name__ == "__main__":
    main()
