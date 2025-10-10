# scripts/audit_alias_coverage.py
import json, re, sys
from pathlib import Path
from rewrite.rewrite import load_config, Rewriter, _ua_norm

def contains(qn: str, t: str) -> bool:
    if not t: return False
    # слово з межами
    if re.search(r"[\w\u0400-\u04FF]", t):
        return re.search(rf"\b{re.escape(t)}\b", qn) is not None
    return t in qn

def main(cfg_path: str, *jsonl_paths: str):
    cfg = load_config(cfg_path)
    rw = Rewriter(cfg)
    terms = list(rw.aliases.keys())
    print(f"[INFO] aliases: {len(terms)}")

    total_q = 0
    hit_q = 0
    misses = []

    for p in jsonl_paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                o = json.loads(line)
                q = o.get("query","")
                qn = _ua_norm(q)
                total_q += 1
                any_hit = False
                for t in terms:
                    if contains(qn, t):
                        any_hit = True
                        break
                if not any_hit:
                    misses.append(q[:120])
                else:
                    hit_q += 1

    print(f"[COVERAGE] queries with ≥1 alias hit: {hit_q}/{total_q} = {hit_q/total_q:.2%}")
    print("Examples WITHOUT alias hit (up to 10):")
    for s in misses[:10]:
        print("  -", s)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python scripts/audit_alias_coverage.py configs/rewrite.yaml file1.jsonl [file2.jsonl ...]")
        sys.exit(1)
    main(sys.argv[1], *sys.argv[2:])
