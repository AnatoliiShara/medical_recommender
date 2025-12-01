import argparse, json
from collections import Counter, defaultdict

def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                raise SystemExit(f"[FAIL] {path}:{i} invalid JSON: {e}\nLINE={line[:200]}...")
            rows.append(obj)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True)
    ap.add_argument("--fail_on_missing_gold", action="store_true")
    args = ap.parse_args()

    rows = read_jsonl(args.path)

    ids = []
    qtxt = []
    missing_gold = []
    bad_gold = []
    for r in rows:
        rid = r.get("id", None)
        if rid is None:
            raise SystemExit("[FAIL] missing 'id' in some row")
        ids.append(rid)

        q = r.get("query", "")
        qtxt.append(q.strip().lower())

        if "gold_ids" not in r:
            missing_gold.append(rid)
        else:
            g = r["gold_ids"]
            if not isinstance(g, list) or any((not isinstance(x, int)) for x in g):
                bad_gold.append(rid)

    dup_ids = [k for k,v in Counter(ids).items() if v > 1]
    dup_q = [k for k,v in Counter(qtxt).items() if v > 1]

    print(f"[OK] file={args.path}")
    print(f"  rows_total        = {len(rows)}")
    print(f"  with_gold_ids     = {len(rows) - len(missing_gold)}")
    print(f"  missing_gold_ids  = {len(missing_gold)}")
    print(f"  bad_gold_format   = {len(bad_gold)}")
    print(f"  duplicate_ids     = {len(dup_ids)}")
    print(f"  duplicate_queries = {len(dup_q)}")

    if dup_ids:
        print("  dup_ids:", dup_ids[:50], ("..." if len(dup_ids) > 50 else ""))
    if missing_gold:
        print("  missing_gold_ids:", missing_gold[:50], ("..." if len(missing_gold) > 50 else ""))
    if bad_gold:
        print("  bad_gold_ids:", bad_gold[:50], ("..." if len(bad_gold) > 50 else ""))

    if dup_ids:
        raise SystemExit("[FAIL] duplicate 'id' values found — fix before scoring.")
    if args.fail_on_missing_gold and missing_gold:
        raise SystemExit("[FAIL] missing gold_ids but --fail_on_missing_gold was set.")
    if bad_gold:
        raise SystemExit("[FAIL] bad gold_ids format (must be list[int]).")

if __name__ == "__main__":
    main()
