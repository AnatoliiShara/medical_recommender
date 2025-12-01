#!/usr/bin/env python3
import argparse, json, re, random
from pathlib import Path

import pandas as pd

# -------------------------
# Normalization helpers
# -------------------------
_UA_RE = re.compile(r"[^0-9a-zа-яіїєґ\+\-\'’\s]+", re.IGNORECASE)

def norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = s.replace("’", "'")
    s = _UA_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def looks_non_med(nm_norm: str) -> bool:
    # Heuristic filter to drop cosmetics/devices that poison gold.
    # Keep it conservative: we filter *obvious* non-med stuff.
    bad = [
        "шампун", "маска для волос", "тонік", "лосьйон", "крем", "міцеляр", "макіяж",
        "репелент", "комар", "спрей від комар", "купероз", "дезодорант",
        "колготи", "пояс", "бандаж", "судно підкладне", "паста зубн", "зубна паста",
        "гель для душ", "пінка для вмиван", "космет",
    ]
    return any(b in nm_norm for b in bad)

def match_doc_ids(df, patterns, max_per_pattern=50):
    """
    patterns: list[str] substrings (already normalized fragments)
    returns: dict[pat] -> list[(doc_id, drug_name)]
    """
    out = {}
    names_norm = df["_name_norm"].tolist()
    doc_ids = df["doc_id"].tolist()
    drug_names = df["drug_name"].tolist()

    for pat in patterns:
        p = norm(pat)
        if not p:
            out[pat] = []
            continue

        hits = []
        for did, dn, dnn in zip(doc_ids, drug_names, names_norm):
            if p in dnn and not looks_non_med(dnn):
                hits.append((int(did), str(dn)))
                if len(hits) >= max_per_pattern:
                    break
        out[pat] = hits
    return out

# -------------------------
# Default patterns (diarrhea)
# -------------------------
DEFAULT_PATTERNS = {
    # very “core” antidiarrheals / adsorption / microflora / ORS
    "core": [
        "лоперам", "імодіум", "imodium", "loperam",
        "ніфурокс", "nifurox", "ентерофурил", "enterofuril", "ерцефурил", "ercefur",
        "смекта", "smecta", "діосмектит", "diosmect",
        "ентеросгель", "enterosgel",
        "атоксил", "atoxil",
        "полісорб", "polysorb",
        "ентерол", "enterol", "saccharomyces",
        "хілак", "hilak",
        "лінекс", "linex",
        "регідрон", "regidron", "rehydrat",
        "сорбекс", "sorbex",
        "активоване вугіл", "activated charcoal",
        # UA brands sometimes used:
        "стопдіар", "stopdiar",
    ],
    # safer leaning subset for child queries (no loperamide by default)
    "child": [
        "смекта", "smecta", "діосмектит", "diosmect",
        "ентеросгель", "enterosgel",
        "атоксил", "atoxil",
        "полісорб", "polysorb",
        "ентерол", "enterol", "saccharomyces",
        "лінекс", "linex",
        "регідрон", "regidron", "rehydrat",
    ],
    # traveler-ish: often nifuroxazide + ORS + probiotics/sorbents
    "travel": [
        "ніфурокс", "nifurox", "ентерофурил", "enterofuril", "ерцефурил", "ercefur", "стопдіар", "stopdiar",
        "регідрон", "regidron", "rehydrat",
        "ентерол", "enterol",
        "ентеросгель", "enterosgel", "полісорб", "polysorb", "атоксил", "atoxil",
        "смекта", "smecta",
    ],
}

def choose_mode(q_norm: str) -> str:
    if any(x in q_norm for x in ["дитин", "дитини", "дитяч", "малюк", "немовл", "підлітк"]):
        return "child"
    if any(x in q_norm for x in ["мандр", "подорож", "travel", "турист", "відпустк"]):
        return "travel"
    return "core"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_queries", type=Path, required=True, help="jsonl з полями id/query (gold може бути зіпсований)")
    ap.add_argument("--docs_meta", type=Path, required=True, help="docs_meta.parquet з doc_id, drug_name")
    ap.add_argument("--out_queries", type=Path, required=True, help="куди писати fixed jsonl")
    ap.add_argument("--condition", type=str, default="diarrhea", help="фільтр по matched_conditions (якщо поле є)")
    ap.add_argument("--max_gold", type=int, default=25, help="макс gold_ids на запит (після union)")
    ap.add_argument("--min_gold", type=int, default=3, help="мінімум gold_ids, інакше рядок відкинемо")
    ap.add_argument("--limit", type=int, default=50, help="скільки запитів записати максимум")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--max_per_pattern", type=int, default=50, help="скільки doc_id брати максимум з 1 патерну")
    ap.add_argument("--report", type=Path, default=None, help="json-репорт матчів патернів (опційно)")
    args = ap.parse_args()

    if not args.docs_meta.exists():
        raise FileNotFoundError(args.docs_meta)
    if not args.in_queries.exists():
        raise FileNotFoundError(args.in_queries)

    df = pd.read_parquet(args.docs_meta, columns=["doc_id", "drug_name"])
    df["doc_id"] = df["doc_id"].astype(int)
    df["drug_name"] = df["drug_name"].astype(str)
    df["_name_norm"] = df["drug_name"].map(norm)

    # Precompute pattern matches once
    pat_hits = {}
    for mode, pats in DEFAULT_PATTERNS.items():
        pat_hits[mode] = match_doc_ids(df, pats, max_per_pattern=args.max_per_pattern)

    # Flatten each mode into doc_id set
    mode_gold = {}
    for mode, mp in pat_hits.items():
        s = set()
        for pat, hits in mp.items():
            for did, _ in hits:
                s.add(int(did))
        mode_gold[mode] = sorted(s)

    # Read queries, filter by condition if present
    random.seed(args.seed)
    rows = []
    with args.in_queries.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            j = json.loads(line)

            # condition filter (only if field exists)
            conds = j.get("matched_conditions")
            if isinstance(conds, list) and args.condition:
                if args.condition not in conds:
                    continue

            q = str(j.get("query", ""))
            qn = norm(q)
            mode = choose_mode(qn)

            gold_ids = mode_gold.get(mode, [])
            if len(gold_ids) < args.min_gold:
                continue
            if args.max_gold and len(gold_ids) > args.max_gold:
                gold_ids = gold_ids[:args.max_gold]

            j["gold_ids"] = [int(x) for x in gold_ids]
            j["gold_mode"] = mode
            j["gold_source"] = "patterns@docs_meta"
            rows.append(j)

    # sample to limit
    if len(rows) > args.limit:
        rows = random.sample(rows, args.limit)

    args.out_queries.parent.mkdir(parents=True, exist_ok=True)
    with args.out_queries.open("w", encoding="utf-8") as out:
        for j in sorted(rows, key=lambda x: int(x.get("id", 0))):
            out.write(json.dumps(j, ensure_ascii=False) + "\n")

    print(f"[OK] wrote {args.out_queries} rows={len(rows)}")

    if args.report:
        rep = {
            "docs_meta": str(args.docs_meta),
            "in_queries": str(args.in_queries),
            "out_queries": str(args.out_queries),
            "condition": args.condition,
            "modes": {},
        }
        for mode, mp in pat_hits.items():
            rep["modes"][mode] = {
                "unique_doc_ids": len(mode_gold[mode]),
                "patterns": {
                    pat: {
                        "hits": len(hits),
                        "examples": [h[1] for h in hits[:5]],
                    }
                    for pat, hits in mp.items()
                }
            }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] wrote report {args.report}")

if __name__ == "__main__":
    main()
