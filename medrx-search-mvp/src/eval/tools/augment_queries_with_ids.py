# src/eval/tools/augment_queries_with_ids.py
# -*- coding: utf-8 -*-
import re, json, csv, unicodedata, argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import pandas as pd

# ---------- нормалізація назв (узгоджена з евалом) ----------
def norm(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = s.replace("®", " ").replace("™", " ").replace("©", " ")
    s = s.replace("’", "'").replace("`", "'").replace("ʼ", "'")
    # прибрати дужки лише для альтернативного варіанта:
    s = re.sub(r"[\"“”«»]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def norm_variants(name: str) -> List[str]:
    """Породжує варіанти для індексування (без дужок, без спецсимволів, без зайвих пробілів)."""
    base = norm(name)
    variants = {base}
    # без умісту в дужках
    no_paren = re.sub(r"\(.*?\)", " ", base)
    no_paren = re.sub(r"\s+", " ", no_paren).strip()
    variants.add(no_paren)
    # прибрати все, окрім літер/цифр/пробілів/дефісів
    clean = re.sub(r"[^0-9a-zа-яіїєґ\-\s]", " ", no_paren)
    clean = re.sub(r"\s+", " ", clean).strip()
    variants.add(clean)
    return [v for v in variants if v]

# ---------- побудова індексу з parquet + aliases ----------
def build_name_index(df: pd.DataFrame, alias_csv: Path = None) -> Tuple[Dict[str, Set[int]], Dict[str, str]]:
    """
    Повертає:
      name2ids: нормалізована назва/аліас -> множина doc_id
      canon_map: alias -> канонічна target (із csv), якщо був
    """
    name2ids: Dict[str, Set[int]] = defaultdict(set)
    canon_map: Dict[str, str] = {}

    has_inn = any(c.lower() in ["inn", "мнг", "мнн", "міжнародна непатентована назва"] for c in df.columns)
    inn_col = None
    for c in df.columns:
        if c.lower() in ["inn", "мнг", "мнн", "міжнародна непатентована назва"]:
            inn_col = c
            break

    # 1) індексуємо назву препарату + інн (якщо є)
    for i, row in df.iterrows():
        doc_id = int(row["__doc_id__"])
        brand = str(row.get("Назва препарату", "") or "").strip()
        if brand:
            for v in norm_variants(brand):
                name2ids[v].add(doc_id)
        if inn_col:
            inn = str(row.get(inn_col, "") or "").strip()
            if inn:
                for v in norm_variants(inn):
                    name2ids[v].add(doc_id)

    # 2) розширюємо alias-ами
    if alias_csv and alias_csv.exists():
        with alias_csv.open("r", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                alias = norm(r.get("alias", ""))
                target = norm(r.get("target", ""))
                if not alias or not target:
                    continue
                canon_map[alias] = target
                # alias → ті ж doc_id, що й target (точний або «містить»)
                if target in name2ids:
                    for did in name2ids[target]:
                        name2ids[alias].add(did)
                else:
                    # пошук «містить» для target
                    # (якщо кілька різних брендів містять target → додаємо всі — потім можна вручну відфільтрувати)
                    for key in list(name2ids.keys()):
                        if target and target in key:
                            name2ids[alias] |= name2ids[key]

    return name2ids, canon_map

def resolve_names_to_ids(names: List[str], name2ids: Dict[str, Set[int]]) -> Tuple[List[int], List[str]]:
    """Повертає (унікальні doc_id, unmatched_назви)."""
    out: Set[int] = set()
    unmatched: List[str] = []
    for n in names or []:
        N = norm(n)
        hits = set()
        if N in name2ids:
            hits |= name2ids[N]
        else:
            # fallback: спроба «містить», якщо N довше 4 символів
            if len(N) >= 4:
                for key, ids in name2ids.items():
                    if N in key:
                        hits |= ids
        if hits:
            out |= hits
        else:
            unmatched.append(n)
    return sorted(out), unmatched

# ---------- intent інференс ----------
_INTENT_RULES = [
    ("dosage",            r"(доз(ування|а|и)|скільки|по\s+\d|\d+\s*мг|доза|частота|max|максимальн)"),
    ("contraindication",  r"(протипоказ(ання|и)|не\s*можна|заборонено|вагітн|лактац|дітям|до\s*18)"),
    ("side_effects",      r"(побічн(і|і|ка)|реакці(ї|я)|небажан(і|і))"),
    ("interaction",       r"(взаємод(ія|ії)|разом\s+з|з\s+алкоголем|комбінувати)"),
    ("indication",        r"(від|лікуванн|симптоматичн|при\s+|стабілізує|допомагає)"),
]

def infer_intent(query: str) -> str:
    q = norm(query)
    for label, pat in _INTENT_RULES:
        if re.search(pat, q):
            return label
    return "indication"  # дефолт

# ---------- перс-кі @k ----------
def infer_k_if_needed(query: str, current_k: int = None) -> int:
    if current_k and current_k > 0:
        return current_k
    q = norm(query)
    # якщо «широкі» формулювання — даємо більш глибоке k
    broad_markers = ["препарат", "препарати", "ліки", "що найкраще", "від ", "при "]
    if any(b in q for b in broad_markers) or len(q.split()) <= 3:
        return 300
    return 100

def main():
    ap = argparse.ArgumentParser("Augment queries JSONL with gold_doc_ids, intent, per-query k")
    ap.add_argument("--parquet", required=True, help="Шлях до compendium_all.parquet")
    ap.add_argument("--queries_in", required=True, help="Вхідний queries_medrx_ua.jsonl")
    ap.add_argument("--queries_out", required=True, help="Вихідний .enhanced.jsonl")
    ap.add_argument("--aliases_csv", required=False, default=None, help="brand_inn_aliases.csv")
    ap.add_argument("--rows", type=int, default=0, help="0 = всі рядки parquet")
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    if args.rows > 0:
        df = df.head(args.rows).copy()
    df = df.reset_index(drop=True)
    df["__doc_id__"] = df.index.astype(int)

    alias_csv = Path(args.aliases_csv) if args.aliases_csv else None
    name2ids, canon_map = build_name_index(df, alias_csv)

    q_in = Path(args.queries_in)
    q_out = Path(args.queries_out)
    items = []
    with q_in.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            items.append(json.loads(line))

    total = len(items)
    matched_any = 0
    gold_total = 0
    gold_matched = 0
    unmatched_examples = []

    out_lines = []
    for it in items:
        gold = it.get("gold_drugs") or it.get("gold") or []
        acceptable = it.get("acceptable", [])
        must_avoid = it.get("must_avoid", [])

        gold_ids, gold_unmatched = resolve_names_to_ids(gold, name2ids)
        acc_ids, _ = resolve_names_to_ids(acceptable, name2ids)
        avoid_ids, _ = resolve_names_to_ids(must_avoid, name2ids)

        gold_total += len(gold)
        gold_matched += (len(gold) - len(gold_unmatched))
        if gold_ids:
            matched_any += 1
        else:
            if gold:
                unmatched_examples.append({"query": it.get("query"), "gold": gold, "unmatched": gold_unmatched})

        # intent + k
        intent = it.get("intent") or infer_intent(it.get("query", ""))
        k = infer_k_if_needed(it.get("query", ""), it.get("k"))

        it["gold_doc_ids"] = gold_ids
        if acc_ids:
            it["acceptable_doc_ids"] = acc_ids
        if avoid_ids:
            it["must_avoid_doc_ids"] = avoid_ids
        it["intent"] = intent
        it["k"] = k

        out_lines.append(json.dumps(it, ensure_ascii=False))

    with q_out.open("w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")

    print(f"Queries: {total}")
    print(f"Queries with at least one gold_doc_id matched: {matched_any}/{total} ({matched_any/total:.1%})")
    if gold_total > 0:
        print(f"Gold name coverage: {gold_matched}/{gold_total} ({gold_matched/gold_total:.1%})")

    if unmatched_examples[:5]:
        print("\nExamples of unmatched gold names (first 5):")
        for ex in unmatched_examples[:5]:
            print(f"- Q: {ex['query']}")
            print(f"  gold: {ex['gold']}")
            print(f"  unmatched: {ex['unmatched']}")
            print()

if __name__ == "__main__":
    main()
