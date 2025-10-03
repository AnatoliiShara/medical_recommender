# -*- coding: utf-8 -*-
import json, re, unicodedata, random, argparse
from pathlib import Path
import pandas as pd

SECTIONS_DEFAULT = [
    "Показання",
    "Спосіб застосування та дози",
    "Протипоказання",
]

INTENT_BY_SECTION = {
    "Показання": "indication",
    "Спосіб застосування та дози": "dosage",
    "Протипоказання": "contraindication",
}

# прості синонімічні заміни / варіанти формулювань українською
PARAPHRASE_VARIANTS = [
    ("порадьте", ["порекомендуйте", "які є", "які препарати допоможуть", "що призначають при"]),
    ("лікування", ["терапія", "лікувати", "чим лікують", "в чому полягає лікування"]),
    ("симптоматичне", ["для зняття симптомів", "полегшення симптомів"]),
    ("що приймати при", ["що допомагає при", "які ліки при", "чим лікувати"]),
    ("дозування", ["як приймати", "яка доза", "схема прийому"]),
]

def norm(s: str) -> str:
    if s is None: return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.lower()
    s = re.sub(r"[®™©“”«»'’`]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def sent_split(text: str):
    # грубий розріз на речення
    parts = re.split(r"(?<=[\.\!\?])\s+", text)
    return [p.strip() for p in parts if len(p.strip()) >= 5]

def make_indication_query(sent: str):
    base = [
        f"що призначають при {sent}",
        f"лікування: {sent}",
        f"які препарати при {sent}",
        f"порадьте засоби від {sent}",
    ]
    return base

def make_dosage_query(drug: str, sent: str):
    base = [
        f"{drug}: дозування — {sent}",
        f"як приймати {drug}: {sent}",
        f"схема прийому {drug}: {sent}",
    ]
    return base

def make_contra_query(sent: str):
    base = [
        f"протипоказання: {sent}",
        f"коли не можна застосовувати: {sent}",
        f"у кого протипоказано: {sent}",
    ]
    return base

def light_paraphrase(q: str, rng: random.Random):
    # застосовуємо 0..2 випадкові заміни
    out = q
    for _ in range(rng.randint(0,2)):
        a, bs = rng.choice(PARAPHRASE_VARIANTS)
        if a in out:
            out = out.replace(a, rng.choice(bs))
    return out

def main():
    ap = argparse.ArgumentParser("Auto-generate Ukrainian eval queries (guarantee target_n via backfill)")
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--rows", type=int, default=0, help="0=all")
    ap.add_argument("--sections", default=",".join(SECTIONS_DEFAULT))
    ap.add_argument("--target_n", type=int, default=240)
    ap.add_argument("--min_len", type=int, default=24, help="min chars for a sentence to use")
    ap.add_argument("--max_per_drug", type=int, default=3, help="max queries per drug before paraphrase backfill")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    sections = [s.strip() for s in args.sections.split(",") if s.strip()]

    df = pd.read_parquet(args.parquet)
    if args.rows > 0:
        df = df.head(args.rows).copy()
    df = df.reset_index(drop=True)
    df["__doc_id__"] = df.index

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    queries = []
    seen = set()  # дедуп по нормалізованому тексту

    for _, row in df.iterrows():
        if len(queries) >= args.target_n:
            break
        drug = str(row.get("Назва препарату", "")).strip()
        doc_id = int(row["__doc_id__"])
        made_for_drug = 0

        for sec in sections:
            if len(queries) >= args.target_n or made_for_drug >= args.max_per_drug:
                break
            raw = str(row.get(sec, "") or "").strip()
            if not raw:
                continue
            sents = [s for s in sent_split(raw) if len(s) >= args.min_len]
            if not sents:
                continue

            # візьмемо 1–2 інформативні речення
            for sent in sents[:2]:
                q_candidates = []
                if sec == "Показання":
                    q_candidates += make_indication_query(sent)
                elif sec == "Спосіб застосування та дози":
                    q_candidates += make_dosage_query(drug, sent)
                elif sec == "Протипоказання":
                    q_candidates += make_contra_query(sent)

                for q in q_candidates:
                    nq = norm(q)
                    if nq in seen:
                        continue
                    seen.add(nq)

                    item = {
                        "query": q,
                        "intent": INTENT_BY_SECTION.get(sec, "generic"),
                        "k": 10,
                        "notes": f"auto from {sec}",
                        # gold — як мінімум цей doc_id; потім руками/алiасами збагачуємо
                        "gold_doc_ids": [doc_id],
                        "gold_drugs": [drug],
                    }
                    queries.append(item)
                    made_for_drug += 1
                    if len(queries) >= args.target_n or made_for_drug >= args.max_per_drug:
                        break

    # бекфіл: парафразимо існуючі, щоб добрати до target_n
    i = 0
    while len(queries) < args.target_n and i < len(queries)*3+500:
        base = rng.choice(queries)
        q2 = light_paraphrase(base["query"], rng)
        nq2 = norm(q2)
        if nq2 not in seen and len(q2) >= args.min_len:
            seen.add(nq2)
            clone = dict(base)
            clone["query"] = q2
            clone["notes"] = base.get("notes","") + " | paraphrase"
            queries.append(clone)
        i += 1

    with out_path.open("w", encoding="utf-8") as f:
        for it in queries[:args.target_n]:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"Generated: {len(queries[:args.target_n])} queries → {out_path}")

if __name__ == "__main__":
    main()
