# scripts/merge_brand_candidates.py
import csv, re, sys, unicodedata, pathlib

def norm(s):
    s = unicodedata.normalize("NFKC", s or "")
    for ch in ["®","™","©","’","‘","`","“","”"]:
        s = s.replace(ch, " ")
    s = " ".join(s.split()).lower()
    return s

def short_alias(a):
    a = norm(a)
    # обрізаємо по дужці, потім прибираємо формо-слова
    a = a.split("(",1)[0].strip()
    a = re.sub(r"\b(sr|xr|mr|retard)\b", "", a)
    a = re.sub(r"\b(табл(етки)?|капсул(и)?|сироп|розчин|крапл(і|и)|для|пероральн\w*|інгаляц\w*)\b", "", a)
    a = re.sub(r"\s{2,}"," ", a).strip()
    return a

def load_brand2inn(path):
    # очікуємо дві колонки: brand, inn (можна вільні заголовки, берем перші 2)
    d={}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        r = csv.reader(f)
        hdr = next(r, None)
        for row in r:
            if not row: continue
            b = norm(row[0]); inn = norm(row[1] if len(row)>1 else "")
            if b and inn:
                d[b]=inn
    return d

def main(aliases_csv, candidates_filled_csv, out_csv):
    brand2inn = load_brand2inn(candidates_filled_csv)
    inp = pathlib.Path(aliases_csv)
    out = []
    with inp.open("r", encoding="utf-8", errors="ignore") as f:
        r = csv.DictReader(f)
        for row in r:
            alias_raw = row.get("alias") or row.get("src") or ""
            target_raw = row.get("target") or row.get("canon") or row.get("inn") or ""
            typ = (row.get("type") or "alias").lower()
            a_short = short_alias(alias_raw)
            # вирівнюємо таргет по еталонній мапі, якщо є
            inn = brand2inn.get(a_short) or brand2inn.get(norm(alias_raw)) or norm(target_raw)
            # якщо все ще виглядає не як INN — як крайній випадок залишаємо старе
            out.append({"alias": a_short, "target": inn, "type": "brand"})
    # унікалізуємо по (alias,target)
    seen=set(); dedup=[]
    for r in out:
        key=(r["alias"], r["target"])
        if r["alias"] and r["target"] and key not in seen:
            seen.add(key); dedup.append(r)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["alias","target","type"])
        w.writeheader()
        w.writerows(dedup)
    print(f"[DONE] wrote {out_csv} rows={len(dedup)}")

if __name__ == "__main__":
    aliases_csv = sys.argv[1]
    candidates_filled_csv = sys.argv[2]
    out_csv = sys.argv[3]
    main(aliases_csv, candidates_filled_csv, out_csv)
