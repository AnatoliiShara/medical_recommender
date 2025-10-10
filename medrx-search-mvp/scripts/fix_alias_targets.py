# -*- coding: utf-8 -*-
import csv, re, sys, unicodedata, pathlib

BAD_FORM_TOKENS = {
    "tablet","tablets","capsule","capsules","syrup","solution","suspension",
    "coated","prolonged","retard","mr","sr","xr","for","oral","inhalation",
    "drops","powder","infusion",
    "таблет","капсул","сироп","розчин","суспензі","покрит","пролонг","ретард",
    "для","пероральн","інгаляц","крапл","порош"
}

INN_RE = re.compile(r"^[a-z][a-z\-]{1,64}$")  # ASCII латиниця + дефіси

def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    for ch in ("®","™","©","’","‘","`","“","”"):
        s = s.replace(ch, " ")
    s = " ".join(s.split()).lower()
    return s

def short_alias(a: str) -> str:
    a = norm(a)
    # обрізати все після "("
    if "(" in a:
        a = a.split("(",1)[0].strip()
    # прибрати типові формо-слова
    a = re.sub(r"\b(sr|xr|mr|retard)\b", "", a)
    a = re.sub(r"\b(табл(етки)?|капсул(и)?|сироп|розчин|крапл(і|и)|для|пероральн\w*|інгаляц\w*|суспензі\w*|порош\w*)\b", "", a)
    a = re.sub(r"\s{2,}", " ", a).strip()
    return a

def looks_like_inn(t: str) -> bool:
    if not t: return False
    if any(tok in t for tok in BAD_FORM_TOKENS): return False
    return bool(INN_RE.fullmatch(t))

def load_brand2inn(p: pathlib.Path) -> dict:
    """
    Очікуємо у brand_candidates_FILLED.csv перші дві колонки:
    brand, inn  (назви заголовків можуть бути будь-які — беремо [0], [1]).
    """
    d = {}
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        r = csv.reader(f)
        hdr = next(r, None)
        for row in r:
            if not row: continue
            brand = norm(row[0])
            inn = norm(row[1] if len(row) > 1 else "")
            if brand and looks_like_inn(inn):
                d[brand] = inn
    return d

def main(in_aliases: str, brand_candidates_filled: str, out_csv: str):
    p_in = pathlib.Path(in_aliases)
    p_map = pathlib.Path(brand_candidates_filled)
    brand2inn = load_brand2inn(p_map)

    kept, fixed, dropped = 0, 0, 0
    out_rows = []
    with p_in.open("r", encoding="utf-8", errors="ignore") as f:
        r = csv.DictReader(f)
        for row in r:
            alias_raw = row.get("alias") or row.get("src") or ""
            target_raw = row.get("target") or row.get("canon") or row.get("inn") or ""
            typ = (row.get("type") or "brand").lower()

            a = short_alias(alias_raw)
            t = norm(target_raw)

            # 1) якщо target і так схожий на INN — ок
            if looks_like_inn(t):
                out_rows.append({"alias": a, "target": t, "type": "brand" if typ=="brand2inn" else typ})
                kept += 1
                continue

            # 2) спробувати мапу brand->inn
            cand = brand2inn.get(a) or brand2inn.get(norm(alias_raw)) or brand2inn.get(t)
            if cand and looks_like_inn(cand):
                out_rows.append({"alias": a, "target": cand, "type": "brand"})
                fixed += 1
                continue

            # 3) інакше — дроп
            dropped += 1

    # дедуп по (alias,target)
    dedup, seen = [], set()
    for r in out_rows:
        key = (r["alias"], r["target"])
        if r["alias"] and r["target"] and key not in seen:
            seen.add(key)
            dedup.append(r)

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["alias","target","type"])
        w.writeheader()
        w.writerows(dedup)

    print(f"[DONE] wrote {out_csv} rows={len(dedup)} | kept={kept} fixed={fixed} dropped={dropped}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python scripts/fix_alias_targets.py <brand_inn_aliases.csv> <brand_candidates_FILLED.csv> <out.csv>")
        sys.exit(2)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
