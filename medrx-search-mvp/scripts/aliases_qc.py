# scripts/aliases_qc.py
import csv, re, sys, unicodedata, collections, pathlib

def norm(s):
    s = unicodedata.normalize("NFKC", s or "")
    for ch in ["®","™","©","’","‘","`","“","”"]:
        s = s.replace(ch, " ")
    return " ".join(s.split()).lower()

BAD_TOKENS = {
  "tablet","tablets","capsule","capsules","syrup","solution","coated",
  "prolonged","for","oral","inhalation","drops","mg","ml",
  "таблет","капсул","сироп","розчин","р-н","для","пероральн","інгаляц"
}

def is_probably_inn(t):
    # латиниця/дефіс/пробіли; без цифр і “форм”
    if any(ch.isdigit() for ch in t): return False
    low = t.lower()
    if any(tok in low for tok in BAD_TOKENS): return False
    return bool(re.fullmatch(r"[a-z][a-z\- ]{1,64}", low))

def main(path):
    p = pathlib.Path(path)
    bad_target, long_alias, with_paren = [], [], []
    seen=0
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        r = csv.DictReader(f)
        for row in r:
            seen += 1
            alias = norm(row.get("alias") or row.get("src") or "")
            target = norm(row.get("target") or row.get("canon") or row.get("inn") or "")
            if "(" in alias or ")" in alias:
                with_paren.append(alias)
            if len(alias) > 48:
                long_alias.append(alias)
            if not is_probably_inn(target):
                bad_target.append((alias, target))
    print(f"[QC] rows: {seen}")
    print(f"[QC] aliases with parentheses: {len(with_paren)}")
    print(f"[QC] aliases too long (>48): {len(long_alias)}")
    print(f"[QC] suspicious targets (not INN): {len(bad_target)}")
    for a,t in bad_target[:12]:
        print("  -", a, "->", t)

if __name__ == "__main__":
    main(sys.argv[1])
