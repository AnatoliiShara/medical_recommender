import csv, re, sys
from pathlib import Path

def norm(s: str) -> str:
    s = (s or "").replace("\u00A0"," ").lower()
    s = re.sub(r"[’‘`']", "", s)
    s = re.sub(r"[-‐-‒–—]", " ", s)
    s = re.sub(r"[®™©“”«»]", " ", s)
    s = re.sub(r"\s+"," ", s).strip()
    return s

def read_aliases(p: Path):
    rows=[]
    with p.open("r", encoding="utf-8-sig") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append({k.strip().lower(): (v or "").strip() for k,v in r.items()})
    return rows

def main(base_csv: str, new_csv: str, out_csv: str, report_txt: str):
    base = Path(base_csv); new = Path(new_csv)
    base_rows = read_aliases(base) if base.exists() else []
    new_rows  = read_aliases(new)

    # map alias->(target,type) normalized
    mp = {}
    conflicts = []
    added = 0
    kept = 0

    for r in base_rows:
        a = norm(r.get("alias","")); t = norm(r.get("target","")); ty = (r.get("type","") or "alias").lower()
        if not a or not t: continue
        mp[a] = (t, ty)

    for r in new_rows:
        a = norm(r.get("alias","")); t = norm(r.get("target","")); ty = (r.get("type","") or "alias").lower()
        if not a or not t:  # пропускаємо незаповнені target
            continue
        if a in mp and mp[a][0] != t:
            conflicts.append((a, mp[a][0], t))
            # пріоритет на існуючий таргет, новий ігноруємо
            continue
        if a not in mp:
            mp[a] = (t, ty)
            added += 1
        else:
            kept += 1

    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        wr.writerow(["alias","target","type"])
        for a,(t,ty) in sorted(mp.items()):
            wr.writerow([a, t, ty])

    with Path(report_txt).open("w", encoding="utf-8") as r:
        r.write(f"Base rows: {len(base_rows)}\n")
        r.write(f"New rows (non-empty target): {sum(1 for r in new_rows if (r.get('target') or '').strip())}\n")
        r.write(f"Added: {added} | Kept: {kept} | Conflicts: {len(conflicts)}\n")
        if conflicts:
            r.write("Conflicts (alias | base_target | new_target):\n")
            for a, t0, t1 in conflicts[:50]:
                r.write(f"  {a} | {t0} | {t1}\n")

    print(f"[DONE] merged → {out} ; report: {report_txt}")

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python scripts/merge_aliases.py base.csv new.csv out.csv report.txt")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
