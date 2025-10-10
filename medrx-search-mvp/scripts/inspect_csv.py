# scripts/inspect_csv.py
import csv, sys, pathlib

p = pathlib.Path(sys.argv[1])
with p.open("r", encoding="utf-8-sig") as f:
    sample = f.read(4096)
    try:
        dialect = csv.Sniffer().sniff(sample)
    except Exception:
        dialect = csv.excel
    f.seek(0)
    rdr = csv.DictReader(f, dialect=dialect)
    headers = rdr.fieldnames or []
    rows = list(rdr)[:5]

print("FILE:", p.resolve())
print("HEADERS:", headers)
print("N_SAMPLES:", len(rows))
for i, r in enumerate(rows, 1):
    print(f"ROW{i}:", {k:(r[k] if k in r else None) for k in headers})
