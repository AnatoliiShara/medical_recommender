#!/usr/bin/env python3
import csv, sys, re
from collections import OrderedDict

def _norm(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("\u00A0", " ")  # NBSP -> space
    s = re.sub(r"\s+", " ", s)
    return s

def _lower_headers(reader):
    # повертає headers у нижньому регістрі, очищені від пробілів
    return [ (h or "").strip().lower() for h in (reader.fieldnames or []) ]

def _pick_key(headers, candidates):
    for k in candidates:
        if k in headers:
            return k
    return None

def _load_csv_generic(path):
    """Зчитати CSV -> (rows, headers_lower, canon_key_detected)"""
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        headers = _lower_headers(reader)
        if not headers:
            raise ValueError(f"{path}: empty or bad CSV")

        alias_key = _pick_key(headers, ["alias"])
        type_key  = _pick_key(headers, ["type"])
        canon_key = _pick_key(headers, ["canon","canon_inn","target"])

        if not alias_key or not canon_key:
            raise ValueError(
                f"{path}: expected headers alias + one of [canon|canon_inn|target] (got: {headers})"
            )

        for r in reader:
            alias = _norm(r.get(alias_key, "")).lower()
            canon = _norm(r.get(canon_key, "")).lower()
            typ   = _norm(r.get(type_key, "")).lower() if type_key else "brand"
            if alias and canon:
                rows.append({"alias": alias, "canon": canon, "type": typ})

    return rows, headers, canon_key

def load_base_csv(path):
    # Повертаємо rows і "режим" виходу: 'target' якщо база з target, інакше 'canon'
    rows, headers, canon_key = _load_csv_generic(path)
    out_mode = "target" if "target" in headers else "canon"
    return rows, out_mode

def load_patches_csv(path):
    # Патчі просто нормалізуємо до (alias, canon, type)
    rows, _, _ = _load_csv_generic(path)
    return rows

def write_csv(path, rows, out_mode="target"):
    # Вихід синхронізуємо з базою:
    if out_mode == "target":
        fieldnames = ["alias","target","type"]
    else:
        fieldnames = ["alias","canon","type"]

    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            row = {"alias": r["alias"], "type": r["type"]}
            row["target" if out_mode=="target" else "canon"] = r["canon"]
            w.writerow(row)

def main():
    if len(sys.argv) != 4:
        print("Usage: apply_alias_patches.py BASE_CSV PATCHES_CSV OUT_CSV", file=sys.stderr)
        sys.exit(2)
    base_csv, patch_csv, out_csv = sys.argv[1:4]

    base_rows, out_mode = load_base_csv(base_csv)
    patches = load_patches_csv(patch_csv)

    idx = OrderedDict()
    for r in base_rows:
        key = (r["alias"], r["type"])
        idx[key] = r

    added, replaced = 0, 0
    for p in patches:
        key = (p["alias"], p["type"])
        if key in idx:
            if idx[key]["canon"] != p["canon"]:
                idx[key] = p
                replaced += 1
        else:
            idx[key] = p
            added += 1

    out = list(idx.values())
    out.sort(key=lambda r: (r["alias"], r["type"], r["canon"]))

    write_csv(out_csv, out, out_mode=out_mode)
    print(f"[DONE] wrote {out_csv} rows={len(out)} | added={added} replaced={replaced} skipped=0 (mode={out_mode})")

if __name__ == "__main__":
    main()
