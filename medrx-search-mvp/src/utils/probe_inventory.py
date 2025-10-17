# src/utils/probe_inventory.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import re
from pathlib import Path
import argparse
import pandas as pd

# --- ключові колонки корпусу
UKR_TEXT_COLS = [
    "Назва препарату","Показання","Фармакологічні властивості","Фармакотерапевтична група",
    "Лікарська форма","Спосіб застосування та дози","Склад",
]

# --- регекси для нашої інвентаризації
ANTIFUNGAL_TERMS = re.compile(
    r"(клотримазол|clotrimazole|ністатин|nystatin|міконазол|miconazole|"
    r"кетоконазол|ketoconazole|натаміцин|natamycin|леворин|levorin|"
    r"пімафуцин|pimafucin|флуконазол|fluconazole)", re.IGNORECASE
)
ORAL_ALLOWED_FORM = re.compile(
    r"(льодяник|розсмокт|таблет(ка|ки).*розсмокт|спрей.*(рот|горл)|аерозол.*(рот|горл)|ополіск|полоск)",
    re.IGNORECASE
)
DERM_BLOCK_FORM = re.compile(r"(крем|мазь|нашкірн|лак для нігт|шампунь|розчин нашкірн)", re.IGNORECASE)

THROAT_ANTISEPTIC = re.compile(
    r"(бензидамін|benzydamine|хлоргексидин|chlorhexidine|мірамістин|miramistin|"
    r"амілметакрезол|amylmetacresol|дихлорбензилов(ий|ого)\s*спирт|dichlorobenzyl\s*alcohol)",
    re.IGNORECASE
)

PPI_H2_ANTACID = re.compile(
    r"(омепразол|пантопразол|езомепразол|лансопразол|рабепразол|"
    r"фамотидин|ранитидин|антацид|альгінат|сукральфат)",
    re.IGNORECASE
)

def get_field(row, key):
    v = row.get(key)
    return v if isinstance(v, str) else ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("data/raw/compendium_all.parquet"))
    ap.add_argument("--max_show", type=int, default=30)
    ap.add_argument("--save_csv", type=Path, default=None, help="опційно: шлях для збереження знайдених збігів у CSV")
    args = ap.parse_args()

    df = pd.read_parquet(args.dataset)
    print(f"[INFO] rows={len(df)}")

    cols = [c for c in UKR_TEXT_COLS if c in df.columns]

    def has_antifungal_blob(row):
        blob = " ".join(get_field(row,c) for c in cols)
        return bool(ANTIFUNGAL_TERMS.search(blob))

    def oral_form(row):
        form = get_field(row,"Лікарська форма") + " " + get_field(row,"Спосіб застосування та дози")
        return ORAL_ALLOWED_FORM.search(form) and not DERM_BLOCK_FORM.search(form)

    def is_throat_antiseptic(row):
        blob = " ".join(get_field(row,c) for c in cols)
        return bool(THROAT_ANTISEPTIC.search(blob))

    def is_ppi_h2_antacid(row):
        blob = " ".join(get_field(row,c) for c in cols)
        return bool(PPI_H2_ANTACID.search(blob))

    # 1) антимікотик + оральна/горлова форма
    mask_antifungal = df.apply(has_antifungal_blob, axis=1)
    mask_oral_form  = df.apply(oral_form, axis=1)
    oral_antifungal = df[mask_antifungal & mask_oral_form]

    # 2) конкуренти: лор-антисептики в оральних формах
    throat_ants = df[df.apply(is_throat_antiseptic, axis=1) & mask_oral_form]

    # 3) контроль для гастриту/кислотності
    acid_ok = df[df.apply(is_ppi_h2_antacid, axis=1)]

    print("\n=== ОРОФАРИНГЕАЛЬНИЙ КАНДИДОЗ: антимікотик + оральна форма ===")
    print(f"count={len(oral_antifungal)}")
    for i, (_, r) in enumerate(oral_antifungal.head(args.max_show).iterrows(), 1):
        name = get_field(r, "Назва препарату")[:80]
        form = get_field(r, "Лікарська форма")[:80]
        comp = get_field(r, "Склад")[:100]
        print(f"{i:02d}. {name} | {form} | {comp}")

    print("\n=== ЛОР-АНТИСЕПТИКИ В ОРАЛЬНИХ ФОРМАХ (конкуренти) ===")
    print(f"count={len(throat_ants)}")
    for i, (_, r) in enumerate(throat_ants.head(args.max_show).iterrows(), 1):
        name = get_field(r, "Назва препарату")[:80]
        form = get_field(r, "Лікарська форма")[:80]
        comp = get_field(r, "Склад")[:100]
        print(f"{i:02d}. {name} | {form} | {comp}")

    print("\n=== КИСЛОТНІСТЬ/ГАСТРИТ: PPI/H2/антацид/альгінат/сукральфат (контроль) ===")
    print(f"count={len(acid_ok)}")
    for i, (_, r) in enumerate(acid_ok.head(args.max_show).iterrows(), 1):
        name = get_field(r, "Назва препарату")[:80]
        group = get_field(r, "Фармакотерапевтична група")[:80]
        print(f"{i:02d}. {name} | {group}")

    if args.save_csv:
        out = pd.DataFrame({
            "oral_antifungal_ids": oral_antifungal.index.tolist(),
            "throat_antiseptic_ids": throat_ants.index.tolist(),
            "acid_ok_ids": acid_ok.index.tolist(),
        }, index=[0])
        args.save_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.save_csv, index=False)
        print(f"\n[SAVED] ids -> {args.save_csv}")

if __name__ == "__main__":
    main()
