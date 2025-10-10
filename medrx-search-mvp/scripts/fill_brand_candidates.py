#!/usr/bin/env python3
"""
Скрипт для автоматичного заповнення target у brand_candidates.csv

Використання:
    python scripts/fill_brand_candidates.py
    
    або з будь-якої папки:
    cd /path/to/project
    python scripts/fill_brand_candidates.py

Автоматично знаходить файли у data/dicts/
"""

import pandas as pd
import re
from pathlib import Path


def find_project_root():
    """Знаходить кореневу папку проєкту (де є папка data/)"""
    current = Path(__file__).resolve().parent
    
    # Шукаємо вгору до 5 рівнів
    for _ in range(5):
        if (current / 'data' / 'dicts').exists():
            return current
        current = current.parent
    
    raise FileNotFoundError(
        "Не можу знайти кореневу папку проєкту! "
        "Переконайся, що запускаєш скрипт з правильного місця."
    )


def extract_inn_from_brackets(alias):
    """Витягує INN з дужок у назві препарату"""
    match = re.search(r'\(([^)]+)\)', alias)
    if match:
        inn = match.group(1).lower().strip()
        
        # ВАЖЛИВО: якщо в дужках тільки цифри або коми - це не INN!
        if re.match(r'^[\d,.\s]+$', inn):
            return None
        
        # Очищаємо від зайвих слів
        inn = re.sub(r"['''`]", '', inn)
        inn = re.sub(r'\s+', ' ', inn)
        inn = re.sub(r'\s+tablets?\s*', ' ', inn, flags=re.IGNORECASE)
        inn = re.sub(r'\s+sublingual\s*', ' ', inn, flags=re.IGNORECASE)
        inn = re.sub(r'\s+атм\s*', ' ', inn, flags=re.IGNORECASE)
        inn = re.sub(r'\s+\d+\s*', ' ', inn)
        inn = re.sub(r'["«»]', '', inn)
        inn = re.sub(r'\s+ebewe\s*', ' ', inn, flags=re.IGNORECASE)
        inn = inn.strip()
        
        # Якщо після очистки залишились тільки коми/цифри - пропускаємо
        if not inn or re.match(r'^[\d,.\s]+$', inn):
            return None
            
        return inn if inn else None
    return None


def main():
    # Знаходимо кореневу папку проєкту
    try:
        project_root = find_project_root()
        print(f"📂 Знайдено кореневу папку проєкту: {project_root}")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    # Шляхи до файлів
    input_file = project_root / 'data' / 'dicts' / 'brand_candidates.csv'
    output_file = project_root / 'data' / 'dicts' / 'brand_candidates_FILLED.csv'
    
    # Перевіряємо чи існує вхідний файл
    if not input_file.exists():
        print(f"❌ Файл не знайдено: {input_file}")
        return
    
    # Читаємо файл
    print(f"\n📖 Читаю {input_file.name}...")
    df = pd.read_csv(input_file)
    
    print(f"   Завантажено {len(df)} записів")
    
    # Обробляємо target - замінюємо 'nan' на None
    df['target'] = df['target'].replace('nan', None)
    df['target'] = df['target'].where(pd.notna(df['target']), None)
    
    # Крок 1: Автоматично витягуємо INN з дужок
    print("\n🤖 Крок 1: Автоматичне заповнення з дужок...")
    auto_filled = 0
    
    for idx, row in df.iterrows():
        if pd.isna(row['target']) or row['target'] == '' or row['target'] is None:
            inn = extract_inn_from_brackets(row['alias'])
            if inn:
                df.at[idx, 'target'] = inn
                df.at[idx, 'note'] = ''  # Очищаємо note
                auto_filled += 1
    
    print(f"   ✅ Автоматично заповнено: {auto_filled} записів")
    
    # Крок 2: Ручне заповнення ТОП-препаратів
    print("\n✋ Крок 2: Ручне заповнення ТОП-препаратів...")
    
    manual_fill_map = {
        "диметилсульфоксид": "dimethyl sulfoxide",
        "озельтамівір": "oseltamivir",
        "линезолид": "linezolid",
        "налбуфін": "nalbuphine",
        "декскетопрофен": "dexketoprofen",
        "воріконазол ромфарм": "voriconazole",
        "ворикоцид": "voriconazole",
        "доцетаксел": "docetaxel",
        "ексіб": "etoricoxib",
        "софген в": "sofosbuvir+velpatasvir",
        "фебумакс": "febuxostat",
        "етора": "etoricoxib",
        "еторикоксиб віста": "etoricoxib",
        "еторіакс": "etoricoxib",
        "вабісмо": "faricimab",
        "вориконазол": "voriconazole",
        "коцитаф": "cytarabine",
        "кококсиб": "etoricoxib",
        "доцет концентрат для р ну для інфузій": "docetaxel",
        "моксотенс": "moxonidine",
        "рокуронію бромід калцекс": "rocuronium",
        "зірабев": "bevacizumab",
        "кеторолак лубнифарм": "ketorolac",
        "беспонза": "tixagevimab+cilgavimab",
        "моксанацін": "moxifloxacin",
        "мозіфер": "iron isomaltoside",
        "ганцил": "ganciclovir",
        "ритовір": "ritonavir",
        "лінезолід новофарм": "linezolid",
        "гадолерій": "gadobutrol",
        "моксин": "moxifloxacin"
    }
    
    manual_filled = 0
    for idx, row in df.iterrows():
        if (pd.isna(row['target']) or row['target'] == '' or row['target'] is None) and row['alias'] in manual_fill_map:
            df.at[idx, 'target'] = manual_fill_map[row['alias']]
            df.at[idx, 'note'] = 'manually filled'
            manual_filled += 1
    
    print(f"   ✅ Вручну заповнено: {manual_filled} записів")
    
    # Крок 3: Сортуємо за freq (спадання)
    print("\n📊 Крок 3: Сортування за частотою...")
    df = df.sort_values('freq', ascending=False)
    
    # Фінальна статистика
    total = len(df)
    filled = df['target'].notna().sum()
    unfilled = total - filled
    
    print("\n" + "="*60)
    print("📈 ФІНАЛЬНА СТАТИСТИКА")
    print("="*60)
    print(f"Всього записів:      {total:,}")
    print(f"Заповнено target:    {filled:,} ({filled/total*100:.1f}%)")
    print(f"Без target:          {unfilled:,} ({unfilled/total*100:.1f}%)")
    
    # ТОП-200 статистика
    top200 = df.head(200)
    top200_filled = top200['target'].notna().sum()
    print(f"\nТОП-200 заповнено:   {top200_filled}/200 ({top200_filled/200*100:.1f}%)")
    
    # Зберігаємо повний результат
    print(f"\n💾 Зберігаю результат у {output_file.name}...")
    df.to_csv(output_file, index=False)
    
    print("\n" + "="*60)
    print("✅ ГОТОВО!")
    print("="*60)
    print(f"\n📁 Файл збережено: {output_file}")
    print(f"📍 Повний шлях: {output_file.resolve()}")
    
    # Показуємо приклади заповнених записів
    print("\n📝 Приклади заповнених записів (перші 10):")
    print("-"*80)
    for idx, (_, row) in enumerate(df.head(10).iterrows(), 1):
        target_display = row['target'] if pd.notna(row['target']) else '(порожньо)'
        alias_short = row['alias'][:50] + '...' if len(row['alias']) > 50 else row['alias']
        print(f"{idx:2}. {alias_short:<53} → {target_display}")
    
    print("\n" + "="*60)
    print("🎯 НАСТУПНІ КРОКИ:")
    print("="*60)
    print("1. Перевір файл: cat data/dicts/brand_candidates_FILLED.csv | head -20")
    print("2. Якщо все ок, заміни оригінал:")
    print("   cp data/dicts/brand_candidates.csv data/dicts/brand_candidates_BACKUP.csv")
    print("   mv data/dicts/brand_candidates_FILLED.csv data/dicts/brand_candidates.csv")
    print("\nАБО залиш як є і використовуй _FILLED версію!")


if __name__ == '__main__':
    main()