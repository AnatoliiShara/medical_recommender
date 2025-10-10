#!/usr/bin/env python3
import pandas as pd
import re

# Читаємо
df = pd.read_csv('data/dicts/brand_candidates_FILLED.csv')

# Функція перевірки чи target є валідним INN
def is_valid_inn(target):
    if pd.isna(target) or not target:
        return False
    target = str(target).strip()
    
    # Відкидаємо якщо:
    # 1. Тільки цифри/коми/крапки
    if re.match(r'^[\d,.\s\-/]+$', target):
        return False
    # 2. Містить багато ком підряд
    if ',,' in target or ', ,' in target:
        return False
    # 3. Дуже короткий (менше 3 символів)
    if len(target) < 3:
        return False
    # 4. Починається з коми або містить тільки службові слова
    if target.startswith(',') or target in ['component', 'компонент', 'ацелюлярний компонент']:
        return False
    
    return True

# Фільтруємо
print(f"До фільтрації: {len(df)}")
df_clean = df[df['target'].apply(is_valid_inn)]
print(f"Після фільтрації: {len(df_clean)}")
print(f"Видалено: {len(df) - len(df_clean)}")

# Зберігаємо
df_clean.to_csv('data/dicts/brand_candidates_CLEANED.csv', index=False)
print("Збережено: data/dicts/brand_candidates_CLEANED.csv")