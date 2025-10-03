import sys
import pandas as pd
sys.path.append('src')

from search.enhanced_medical_assistant import EnhancedMedicalAssistant, SearchConfig

print("=== ДІАГНОСТИКА ПРОБЛЕМ ПОШУКУ ===")

# Завантажуємо дані
df = pd.read_parquet("/home/anatolii-shara/Documents/scraping_compendium/compendium_all.parquet")
test_df = df.head(20).copy()

assistant = EnhancedMedicalAssistant()

# Тест 1: Перевірка token lengths
print("\n[1] АНАЛІЗ ДОВЖИНИ CHUNKS:")
assistant.build_from_dataframe(test_df, encoder_model=None, medical_chunking=True, max_chunk_tokens=128)  # Зменшуємо

token_lengths = []
for i, passage in enumerate(assistant.passages[:10]):
    if assistant.medical_chunker._tokenizer:
        tokens = len(assistant.medical_chunker._tokenizer.encode(passage))
        token_lengths.append(tokens)
        print(f"Passage {i}: {tokens} токенів, {len(passage)} символів")
    else:
        word_count = len(passage.split())
        print(f"Passage {i}: ~{word_count} слів, {len(passage)} символів")

# Тест 2: Перевірка BM25 токенізації
print(f"\n[2] BM25 ТОКЕНІЗАЦІЯ:")
query = "артеріальна гіпертензія"
from search.enhanced_medical_assistant import _bm25_tokens
tokens = _bm25_tokens(query)
print(f"Запит: '{query}'")
print(f"BM25 токени: {tokens}")

# Тест 3: Пошук препаратів що точно мають гіпертензію
print(f"\n[3] ПРЕПАРАТИ З 'ГІПЕРТЕНЗІЯ' У ПОКАЗАННЯХ:")
hypertension_drugs = test_df[test_df['Показання'].str.contains('гіпертензі', case=False, na=False)]
print(f"Знайдено {len(hypertension_drugs)} препаратів з гіпертензією:")
for idx, drug in hypertension_drugs.iterrows():
    name = drug['Назва препарату']
    indications = drug['Показання'][:100]
    print(f"  - {name}: {indications}...")

# Тест 4: Перевірка чи ці препарати є в passages
print(f"\n[4] ЧИ ПРЕПАРАТИ З ГІПЕРТЕНЗІЄЮ Є В PASSAGES:")
for passage in assistant.passages[:20]:
    if 'гіпертензі' in passage.lower():
        drug_name = assistant.meta[assistant.passages.index(passage)]['name']
        print(f"ЗНАЙДЕНО: {drug_name}")
        print(f"  Text: {passage[:150]}...")
        break
else:
    print("НЕ ЗНАЙДЕНО жодного passage з 'гіпертензі' в перших 20!")

